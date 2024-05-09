"""Defines the StringFilter class which is used to filter Mattermost messages"""

from __future__ import annotations

import logging
import os
import pathlib
import pprint
import shutil
import warnings
from collections import abc
from dataclasses import dataclass

# import csv
# import inspect
from enum import Enum
from functools import partialmethod, singledispatchmethod
from pathlib import Path
from typing import List, Optional, SupportsIndex

import dill
import joblib
import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from rich.console import Console
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaseEnsemble
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from snorkel.labeling import LabelingFunction, PandasLFApplier
from snorkel.labeling.model import LabelModel

# import rich
from tqdm import tqdm

from language_toolkit.filters.labeling_function_collection import (
    LabelFunctionItem,
    LabelingFunctionCollection,
)
from language_toolkit.filters.preprocessor_stack import PreprocessorStack
from language_toolkit.logger import logger

console = Console()
showwarning_ = warnings.showwarning


def showwarning(message, *args, **kwargs):
    logger.opt(depth=2).warning(message)
    showwarning_(message, *args, **kwargs)


warnings.showwarning = showwarning


class FilterResult(Enum):
    """Enumeration of categories for each message"""

    ABSTAIN = -1
    ACTION = 0
    REVIEW = 1
    RECYCLE = 2


# <editor-fold desc="Custom Types">
# Preprocessor: TypeAlias = (
#     abc.Callable
#     | pathlib.Path
#     | Iterable[abc.Callable | pathlib.Path]
#     | Iterable[tuple[abc.Callable | pathlib.Path, int]]
# )
#
# LabelingFunctionItem: TypeAlias = (
#     abc.Callable
#     | LabelingFunction
#     | BaseEstimator
#     | Iterable[abc.Callable | LabelingFunction | BaseEstimator]
# )
# </editor-fold>


@dataclass
class TrainingResult:
    results: pd.DataFrame | pd.Series
    accuracy: float
    precision: float
    n_correct: int
    n_incorrect: int


class StringFilter:
    def __init__(self, train_col: str):
        self.train_col = train_col
        self.train_col_idx = None
        self._preprocessors = PreprocessorStack()
        self._labeling_fns = LabelingFunctionCollection(train_col)
        self._count_vectorizer = None
        self.label_model = None

    def predict(
        self,
        data: pd.DataFrame,
        use_template_miner: Optional[bool] = False,
        memoize: Optional[bool] = False,
        lru_cache_size: Optional[int] = 128,
        dask_client: Optional[bool] = None,
        dask_scheduling_strategy: Optional[str] = "threads",
    ) -> pd.DataFrame | pd.Series:
        data = self._preprocessors(data, self.train_col_idx)

        if use_template_miner:
            data[self.train_col] = data.apply(self._transform_template, axis=1)

        data = self.applier.apply(data)
        predictions = self.label_model.predict(data, self.train_col)
        return predictions

    @staticmethod
    def invoke_sklearn(fn: BaseEstimator | BaseEnsemble, X: np.ndarray) -> np.ndarray:
        """Attempts to call predict or transform on the provided sklearn estimator"""
        if hasattr(fn, "predict"):
            y_pred = fn.predict(X)
        elif hasattr(fn, "transform"):
            y_pred = fn.transform(X)
        else:
            raise ValueError(f"Function: {fn} does not have the correct methods!")
        return y_pred

    @partialmethod
    def _transform_template(self, ds: pd.Series) -> str:
        """Transform a string into a matching template"""
        if not hasattr(self, "template_miner"):
            raise ValueError("Template transformation called without template_miner")
        query_str = ds[self.train_col]
        cluster = self.template_miner.match(query_str)
        if cluster:
            return cluster.get_template()
        else:
            return query_str

    def _vectorize(self, text: pd.Series) -> np.ndarray:
        """Transform a string into a vector of one hot encodings"""
        if not self._count_vectorizer:
            raise ValueError("Count vectorizer cannot be found!")
        return self._count_vectorizer.transform(text)

    """
    +--------------------------------------------------------------------------------+
    | Preprocessors CR_D                                                             |
    +--------------------------------------------------------------------------------+
    """
    # <editor-fold desc="Preprocessor CR_D Methods">

    @singledispatchmethod
    def add_preprocessor(
        self,
        fn,
        position: Optional[int] = None,
    ) -> None:
        raise NotImplementedError("Invalid type for preprocessor")

    @add_preprocessor.register
    def _(self, fn: abc.Callable, position: Optional[int] = -1) -> None:
        self._preprocessors.add(fn, position)

    @add_preprocessor.register
    def _(self, fn: pathlib.Path, position: Optional[int] = -1) -> None:
        self._preprocessors.add_csv_preprocessor(fn, order=position)

    @add_preprocessor.register
    def _(self, fn: abc.Iterable) -> None:
        if isinstance(fn[0], abc.Callable) or isinstance(fn[0], pathlib.Path):
            n_pre = len(self._preprocessors)
            _l = [(f, n_pre + i) for i, f in enumerate(fn)]
            self._preprocessors.add_multiple(_l)
        else:
            self._preprocessors.add_multiple(fn)

    @singledispatchmethod
    def remove_preprocessor(self, item) -> None:
        raise NotImplementedError(
            f"Cannot remove based on type {item.__class__.__name__}"
        )

    @singledispatchmethod
    def get_preprocessor(self, item):
        raise IndexError("Getters only support strings and indexers")

    @get_preprocessor.register
    def _(self, item: str):
        return self._preprocessors.get(item)

    @get_preprocessor.register
    def _(self, item: SupportsIndex):
        return self._preprocessors[item]

    @remove_preprocessor.register(str)
    @remove_preprocessor.register(abc.Callable)
    def _(self, item) -> None:
        self._preprocessors.remove(item)

    @remove_preprocessor.register
    def _(self, item: SupportsIndex) -> None:
        del self._preprocessors[item]

    # </editor-fold>
    """
    +--------------------------------------------------------------------------------+
    | Labeling Function CR_D                                                         |
    +--------------------------------------------------------------------------------+
    """
    # <editor-fold desc="Labeling Functions CR_D Methods">

    @singledispatchmethod
    def add_labeling_function(
        self,
    ) -> None:
        raise NotImplementedError("Invalid type for labeling function")

    @add_labeling_function.register(abc.Callable)
    @add_labeling_function.register(LabelingFunction)
    @add_labeling_function.register(BaseEstimator)
    @add_labeling_function.register(BaseEnsemble)
    def _(self, fn) -> None:
        """Handles the single addition case"""
        self._labeling_fns.add(fn)

    @add_labeling_function.register
    def _(self, fn: abc.Iterable) -> None:
        """Handles the multi-addition case"""
        self._labeling_fns.extend(fn)

    @singledispatchmethod
    def get_labeling_function(self, item) -> LabelFunctionItem:
        raise IndexError("Expected strings or an object which supports indexing!")

    @get_labeling_function.register
    def _(self, item: str) -> LabelFunctionItem:
        return self._labeling_fns.get(item)

    @get_labeling_function.register
    def _(self, item: SupportsIndex) -> LabelFunctionItem:
        return self._labeling_fns[item]

    def remove_labeling_function(self, item: str) -> None:
        self._labeling_fns.remove(item)

    # </editor-fold>
    """
    +--------------------------------------------------------------------------------+
    | Training                                                                       |
    +--------------------------------------------------------------------------------+
    """
    # <editor-fold desc="Training Methods">

    @staticmethod
    def train_test_split(
        data: pd.DataFrame | pd.Series,
        train_size: Optional[float] = 0.8,
        shuffle: Optional[bool] = False,
        seed: Optional[int] = 0,
    ) -> tuple:
        r"""Split the data into training and testing sets. This is a convenience function,
        so you do not need to import the train_test_split function from sklearn.

        Args:
            data (pd.DataFrame | pd.Series): The data to split
            train_size (float, optional): The size of the training set
            shuffle (bool, optional): Whether to shuffle the data before splitting
            seed (int, optional): The seed for the random number generator

        Returns:
            tuple: A tuple containing the training and testing sets

        Example:
            >>> from language_toolkit.filters.string_filter import StringFilter
            >>> sf = StringFilter()
            >>> train, test = sf.train_test_split(data, train_size=0.8, shuffle=True)
        """
        return train_test_split(
            data, train_size=train_size, shuffle=shuffle, random_state=seed
        )

    def _fit_template_miner(self, data: pd.DataFrame):
        """Train the drain3 template miner first on all available data"""
        if not hasattr(self, "template_miner"):
            self.template_miner = TemplateMiner()

        if not Path("./drain3.ini").exists():
            logger.warning("Cannot find a drain3.ini in the current working directory!")
            logger.info("Creating a default drain3.ini file")
            file_gen = Path().glob("**/*.ini")
            found_drain_config = False
            for file in file_gen:
                if file.stem == "drain3":
                    found_drain_config = True
                    logger.info("Loading default drain3 config!")
                    default_drain3_config_path = file.absolute()
                    new_drain3_config_path = Path("./drain3.ini")
                    shutil.copy(default_drain3_config_path, new_drain3_config_path)
                    break
            if not found_drain_config:
                raise ValueError(
                    "Cannot find example drain3.ini! Suggest re-downloading the toolkit."
                )

        # Increment by 1 since itertuples has the index as the first item in the tuple
        for log_line in tqdm(data.itertuples()):
            _ = self.template_miner.add_log_message(log_line[self.train_col_idx + 1])

        logger.info("Template miner training complete!")

    def get_template_miner(self) -> TemplateMiner | None:
        if hasattr(self, "template_miner"):
            return self.template_miner
        else:
            logger.warning("No template miner!")
            return None

    def get_template_miner_clusters(self) -> List[str] | None:
        if hasattr(self, "template_miner"):
            return self.template_miner.drain.clusters
        else:
            logger.warning("No template miner!")
            return None

    def fit(
        self,
        training_data: pd.DataFrame,
        target_values: Optional[pd.Series] = None,
        train_col: Optional[str | int] = None,
        target_col: Optional[str | int] = None,
        template_miner: Optional[bool] = False,
        ensemble_split: Optional[float] = 0.5,
        show_progress_bar: Optional[bool] = False,
        visualize: Optional[bool] = False,
    ) -> TrainingResult:
        r"""Fit the filter to the training data

        Args:
            training_data (pd.DataFrame | pd.Series): The training data to fit the filter
            target_values (pd.Series, optional): The target values for the training data
            train_col (str | int, optional): The name of the column to train on
            target_col (str | int, optional): The name of the target column
            template_miner (bool, optional): Train the template miner on the training data
            ensemble_split (float, optional): The split between the data used to train
                the weak learners and the label model
            show_progress_bar (bool, optional): Show a progress bar while training
            visualize (bool, optional): Visualize the training data

        Returns:
            TrainingResult: A dataclass containing the accuracy, precision, and number of
                correct and incorrect predictions

        Example:
            >>> from language_toolkit.filters.string_filter import StringFilter
            >>> sf = StringFilter()
            >>> data = ...  # Pandas DataFrame
            >>> test, train = sf.train_test_split(data, train_size=0.8, shuffle=True)
            >>> training_results = sf.fit(
            >>>    train,
            >>>    target_values,
            >>>    template_miner=True,
            >>>)
        """

        self.train_col = train_col
        self.train_col_idx = training_data.columns.get_loc(self.train_col)

        # No targets provided
        if not target_col and not target_values:
            raise ValueError("No target column or target values provided!")

        # Ambiguous targets provided
        if target_col and target_values:
            raise ValueError("Both target column and target values provided!")

        training_data = self._preprocessors(training_data, self.train_col_idx)

        # The user may provide a custom count vectorizer, provide default otherwise
        if not self._count_vectorizer:
            self._count_vectorizer = CountVectorizer()

        self._count_vectorizer.fit(training_data[self.train_col])
        self._labeling_fns.m_vectorizer = self._count_vectorizer
        if template_miner:
            self._fit_template_miner(training_data)
            training_data[self.train_col] = training_data.apply(
                self._transform_template, axis=1
            )

        # split the dataset for the ensemble, recommend fewer data for the ensemble
        labels = training_data[target_col] if target_col else target_values
        # split_amt = int(len(training_data) * ensemble_split)
        # train_labeling_fns, label_labeling_fns = (
        #     training_data.iloc[:split_amt],
        #     labels.iloc[:split_amt],
        # )
        # train_ensemble, label_ensemble = (
        #     training_data.iloc[split_amt:],
        #     labels.iloc[split_amt:],
        # )

        # TODO: Should probably move this into the labeling function collection
        train_labeling_fns_vec = self._vectorize(training_data[self.train_col])

        training_metrics = self._train_weak_learners(train_labeling_fns_vec, labels)

        ensemble_train_metrics = self._train_ensemble(training_data, labels)
        training_metrics.update(ensemble_train_metrics)

        pprint.pprint(training_metrics)

        return TrainingResult(
            results=None, accuracy=0.0, precision=0.0, n_correct=0, n_incorrect=0
        )

    # TODO: Resolve these issues:

    # Problem 1
    #
    # SKLearn does not stream training metrics during the fit() call.
    # The only way to capture these metrics is via reading the std_out during fit().
    # We can capture standard out by using a context manager with IO.ReadStream
    # https://stackoverflow.com/questions/44443479/python-sklearn-show-loss-values-during-training

    def _train_weak_learners(self, data: np.ndarray, labels: pd.Series) -> dict:
        training_results = {}
        split_amt = int(data.shape[0] * 0.9)
        train_data, train_labels = (
            data[:split_amt],
            labels.iloc[:split_amt],
        )
        test_data, test_labels = (
            data[split_amt:],
            labels.iloc[split_amt:],
        )

        def learn(lf_item: LabelFunctionItem):
            logger.info(f"Training weak learner: {lf_item.labeling_function.name}")
            if lf_item.type == "sklearn":
                trained_estimator = lf_item.estimator.fit(train_data, train_labels)
                y_pred = self.invoke_sklearn(trained_estimator, test_data)
                training_results[lf_item.labeling_function.name] = self._get_metrics(
                    test_labels.to_numpy(), y_pred
                )
                return trained_estimator
            else:
                raise NotImplementedError(f"Type: {lf_item.type} not supported")

        for item in self._labeling_fns.values():
            if item.learnable:
                item.estimator = learn(item)

        return training_results

    def _train_ensemble(self, data: pd.DataFrame, labels: pd.Series) -> dict:
        if not hasattr(self, "applier"):
            self.applier = PandasLFApplier(lfs=self._labeling_fns.as_list())

        split_amt = int(len(data) * 0.9)
        train_data, train_label = (
            data.iloc[:split_amt],
            labels.iloc[:split_amt],
        )
        test_data, test_label = (
            data.iloc[split_amt:],
            labels.iloc[split_amt:],
        )

        train_label_array = self.applier.apply(train_data)
        test_label_array = self.applier.apply(test_data)
        cardinality = len(self._labeling_fns)
        self.label_model = LabelModel(cardinality=4, verbose=True)
        self.label_model.fit(
            L_train=train_label_array, n_epochs=500, log_freq=100, seed=123
        )
        return {
            "label_model": self._get_metrics(
                test_label.to_numpy(), self.label_model.predict(test_label_array)
            )
        }

    # </editor-fold>
    """
    +--------------------------------------------------------------------------------+
    | Evaluation                                                                     |
    +--------------------------------------------------------------------------------+
    """

    @staticmethod
    def _get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

    def eval(self, test_data: pd.DataFrame, data_col: str, label_col: str) -> dict:
        # test_data_arr = self.applier.apply(test_data)
        predictions = self.predict(test_data, use_template_miner=False)
        return self._get_metrics(test_data[label_col].to_numpy(), predictions[0])

    """
    +--------------------------------------------------------------------------------+
    | Information                                                                    |
    +--------------------------------------------------------------------------------+
    """

    def print_preprocessors(self):
        pass

    def print_labeling_functions(self):
        pass

    def __repr__(self):
        pass

    """
    +--------------------------------------------------------------------------------+
    | Serialization/De-serialization                                                 |
    +--------------------------------------------------------------------------------+
    """

    def save(self, save_path_stub: Path = Path("./model")) -> None:
        """Save trained models to directory with a random uuid to prevent collisions"""
        save_path = str(save_path_stub)
        os.makedirs(save_path, exist_ok=True)
        console.log(f"Saving models to {save_path}")
        console.log("================================================================")
        console.log(f"Saving string filter to {save_path + '/string_filter.pkl'}")
        _string_filter_byte_str = dill.dumps(self)
        joblib.dump(_string_filter_byte_str, save_path + "/string_filter.pkl")
        console.log("================================================================")
        console.log("Finished!")

    @classmethod
    def load(cls, model_dir: Path) -> StringFilter:
        """Restore models from a directory"""
        assert (
            model_dir.absolute().exists()
        ), f"Cannot find directory at path: {model_dir}!"
        assert (
            model_dir.absolute().is_dir()
        ), f"Provided path is not a directory: {model_dir}!"

        models = os.listdir(model_dir)
        model_names = [
            "string_filter.pkl",
        ]

        for model_name in model_names:
            assert model_name in models, f"Cannot find model at path: {model_dir}!"

        console.log("Models found! Starting restoration...")
        model_dir_path = str(model_dir.absolute())
        model_dir_rel = str(model_dir)

        def msg_factory(m):
            return f"❌ {m} is corrupted and cannot be loaded!"

        console.log("================================================================")
        _string_filter_byte_str = joblib.load(model_dir_path + "/string_filter.pkl")
        new_filter = dill.loads(_string_filter_byte_str)
        assert isinstance(new_filter, StringFilter), msg_factory("Vectorizer")
        console.log(
            f"Loading StringFilter from {model_dir_rel + '/string_filter.pkl'}... ✅"
        )
        console.log("================================================================")
        console.log("Complete!")
        return new_filter

    def add_labeling_fn(self, labeling_fn: LabelingFunction) -> None:
        r"""Adds a labeling function to the filter

        Args:
            labeling_fn (LabelingFunction): A Snorkel labeling function to be used in the ensemble. The labeling
                function takes in a Panda's Series and returns an integer representing the label. See the provided
                example for more information.
        Returns:
            None
        Raises:
            ValueError: If the supplied function is not a Snorkel labeling function
        Example:
            >>> from language_toolkit.filters.string_filter import StringFilter
            >>> from snorkel.labeling import LabelingFunction
            >>> sf = StringFilter()
            >>> @labeling_function()
            >>> def lf_example(ds: pd.Series) -> int:
            >>>     # This function will test for string lengths greater than 10
            >>>     col_name = "Test"
            >>>     if len(ds[col_name]) >= 10:
            >>>         return 1
            >>>     return 0
            >>> sf.add_labeling_fn(lf_example)  # noqa
        """
        if not isinstance(labeling_fn, LabelingFunction):
            raise ValueError(
                f"Supplied function must be a Snorkel labeling function; got {type(labeling_fn)}"
            )
        self._labeling_fns.append(labeling_fn)

    # TODO: Finish this function
    def add_multiple_labeling_fns(
        self, labeling_fn_list: list[LabelingFunction]
    ) -> None:
        r"""Convenience function to add multiple labeling functions to the filter

        Args:
            labeling_fn_list (list[LabelingFunction]): List of Snorkel labeling functions to be added

        Returns:
             None

        Raises:


        """
        if len(labeling_fn_list) == 0:
            logging.warning("No labeling functions supplied, skipping registration")
            return
        for fn in labeling_fn_list:
            if not isinstance(fn, LabelingFunction):
                raise ValueError(f"{fn.__name__} is not a labeling function.")
            self.add_labeling_fn(fn)

    def remove_labeling_fn(self, labeling_fn: LabelingFunction) -> None:
        r"""Remove a labeling function from the filter.

        Args:
            labeling_fn (LabelingFunction): Labeling function to remove from the filter

        Returns:
            None

        Raises:
            ValueError: If the labeling function is not in the filter

        Example:
            >>> from language_toolkit.filters.string_filter import StringFilter
            >>> from snorkel.labeling import LabelingFunction
            >>> sf = StringFilter()
            >>> # Define a labeling function
            >>> @labeling_function()
            >>> def lf_example(ds: pd.Series) -> int:
            >>>     # This function will test for string lengths greater than 10
            >>>     col_name = "Test"
            >>>     if len(ds[col_name]) >= 10:
            >>>         return 1
            >>>     return 0
            >>> sf.add_labeling_fn(lf_example)  # noqa
            >>> # Remove the previously added labeling function
            >>> sf.remove_labeling_fn(lf_example)  # noqa
        """
        for existing_fn in self._labeling_fns:
            # Snorkel's labeling function does not have a _eq_ method, fallback to using __name__
            if existing_fn.__name__ == labeling_fn.__name__:
                try:
                    self._labeling_fns.remove(existing_fn)
                except ValueError:
                    logging.warning(
                        f"Labeling function: {labeling_fn} not in the filter"
                    )

    def update_labeling_fn(self, labeling_fn: LabelingFunction) -> None:
        r"""Update an existing labeling function in the filter

        Args:
            labeling_fn (LabelingFunction): Updated labeling function

        Returns:
            None

        Raises:
            ValueError: If the labeling function is not in the filter

        Example:
            >>> from language_toolkit.filters.string_filter import StringFilter
        """
        self.remove_labeling_fn(labeling_fn)
        self.add_labeling_fn(labeling_fn)

    # ========================================================================

    def vectorize_text(self, ds: pd.Series) -> np.array:
        r"""Helper function to vectorize the messages in a pandas Series"""
        msg = ds["Message"]
        ds = [msg]
        ds_arr = self.cv.transform(ds)
        return ds_arr

    @staticmethod
    def template_miner_transform(in_row: pd.Series, tm: TemplateMiner) -> str:
        r"""Helper function to transform messages into their cluster templates"""
        msg = in_row["Message"]
        cluster = tm.match(msg)
        if cluster:
            return cluster.get_template()
        return msg

    def evaluate(
        self,
        test_data: pd.DataFrame | np.array,
        test_labels: pd.Series,
        classifier_id: str = "rf",
    ) -> None:
        r"""Evaluate trained weak learners."""
        # table_title = ""
        # match classifier_id:
        #     case "rf":
        #         table_title = "Random Forest Performance"
        #         classifier = self.rf
        #         test_data = self.cv.transform(test_data["Message"])
        #     case "mlp":
        #         table_title = "Perceptron Performance"
        #         classifier = self.mlp
        #         test_data = self.cv.transform(test_data["Message"])
        #     case "label_model":
        #         table_title = "Label Model Performance"
        #         classifier = self.label_model
        #     case _:
        #         classifier = None
        #
        # y_pred = classifier.predict(test_data)
        # mask = np.array(y_pred == test_labels)
        #
        # stats_table = Table(title=table_title, show_lines=True)
        # stats_table.add_column("Metric")
        # stats_table.add_column("Value")
        # stats_table.add_column("Percentage")
        # stats_table.add_column("Total")
        #
        # num_correct = mask.sum()
        # num_incorrect = len(mask) - num_correct
        # stats_table.add_row(
        #     "Correct",
        #     str(num_correct),
        #     f"{100 * num_correct / len(mask):.2f}%",
        #     str(len(mask)),
        #     style="dark_sea_green4",
        # )
        # stats_table.add_row(
        #     "Incorrect",
        #     str(num_incorrect),
        #     f"{100 * num_incorrect / len(mask):.2f}%",
        #     str(len(mask)),
        #     style="dark_orange",
        # )
        # console.print(stats_table)
