"""Defines the StringFilter class which is used to filter Mattermost messages"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import warnings
import shutil
import time
import traceback
from collections import abc
from functools import singledispatchmethod, partialmethod

# import csv
import uuid

# import inspect
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Callable, TypeAlias, Iterable, SupportsIndex, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaseEnsemble
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    class_likelihood_ratios,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.svm import SVC
from snorkel.labeling import LFAnalysis
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import labeling_function, LabelingFunction
from snorkel.labeling.model import LabelModel

# import rich
from tqdm import tqdm

from language_toolkit.filters.preprocessor_stack import PreprocessorStack
from language_toolkit.filters.weak_learner_collection import WeakLearners, LearnerItem
from language_toolkit.logger import logger

# from loguru import logger as log

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


Preprocessor: TypeAlias = (
    abc.Callable
    | pathlib.Path
    | Iterable[abc.Callable | pathlib.Path]
    | Iterable[tuple[abc.Callable | pathlib.Path, int]]
)

LabelingFunctionItem: TypeAlias = (
    abc.Callable
    | LabelingFunction
    | BaseEstimator
    | Iterable[abc.Callable | LabelingFunction | BaseEstimator]
)


@dataclass
class TrainingResult:
    results: pd.DataFrame | pd.Series
    accuracy: float
    precision: float
    n_correct: int
    n_incorrect: int


class StringFilter:
    def __init__(self, col_name: str):
        self._preprocessors = PreprocessorStack()
        self._labeling_fns = WeakLearners(col_name)
        self._count_vectorizer = None

    def predict(
        self,
        data: pd.DataFrame | pd.Series,
        col_name: Optional[str],
        use_template_miner: Optional[bool] = False,
        memoize: Optional[bool] = False,
        lru_cache_size: Optional[int] = 128,
        return_dataframe: Optional[bool] = False,
        dask_client: Optional[bool] = None,
        dask_scheduling_strategy: Optional[str] = "threads",
    ) -> pd.DataFrame | pd.Series:
        pass

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
    def _transform_template(self, text: str) -> str:
        """Transform a string into a matching template"""
        if not hasattr(self, "template_miner"):
            raise ValueError("Template transformation called without template_miner")
        cluster = self.template_miner.match(text)
        if cluster:
            return cluster.get_template()
        else:
            return text

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

    @singledispatchmethod
    def add_preprocessor(
        self,
        fn: Preprocessor | Iterable[Preprocessor],
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
    def get_preprocessor(self, item) -> Preprocessor:
        raise IndexError("Getters only support strings and indexers")

    @get_preprocessor.register
    def _(self, item: str) -> Preprocessor:
        return self._preprocessors.get(item)

    @get_preprocessor.register
    def _(self, item: SupportsIndex) -> Preprocessor:
        return self._preprocessors[item]

    @remove_preprocessor.register(str)
    @remove_preprocessor.register(abc.Callable)
    def _(self, item) -> None:
        self._preprocessors.remove(item)

    @remove_preprocessor.register
    def _(self, item: SupportsIndex) -> None:
        del self._preprocessors[item]

    """
    +--------------------------------------------------------------------------------+
    | Labeling Function CR_D                                                         |
    +--------------------------------------------------------------------------------+
    """

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
    def get_labeling_function(self, item) -> LearnerItem:
        raise IndexError("Expected strings or an object which supports indexing!")

    @get_labeling_function.register
    def _(self, item: str) -> LearnerItem:
        return self._labeling_fns.get(item)

    @get_labeling_function.register
    def _(self, item: SupportsIndex) -> LearnerItem:
        return self._labeling_fns[item]

    def remove_labeling_function(self, item: str) -> None:
        self._labeling_fns.remove(item)

    """
    +--------------------------------------------------------------------------------+
    | Training                                                                       |
    +--------------------------------------------------------------------------------+
    """

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

    def _fit_template_miner(self, data: pd.DataFrame | pd.Series):
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
                    logger.info(f"Loading default drain3 config!")
                    default_drain3_config_path = file.absolute()
                    new_drain3_config_path = Path("./drain3.ini")
                    shutil.copy(default_drain3_config_path, new_drain3_config_path)
                    break
            if not found_drain_config:
                raise ValueError(
                    "Cannot find example drain3.ini! Suggest redownloading the toolkit."
                )

        match data.__class__.__name__:
            case "DataFrame":
                for log_line in tqdm(data.itertuples()):
                    _ = self.template_miner.add_log_message(log_line[2])
            case "Series":
                for log_line in tqdm(data.items()):
                    _ = self.template_miner.add_log_message(log_line[1])
            case _:
                raise ValueError(f"Cannot train on dtype: {data.__class__.__name__}")

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
        training_data: pd.DataFrame | pd.Series,
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

        if not self._count_vectorizer:
            self._count_vectorizer = CountVectorizer()

        def pass_by_df() -> str:
            nonlocal training_data
            nonlocal train_col
            nonlocal target_col
            nonlocal target_values

            if isinstance(training_data, pd.DataFrame) and train_col and target_col:
                return "frame"
            elif (
                isinstance(training_data, pd.Series)
                and isinstance(target_values, pd.Series)
                and not (train_col and target_col)
            ):
                return "series"
            elif isinstance(training_data, pd.DataFrame) and not (
                train_col and target_col
            ):
                raise ValueError(
                    "DataFrame provided, but missing train_col or target_col"
                )
            elif isinstance(training_data, pd.DataFrame) and isinstance(
                target_values, pd.Series
            ):
                raise ValueError(
                    "Received DataFrame for training and target_values! "
                    "Please use column names if working with DataFrames!"
                )
            elif (
                isinstance(training_data, pd.Series)
                and isinstance(target_values, pd.Series)
                and target_col
                and train_col
            ):
                warnings.warn(
                    "Provided two series but also column names. "
                    "When working with series column names are not needed.",
                    RuntimeWarning,
                )
                return "series"

        X: pd.Series = (
            training_data[train_col] if pass_by_df() == "frame" else training_data
        )
        y: pd.Series = (
            training_data[target_col] if pass_by_df() == "frame" else target_values
        )

        self._count_vectorizer.fit(X)
        if template_miner:
            self._fit_template_miner(X)
            X = X.apply(self._transform_template)

        X_vec = self._vectorize(X)

        # context manager goes here
        training_metrics = self._train_weak_learners(X_vec, y)

        import pprint

        pprint.pprint(training_metrics)

        return TrainingResult(
            results=X, accuracy=0.0, precision=0.0, n_correct=0, n_incorrect=0
        )

    # TODO: Resolve these issues:

    # Problem 1
    #
    # SKLearn does not stream training metrics during the fit() call.
    # The only way to capture these metrics is via reading the std_out during fit().
    # We can capture standard out by using a context manager with IO.ReadStream
    # https://stackoverflow.com/questions/44443479/python-sklearn-show-loss-values-during-training

    def _train_weak_learners(self, X: np.ndarray, y: pd.Series) -> dict:
        training_results = {}

        def learn(item):
            logger.info(f"Training weak learner: {item.fn.name}")
            if item.item_type == "sklearn":
                _f = self._labeling_fns.m_learners[item.fn.name]
                _f.fit(X, y)
                y_pred = self.invoke_sklearn(_f, X)
                training_results[item.fn.name] = self._get_metrics(y.to_numpy(), y_pred)
            else:
                raise NotImplementedError(f"Type: {item.item_type} not supported")

        for labeling_item in self._labeling_fns:
            if labeling_item.learnable:
                learn(labeling_item)

        return training_results

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
            "class_likelihood_ratio": class_likelihood_ratios(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "log_loss": log_loss(y_true, y_pred),
        }

    #  TODO: Add warning if use_template_miner is false

    # def fit_transform(
    #     self,
    #     training_data: pd.DataFrame | pd.Series,
    #     target_values: Optional[pd.Series] = None,
    #     train_col: Optional[str | int] = None,
    #     target_col: Optional[str | int] = None,
    #     template_miner: Optional[bool] = False,
    #     visualize: Optional[bool] = False,
    # ) -> tuple[pd.DataFrame | pd.Series, TrainingResult]:
    #     pass

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

    def save(self):
        pass

    def load(self):
        pass

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
        table_title = ""
        match classifier_id:
            case "rf":
                table_title = "Random Forest Performance"
                classifier = self.rf
                test_data = self.cv.transform(test_data["Message"])
            case "mlp":
                table_title = "Perceptron Performance"
                classifier = self.mlp
                test_data = self.cv.transform(test_data["Message"])
            case "label_model":
                table_title = "Label Model Performance"
                classifier = self.label_model
            case _:
                classifier = None

        y_pred = classifier.predict(test_data)
        mask = np.array(y_pred == test_labels)

        stats_table = Table(title=table_title, show_lines=True)
        stats_table.add_column("Metric")
        stats_table.add_column("Value")
        stats_table.add_column("Percentage")
        stats_table.add_column("Total")

        num_correct = mask.sum()
        num_incorrect = len(mask) - num_correct
        stats_table.add_row(
            "Correct",
            str(num_correct),
            f"{100 * num_correct / len(mask):.2f}%",
            str(len(mask)),
            style="dark_sea_green4",
        )
        stats_table.add_row(
            "Incorrect",
            str(num_incorrect),
            f"{100 * num_incorrect / len(mask):.2f}%",
            str(len(mask)),
            style="dark_orange",
        )
        console.print(stats_table)

    # def stage_one_train(self, in_data: pd.DataFrame, train_config: dict):
    #     """Train the MLP and RF on the reserved stage one training data"""
    #
    #     # train vectorizer on entire data set
    #     _ = self.cv.fit(in_data["Message"])
    #
    #     # produce test-train split
    #     amt = train_config["amt"]
    #     split_p = train_config["split"]
    #     df = in_data[:amt]
    #     training_mask = np.random.rand(len(df)) < split_p
    #     training_set = df[training_mask]
    #     training_set.iloc[:, 11] = training_set.apply(
    #         self.template_miner_transform, axis=1
    #     )
    #     self.stage_two_train_data = training_set
    #     training_set = self.cv.transform(training_set["Message"])
    #
    #     test_set = df[~training_mask]
    #     test_set.iloc[:, 11] = test_set.apply(self.template_miner_transform, axis=1)
    #     self.stage_one_test_data = test_set
    #
    #     training_labels = df["labels"][training_mask]
    #     test_labels = df["labels"][~training_mask]
    #
    #     # train RF
    #     start_time = time.time()
    #     self.rf.fit(training_set, training_labels)
    #     svm_finish_time = time.time()
    #     elapsed_time = svm_finish_time - start_time
    #     if self.verbose:
    #         console.log(f"RF training complete: elapsed time {elapsed_time}s")
    #     self.evaluate(test_set, test_labels, "rf")
    #
    #     # train MLP
    #     start_time = time.time()
    #     self.mlp.fit(training_set, training_labels)
    #     mlp_finish_time = time.time()
    #     elapsed_time = mlp_finish_time - start_time
    #     if self.verbose:
    #         console.log(f"MLP training complete: elapsed time {elapsed_time}")
    #     self.evaluate(test_set, test_labels, "mlp")
    #
    # def save_template_miner_cluster_information(self) -> None:
    #     """Save template miner clusters to a JSON for analysis"""
    #     clusters = []
    #     for cluster in self.template_miner.drain.clusters:
    #         clusters.append(
    #             {
    #                 "id": cluster.cluster_id,
    #                 "template": cluster.get_template(),
    #                 "size": cluster.size,
    #             }
    #         )
    #         clusters = sorted(clusters, key=lambda x: x["size"], reverse=True)
    #         with open("clusters.json", "w", encoding="utf-8") as f:
    #             json.dump(clusters, f, indent=4)
    #
    # def stage_two_train(self, in_data: pd.DataFrame, train_config: Dict):
    #     """Train the ensemble on the reserved stage two training data"""
    #
    #     amt = train_config["amt"]
    #     split_p = train_config["split"]
    #     df = in_data[-amt:]
    #     training_mask = np.random.rand(len(df)) < split_p
    #     training_set = df[training_mask]
    #     training_set.iloc[:, 11] = training_set.apply(
    #         self.template_miner_transform, axis=1
    #     )
    #     self.stage_two_train_data = training_set
    #     test_set = df[~training_mask]
    #     test_labels = df["labels"][~training_mask]
    #     test_set.iloc[:, 11] = test_set.apply(self.template_miner_transform, axis=1)
    #     self.stage_two_test_data = test_set
    #     test_set = self.applier.apply(test_set)
    #     l_train = self.applier.apply(training_set)
    #     self.label_model = LabelModel(cardinality=7, verbose=True)
    #     start_time = time.time()
    #     self.label_model.fit(L_train=l_train, n_epochs=500, log_freq=100, seed=123)
    #     label_finish_time = time.time()
    #     elapsed_time = label_finish_time - start_time
    #     if self.verbose:
    #         console.log("Label model training complete: elapsed time %s", elapsed_time)
    #     self.evaluate(test_set, test_labels, "label_model")

    # def predict(self, in_data: pd.DataFrame) -> np.ndarray:
    #     """Predict the labels for a supplied Pandas data frame"""
    #     in_data.loc[:, "Message"] = in_data.apply(self.template_miner_transform, axis=1)
    #     in_data = self.applier.apply(in_data)
    #     return self.label_model.predict(in_data)
    #
    # def train(self, in_data: pd.DataFrame, train_conf: Dict, serialize=False):
    #     """Trains both the first and second stages"""
    #     # Train template miner
    #     self.train_template_miner(in_data)
    #
    #     # Train stage one
    #     stage_one_conf = train_conf["stage-one"]
    #     self.stage_one_train(in_data, stage_one_conf)
    #
    #     # Train stage two
    #     stage_two_conf = train_conf["stage-two"]
    #     self.stage_two_train(in_data, stage_two_conf)
    #
    #     # TODO: Use save models here
    #     if serialize:
    #         self.save_models(Path("./models"))

    def save_models(self, save_path_stub: Path) -> None:
        """Save trained models to directory with a random uuid to prevent collisions"""
        uuid_str = uuid.uuid4().hex
        uuid_str = str(uuid_str)[:4]
        save_path = str(save_path_stub) + uuid_str
        os.makedirs(save_path, exist_ok=True)
        console.log(f"Saving models to {save_path}")
        console.log("================================================================")
        console.log(f"Saving vectorizer to {save_path + '/vectorizer.pkl'}")
        joblib.dump(self.cv, save_path + "/vectorizer.pkl")
        console.log(f"Saving template miner to {save_path + '/template_miner.pkl'}")
        joblib.dump(self.template_miner, save_path + "/template_miner.pkl")
        console.log(f"Saving random forest to {save_path + '/random_forest.pkl'}")
        joblib.dump(self.rf, save_path + "/random_forest.pkl")
        console.log(f"Saving MLP to {save_path + '/mlp.pkl'}")
        joblib.dump(self.mlp, save_path + "/mlp.pkl")
        console.log(f"Saving label model to {save_path + '/label_model.pkl'}")
        joblib.dump(self.label_model, save_path + "/label_model.pkl")
        console.log("================================================================")
        console.log("Finished!")

    def load_models(self, model_dir: Path) -> None:
        """Restore models from a directory"""
        assert (
            model_dir.absolute().exists()
        ), f"Cannot find directory at path: {model_dir}!"
        assert (
            model_dir.absolute().is_dir()
        ), f"Provided path is not a directory: {model_dir}!"

        models = os.listdir(model_dir)
        model_names = [
            "vectorizer.pkl",
            "template_miner.pkl",
            "random_forest.pkl",
            "mlp.pkl",
            "label_model.pkl",
        ]

        for model_name in model_names:
            assert model_name in models, f"Cannot find model at path: {model_dir}!"

        console.log("Models found! Starting restoration...")
        model_dir_path = str(model_dir.absolute())
        model_dir_rel = str(model_dir)

        def msg_factory(m):
            return f"❌ {m} is corrupted and cannot be loaded!"

        console.log("================================================================")
        self.cv = joblib.load(model_dir_path + "/vectorizer.pkl")
        assert isinstance(self.cv, CountVectorizer), msg_factory("Vectorizer")
        console.log(
            f"Loading vectorizer from {model_dir_rel + '/vectorizer.pkl'}... ✅"
        )
        self.template_miner = joblib.load(model_dir_path + "/template_miner.pkl")
        assert isinstance(self.template_miner, TemplateMiner), msg_factory(
            "Template miner"
        )
        console.log(
            f"Loading template miner from {model_dir_rel + '/template_miner.pkl'}... ✅"
        )
        self.rf = joblib.load(model_dir_path + "/random_forest.pkl")
        assert isinstance(self.rf, RandomForestClassifier), msg_factory("Random forest")
        console.log(
            f"Loading random forest from {model_dir_rel + '/random_forest.pkl'}... ✅"
        )
        self.mlp = joblib.load(model_dir_path + "/mlp.pkl")
        assert isinstance(self.mlp, MLPClassifier), msg_factory("MLP")
        console.log(f"Loading MLP from {model_dir_rel + '/mlp.pkl'}... ✅")
        self.label_model = joblib.load(model_dir_path + "/label_model.pkl")
        assert isinstance(self.label_model, LabelModel), msg_factory("Label model")
        console.log(
            f"Loading label model from {model_dir_rel + '/label model.pkl'}... ✅"
        )
        console.log("================================================================")
        console.log("Complete!")
