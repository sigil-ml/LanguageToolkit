"""Defines the StringFilter class which is used to filter Mattermost messages"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import time
import traceback
from collections import abc

# import csv
import uuid

# import inspect
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Callable, TypeAlias, Iterable, SupportsIndex, Optional
from sklearn.base import BaseEstimator
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
from sklearn.svm import SVC
from snorkel.labeling import LFAnalysis
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import labeling_function, LabelingFunction
from snorkel.labeling.model import LabelModel

# import rich
from tqdm import tqdm

from at_nlp.filters.preprocessor_stack import PreprocessorStack
from at_nlp.filters.weak_learner_collection import WeakLearners, LearnerItem

# from loguru import logger as log

console = Console()


def custom_except_hook():
    _, exc_value, _ = sys.exc_info()
    """Custom exception hook to print errors to the console"""
    message = traceback.format_exception(exc_value)
    console.log("".join(message), style="bold red")


sys.excepthook = custom_except_hook


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
    accuracy: float
    precision: float
    n_correct: int
    n_incorrect: int


class StringFilter:
    def __init__(self, col_name: str):
        self._preprocessors = PreprocessorStack()
        self._labeling_fns = WeakLearners(col_name)
        self._count_vectorizer = None

    def add_preprocessor(
        self, fn: Preprocessor, position: Optional[int] = None
    ) -> None:
        pass

    def add_labeling_function(self, fn: LabelingFunctionItem) -> None:
        pass

    def train_test_split(
        self,
        data: pd.DataFrame | pd.Series,
        train_size: float = 0.8,
        shuffle: Optional[bool] = False,
    ) -> tuple:
        r"""Split the data into training and testing sets

        Args:
            data (pd.DataFrame | pd.Series): The data to split
            train_size (float, optional): The size of the training set
            shuffle (bool, optional): Whether to shuffle the data before splitting

        Returns:
            tuple: A tuple containing the training and testing sets

        Example:
            >>> from at_nlp.filters.string_filter import StringFilter
            >>> sf = StringFilter()
            >>> train, test = sf.train_test_split(data, train_size=0.8, shuffle=True)
        """

    def fit(
        self,
        training_data: pd.DataFrame | pd.Series,
        target_values: Optional[pd.Series] = None,
        train_col: Optional[str | int] = None,
        target_col: Optional[str | int] = None,
        template_miner: Optional[bool] = False,
        ensemble_split: Optional[float] = 0.8,
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
            >>> from at_nlp.filters.string_filter import StringFilter
            >>> sf = StringFilter()
            >>> data = ...  # Pandas DataFrame
            >>> test, train = sf.train_test_split(data, train_size=0.8, shuffle=True)
            >>> training_results = sf.fit(
            >>>    train,
            >>>    target_values,
            >>>    template_miner=True,
            >>>)
        """

    def predict(
        self,
        data: pd.DataFrame | pd.Series,
        col_name: Optional[str],
        use_template_miner: Optional[bool] = None,
        memoize: Optional[bool] = False,
        lru_cache_size: Optional[int] = 128,
        return_dataframe: Optional[bool] = False,
        dask_client: Optional[bool] = None,
        dask_scheduling_strategy: Optional[str] = "threads",
    ) -> pd.DataFrame | pd.Series:
        pass

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

    def print_preprocessors(self):
        pass

    def print_labeling_functions(self):
        pass

    def __repr__(self):
        pass

    def remove_preprocessor(self, item: Preprocessor | str | SupportsIndex) -> None:
        pass

    def remove_labeling_function(self, item: str) -> None:
        pass

    def save(self):
        pass

    def load(self):
        pass

    def eval(self):
        pass

    def get_preprocessor(self, item: str | SupportsIndex) -> Preprocessor:
        pass

    def get_labeling_function(self, item: str) -> LearnerItem:
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
            >>> from at_nlp.filters.string_filter import StringFilter
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
            >>> from at_nlp.filters.string_filter import StringFilter
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
            >>> from at_nlp.filters.string_filter import StringFilter
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

    def transform(
        self,
        in_data: np.array,
        pred_fun: MLPClassifier | SVC | RandomForestClassifier,
    ) -> np.array:
        """Generic prediction function that calls the predict method of the supplied callable"""
        y_prob = pred_fun.predict_proba(in_data)
        max_prob = np.max(y_prob)
        if max_prob < self.class_likelihood:
            return self.filter_result.ABSTAIN.value
        return np.argmax(y_prob)

    def train_template_miner(self, in_data: pd.DataFrame) -> None:
        """Train the drain3 template miner first on all available data"""
        for log_line in tqdm(in_data["Message"]):
            _ = self.template_miner.add_log_message(log_line)
        console.log("Template miner training complete!")

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

    def latency_trace(self, test_data: pd.DataFrame | np.array) -> None:
        """Evaluate the inference speed of the classifiers"""

        self.trace_mode = True
        data_quantity = len(test_data)
        self.trace_stack = dict()

        # Drain3
        start_drain = time.perf_counter()
        test_data.loc[:, "Message"] = test_data.apply(
            self.template_miner_transform, axis=1
        )
        end_drain = time.perf_counter()
        drain_time = end_drain - start_drain
        drain_time = drain_time / data_quantity
        self.trace_stack["drain_time"] = drain_time

        # Applier
        start_apply = time.perf_counter()
        test_data = self.applier.apply(test_data)
        end_apply = time.perf_counter()
        apply_time = end_apply - start_apply
        apply_time = apply_time / data_quantity
        self.trace_stack["apply_time"] = apply_time

        # Prediction
        prediction_start = time.perf_counter()
        _ = self.label_model.predict(test_data)
        prediction_end = time.perf_counter()
        prediction_time = prediction_end - prediction_start
        prediction_time = prediction_time / data_quantity
        self.trace_stack["prediction_time"] = prediction_time

        console.log(self.trace_stack)
        self.trace_mode = False
        self.trace_stack = dict()

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

    def print_weak_learner_info(self, l_train):
        """Prints the weak learners collisions, etc."""
        console.log(LFAnalysis(L=l_train, lfs=self.labeling_functions).lf_summary())

    def stage_one_train(self, in_data: pd.DataFrame, train_config: dict):
        """Train the MLP and RF on the reserved stage one training data"""

        # train vectorizer on entire data set
        _ = self.cv.fit(in_data["Message"])

        # produce test-train split
        amt = train_config["amt"]
        split_p = train_config["split"]
        df = in_data[:amt]
        training_mask = np.random.rand(len(df)) < split_p
        training_set = df[training_mask]
        training_set.iloc[:, 11] = training_set.apply(
            self.template_miner_transform, axis=1
        )
        self.stage_two_train_data = training_set
        training_set = self.cv.transform(training_set["Message"])

        test_set = df[~training_mask]
        test_set.iloc[:, 11] = test_set.apply(self.template_miner_transform, axis=1)
        self.stage_one_test_data = test_set

        training_labels = df["labels"][training_mask]
        test_labels = df["labels"][~training_mask]

        # train RF
        start_time = time.time()
        self.rf.fit(training_set, training_labels)
        svm_finish_time = time.time()
        elapsed_time = svm_finish_time - start_time
        if self.verbose:
            console.log(f"RF training complete: elapsed time {elapsed_time}s")
        self.evaluate(test_set, test_labels, "rf")

        # train MLP
        start_time = time.time()
        self.mlp.fit(training_set, training_labels)
        mlp_finish_time = time.time()
        elapsed_time = mlp_finish_time - start_time
        if self.verbose:
            console.log(f"MLP training complete: elapsed time {elapsed_time}")
        self.evaluate(test_set, test_labels, "mlp")

    def save_template_miner_cluster_information(self) -> None:
        """Save template miner clusters to a JSON for analysis"""
        clusters = []
        for cluster in self.template_miner.drain.clusters:
            clusters.append(
                {
                    "id": cluster.cluster_id,
                    "template": cluster.get_template(),
                    "size": cluster.size,
                }
            )
            clusters = sorted(clusters, key=lambda x: x["size"], reverse=True)
            with open("clusters.json", "w", encoding="utf-8") as f:
                json.dump(clusters, f, indent=4)

    def stage_two_train(self, in_data: pd.DataFrame, train_config: Dict):
        """Train the ensemble on the reserved stage two training data"""

        amt = train_config["amt"]
        split_p = train_config["split"]
        df = in_data[-amt:]
        training_mask = np.random.rand(len(df)) < split_p
        training_set = df[training_mask]
        training_set.iloc[:, 11] = training_set.apply(
            self.template_miner_transform, axis=1
        )
        self.stage_two_train_data = training_set
        test_set = df[~training_mask]
        test_labels = df["labels"][~training_mask]
        test_set.iloc[:, 11] = test_set.apply(self.template_miner_transform, axis=1)
        self.stage_two_test_data = test_set
        test_set = self.applier.apply(test_set)
        l_train = self.applier.apply(training_set)
        self.label_model = LabelModel(cardinality=7, verbose=True)
        start_time = time.time()
        self.label_model.fit(L_train=l_train, n_epochs=500, log_freq=100, seed=123)
        label_finish_time = time.time()
        elapsed_time = label_finish_time - start_time
        if self.verbose:
            console.log("Label model training complete: elapsed time %s", elapsed_time)
        self.evaluate(test_set, test_labels, "label_model")

    def predict(self, in_data: pd.DataFrame) -> np.ndarray:
        """Predict the labels for a supplied Pandas data frame"""
        in_data.loc[:, "Message"] = in_data.apply(self.template_miner_transform, axis=1)
        in_data = self.applier.apply(in_data)
        return self.label_model.predict(in_data)

    def train(self, in_data: pd.DataFrame, train_conf: Dict, serialize=False):
        """Trains both the first and second stages"""
        # Train template miner
        self.train_template_miner(in_data)

        # Train stage one
        stage_one_conf = train_conf["stage-one"]
        self.stage_one_train(in_data, stage_one_conf)

        # Train stage two
        stage_two_conf = train_conf["stage-two"]
        self.stage_two_train(in_data, stage_two_conf)

        # TODO: Use save models here
        if serialize:
            self.save_models(Path("./models"))

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
        console.log(f"Loading vectorizer from {model_dir_rel + '/vectorizer.pkl'}... ✅")
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

    @classmethod
    def reset(cls) -> None:
        r"""Reset the class to its default state

        Returns:
            None

        Example:
            >>> sf = StringFilter()
            >>> sf.reset()
        """
        cls._preprocessor_stack = []
