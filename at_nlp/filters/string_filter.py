"""Defines the StringFilter class which is used to filter Mattermost messages"""
import logging
import os
import sys
# import csv
import uuid
import json
import time
import traceback
# import inspect
from collections import defaultdict
from string import punctuation
from functools import partial
from pathlib import Path
from enum import Enum
from typing import Union, List, Dict, Callable, Optional

import joblib
import pandas as pd
import dask.dataframe as dd
import numpy as np
# import rich
from tqdm import tqdm

# from loguru import logger as log

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from snorkel.labeling import labeling_function, LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from rich.table import Table
from rich.console import Console


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


class StringFilter:
    r"""This is a conceptual class representation of a simple BLE device
    (GATT Server). It is essentially an extended combination of the

    """





    mlp = MLPClassifier(alpha=1, max_iter=1000)
    """Simple MLP classifier for ensemble of weak learners"""
    class_likelihood = 0.6
    """Threshold for class probabilities"""
    salutations = [
        "hello",
        "hola",
        "aloha",
        "mornin",  # noqa
        "ello govna", # noqa
        "good morning",
        "good evening",
        "good night",
        "good <:*:>",
        "hey <:*:>",
        "hi <:*:>",
        "haha <:*:>",
    ]
    """List of salutations to filter"""
    verbose: bool
    """Whether to print diagnostic information to stdout"""
    msg_len_cutoff = 7
    """Number of characters in a message to define a short message"""
    applier: PandasLFApplier
    """Applies functions to Pandas DataFrame"""
    label_model: LabelModel
    """Ensemble of labeling models"""
    drain_config: TemplateMinerConfig
    """Configuration dictionary from drain3.ini file in cur dir"""
    drain_config_path: Path
    """Path to drain3.ini file as pathlib.Path object"""
    template_miner: TemplateMiner
    """Drain3 template miner to convert log messages to cluster templates"""
    stage_one_train_data: pd.DataFrame
    """Data used to train stage one"""
    stage_one_test_data: pd.DataFrame
    """Data used to test stage one"""
    stage_two_train_data: pd.DataFrame
    """Data used to train stage two"""
    stage_two_test_data: pd.DataFrame
    """Data used to test stage two"""
    trace_stack: Dict = defaultdict(lambda: list())
    """Retain performance metrics for each classifier"""
    trace_mode: bool = False
    """Toggle tracing mode"""
    keyword_register: List[str] = []
    """Iterable of keywords to strain"""
    min_str_len: int = None
    """Minimum length of a message"""
    max_str_len: int = None
    """Maximum length of a message"""
    _preprocessor_stack: list[Callable[[pd.Series, int], pd.Series]] = []
    """Call stack of preprocessing functions"""

    # TODO: add source dir as a parameter to make importing the CSV and INI files easier
    def __init__(self, verbose=True, model_path: Path = None):
        self.acronym_mapping: Dict[str, str] = dict()                 # marked for deletion
        """Mapping of acronyms to their meanings from provided CSV"""
        self.filter_result: FilterResult = FilterResult
        """Enumeration of categories for each message"""
        self._labeling_fns: list[LabelingFunction] = []
        self.cv = CountVectorizer()
        """Coverts messages to a sparse matrix of token counts"""
        self.rf = RandomForestClassifier()
        """Random forest classifier for ensemble of weak learners"""


        # assert acronyms_path is not None, "Acronyms CSV is required!"
        # assert (
        #     acronyms_path.exists()
        # ), f"Provided acronyms path: {acronyms_path.absolute()} does not exist!"
        # assert drain_config_path is not None, "Drain3 config file is required!"
        # assert (
        #     drain_config_path.exists()
        # ), f"Provided drain3.ini path: {drain_config_path.absolute()} does not exist!"

        self.verbose = verbose

        # with open(acronyms_path.absolute(), "r", encoding="utf-8-sig") as f:
        #     for line in csv.DictReader(f):
        #         line: dict
        #         self.acronym_mapping[line["Jargon"]] = (
        #             line["Meaning"] + " (" + line["Jargon"] + ")"
        #         )

        # self.drain_config = TemplateMinerConfig()
        # self.drain_config.load(str(drain_config_path))
        # self.drain_config.profiling_enabled = False
        # self.template_miner = TemplateMiner(None, self.drain_config)

        self.keyword_register = []

        @labeling_function()
        def string_keyword_strainer(ds: pd.Series) -> int:
            """
            Return a list of random ingredients as strings.
            """
            pred = self.filter_result.ABSTAIN.value
            for keyword in self.keyword_register:
                if ds["Message"].find(keyword) >= 0:
                    pred = self.filter_result.RECYCLE.value
                    break
            return pred

        @labeling_function()
        def string_len_strainer() -> int:
            """Classify based on the length of the string"""
            pass

        @labeling_function()
        def lf_rf_rbf(ds: pd.Series) -> int:
            """Classify a post using the SVC"""
            start_time = time.time()
            ds_arr = self.vectorize_text(ds)
            ds_arr = self.transform(ds_arr, self.rf)
            end_time = time.time()
            if self.trace_mode:
                total_time = end_time - start_time
                self.trace_stack['rf'] += total_time
            return ds_arr

        @labeling_function()
        def lf_mlp(ds: pd.Series) -> int:
            """Classify a post using the MLP"""
            start_time = time.time()
            ds_arr = self.vectorize_text(ds)
            ds_arr = self.transform(ds_arr, self.mlp)
            end_time = time.time()
            if self.trace_mode:
                total_time = end_time - start_time
                self.trace_stack['mlp'] += total_time
            return ds_arr

        # @labeling_function()
        # def lf_channel(ds: pd.Series) -> int:
        #     """Classify a post as RECYCLE if it's a system message otherwise ABSTAIN"""
        #     start_time = time.time()
        #     msg = ds["Message"]

        #     # TODO: Prune with Mercury
        #     r_val = self.filter_result.ABSTAIN.value
        #     if (
        #         msg.find("joined the channel") >= 0
        #         or msg.find("added to the channel") >= 0
        #     ):
        #         r_val = self.filter_result.RECYCLE.value
        #     end_time = time.time()
        #     if self.trace_mode:
        #         total_time = end_time - start_time
        #         self.trace_stack.append(["channel", f"{total_time:.4f}s"])
        #     return r_val

        @labeling_function()
        def lf_length(ds: pd.Series) -> int:
            start_time = time.time()
            msg = ds["Message"]
            msg_len = len(msg)
            r_val = self.filter_result.ABSTAIN.value
            if msg_len < self.msg_len_cutoff:
                r_val = self.filter_result.RECYCLE.value
            end_time = time.time()
            if self.trace_mode:
                total_time = end_time - start_time
                self.trace_stack['length'] += total_time
            return r_val

        # @labeling_function()
        # def lf_hello(ds: pd.Series) -> int:
        #     """Classify a post as RECYCLE if it's a greeting/farewell otherwise ABSTAIN"""
        #     start_time = time.time()
        #     msg = ds["Message"].lower()
        #     for item in self.salutations:
        #         if msg.find(item.lower()) >= 0:
        #             end_time = time.time()
        #             if self.trace_mode:
        #                 total_time = end_time - start_time
        #                 self.trace_stack.append(["hello", f"{total_time:.4f}s"])
        #             return self.filter_result.RECYCLE.value

        #     end_time = time.time()
        #     if self.trace_mode:
        #         total_time = end_time - start_time
        #         self.trace_stack.append(["hello", f"{total_time:.4f}s"])
        #     return self.filter_result.ABSTAIN.value

        # @labeling_function()
        # def lf_roger(ds: pd.Series) -> int:
        #     """Classify a post as RECYCLE if it's just a confirmation otherwise ABSTAIN"""
        #     start_time = time.time()
        #     msg = ds["Message"].lower()
        #     r_val = self.filter_result.ABSTAIN.value
        #     if msg.find("rgr") >= 0 or msg.find("roger") >= 0:
        #         r_val = self.filter_result.RECYCLE.value
        #     end_time = time.time()
        #     if self.trace_mode:
        #         total_time = end_time - start_time
        #         self.trace_stack.append(["roger", f"{total_time:.4f}s"])
        #     return r_val

        # @labeling_function()
        # def lf_lunch(ds: pd.Series) -> int:
        #     """Classify a post as RECYCLE if it is about food otherwise ABSTAIN"""
        #     start_time = time.time()
        #     msg = ds["Message"].lower()
        #     for item in ["lunch", "dinner", "food"]:
        #         if msg.find(item) >= 0:
        #             end_time = time.time()
        #             if self.trace_mode:
        #                 total_time = end_time - start_time
        #                 self.trace_stack.append(["lunch", f"{total_time:.4f}s"])
        #             return self.filter_result.RECYCLE.value
        #     end_time = time.time()
        #     if self.trace_mode:
        #         total_time = end_time - start_time
        #         self.trace_stack.append(["lunch", f"{total_time:.4f}s"])
        #     return self.filter_result.ABSTAIN.value

        if model_path is not None and model_path.exists():
            console.log("Loading from directory: %s", model_path)
            self.load_models(model_path)

        # self.transform_template = partial(
        #     self.template_miner_transform, tm=self.template_miner
        # )
        # self.labeling_functions = [
        #     lf_rf_rbf,
        #     lf_mlp,
        #     string_keyword_strainer,
        #     lf_length,
        # ]
        # self.update_applier()

    def register_keywords(
        self, keywords: List[str], make_lowercase: bool = True
    ) -> None:
        """Register new keywords to be used in the labeling functions"""
        assert len(keywords) != 0, "No keywords supplied!"
        if make_lowercase:
            keywords = [x.lower() for x in keywords]
        self.keyword_register.extend(keywords)

    def set_string_len_bounds(self, lower_bound: int, upper_bound: int) -> None:
        """Set the lower and upper bounds for the string length labeling function"""
        assert lower_bound < upper_bound, "Lower bound must be less than upper bound!"
        assert lower_bound > 0, "Lower bound must be greater than 0!"
        self.min_str_len = lower_bound
        self.max_str_len = upper_bound

    def update_applier(self):
        """Update applier with new lists"""
        assert len(self.labeling_functions) != 0, "No labeling functions supplied!"
        for fn in self.labeling_functions:
            assert isinstance(
                fn, LabelingFunction
            ), f"{fn.__name__} is not a labeling function!"
        self.applier = PandasLFApplier(self.labeling_functions)

    def register_csv_preprocessor(
        self,
        csv_path: Path,
        search_idx: int = 0,
        replace_idx: int = 1,
        order: Optional[int] = None,
    ) -> None:
        r"""Registers a CSV file to be used for preprocessing. This is different from
        registering a CSV file for weak learning since we replace the strings
        before the weak learners are trained and applied. If you wish to use the CSV
        for weak learning then use the :meth:`register_csv_weak_learner` method instead.

        Note:
            The CSV file will not be serialized when saving the StringFilter object.
            Internally we will store the search and replacement strings in a dictionary
            that will get pickled with the object. Thus, when loading the object the
            CSV file is not necessary.

        Args:
            csv_path (Path): Path to the CSV file.
            search_idx (int): Index of the column containing the string to be replaced (Defaults to 0).
            replace_idx (int): Index of the column containing the replacement string (Defaults to 1).
            order (int): The position in the call stack to place the preprocessor function.
                The default is None which places the caller at the end of the stack.

        Returns:
            None

        Raises:
            AssertionError: raised if the CSV file does not exist, or the indices are not integers.

        Example:
            >>> sf = StringFilter()
            >>> sf.register_csv_preprocessor(Path("replacement_text1.csv"), 0, 1)
            >>> sf.register_csv_preprocessor(Path("replacement_text2.csv"))
        """
        console.log(f"Registering CSV for preprocessing: {csv_path.stem}")
        assert isinstance(csv_path, Path), "CSV path must be a pathlib.Path object!"
        assert csv_path.exists(), f"CSV file {csv_path.absolute()} does not exist!"
        assert search_idx >= 0, "Search index must be greater than or equal to 0!"
        assert replace_idx >= 0, "Replace index must be greater than or equal to 0!"
        assert (
            search_idx != replace_idx
        ), "Search and replace indices must be different!"
        csv_df = pd.DataFrame(pd.read_csv(csv_path))
        num_cols = csv_df.columns.size
        assert search_idx <= num_cols, "Search index out of range!"
        assert replace_idx <= num_cols, "Replace index out of range!"
        assert num_cols > 1, "CSV file must have at least two columns!"
        csv_name = csv_path.stem
        search_col_name = csv_df.columns[search_idx]
        replace_col_name = csv_df.columns[replace_idx]
        csv_df = csv_df.set_index(search_col_name)
        self.__dict__[f"{csv_name}_data"] = csv_df[replace_col_name].to_dict()

        def preprocessor(ds: pd.Series, col_idx: int) -> pd.Series:
            _d: dict = self.__dict__[f"{csv_name}_data"]
            _s: str = ds.iat[col_idx]
            for key, val in _d.items():
                if _s.find(key) != -1:
                    _s = _s.replace(key, val)
            ds.iat[col_idx] = _s
            return ds

        preprocessor.__name__ = f"{csv_name}_{preprocessor.__name__}"
        processor_stack_size = len(self._preprocessor_stack)
        if order is None:
            self.register_preprocessor([(processor_stack_size, preprocessor)])
        else:
            self.register_preprocessor([(order, preprocessor)])
        console.log(
            f"CSV registered successfully at callstack position: {processor_stack_size}!"
        )

    def register_preprocessor(
        self, fns: list[tuple[int, Callable[[pd.Series, int], pd.Series]]]
    ) -> None:
        r"""Registers one or more preprocessing functions.

        Args:
            fns (list[tuple[int, Callable]]): A list of tuples where each tuple contains the index
                in the call stack the function should be placed and the function itself.

        Returns:
            None

        Raises:
            AssertionError: raised if no tuples are supplied, the index is not an integer, or the function is not
                callable.
        """
        assert len(fns) != 0, "No preprocessing functions supplied!"
        for fn_tuple in fns:
            idx = fn_tuple[0]
            _fn = fn_tuple[1]
            assert callable(_fn), "Second element must be a callable function!"
            if idx == -1:
                self._preprocessor_stack.append(_fn)
            else:
                self._preprocessor_stack.insert(idx, _fn)

    def preprocess(
        self,
        df: pd.DataFrame,
        col_idx: int = 0,
        use_multi_processing: bool = False,
        num_processors: int = 2
    ) -> pd.DataFrame:
        r"""Sequentially execute functions in the preprocessor stack"""
        col_name = df.columns[col_idx]
        if use_multi_processing:
            ddf = dd.from_pandas(df, npartitions=num_processors)
            for _fn in self._preprocessor_stack:
                partial_fn = partial(_fn, col_idx=col_idx)
                df = ddf.apply(partial_fn, axis=1, meta=ddf)
        else:
            for _fn in self._preprocessor_stack:
                partial_fn = partial(_fn, col_idx=col_idx)
                df = df.apply(partial_fn, axis=1)
        return df

    def print_preprocessor_stack(self) -> None:
        r"""Prints the current preprocessor callstack."""
        table = Table(title="Preprocessor Callstack", show_lines=True)
        table.add_column("Index")
        table.add_column("Function", style="bold")
        table.add_column("Type")
        for idx, fn in enumerate(self._preprocessor_stack):
            table.add_row(str(idx), fn.__name__, str(type(fn)))
        console.print(table)

    def use_random_forest(self, random_forest: RandomForestClassifier) -> None:
        r"""Sets up the StringFilter to use a provided RandomForest Classifier from sklearn

        Args:
            random_forest (RandomForestClassifier): Sklearn RandomForest Classifier reference

        Returns:
            None

        Example:
            >>> from at_nlp.filters.string_filter import StringFilter
            >>> from sklearn.ensemble import RandomForestClassifier  # noqa
            >>> random_forest = RandomForestClassifier()  # noqa
            >>> string_filter = StringFilter()
            >>> @labeling_function()
            >>> def new_labeling_function(ds: pd.Series) -> pd.Series:
                    x = cv.transform(ds)
                    x =
            >>>
            >>> string_filter.use_random_forest(random_forest)
        """

    def add_labeling_fns(self, labeling_fn_list: list[LabelingFunction]):
        r"""Register functions that perform labeling operations

            Args:
                labeling_fn_list (list[LabelingFunction]): List of Snorkel labeling functions
        """
        if len(labeling_fn_list) == 0:
            logging.warning("No labeling functions supplied, skipping registration")
            exit()
        for fn in fns:
            assert isinstance(
                fn, LabelingFunction
            ), f"{fn.__name__} is not a labeling function!"
            self.labeling_functions.append(fn)
        self.update_applier()

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
            raise ValueError(f"Supplied function must be a Snorkel labeling function; got {type(labeling_fn)}")
        self._labeling_fns.append(labeling_fn)

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
                    logging.warning(f"Labeling function: {labeling_fn} not in the filter")

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

    def replace_acronyms(self, in_text: str) -> str:
        """Replace acronyms with their meaningful names"""
        return " ".join(
            self.acronym_mapping[x.upper()] if x.upper() in self.acronym_mapping else x
            for x in in_text.split()
        )

    def fix_text(self, in_text: str) -> str:
        if in_text == "":
            return ""

        end_of_string = in_text[-1]
        if end_of_string not in punctuation:
            return self.replace_acronyms(in_text)

        i = 0
        end_punctuation = ""
        while end_of_string in punctuation:
            end_of_string = in_text[-1 - i]
            end_punctuation = end_of_string + end_punctuation
            i -= 1

        fixed_text = self.replace_acronyms(in_text.rstrip(punctuation))
        return fixed_text + end_punctuation

    def vectorize_text(self, ds: pd.Series) -> np.array:
        r"""Helper function to vectorize the messages in a pandas Series"""
        msg = ds["Message"]
        ds = [msg]
        ds_arr = self.cv.transform(ds)
        return ds_arr

    def transform(
        self,
        in_data: np.array,
        pred_fun: Union[MLPClassifier, SVC, RandomForestClassifier],
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
        test_data: Union[pd.DataFrame, np.array],
        test_labels: pd.Series,
        classifier_id: str = "rf",
    ) -> None:
        r"""Evaluate trained weak learners.

        """
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

    def latency_trace(self, test_data: Union[pd.DataFrame, np.array]) -> None:
        """Evaluate the inference speed of the classifiers"""

        self.trace_mode = True
        data_quantity = len(test_data)
        self.trace_stack = dict()

        # Drain3
        start_drain = time.perf_counter()
        test_data.loc[:, "Message"] = test_data.apply(self.template_miner_transform, axis=1)
        end_drain = time.perf_counter()
        drain_time = end_drain - start_drain
        drain_time = drain_time / data_quantity
        self.trace_stack['drain_time'] = drain_time

        # Applier
        start_apply = time.perf_counter()
        test_data = self.applier.apply(test_data)
        end_apply = time.perf_counter()
        apply_time = end_apply - start_apply
        apply_time = apply_time / data_quantity
        self.trace_stack['apply_time'] = apply_time

        # Prediction
        prediction_start = time.perf_counter()
        _ = self.label_model.predict(test_data)
        prediction_end = time.perf_counter()
        prediction_time = prediction_end - prediction_start
        prediction_time = prediction_time / data_quantity
        self.trace_stack['prediction_time'] = prediction_time

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

    def stage_one_train(self, in_data: pd.DataFrame, train_config: Dict):
        """Train the MLP and RF on the reserved stage one training data"""

        # train vectorizer on entire data set
        _ = self.cv.fit(in_data["Message"])

        # produce test-train split
        amt = train_config["amt"]
        split_p = train_config["split"]
        df = in_data[:amt]
        training_mask = np.random.rand(len(df)) < split_p
        training_set = df[training_mask]
        training_set.iloc[:, 11] = training_set.apply(self.template_miner_transform, axis=1)
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
        training_set.iloc[:, 11] = training_set.apply(self.template_miner_transform, axis=1)
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
