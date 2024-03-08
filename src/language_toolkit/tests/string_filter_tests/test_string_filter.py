r"""
   ______      _             _____ ____            ______        __
  / __/ /_____(_)__  ___ _  / __(_) / /____ ____  /_  __/__ ___ / /_
 _\ \/ __/ __/ / _ \/ _ `/ / _// / / __/ -_) __/   / / / -_|_-</ __/
/___/\__/_/ /_/_//_/\_, / /_/ /_/_/\__/\__/_/     /_/  \__/___/\__/
                   /___/


This file tests the various functions of the StringFilter. Specific tests for the
WeakLearners collection are in test_weak_learner_collection.py, and tests for the
PreprocessorStack are in test_preprocessor_stack.py.

We will test the following combination of functions:

D := 1 if using template miner, 0 otherwise

Preprocessors:
P1 := AcronymExpansion
P2 := Strip Non-ASCII
P3 := Link remover

Weak Learners:
W1 := Profanity Learner                 # Labeling function Test
W2 := sklearn RandomForestClassifier    # sklearn test
W3 := Length Learner                    # Primitive test


There are 2^4 * (2^3-1) = 112 possible combinations of tests. Testing the pre-processors
individually is un-necessary since they are already tested in test_preprocessor_stack.py.
Therefore, there are 2^4 possible configurations of pre-weak learner tests. We do not
consider the case when all the weak learners are turned off, so we subtract one.

We are interested in testing the interaction between different types of pre-processors
and the weak learners. There are three archetypal pre-processors: csv mappings (P1),
symbol-based (P2), and regex-based (P3). We will also test each of the accepted weak
learner types: Snorkel labeling function (W1), Sci-kit Learn (W2), and primitives (W3)
to ensure that they are functioning correctly.


+========================================================+
|                      TEST CHART                        |
+=========+================+==============+==============+
| TEST ID | Template Miner | Preprocessors| Weak Learners|
+---------+----------------+--------------+--------------+
|         |       D        | P1 | P2 | P3 | W1 | W2 | W3 |
+---------+----------------+----+----+----+----+----+----+
| 1       |       0        |  0 |  0 |  0 |  1 |  0 |  0 |
| 2       |       0        |  0 |  0 |  0 |  1 |  1 |  0 |
| 3       |       0        |  0 |  0 |  0 |  1 |  0 |  1 |
| 4       |       0        |  0 |  0 |  0 |  0 |  1 |  1 |
| 5       |       0        |  0 |  0 |  0 |  1 |  1 |  1 |
| 6       |       0        |  0 |  0 |  0 |  0 |  0 |  1 |
| 7       |       0        |  0 |  0 |  0 |  0 |  1 |  0 |
| 9       |       0        |  1 |  0 |  0 |  1 |  0 |  0 |
| 9       |       0        |  1 |  0 |  0 |  1 |  1 |  0 |
| 10      |       0        |  1 |  0 |  0 |  1 |  0 |  1 |
| 11      |       0        |  1 |  0 |  0 |  0 |  1 |  1 |
| 12      |       0        |  1 |  0 |  0 |  1 |  1 |  1 |
| 13      |       0        |  1 |  0 |  0 |  0 |  0 |  1 |
| 14      |       0        |  1 |  0 |  0 |  0 |  1 |  0 |
| 15      |       0        |  0 |  1 |  0 |  1 |  0 |  0 |
| 16      |       0        |  0 |  1 |  0 |  1 |  1 |  0 |
| 17      |       0        |  0 |  1 |  0 |  1 |  0 |  1 |
| 18      |       0        |  0 |  1 |  0 |  0 |  1 |  1 |
| 19      |       0        |  0 |  1 |  0 |  1 |  1 |  1 |
| 20      |       0        |  0 |  1 |  0 |  0 |  0 |  1 |
| 21      |       0        |  0 |  1 |  0 |  0 |  1 |  0 |
| 22      |       0        |  0 |  0 |  1 |  1 |  0 |  0 |
| 23      |       0        |  0 |  0 |  1 |  1 |  1 |  0 |
| 24      |       0        |  0 |  0 |  1 |  1 |  0 |  1 |
| 25      |       0        |  0 |  0 |  1 |  0 |  1 |  1 |
| 26      |       0        |  0 |  0 |  1 |  1 |  1 |  1 |
| 27      |       0        |  0 |  0 |  1 |  0 |  0 |  1 |
| 28      |       0        |  0 |  0 |  1 |  0 |  1 |  0 |
| 29      |       0        |  1 |  1 |  0 |  1 |  0 |  0 |
| 30      |       0        |  1 |  1 |  0 |  1 |  1 |  0 |
| 31      |       0        |  1 |  1 |  0 |  1 |  0 |  1 |
| 32      |       0        |  1 |  1 |  0 |  0 |  1 |  1 |
| 33      |       0        |  1 |  1 |  0 |  1 |  1 |  1 |
| 34      |       0        |  1 |  1 |  0 |  0 |  0 |  1 |
| 35      |       0        |  1 |  1 |  0 |  0 |  1 |  0 |
| 36      |       0        |  1 |  0 |  1 |  1 |  0 |  0 |
| 37      |       0        |  1 |  0 |  1 |  1 |  1 |  0 |
| 38      |       0        |  1 |  0 |  1 |  1 |  0 |  1 |
| 39      |       0        |  1 |  0 |  1 |  0 |  1 |  1 |
| 40      |       0        |  1 |  0 |  1 |  1 |  1 |  1 |
| 41      |       0        |  1 |  0 |  1 |  0 |  0 |  1 |
| 42      |       0        |  1 |  0 |  1 |  0 |  1 |  0 |
| 43      |       0        |  0 |  1 |  1 |  1 |  0 |  0 |
| 44      |       0        |  0 |  1 |  1 |  1 |  1 |  0 |
| 45      |       0        |  0 |  1 |  1 |  1 |  0 |  1 |
| 46      |       0        |  0 |  1 |  1 |  0 |  1 |  1 |
| 47      |       0        |  0 |  1 |  1 |  1 |  1 |  1 |
| 48      |       0        |  0 |  1 |  1 |  0 |  0 |  1 |
| 49      |       0        |  0 |  1 |  1 |  0 |  1 |  0 |
| 50      |       0        |  1 |  1 |  1 |  1 |  0 |  0 |
| 51      |       0        |  1 |  1 |  1 |  1 |  1 |  0 |
| 52      |       0        |  1 |  1 |  1 |  1 |  0 |  1 |
| 53      |       0        |  1 |  1 |  1 |  0 |  1 |  1 |
| 54      |       0        |  1 |  1 |  1 |  1 |  1 |  1 |
| 55      |       0        |  1 |  1 |  1 |  0 |  0 |  1 |
| 56      |       0        |  1 |  1 |  1 |  0 |  1 |  0 |
| 57      |       1        |  0 |  0 |  0 |  1 |  0 |  0 |
| 58      |       1        |  0 |  0 |  0 |  1 |  1 |  0 |
| 59      |       1        |  0 |  0 |  0 |  1 |  0 |  1 |
| 60      |       1        |  0 |  0 |  0 |  0 |  1 |  1 |
| 61      |       1        |  0 |  0 |  0 |  1 |  1 |  1 |
| 62      |       1        |  0 |  0 |  0 |  0 |  0 |  1 |
| 63      |       1        |  0 |  0 |  0 |  0 |  1 |  0 |
| 64      |       1        |  1 |  0 |  0 |  1 |  0 |  0 |
| 65      |       1        |  1 |  0 |  0 |  1 |  1 |  0 |
| 66      |       1        |  1 |  0 |  0 |  1 |  0 |  1 |
| 67      |       1        |  1 |  0 |  0 |  0 |  1 |  1 |
| 68      |       1        |  1 |  0 |  0 |  1 |  1 |  1 |
| 69      |       1        |  1 |  0 |  0 |  0 |  0 |  1 |
| 70      |       1        |  1 |  0 |  0 |  0 |  1 |  0 |
| 71      |       1        |  0 |  1 |  0 |  1 |  0 |  0 |
| 72      |       1        |  0 |  1 |  0 |  1 |  1 |  0 |
| 73      |       1        |  0 |  1 |  0 |  1 |  0 |  1 |
| 74      |       1        |  0 |  1 |  0 |  0 |  1 |  1 |
| 75      |       1        |  0 |  1 |  0 |  1 |  1 |  1 |
| 76      |       1        |  0 |  1 |  0 |  0 |  0 |  1 |
| 77      |       1        |  0 |  1 |  0 |  0 |  1 |  0 |
| 78      |       1        |  0 |  0 |  1 |  1 |  0 |  0 |
| 79      |       1        |  0 |  0 |  1 |  1 |  1 |  0 |
| 80      |       1        |  0 |  0 |  1 |  1 |  0 |  1 |
| 81      |       1        |  0 |  0 |  1 |  0 |  1 |  1 |
| 82      |       1        |  0 |  0 |  1 |  1 |  1 |  1 |
| 83      |       1        |  0 |  0 |  1 |  0 |  0 |  1 |
| 84      |       1        |  0 |  0 |  1 |  0 |  1 |  0 |
| 85      |       1        |  1 |  1 |  0 |  1 |  0 |  0 |
| 86      |       1        |  1 |  1 |  0 |  1 |  1 |  0 |
| 87      |       1        |  1 |  1 |  0 |  1 |  0 |  1 |
| 88      |       1        |  1 |  1 |  0 |  0 |  1 |  1 |
| 89      |       1        |  1 |  1 |  0 |  1 |  1 |  1 |
| 90      |       1        |  1 |  1 |  0 |  0 |  0 |  1 |
| 91      |       1        |  1 |  1 |  0 |  0 |  1 |  0 |
| 92      |       1        |  1 |  0 |  1 |  1 |  0 |  0 |
| 93      |       1        |  1 |  0 |  1 |  1 |  1 |  0 |
| 94      |       1        |  1 |  0 |  1 |  1 |  0 |  1 |
| 95      |       1        |  1 |  0 |  1 |  0 |  1 |  1 |
| 96      |       1        |  1 |  0 |  1 |  1 |  1 |  1 |
| 97      |       1        |  1 |  0 |  1 |  0 |  0 |  1 |
| 98      |       1        |  1 |  0 |  1 |  0 |  1 |  0 |
| 99      |       1        |  0 |  1 |  1 |  1 |  0 |  0 |
| 100     |       1        |  0 |  1 |  1 |  1 |  1 |  0 |
| 101     |       1        |  0 |  1 |  1 |  1 |  0 |  1 |
| 102     |       1        |  0 |  1 |  1 |  0 |  1 |  1 |
| 103     |       1        |  0 |  1 |  1 |  1 |  1 |  1 |
| 104     |       1        |  0 |  1 |  1 |  0 |  0 |  1 |
| 105     |       1        |  0 |  1 |  1 |  0 |  1 |  0 |
| 106     |       1        |  1 |  1 |  1 |  1 |  0 |  0 |
| 107     |       1        |  1 |  1 |  1 |  1 |  1 |  0 |
| 108     |       1        |  1 |  1 |  1 |  1 |  0 |  1 |
| 109     |       1        |  1 |  1 |  1 |  0 |  1 |  1 |
| 110     |       1        |  1 |  1 |  1 |  1 |  1 |  1 |
| 111     |       1        |  1 |  1 |  1 |  0 |  0 |  1 |
| 112     |       1        |  1 |  1 |  1 |  0 |  1 |  0 |
+=========+================+====+====+====+====+====+====+

"""

from pathlib import Path
import zipfile
import pytest
import pandas as pd

from loguru import logger
from at_nlp.filters.string_filter import StringFilter

compressed_test_data_path = Path("./tests/test_data.zip")
assert compressed_test_data_path.exists(), "Cannot find test data!"

test_data_path = Path("../tests/spam.csv")
if not test_data_path.exists():
    with zipfile.ZipFile(compressed_test_data_path, "r") as z:
        z.extractall(Path("../tests/"))

test_data = pd.read_csv(test_data_path.absolute(), encoding="ISO-8859-1")
test_data.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
test_data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)


def preprocess(s: str) -> int:
    match s:
        case "ham":
            return 0
        case "spam":
            return 2
        case _:
            return -1


test_data["label"] = test_data["label"].apply(preprocess)

# Clear old log files
log_path = Path("./tests/tests.log")
logger.info("Cleaning previous test's log files...")
try:
    log_path.unlink()
    logger.info(f"{log_path} has been removed successfully")
except FileNotFoundError:
    logger.error(f"The file {log_path} does not exist")
except PermissionError:
    logger.error(f"Permission denied: unable to delete {log_path}")
except Exception as e:
    logger.error(f"Error occurred: {e}")


class TestTrainStringFilter:
    @pytest.fixture
    def new_filter_wo_template_miner(self):
        str_filter = StringFilter(
            training_col_name="text", label_col_name="label", use_template_miner=False
        )
        yield str_filter

    @pytest.fixture
    def new_filter_template_miner(self):
        str_filter = StringFilter(
            training_col_name="text", label_col_name="label", use_template_miner=True
        )
        yield str_filter

    def test_training_data_exists(self):
        assert len(test_data) == 5_572
        logger.info("Training data loaded!")
        logger.info("====================================")
        logger.info(test_data.head())
        logger.info(f"Num rows: {len(test_data)}")
        logger.info(f"Num cols: {len(test_data.columns)}")
        logger.info(f"Columns: {test_data.columns}")
        logger.info("====================================")

    def test_training_weak_learners_no_templates(self, new_filter_wo_template_miner):
        train, test = new_filter_wo_template_miner.train_test_split(test_data)
        training_results = new_filter_wo_template_miner.fit(train)
        test_results = new_filter_wo_template_miner.eval(test)
        assert test_results.acc > 0.9, f"Training accuracy: {test_results.acc}"

    def test_training_weak_learners_no_templates(self, new_filter_template_miner):
        train, test = new_filter_template_miner.train_test_split(test_data)
        training_results = new_filter_template_miner.fit(train)
        test_results = new_filter_template_miner.eval(test)
        assert test_results.acc > 0.9, f"Training accuracy: {test_results.acc}"

    # def test_train_string_filter(self, new_filter):
    #     new_filter.fit()


#
#
# TEST_MSGS = ["1", "22", "333", "4444", "55555", "666666", "7777777" "hello", "hola"]
#
# # acronyms_path = Path("../../../nitmre/data/acronyms.csv")
# # assert acronyms_path.exists(), "Cannot find acronyms data"
# # drain3_conf = Path(
# #     "/Users/dalton/dev/SIGIL/natural_language_processing/language_toolkit/language_toolkit/filters/drain3.ini"
# # )
# # assert drain3_conf.exists(), "Cannot find drain3.ini"
#
# # test_data = pd.DataFrame(TEST_MSGS, columns=["Message"])
# # sf = StringFilter()
# # sf.load_models(
# #     Path(
# #         "/Users/dalton/dev/SIGIL/natural_language_processing/language_toolkit/language_toolkit/filters/test_model9345"
# #     )
# # )
#
#
# # TODO: Add preprocess calls
# class TestCSVPreprocessor:
#     csv_path = Path("./tests/test.csv").absolute()
#     test_df = pd.DataFrame(
#         [[0, "test"], [1, "test2"], [2, "csv"], [3, "test3"], [4, "APL"]],
#         columns=["id", "text"],
#     )
#     true_df = pd.DataFrame(
#         [
#             [0, "test"],
#             [1, "test2"],
#             [2, "Comma-seperated Values"],
#             [3, "test3"],
#             [4, "A Programming Language"],
#         ],
#         columns=["id", "text"],
#     )
#
#     @pytest.fixture
#     def sf(self):
#         _filter = StringFilter()
#         _filter.reset()
#         yield _filter
#
#     def test_csv_register_full(self, sf):
#         sf.register_csv_preprocessor(self.csv_path, 0, 1)
#         assert sf.test_data is not None, "CSV data not loaded"
#         assert (
#             sf._preprocessor_stack[-1].__name__ == "test_preprocessor"
#         ), "Preprocessor not registered"
#         assert callable(sf._preprocessor_stack[-1]), "Preprocessor not callable"
#
#     def test_csv_register_partial(self, sf):
#         sf.register_csv_preprocessor(self.csv_path)
#         assert sf.test_data is not None, "CSV data not loaded"
#         assert (
#             sf._preprocessor_stack[-1].__name__ == "test_preprocessor"
#         ), "Preprocessor not registered"
#         assert callable(sf._preprocessor_stack[-1]), "Preprocessor not callable"
#
#     def test_csv_partial_ordering_start(self, sf):
#         sf.register_csv_preprocessor(self.csv_path, order=0)
#         assert (
#             sf._preprocessor_stack[0].__name__ == "test_preprocessor"
#         ), "Order is incorrect!"
#
#     def test_csv_partial_ordering_end(self, sf):
#         def l1(x):
#             return x + 1
#
#         def l2(x):
#             return x + 2
#
#         sf.register_preprocessor([(0, l1), (1, l2)])
#         sf.register_csv_preprocessor(self.csv_path, order=-1)
#         assert (
#             sf._preprocessor_stack[-1].__name__ == "test_preprocessor"
#         ), "Order is incorrect!"
#
#     def test_csv_partial_ordering_end2(self, sf):
#         def l1(x):
#             return x + 1
#
#         def l2(x):
#             return x + 2
#
#         sf.register_preprocessor([(0, l1), (1, l2)])
#         sf.register_csv_preprocessor(self.csv_path)
#         assert (
#             sf._preprocessor_stack[-1].__name__ == "test_preprocessor"
#         ), "Order is incorrect!"
#
#     def test_csv_non_int_order(self, sf):
#         with pytest.raises(TypeError):
#             sf.register_csv_preprocessor(self.csv_path, order=0.1)
#
#     def test_csv_non_int_search_idx(self, sf):
#         with pytest.raises(IndexError):
#             sf.register_csv_preprocessor(self.csv_path, search_idx=0.1)
#
#     def test_csv_non_int_replace_idx(self, sf):
#         with pytest.raises(IndexError):
#             sf.register_csv_preprocessor(self.csv_path, replace_idx=0.1)
#
#     def test_csv_non_positive_search_idx(self, sf):
#         with pytest.raises(AssertionError):
#             sf.register_csv_preprocessor(self.csv_path, search_idx=-1)
#
#     def test_csv_non_positive_replace_idx(self, sf):
#         with pytest.raises(AssertionError):
#             sf.register_csv_preprocessor(self.csv_path, replace_idx=-1)
#
#     def test_csv_search_eq_replace_idx(self, sf):
#         with pytest.raises(AssertionError):
#             sf.register_csv_preprocessor(self.csv_path, search_idx=0, replace_idx=0)
#
#     def test_csv_too_large_search_idx(self, sf):
#         with pytest.raises(AssertionError):
#             sf.register_csv_preprocessor(self.csv_path, search_idx=100)
#
#     def test_csv_too_large_replace_idx(self, sf):
#         with pytest.raises(AssertionError):
#             sf.register_csv_preprocessor(self.csv_path, replace_idx=100)
#
#     def test_preprocessor_call_single(self, sf):
#         sf.register_csv_preprocessor(self.csv_path)
#         return_df = sf.preprocess(self.test_df, False, col_idx=1)
#         assert return_df == self.true_df
#
#
# # class TestStringFilterLen:
# #     y_true = np.array([2] * len(TEST_MSGS))
# #     """All items should be recycled besides 7777777"""
#
# #     def test_string_len(self):
# #         y_pred = sf.predict(test_data)
# #         percentage_correct = np.sum(y_pred == self.y_true) / len(self.y_true)
# #         assert (
# #             percentage_correct > 0.9
# #         ), f"String length test failed. Percentage correct: {percentage_correct}"
#
# #     def test_string_bounds(self):
# #         sf.set_string_len_bounds(1, 2)
# #         assert (
# #             sf.min_str_len == 1 and sf.max_str_len == 2
# #         ), "String bounds not set correctly"
#
# #     def test_string_bounds_ordering(self):
# #         with pytest.raises(AssertionError):
# #             sf.set_string_len_bounds(1, 0)
#
# #     def test_string_lowerbound_eq_zero(self):
# #         with pytest.raises(AssertionError):
# #             sf.set_string_len_bounds(0, 1)
#
#
# # class TestRegisterKeywords:
#
# #     def test_no_keywords_provided(self):
# #         with pytest.raises(AssertionError):
# #             sf.register_keywords([])
