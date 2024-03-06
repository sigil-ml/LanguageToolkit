from pathlib import Path
import zipfile
import pytest
import pandas as pd

from loguru import logger
from at_nlp.filters.string_filter import StringFilter

compressed_test_data_path = Path("./tests/News_Category_Dataset_v3.json.zip")
assert compressed_test_data_path.exists(), "Cannot find test data!"

test_data_path = Path("./tests/News_Category_Dataset_v3.json")
if not test_data_path.exists():
    with zipfile.ZipFile(compressed_test_data_path, 'r') as z:
        z.extractall(Path('./tests/'))

test_data = pd.read_json(test_data_path.absolute(), lines=True)

# Clear old log files
log_path = Path('./tests/tests.log')
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

    # self.training_data = pd

    @pytest.fixture
    def new_filter(self):
        str_filter = StringFilter()
        yield str_filter

    def test_training_data_exists(self):
        assert len(test_data) == 209_527
        logger.info("Training data loaded!")
        logger.info("====================================")
        logger.info(f"Num rows: {len(test_data)}")
        logger.info(f"Num cols: {len(test_data.columns)}")
        logger.info(f"Columns: {test_data.columns}")
        logger.info("====================================")


    # def test_train_string_filter(self, new_filter):
    #     new_filter.fit()






#
#
# TEST_MSGS = ["1", "22", "333", "4444", "55555", "666666", "7777777" "hello", "hola"]
#
# # acronyms_path = Path("../../../nitmre/data/acronyms.csv")
# # assert acronyms_path.exists(), "Cannot find acronyms data"
# # drain3_conf = Path(
# #     "/Users/dalton/dev/SIGIL/natural_language_processing/at_nlp/at_nlp/filters/drain3.ini"
# # )
# # assert drain3_conf.exists(), "Cannot find drain3.ini"
#
# # test_data = pd.DataFrame(TEST_MSGS, columns=["Message"])
# # sf = StringFilter()
# # sf.load_models(
# #     Path(
# #         "/Users/dalton/dev/SIGIL/natural_language_processing/at_nlp/at_nlp/filters/test_model9345"
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
