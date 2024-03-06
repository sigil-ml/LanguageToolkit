from pathlib import Path

import pandas as pd
import pytest
from zipfile import ZipFile
from loguru import logger
from at_nlp.filters.string_filter import StringFilter

compressed_test_data_path = Path("./tests/test_data.zip")
assert compressed_test_data_path.exists(), "Cannot find test data!"

test_data_path = Path("./tests/spam.csv")
if not test_data_path.exists():
    with ZipFile(compressed_test_data_path, 'r') as z:
        z.extractall(Path('./tests/'))

test_data = pd.read_csv(test_data_path.absolute(), encoding="ISO-8859-1")
test_data.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
test_data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)

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


def preprocess(s: str) -> int:
    match s:
        case "ham":
            return 0
        case "spam":
            return 2
        case _:
            return -1


test_data["label"] = test_data["label"].apply(preprocess)

@pytest.fixture
def empty_filter():
    yield StringFilter()


class TestTrainTestSplit:

    @pytest.fixture(scope="class")
    def splits(self, empty_filter):
        train, test = empty_filter.train_test_split(test_data, train_size=0.8)
        yield train, test

    def test_split_amt(self, splits):
        test_data_len = len(test_data)
        train, test = splits
        assert int(0.8 * test_data_len) == len(train)
        assert int(0.2 * test_data_len) == len(test)
        assert test_data_len == len(train) + len(test)

    def test_test_data(self, splits):
        _, test = splits
        assert isinstance(test, pd.DataFrame)
        assert test.columns == test_data.columns

    def test_train_data(self, splits):
        train, _ = splits
        assert isinstance(train, pd.DataFrame)
        assert train.columns == test_data.columns

    def test_train_data_shuffle(self, empty_filter, splits):
        train, test = splits
        train_shuffle, test_shuffle = empty_filter.train_test_split(test_data, train_size=0.8, shuffle=True)
        test_data_len = len(test_data)
        assert not train.equals(train_shuffle)
        assert not test.equals(test_shuffle)
        assert int(0.8 * test_data_len) == len(train_shuffle)
        assert int(0.2 * test_data_len) == len(test_shuffle)
        assert test_data_len == len(train_shuffle) + len(test_shuffle)

    def test_train_data_1_normal(self, empty_filter):
        with pytest.raises(ValueError):
            train, test = empty_filter.train_test_split(test_data, train_size=0.8)

    def test_train_data_1_shuffle(self, empty_filter):
        with pytest.raises(ValueError):
            train, test = empty_filter.train_test_split(test_data, train_size=0.8, shuffle=True)

    def test_train_data_invalid_size_1(self, empty_filter):
        with pytest.raises(ValueError):
            train, test = empty_filter.train_test_split(test_data, train_size=0, shuffle=True)

    def test_train_data_invalid_size_2(self, empty_filter):
        with pytest.raises(ValueError):
            train, test = empty_filter.train_test_split(test_data, train_size=-1, shuffle=True)

    def test_train_data_invalid_size_3(self, empty_filter):
        with pytest.raises(ValueError):
            train, test = empty_filter.train_test_split(test_data, train_size=1, shuffle=True)

    def test_train_data_invalid_size_4(self, empty_filter):
        with pytest.raises(ValueError):
            train, test = empty_filter.train_test_split(test_data, train_size=2, shuffle=True)

    def test_train_data_invalid_size_5(self, empty_filter):
        with pytest.raises(ValueError):
            train, test = empty_filter.train_test_split(test_data, train_size="2", shuffle=True)



class TestFit:

    @pytest.fixture(scope="class")
    def splits(self, empty_filter):
        train, test = empty_filter.train_test_split(test_data, 0.8)
        yield train, test

    def test_fit(self, splits, empty_filter):
        train, test = splits
        train_metrics = empty_filter.fit(train)
        assert isinstance(train_metrics, TrainingMetrics)

class TestTransform:
    pass

class TestFitTransform:
    pass

class TestPrintPreprocessors:
    pass

class TestPrintWeakLearners:
    pass

class TestPrintFilter:
    pass

class TestRemovePreprocessor:
    pass

class TestRemoveWeakLearner:
    pass

class TestLoad:
    pass

class TestSave:
    pass

class TestEval:
    pass

class TestMetrics:
    pass

class TestGetPreprocessor:
    pass

class TestGetWeakLearner:
    pass