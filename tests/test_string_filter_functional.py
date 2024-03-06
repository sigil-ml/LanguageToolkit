from pathlib import Path

import pandas as pd
import pytest
from zipfile import ZipFile
from loguru import logger
from at_nlp.filters.string_filter import StringFilter
from snorkel.labeling import labeling_function, LabelingFunction
from enum import Enum, unique
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

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

csv_path = Path("test.csv").absolute()


def pre_fn_ex0(ds: pd.Series, position: int) -> pd.Series:
    r"""Test function for testing CRUD operations"""
    s: str = ds.iat[position]
    ds.iat[position] = s.lower()
    return ds


def pre_fn_ex1(ds: pd.Series, position: int) -> pd.Series:
    r"""Test function for testing CRUD operations"""
    s: str = ds.iat[position]
    ds.iat[position] = s.upper()
    return ds


def pre_fn_ex2(ds: pd.Series, position: int) -> pd.Series:
    r"""Test function for testing CRUD operations"""
    s: str = ds.iat[position]
    ds.iat[position] = s.capitalize()
    return ds

@unique
class FilterResult(Enum):
    """Enumeration of categories for each message"""

    ABSTAIN = -1
    ACTION = 0
    REVIEW = 1
    RECYCLE = 2


# Create some labeling functions
rsrcs = dict(col_name="col_name")


@labeling_function(name="test_weak_learner_01", resources=rsrcs)
def lf_fn_ex_01(series: pd.Series, col_name: str) -> int:
    s = series[col_name]
    if len(s) > 6:
        return FilterResult.ABSTAIN.value
    return FilterResult.RECYCLE.value


@labeling_function(name="test_weak_learner_02", resources=rsrcs)
def lf_fn_ex_02(series: pd.Series, col_name: str) -> int:
    s: str = series[col_name]
    if s.find("3") != -1:
        return FilterResult.ABSTAIN.value
    return FilterResult.RECYCLE.value


# Create some primitive fns
def pr_fn_ex_01(s: str) -> int:
    if s.lower() == s:
        return FilterResult.ABSTAIN.value
    return FilterResult.RECYCLE.value


def pr_fn_ex_02(s: str) -> int:
    if s.upper() == s:
        return FilterResult.RECYCLE.value
    return FilterResult.ABSTAIN.value


# Create some sklearn estimators
rf = RandomForestClassifier()
sv = SVC()
mlp = MLPClassifier()


@pytest.fixture
def empty_filter():
    yield StringFilter()

@pytest.fixture
def full_pre_filter():
    sf = StringFilter()
    sf.add_preprocessor(pre_fn_ex0)
    sf.add_preprocessor(pre_fn_ex1)
    sf.add_preprocessor(pre_fn_ex2)
    yield sf

@pytest.fixture
def full_lf_filter():
    sf = StringFilter()
    sf.add_labeling_function(lf_fn_ex_01)
    sf.add_labeling_function(pr_fn_ex_01)
    sf.add_labeling_function(rf)
    yield sf

@pytest.fixture
def std_filter():
    sf = StringFilter()
    sf.add_preprocessor(csv_path)
    sf.add_preprocessor(pre_fn_ex0)
    sf.add_labeling_function(lf_fn_ex_01)
    sf.add_labeling_function(rf)
    yield sf

@pytest.fixture
def no_learners_filter():
    sf = StringFilter()
    sf.add_preprocessor(csv_path)
    sf.add_preprocessor(pre_fn_ex0)
    sf.add_labeling_function(lf_fn_ex_01)
    yield sf

class TestAddPreprocessor:

    def test_add_empty_csv_default(self, empty_filter):
        empty_filter.add_preprocessor(csv_path)
        assert len(empty_filter.preprocessors) == 1
        assert empty_filter.preprocessors.__dict__["test_data"] is not None
        assert isinstance(empty_filter.preprocessors.__dict__["test_data"], dict)
        assert empty_filter.preprocessors[-1].__name__ == "test_preprocessor"
        assert callable(empty_filter.preprocessors[-1])

    def test_add_full_csv_default(self, full_pre_filter):
        full_pre_filter.add_preprocessor(csv_path)
        assert len(full_pre_filter.preprocessors) == 4
        assert full_pre_filter.preprocessors.__dict__["test_data"] is not None
        assert isinstance(full_pre_filter.preprocessors.__dict__["test_data"], dict)
        assert full_pre_filter.preprocessors[-1].__name__ == "test_preprocessor"
        assert callable(full_pre_filter.preprocessors[-1])

    def test_add_empty_pre(self, empty_filter):
        empty_filter.add_preprocessor(pre_fn_ex0)
        assert len(empty_filter.preprocessors) == 1
        assert empty_filter.preprocessors[-1].__name__ == "pre_fn_ex0"
        assert callable(empty_filter.preprocessors[-1])

    def test_add_full_pre(self, full_pre_filter):
        full_pre_filter.add_preprocessor(pre_fn_ex0)
        assert len(full_pre_filter.preprocessors) == 4
        assert full_pre_filter.preprocessors[-1].__name__ == "pre_fn_ex0"
        assert callable(full_pre_filter.preprocessors[-1])

    def test_add_full_pre_pos(self, full_pre_filter):
        full_pre_filter.add_preprocessor(pre_fn_ex0, 3)
        assert len(full_pre_filter.preprocessors) == 4
        assert full_pre_filter.preprocessors[3].__name__ == "pre_fn_ex0"
        assert callable(full_pre_filter.preprocessors[3])

    def test_add_multiple_empty(self, empty_filter):
        empty_filter.add_preprocessor([pre_fn_ex0, csv_path, pre_fn_ex2])
        assert len(empty_filter.preprocessors) == 3
        assert empty_filter.preprocessors[0].__name__ == "pre_fn_ex0"
        assert empty_filter.preprocessors[1].__name__ == "test_preprocessor"
        assert empty_filter.preprocessors[2].__name__ == "pre_fn_ex2"
        assert all([callable(fn) for fn in empty_filter.preprocessors])

    def test_add_multiple_full(self, full_pre_filter):
        full_pre_filter.add_preprocessor([pre_fn_ex0, csv_path, pre_fn_ex2])
        assert len(full_pre_filter.preprocessors) == 3
        assert full_pre_filter.preprocessors[0].__name__ == "pre_fn_ex0"
        assert full_pre_filter.preprocessors[1].__name__ == "test_preprocessor"
        assert full_pre_filter.preprocessors[2].__name__ == "pre_fn_ex2"
        assert all([callable(fn) for fn in full_pre_filter.preprocessors])

    def test_add_multiple_empty_pos(self, empty_filter):
        empty_filter.add_preprocessor([
            (pre_fn_ex0, 2),
            (csv_path, 0),
            (pre_fn_ex2, 1)
        ])
        assert len(empty_filter.preprocessors) == 3
        assert empty_filter.preprocessors[2].__name__ == "pre_fn_ex0"
        assert empty_filter.preprocessors[0].__name__ == "test_preprocessor"
        assert empty_filter.preprocessors[1].__name__ == "pre_fn_ex2"
        assert all([callable(fn) for fn in empty_filter.preprocessors])

    def test_add_multiple_full_pos(self, full_pre_filter):
        full_pre_filter.add_preprocessor([
            (pre_fn_ex0, 4),
            (csv_path, 5),
            (pre_fn_ex2, 2)
        ])
        assert len(full_pre_filter.preprocessors) == 3
        assert full_pre_filter.preprocessors[4].__name__ == "pre_fn_ex0"
        assert full_pre_filter.preprocessors[5].__name__ == "test_preprocessor"
        assert full_pre_filter.preprocessors[2].__name__ == "pre_fn_ex2"
        assert all([callable(fn) for fn in full_pre_filter.preprocessors])


class TestAddLabelingFunctions:

    def test_add_empty_labeling_fn(self, empty_filter):
        empty_filter.add_labeling_function(lf_fn_ex_01)
        assert len(empty_filter.labeling_fns) == 1
        assert empty_filter.labeling_fns[0].fn.name == "test_weak_learner_01"
        assert callable(empty_filter.labeling_fns[0].fn)

    def test_add_empty_primitive_fn(self, empty_filter):
        empty_filter.add_labeling_function(pr_fn_ex_01)
        assert len(empty_filter.labeling_fns) == 1
        assert empty_filter.labeling_fns[0].fn.__name__ == "pr_fn_ex_01"
        assert callable(empty_filter.labeling_fns[0].fn)

    def test_add_empty_sklearn_estimator(self, empty_filter):
        empty_filter.add_labeling_function(rf)
        assert len(empty_filter.labeling_fns) == 1
        assert empty_filter.labeling_fns[0].fn.__class__ == RandomForestClassifier
        assert callable(empty_filter.labeling_fns[0].fn)

    def test_add_full_labeling_fn(self, full_lf_filter):
        full_lf_filter.add_labeling_function(lf_fn_ex_01)
        assert len(full_lf_filter.labeling_fns) == 4
        assert full_lf_filter.labeling_fns[-1].fn.name == "test_weak_learner_01"
        assert callable(full_lf_filter.labeling_fns[-1].fn)

    def test_add_full_primitive_fn(self, full_lf_filter):
        full_lf_filter.add_labeling_function(pr_fn_ex_01)
        assert len(full_lf_filter.labeling_fns) == 4
        assert full_lf_filter.labeling_fns[-1].fn.__name__ == "PR_pr_fn_ex_01"
        assert callable(full_lf_filter.labeling_fns[-1].fn)

    def test_add_full_sklearn_estimator(self, full_lf_filter):
        full_lf_filter.add_labeling_function(rf)
        assert len(full_lf_filter.labeling_fns) == 4
        assert full_lf_filter.labeling_fns[-1].fn.__class__ == RandomForestClassifier
        assert callable(full_lf_filter.labeling_fns[-1].fn)

    def test_add_empty_multiple(self, empty_filter):
        empty_filter.add_labeling_function([lf_fn_ex_01, pr_fn_ex_01, rf])
        assert len(empty_filter.labeling_fns) == 3
        assert empty_filter.labeling_fns[0].fn.name == "test_weak_learner_01"
        assert empty_filter.labeling_fns[1].fn.name == "PR_pr_fn_ex_01"
        assert empty_filter.labeling_fns[2].fn.name == "SK_RandomForestClassifier"
        assert all([callable(wl.fn) for wl in empty_filter.labeling_fns])

    def test_add_full_multiple(self, full_lf_filter):
        full_lf_filter.add_labeling_function([lf_fn_ex_01, pr_fn_ex_01, rf])
        assert len(full_lf_filter.labeling_fns) == 6
        assert full_lf_filter.labeling_fns[3].fn.name == "test_weak_learner_01"
        assert full_lf_filter.labeling_fns[4].fn.name == "PR_pr_fn_ex_01"
        assert full_lf_filter.labeling_fns[5].fn.name == "SK_RandomForestClassifier"
        assert all([callable(wl.fn) for wl in full_lf_filter.labeling_fns])

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
            _ = empty_filter.train_test_split(test_data, train_size=0.8)

    def test_train_data_1_shuffle(self, empty_filter):
        with pytest.raises(ValueError):
            _ = empty_filter.train_test_split(test_data, train_size=0.8, shuffle=True)

    def test_train_data_invalid_size_1(self, empty_filter):
        with pytest.raises(ValueError):
            _ = empty_filter.train_test_split(test_data, train_size=0, shuffle=True)

    def test_train_data_invalid_size_2(self, empty_filter):
        with pytest.raises(ValueError):
            _ = empty_filter.train_test_split(test_data, train_size=-1, shuffle=True)

    def test_train_data_invalid_size_3(self, empty_filter):
        with pytest.raises(ValueError):
            _ = empty_filter.train_test_split(test_data, train_size=1, shuffle=True)

    def test_train_data_invalid_size_4(self, empty_filter):
        with pytest.raises(ValueError):
            _ = empty_filter.train_test_split(test_data, train_size=2, shuffle=True)

    def test_train_data_invalid_size_5(self, empty_filter):
        with pytest.raises(ValueError):
            _ = empty_filter.train_test_split(test_data, train_size="2", shuffle=True)



class TestFit:

    @pytest.fixture(scope="class")
    def splits(self, std_filter):
        train, test = std_filter.train_test_split(test_data, train_size=0.8)
        yield train, test

    def test_fit_no_learners(self, empty_filter):
        train, test = empty_filter.train_test_split(test_data, train_size=0.8)
        with pytest.raises(ValueError):  # no weak learners
            _ = empty_filter.fit(train, col_name="text")

    def test_fit_no_learners_2(self, no_learners_filter):
        train, test = no_learners_filter.train_test_split(test_data, train_size=0.8)
        with pytest.raises(ValueError):  # no weak learners
            _ = no_learners_filter.fit(train, col_name="text")

    def test_fit_wrong_type(self, std_filter):
        train = [1, 2, 3, 4, 5]
        with pytest.raises(TypeError):
            _ = std_filter.fit(train)

    def test_fit_unsupplied_column_name(self, std_filter, splits):
        train, test = splits
        with pytest.raises(ValueError):
            _ = std_filter.fit(train)

    def test_with_template_miner(self, std_filter, splits):
        train, test = splits
        _ = std_filter.fit(train, col_name="text", template_miner=True)


class TestTransform:

    @pytest.fixture(scope="class")
    def splits(self, std_filter):
        train, test = std_filter.train_test_split(test_data, train_size=0.8)
        yield train, test

    def test_transform_wrong_type(self, std_filter):
        train = [1, 2, 3, 4, 5]
        with pytest.raises(TypeError):
            _ = std_filter.transform(train)

    def test_transform_unsupplied_column_name(self, std_filter, splits):
        train, test = splits
        with pytest.raises(ValueError):
            _ = std_filter.transform(train)



class TestFitTransform:

    @pytest.fixture(scope="class")
    def splits(self, std_filter):
        train, test = std_filter.train_test_split(test_data, train_size=0.8)
        yield train, test

    def test_fit_transform_no_learners(self, empty_filter):
        train, test = empty_filter.train_test_split(test_data, train_size=0.8)
        with pytest.raises(ValueError):
            _ = empty_filter.fit_transform(train, col_name="text")

    def test_fit_transform_no_learners_2(self, no_learners_filter):
        train, test = no_learners_filter.train_test_split(test_data, train_size=0.8)
        with pytest.raises(ValueError):
            _ = no_learners_filter.fit_transform(train, col_name="text")

    def test_fit_transform_wrong_type(self, std_filter):
        train = [1, 2, 3, 4, 5]
        with pytest.raises(TypeError):
            _ = std_filter.fit_transform(train)

    def test_fit_transform_unsupplied_column_name(self, std_filter, splits):
        train, test = splits
        with pytest.raises(ValueError):
            _ = std_filter.fit_transform(train)

    def test_with_template_miner(self, std_filter, splits):
        train, test = splits
        _ = std_filter.fit_transform(train, col_name="text", template_miner=True)

class TestPrintPreprocessors:  # TODO
    pass

class TestPrintWeakLearners:  # TODO
    pass

class TestPrintFilter:  # TODO
    pass

class TestRemovePreprocessor:

    def test_del(self, full_pre_filter):
        del full_pre_filter.preprocessors[1]
        assert len(full_pre_filter.preprocessors) == 2
        assert full_pre_filter.preprocessors[0].__name__ == "pre_fn_ex0"
        assert full_pre_filter.preprocessors[1].__name__ == "pre_fn_ex2"
        assert all([callable(fn) for fn in full_pre_filter.preprocessors])

    def test_remove_via_method(self, full_pre_filter):
        full_pre_filter.remove_preprocessor("pre_fn_ex0")
        assert len(full_pre_filter.preprocessors) == 2
        assert full_pre_filter.preprocessors[0].__name__ == "pre_fn_ex1"
        assert full_pre_filter.preprocessors[1].__name__ == "pre_fn_ex2"
        assert all([callable(fn) for fn in full_pre_filter.preprocessors])

    def test_remove_via_method(self, full_pre_filter):
        full_pre_filter.remove_preprocessor(pre_fn_ex0)
        assert len(full_pre_filter.preprocessors) == 2
        assert full_pre_filter.preprocessors[0].__name__ == "pre_fn_ex1"
        assert full_pre_filter.preprocessors[1].__name__ == "pre_fn_ex2"
        assert all([callable(fn) for fn in full_pre_filter.preprocessors])

    def test_remove_via_method_pos(self, full_pre_filter):
        full_pre_filter.remove_preprocessor(0)
        assert len(full_pre_filter.preprocessors) == 2
        assert full_pre_filter.preprocessors[0].__name__ == "pre_fn_ex1"
        assert full_pre_filter.preprocessors[1].__name__ == "pre_fn_ex2"
        assert all([callable(fn) for fn in full_pre_filter.preprocessors])

    def test_remove_non_existent(self, empty_filter):
        with pytest.raises(ValueError):
            empty_filter.remove_preprocessor("pre_fn_ex0")
        with pytest.raises(ValueError):
            empty_filter.remove_preprocessor(pre_fn_ex0)
        with pytest.raises(ValueError):
            empty_filter.remove_preprocessor(0)

class TestRemoveWeakLearner:

    def test_del(self, full_lf_filter):
        del full_lf_filter.labeling_fns[1]
        assert len(full_lf_filter.labeling_fns) == 2
        assert all([isinstance(fn, LabelingFunction) for fn in full_pre_filter.labeling_fns])

    def test_remove_via_method(self, full_lf_filter):
        full_lf_filter.remove_labeling_function("rf")
        assert len(full_lf_filter.labeling_fns) == 2
        assert all([isinstance(fn, LabelingFunction) for fn in full_pre_filter.labeling_fns])

    def test_remove_non_existent(self, empty_filter):
        with pytest.raises(ValueError):
            empty_filter.remove_labeling_function("rf")

class TestLoad:  # TODO
    pass

class TestSave:  # TODO
    pass

class TestEval:  # TODO
    pass

class TestMetrics:  # TODO
    pass

class TestGetPreprocessor:

    def test_get_by_name(self, full_pre_filter):
        item = full_pre_filter.get_preprocessor("pre_fn_ex0")
        assert callable(item)

    def test_get_by_position(self, full_pre_filter):
        item = full_pre_filter.get_preprocessor(0)
        assert item.name == "pre_fn_ex0"
        assert callable(item)

    def test_wrong_type(self, full_pre_filter):
        with pytest.raises(IndexError):
            full_pre_filter.get_preprocessor(0.1)

    def test_non_existent_name(self, empty_filter):
        with pytest.raises(ValueError):
            empty_filter.get_preprocessor("pre_fn_ex100")

class TestGetWeakLearner:

    def test_get_by_name(self, full_lf_filter):
        item = full_lf_filter.get_labeling_function("test_weak_learner_01")
        assert isinstance(item.fn, LabelingFunction)
        assert item.fn.name == "test_weak_learner_01"
        assert not item.learnable
        assert item.item_type is None

    def test_get_sklearn_estimator(self, full_lf_filter):
        item = full_lf_filter.get_labeling_function("SK_RandomForestClassifier")
        assert isinstance(item.fn, RandomForestClassifier)
        assert item.fn.name == "SK_RandomForestClassifier"
        assert item.learnable
        assert item.item_type == "sklearn"

    def test_get_wrong_type(self, full_lf_filter):
        with pytest.raises(IndexError):
            full_lf_filter.get_labeling_function(0)

    def test_non_existent_name(self, empty_filter):
        with pytest.raises(ValueError):
            empty_filter.get_labeling_function("test_weak_learner_100")