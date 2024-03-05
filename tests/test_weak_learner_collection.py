from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from snorkel.labeling import labeling_function
from enum import Enum, unique

import pandas as pd
import pytest

from at_nlp.filters.weak_learner_collection import WeakLearners

TEST_MSGS = ["1", "22", "333", "4444", "55555", "666666", "7777777" "hello", "hola"]


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


# Create some primative fns
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


# TODO: Add preprocess calls
class TestAddLabelingFunctions:
    # test_df = pd.DataFrame(
    #     [[0, "test"], [1, "test2"], [2, "csv"], [3, "test3"], [4, "APL"]],
    #     columns=["id", "text"],
    # )

    @pytest.fixture
    def empty_learner_collection(self):
        wl_col = WeakLearners(col_name="text")
        yield wl_col

    # @pytest.fixture
    # def full_stack(self):
    #     ps = PreprocessorStack()
    #     ps.add_multiple(
    #         [(t_preprocessor_fn0, 0), (t_preprocessor_fn1, 1), (t_preprocessor_fn2, 2)])
    #     yield ps

    def test_add_base(self, empty_learner_collection):
        empty_learner_collection.add(lf_fn_ex_01)
        assert len(empty_learner_collection) == 1
        assert empty_learner_collection[0].fn.name == "test_weak_learner_01"

    def test_add_induction(self, empty_learner_collection):
        empty_learner_collection.add(lf_fn_ex_01)
        empty_learner_collection.add(lf_fn_ex_02)
        assert len(empty_learner_collection) == 2
        assert empty_learner_collection[0].fn.name == "test_weak_learner_01"
        assert empty_learner_collection[1].fn.name == "test_weak_learner_02"

    def test_add_multi_lf(self, empty_learner_collection):
        empty_learner_collection.extend([lf_fn_ex_01, lf_fn_ex_02])
        assert len(empty_learner_collection) == 2
        assert empty_learner_collection[0].fn.name == "test_weak_learner_01"
        assert empty_learner_collection[1].fn.name == "test_weak_learner_02"


class TestAddPrimativeFunctions:
    # test_df = pd.DataFrame(
    #     [[0, "test"], [1, "test2"], [2, "csv"], [3, "test3"], [4, "APL"]],
    #     columns=["id", "text"],
    # )

    @pytest.fixture
    def empty_learner_collection(self):
        wl_col = WeakLearners(col_name="text")
        yield wl_col

    # @pytest.fixture
    # def full_stack(self):
    #     ps = PreprocessorStack()
    #     ps.add_multiple(
    #         [(t_preprocessor_fn0, 0), (t_preprocessor_fn1, 1), (t_preprocessor_fn2, 2)])
    #     yield ps

    def test_add_base(self, empty_learner_collection):
        empty_learner_collection.add(pr_fn_ex_01)
        assert len(empty_learner_collection) == 1
        assert empty_learner_collection[0].fn.name == "PR_pr_fn_ex_01"

    def test_add_induction(self, empty_learner_collection):
        empty_learner_collection.add(pr_fn_ex_01)
        empty_learner_collection.add(pr_fn_ex_02)
        assert len(empty_learner_collection) == 2
        assert empty_learner_collection[0].fn.name == "PR_pr_fn_ex_01"
        assert empty_learner_collection[1].fn.name == "PR_pr_fn_ex_02"

    def test_add_multi_pr(self, empty_learner_collection):
        empty_learner_collection.extend([pr_fn_ex_01, pr_fn_ex_02])
        assert len(empty_learner_collection) == 2
        assert empty_learner_collection[0].fn.name == "PR_pr_fn_ex_01"
        assert empty_learner_collection[1].fn.name == "PR_pr_fn_ex_02"


class TestAddSKLearnFunctions:
    # test_df = pd.DataFrame(
    #     [[0, "test"], [1, "test2"], [2, "csv"], [3, "test3"], [4, "APL"]],
    #     columns=["id", "text"],
    # )

    @pytest.fixture
    def empty_learner_collection(self):
        wl_col = WeakLearners(col_name="text")
        yield wl_col

    # @pytest.fixture
    # def full_stack(self):
    #     ps = PreprocessorStack()
    #     ps.add_multiple(
    #         [(t_preprocessor_fn0, 0), (t_preprocessor_fn1, 1), (t_preprocessor_fn2, 2)])
    #     yield ps

    def test_add_base(self, empty_learner_collection):
        empty_learner_collection.add(rf)
        assert len(empty_learner_collection) == 1
        assert empty_learner_collection[0].fn.name == "SK_RandomForestClassifier"

    def test_add_induction(self, empty_learner_collection):
        empty_learner_collection.add(rf)
        empty_learner_collection.add(sv)
        empty_learner_collection.add(mlp)
        assert len(empty_learner_collection) == 3
        assert empty_learner_collection[0].fn.name == "SK_RandomForestClassifier"
        assert empty_learner_collection[1].fn.name == "SK_SVC"
        assert empty_learner_collection[2].fn.name == "SK_MLPClassifier"

    def test_add_multi_sk(self, empty_learner_collection):
        empty_learner_collection.extend([rf, sv, mlp])
        assert len(empty_learner_collection) == 3
        assert empty_learner_collection[0].fn.name == "SK_RandomForestClassifier"
        assert empty_learner_collection[1].fn.name == "SK_SVC"
        assert empty_learner_collection[2].fn.name == "SK_MLPClassifier"
