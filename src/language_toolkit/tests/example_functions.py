import pandas as pd
from language_toolkit.filters.string_filter import FilterResult
from snorkel.labeling import labeling_function
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

resources = dict(col_name="col_name")


@labeling_function(name="test_weak_learner_01", resources=resources)
def lf_fn_ex_01(series: pd.Series, col_name: str) -> int:
    s = series[col_name]
    if len(s) > 6:
        return FilterResult.ABSTAIN.value
    return FilterResult.RECYCLE.value


@labeling_function(name="test_weak_learner_02", resources=resources)
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


# Create some sklearn estimators
rf = RandomForestClassifier()
sv = SVC()
mlp = MLPClassifier()
