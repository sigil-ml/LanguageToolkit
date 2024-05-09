from __future__ import annotations

from collections import abc
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import TypeVar

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaseEnsemble
from snorkel.labeling import LabelingFunction, labeling_function

from language_toolkit.logger import logger
from language_toolkit.utils import get_class_name

_LF = TypeVar("_LF", LabelingFunction, abc.Callable, BaseEstimator)


@dataclass
class LabelFunctionItem:
    labeling_function: LabelingFunction
    estimator: BaseEstimator | BaseEnsemble | None
    learnable: bool
    type: str


class LabelingFunctionCollection:
    """A collection of labeling functions that will be used by Snorkel"""

    def __init__(self, train_col: str):
        self.m_register = {}
        self.m_col_name = None
        self.m_vectorizer = None
        self.m_resources = {"col_name": train_col}

    def register(
        self,
        labeling_fn: LabelingFunction,
        estimator: BaseEstimator | BaseEnsemble = None,
        learnable: bool = False,
        item_type: str | None = None,
    ) -> None:
        logger.trace(f"Registering {labeling_fn.name}")
        self.m_register[labeling_fn.name] = LabelFunctionItem(
            labeling_function=labeling_fn,
            estimator=estimator,
            learnable=learnable,
            type=item_type,
        )

    def create_id(self) -> str:
        """Creates a unique id for the labeling function to stop name collisions"""
        return f"{len(self.m_register) + 1}"

    @singledispatchmethod
    def add(
        self,
        fn: _LF,
        learnable: bool = False,
        item_type: str = None,
    ) -> None:
        r"""Dispatches to one of the following methods:
            1. :meth:`add_labeling_function`
            2. :meth:`add_primitive`
            3. :meth:`add_sklearn`

        Raises:
            TypeError: If the provided fn is not the correct type

        """
        raise TypeError(
            "Supplied function is not an accepted labeling function. Received"
            "{}, but expected either Callable[[pd.Series], int] or a Snorkel"
            "LabelingFunction".format(get_class_name(fn))
        )

    # noinspection GrazieInspection
    @add.register
    def add_labeling_function(self, fn: LabelingFunction) -> None:
        r"""Appends a Snorkel labeling function to the collection. If this function has
        trainable parameters, then ``learnable`` should be `True`. If ``learnable`` is set
        to `True`, then ``item_type`` should be a string representing the class of learner
        ``fn``. For instance, if ``fn`` is a Sci-kit learn model, then ``item_type``
        should be set to ``sklearn``.

        Args:
            fn (LabelingFunction): The function to append.

        Raises:
            None

        Returns:
            None

        Examples:
            >>> from language_toolkit.filters.labeling_function_collection import LabelingFunctionCollection
            >>> from snorkel.labeling import labeling_function
            >>> wl_col = LabelingFunctionCollection()
            >>> resources = {"col_name": "Messages"}
            >>> @labeling_function(name="Example", resources=resources)
            >>> def fn_ex0(series: pd.Series, col_name: str) -> int:
            >>>     if len(series[col_name]) > 6:
            >>>         return 1
            >>>     else:
            >>>         return 0
            >>> wl_col.add(fn_ex0, False)
        """
        self.register(fn, None, False, "labeling_function")

    @add.register
    def add_primitive(self, fn: abc.Callable) -> None:
        """:meth:`add_primitive` creates a labeling function from a simple python
        function ``fn``. The function ``fn`` should accept a `str` and produce an `int`.
        We will wrap this function so that it is a proper Snorkel labeling function.

        Args:
            fn (abc.callable): A callable that has the following signature:
                `fn(s: str) -> int`

        Raises:
            None

        Returns:
            None

        Examples:
            >>> from language_toolkit.filters.labeling_function_collection import LabelingFunctionCollection
            >>> wl_col = LabelingFunctionCollection()
            >>>
            >>> # Ex1
            >>> wl_col.add(lambda s: int(len(s) > 6))
            >>>
            >>> # Ex2
            >>> def primitive_ex2(s: str) -> int:
            >>>     if s.find("some substring to find") == -1:
            >>>         return 0
            >>>     return 1
            >>> wl_col.add(primitive_ex2)

        """
        if fn.__name__ == "<lambda>":
            fn.__name__ = "anon" + self.create_id()

        @labeling_function(name=f"PR_{fn.__name__}", resources=self.m_resources)
        def wrapper(series: pd.Series, col_name: str) -> int:
            s = series[col_name]
            return fn(s)

        self.register(wrapper, None, False, "primitive")

    # noinspection GrazieInspection
    @add.register(BaseEstimator)
    @add.register(BaseEnsemble)
    def add_sklearn(self, estimator):
        r"""Adds an estimator from Sci-kit learn to the collection. Assumes the function
        has trainable parameters. All inputs will be vectorized beforehand.

        Args:
            estimator (BaseEstimator | BaseEnsemble): A valid sklearn estimator

        Raises:
            None

        Returns:
            None

        Examples:
            >>> from language_toolkit.filters.labeling_function_collection import LabelingFunctionCollection
            >>> from sklearn.ensemble import RandomForestClassifier  # noqa
            >>> wl_col = LabelingFunctionCollection()
            >>> rf = RandomForestClassifier(max_depth=2, random_state=0)
            >>> wl_col.add(rf)
        """

        name = f"SK_{get_class_name(estimator)}" + self.create_id()

        # noinspection PyUnresolvedReferences
        def wrapper(series: pd.Series, **kwargs) -> int:
            col_name = kwargs["col_name"]
            s = series[col_name]
            s = self.m_vectorizer.transform([s])
            return self.call_sklearn(estimator)(s)[0]

        _lfn = LabelingFunction(name, wrapper, self.m_resources, [])

        self.register(_lfn, estimator, True, "sklearn")

    def call_sklearn(self, estimator: BaseEstimator | BaseEnsemble):
        if hasattr(estimator, "predict"):
            return estimator.predict
        elif hasattr(estimator, "transform"):
            return estimator.transform
        else:
            raise ValueError(f"Estimator {estimator} does not have a call method")

    def extend(self, fns: abc.Iterable[_LF]) -> None:
        """Extends internal collection with a list of fns. Assumes each function is
        non-trainable unless it subclasses BaseEstimator.
        """
        for fn in fns:
            self.add(fn)

    def remove(self, fn_name: str) -> None:
        r"""Remove a weak learner from the collection.

        Args:
            fn_name (str): Snorkel LabelingFunction name

        Returns:
            None

        Raises:
            ValueError: If the given fn_name is not in the collection

        Example:
            >>> from language_toolkit.filters.labeling_function_collection import LabelingFunctionCollection
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from snorkel.labeling import labeling_function
            >>>
            >>> wl_col = LabelingFunctionCollection()
            >>>
            >>> rf = RandomForestClassifier()
            >>> wl_col.add(rf)
            >>> wl_col.add(lambda s: int(len(s) > 6))
            >>>
            >>> common = {
            >>>     "name": "test_fn",
            >>>     "resources": {
            >>>         "col_name": "Messages"
            >>>     }
            >>> }
            >>> @labeling_function(**common)
            >>> def test_fn_01(series: pd.Series, col_name: str) -> int:
            >>>     s: str = series[col_name]
            >>>     if s == s.lower()
            >>>         return 1
            >>>     return 0
            >>> wl_col.add(test_fn_01)
            >>>
            >>> assert len(wl_col) == 3
            >>> wl_col.remove("test_fn")
            >>> assert len(wl_col) == 2
            >>> wl_col.remove("SK_RandomForestClassifier")
            >>> assert len(wl_col) == 1
            >>> wl_col.remove("PR_anon1")
            >>> assert len(wl_col) == 0
        """
        name_located = False
        for k, v in self.m_register.items():
            if fn_name == k:
                del self.m_register[k]
                name_located = True

        if not name_located:
            raise ValueError(f"Function with name {fn_name} not found")

    def get(self, fn_name: str) -> LabelFunctionItem:
        r"""Returns the first LearnerItem that matches the supplied name.

        Args:
            fn_name (str): The learner to retrieve

        Returns:
            LearnerItem: The first LearnerItem that matches the supplied name. Learner
                items have the following structure:
                1. fn = The labeling function
                2. learnable = Whether the function contains trainable parameters
                3. item_type = The underlying type of the fn

        Raises:
            ValueError: If the function is not in the collection

        Examples:
            >>> from language_toolkit.filters.labeling_function_collection import LabelingFunctionCollection
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from snorkel.labeling import labeling_function
            >>>
            >>> wl_col = LabelingFunctionCollection()
            >>>
            >>> # ex1
            >>> rf = RandomForestClassifier()
            >>> wl_col.add(rf)
            >>> sk_item = wl_col.get("SK_RandomForestClassifier")
            >>> assert isinstance(sk_item.labeling_function, LabelingFunction)
            >>> assert sk_item.labeling_function.name == "SK_RandomForestClassifier"
            >>> assert sk_item.learnable
            >>> assert sk_item.type == "sklearn"
            >>>
            >>> # ex2
            >>> wl_col.add(lambda s: int(len(s) > 6))
            >>> pr_item = wl_col.get('PR_anon2')
            >>> assert isinstance(pr_item.labeling_function, LabelingFunction)
            >>> assert pr_item.labeling_function.name == 'PR_anon2'
            >>> assert not pr_item.learnable
            >>> assert pr_item.type is None
            >>>
            >>> # ex3
            >>> resources = dict(col_name = "Messages")
            >>> @labeling_function(name="test_fn", resources=resources)
            >>> def test_fn_01(series: pd.Series, col_name: str) -> int:
            >>>     s: str = series[col_name]
            >>>     if s == s.lower()
            >>>         return 1
            >>>     return 0
            >>> wl_col.add(test_fn_01)
            >>> lf_item = wl_col.get("test_fn")
            >>> assert isinstance(lf_item.labeling_function, LabelingFunction)
            >>> assert lf_item.labeling_function.name = "test_fn"
            >>> assert not lf_item.learnable
            >>> assert lf_item.type is None

        """
        for k, i in self.m_register.items():
            if fn_name == k:
                return i
        raise ValueError(f"Function with name {fn_name} not found")

    def as_list(self) -> list[LabelingFunction]:
        """Returns the collection as a list of labeling functions"""
        return [v.labeling_function for v in self.m_register.values()]

    def items(self):
        return self.m_register.items()

    def values(self):
        return self.m_register.values()

    def __len__(self):
        return len(self.m_register)
