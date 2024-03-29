from __future__ import annotations

from functools import singledispatchmethod
from dataclasses import dataclass
from collections import abc
from typing import SupportsIndex

import pandas as pd
from snorkel.labeling import LabelingFunction, labeling_function
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class LearnerItem:
    fn: abc.Callable | LabelingFunction
    learnable: bool
    item_type: str | None


class WeakLearners:
    """A collection of weak learners that will be used by Snorkel"""

    def __init__(self, col_name: str):
        self.m_labeling_fns: list[LearnerItem] = []
        self.m_learners = {}
        self.m_col_name = col_name
        self.m_vectorizer = CountVectorizer()
        self.m_rsrcs = dict(col_name=self.m_col_name)
        self.m_idx = 0

    @singledispatchmethod
    def add(
        self,
        fn: abc.Callable | LabelingFunction | BaseEstimator,
        learnable: bool = False,
        item_type: str | None = None,
    ) -> None:
        r"""Dispatches to one of the following methods:
            1. :meth:`add_labeling_function`
            2. :meth:`add_primative`
            3. :meth:`add_sklearn`

        Raises:
            TypeError: If the provided fn is not the correct type

        """
        raise TypeError(
            "Supplied function is not an accepted labeling function. Received"
            "{}, but expected either Callable[[pd.Series], int] or a Snorkel"
            "LabelingFunction" % fn.__class__.__name__
        )

    # noinspection GrazieInspection
    @add.register
    def add_labeling_function(
        self,
        fn: LabelingFunction,
        learnable: bool = False,
        item_type: str | None = None,
    ) -> None:
        r"""Appends a Snorkel labeling function to the collection. If this function has
        trainable parameters, then ``learnable`` should be `True`. If ``learnable`` is set
        to `True`, then ``item_type`` should be a string representing the class of learner
        ``fn``. For instance, if ``fn`` is a Sci-kit learn model, then ``item_type``
        should be set to ``sklearn``.

        Args:
            fn (LabelingFunction): The function to append.
            learnable (bool): If `True`, the function will be marked as
                trainable and will be included in any training methods.
            item_type (str, optional): Optional parameter indicating the type of the
                supplied function.

        Raises:
            None

        Returns:
            None

        Examples:
            >>> from language_toolkit.filters.weak_learner_collection import WeakLearners
            >>> from snorkel.labeling import labeling_function
            >>> wl_col = WeakLearners()
            >>> resources = {"col_name": "Messages"}
            >>> @labeling_function(name="Example", resources=resources)
            >>> def fn_ex0(series: pd.Series, col_name: str) -> int:
            >>>     if len(series[col_name]) > 6:
            >>>         return 1
            >>>     else:
            >>>         return 0
            >>> wl_col.add(fn_ex0, False)
        """
        item = LearnerItem(fn, learnable=learnable, item_type=item_type)
        self.m_labeling_fns.append(item)
        if learnable:
            self.m_learners[fn.name] = fn

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
            >>> from language_toolkit.filters.weak_learner_collection import WeakLearners
            >>> wl_col = WeakLearners()
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
            fn.__name__ = "anon" + str(len(self) + 1)

        @labeling_function(name=f"PR_{fn.__name__}", resources=self.m_rsrcs)
        def wrapper(series: pd.Series, col_name: str) -> int:
            s = series[col_name]
            return fn(s)

        item = LearnerItem(wrapper, learnable=False, item_type=None)
        self.m_learners.append(item)

    # TODO: This should include a uuid to avoid name collisions
    # noinspection GrazieInspection
    @add.register
    def add_sklearn(self, fn: BaseEstimator):
        r"""Adds an estimator from Sci-kit learn to the collection. Assumes the function
        has trainable parameters. All inputs will be vectorized beforehand.

        Args:
            fn (BaseEstimator): A valid sklearn estimator

        Raises:
            None

        Returns:
            None

        Examples:
            >>> from language_toolkit.filters.weak_learner_collection import WeakLearners
            >>> from sklearn.ensemble import RandomForestClassifier  # noqa
            >>> wl_col = WeakLearners()
            >>> rf = RandomForestClassifier(max_depth=2, random_state=0)
            >>> wl_col.add(rf)
        """

        # noinspection PyUnresolvedReferences
        @labeling_function(name=f"SK_{fn.__class__.__name__}", resources=self.m_rsrcs)
        def wrapper(series: pd.Series, col_name: str) -> int:
            s = series[col_name]
            s = self.m_vectorizer.transform(s)
            return fn.transform(s)

        item = LearnerItem(wrapper, learnable=True, item_type="sklearn")
        self.m_labeling_fns.append(item)
        self.m_learners[f"SK_{fn.__class__.__name__}"] = fn

    def extend(
        self, fns: abc.Iterable[LabelingFunction | abc.Callable | BaseEstimator]
    ) -> None:
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
            >>> from language_toolkit.filters.weak_learner_collection import WeakLearners
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from snorkel.labeling import labeling_function
            >>>
            >>> wl_col = WeakLearners()
            >>>
            >>> rf = RandomForestClassifier()
            >>> wl_col.add(rf)
            >>> wl_col.add(lambda s: int(len(s) > 6))
            >>>
            >>> rscrs = dict(col_name = "Messages")
            >>> @labeling_function(name="test_fn", resources=rscrs)
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
        for idx, item in enumerate(self.m_labeling_fns):
            if fn_name == item.fn.name:
                del self.m_labeling_fns[idx]
                name_located = True

        if not name_located:
            raise ValueError(f"Function with name {fn_name} not found")

    def get(self, fn_name: str) -> LearnerItem:
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
            >>> from language_toolkit.filters.weak_learner_collection import WeakLearners
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from snorkel.labeling import labeling_function
            >>>
            >>> wl_col = WeakLearners()
            >>>
            >>> # ex1
            >>> rf = RandomForestClassifier()
            >>> wl_col.add(rf)
            >>> sk_item = wl_col.get("SK_RandomForestClassifier")
            >>> assert isinstance(sk_item.fn, LabelingFunction)         # True
            >>> assert sk_item.fn.name == "SK_RandomForestClassifier"   # True
            >>> assert sk_item.learnable                                # True
            >>> assert sk_item.item_type == "sklearn"                   # True
            >>>
            >>> # ex2
            >>> wl_col.add(lambda s: int(len(s) > 6))
            >>> pr_item = wl_col.get('PR_anon2')
            >>> assert isinstance(pr_item.fn, LabelingFunction) # True
            >>> assert pr_item.fn.name == 'PR_anon2'            # True
            >>> assert not pr_item.learnable                    # True
            >>> assert pr_item.item_type is None                # True
            >>>
            >>> # ex3
            >>> rscrs = dict(col_name = "Messages")
            >>> @labeling_function(name="test_fn", resources=rscrs)
            >>> def test_fn_01(series: pd.Series, col_name: str) -> int:
            >>>     s: str = series[col_name]
            >>>     if s == s.lower()
            >>>         return 1
            >>>     return 0
            >>> wl_col.add(test_fn_01)
            >>> lf_item = wl_col.get("test_fn")
            >>> assert isinstance(lf_item.fn, LabelingFunction) # True
            >>> assert lf_item.fn.name = "test_fn"              # True
            >>> assert not lf_item.learnable                    # True
            >>> assert lf_item.item_type is None                # True

        """
        for item in self.m_labeling_fns:
            if fn_name == item.fn.name:
                return item
        raise ValueError(f"Function with name {fn_name} not found")

    # TODO: Implement __call__?, train_wl, train_ens, save?, load?, print,
    # TODO: display_train_wl_to_term, display_train_ens_to_term

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.m_labeling_fns)

    def __getitem__(self, item: SupportsIndex) -> LearnerItem:
        if not isinstance(item, SupportsIndex):
            raise TypeError("Collection indices must be integers")
        return self.m_labeling_fns[item]

    def __setitem__(self, item: SupportsIndex, learner_item: LearnerItem) -> None:
        if not isinstance(item, SupportsIndex):
            raise TypeError("Collection indices must be integers")
        self.m_labeling_fns[item] = learner_item

    def __delitem__(self, item: SupportsIndex) -> None:
        if not isinstance(item, SupportsIndex):
            raise TypeError("Collection indices do not support indexing!")
        del self.m_labeling_fns[item]

    def __next__(self) -> LearnerItem:
        self.m_idx += 1
        try:
            return self.m_labeling_fns[self.m_idx - 1]
        except IndexError:
            self.m_idx = 0
            raise StopIteration

    def __repr__(self):
        print(self.m_labeling_fns)
