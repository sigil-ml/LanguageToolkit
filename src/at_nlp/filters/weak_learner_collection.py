from __future__ import annotations

from functools import singledispatchmethod
from dataclasses import dataclass
from collections import abc
from typing import SupportsIndex, TypeAlias, Callable, Iterable, Optional

import pandas as pd
from snorkel.labeling import LabelingFunction, labeling_function
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from snorkel.preprocess import preprocessor


# LabelingFunctionPrimative: TypeAlias = abc.Callable[pd.Series], int]]


@dataclass
class LearnerItem:
    fn: abc.Callable | LabelingFunction
    learnable: bool
    item_type: str | None


class WeakLearners:
    """A collection of weak learners that will be used by Snorkel"""

    def __init__(self, col_name: str):
        self.m_learners: list[LearnerItem] = []
        self.m_col_name = col_name
        self.m_vectorizer = CountVectorizer()
        self.m_rsrcs = dict(col_name=self.m_col_name)
        self.m_idx = 0

    @singledispatchmethod
    def add(
            self,
            fn: abc.Callable | LabelingFunction | BaseEstimator,
            learnable: bool = False,
            item_type: str | None = None
    ) -> None:
        r"""Dispatches to one of the following methods:
            1. :meth:`add_labeling_function`
            2. :meth:`add_primative`
            3. :meth:`add_sklearn`

        Raises:
            TypeError: If the provided fn is not the correct type

        """
        raise TypeError("Supplied function is not an accepted labeling function. Received"
                        "{}, but expected either Callable[[pd.Series], int] or a Snorkel"
                        "LabelingFunction" % fn.__class__.__name__)

    # noinspection GrazieInspection
    @add.register
    def add_labeling_function(self, fn: LabelingFunction, learnable: bool = False,
                              item_type: str | None = None) -> None:
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
            >>> from at_nlp.filters.weak_learner_collection import WeakLearners
            >>> from snorkel.labeling import labeling_function
            >>> wl_col = WeakLearners()
            >>> rsrcs = {"col_name": "Messages"}
            >>> @labeling_function(name="Example", resources=rsrcs)
            >>> def fn_ex0(series: pd.Series, col_name: str) -> int:
            >>>     if len(series[col_name]) > 6:
            >>>         return 1
            >>>     else:
            >>>         return 0
            >>> wl_col.add(fn_ex0, False)
        """
        item = LearnerItem(fn, learnable=learnable, item_type=item_type)
        self.m_learners.append(item)

    @add.register
    def add_primative(self, fn: abc.Callable) -> None:
        """:meth:`add_primative` creates a labeling function from a simple python
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
            >>> from at_nlp.filters.weak_learner_collection import WeakLearners
            >>> wl_col = WeakLearners()
            >>>
            >>> # Ex1
            >>> wl_col.add(lambda s: int(len(s) > 6))
            >>>
            >>> # Ex2
            >>> def primative_ex2(s: str) -> int:
            >>>     if s.find("some substring to find") == -1:
            >>>         return 0
            >>>     return 1
            >>> wl_col.add(primative_ex2)

        """

        @labeling_function(name=f"PR_{fn.__name__}", resources=self.m_rsrcs)
        def wrapper(series: pd.Series, col_name: str) -> int:
            s = series[col_name]
            return fn(s)

        item = LearnerItem(wrapper, learnable=False, item_type=None)
        self.m_learners.append(item)

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
            >>> from at_nlp.filters.weak_learner_collection import WeakLearners
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
        self.m_learners.append(item)

    def extend(self,
               fns: abc.Iterable[LabelingFunction | abc.Callable | BaseEstimator]
               ) -> None:
        """Extends internal collection with a list of fns. Assumes each function is
        non-trainable unless it subclasses BaseEstimator.
        """
        for fn in fns:
            self.add(fn)

    # def remove(self, preprocessor: Preprocessor):
    #     r"""Remove a weak learner from the collection.
    #
    #     Args:
    #         preprocessor (Callable[[pd.Series, int | str], pd.Series]): Preprocessor reference to be removed
    #
    #     Returns:
    #         None
    #
    #     Raises:
    #         ValueError: If the preprocessor is not callable or the preprocessor is not in the stack.
    #
    #     Example:
    #         >>> from at_nlp.filters.preprocessor_stack import PreprocessorStack
    #         >>> stack = PreprocessorStack()
    #         >>> # Define a preprocessor
    #         >>> def example_preprocessor(ds: pd.Series, position: int) -> pd.Series:
    #         >>>     # This function will test for string lengths greater than 10
    #         >>>     if len(ds.iat[position]) >= 10:
    #         >>>         return ds
    #         >>>     ds.iat[position] = ""
    #         >>>     return ds
    #         >>> stack.append(example_preprocessor)
    #         >>> # Remove the previously added preprocessor
    #         >>> stack.remove(example_preprocessor)
    #     """
    #     if not callable(preprocessor):
    #         raise ValueError("Must provide preprocessor reference")
    #
    #     try:
    #         self._stack.remove(preprocessor)
    #     except ValueError:
    #         logger.warning(
    #             f"Preprocessing function: {preprocessor} not in the stack."
    #         )
    #
    # def update(self, preprocessor: Preprocessor):
    #     r"""Update an existing preprocessor in the stack
    #
    #     Args:
    #         preprocessor (Preprocessor): Updated preprocessor
    #
    #     Returns:
    #         None
    #
    #     Raises:
    #         ValueError: If the preprocessor is not in the stack.
    #
    #     Example:
    #         >>> from at_nlp.filters.preprocessor_stack import PreprocessorStack
    #         >>> stack = PreprocessorStack()
    #         >>> # Define a preprocessor
    #         >>> def example_preprocessor(ds: pd.Series, position: int) -> pd.Series:
    #         >>>     s: str = ds.iat[position]
    #         >>>     ds.iat[position] = s.lower()
    #         >>>     return ds
    #         >>> stack.append(example_preprocessor)
    #         >>> # event necessitates changing a preprocessor
    #         >>> def example_preprocessor(ds: pd.Series, position: int) -> pd.Series:
    #         >>>     s: str = ds.iat[position]
    #         >>>     ds.iat[position] = s.upper() # <-- change
    #         >>>     return ds
    #         >>> stack.update(example_preprocessor)
    #     """
    #     for idx, fn in enumerate(self._stack):
    #         if preprocessor.__name__ == fn.__name__:
    #             self.remove(preprocessor)
    #             self.add(preprocessor, idx - 1)
    #
    # def __call__(
    #         self,
    #         df: pd.DataFrame,
    #         col_idx: int = 0,
    #         parallel: bool = False,
    #         num_partitions: int = 2,
    # ) -> pd.DataFrame:
    #     r"""Sequentially execute functions in the preprocessor stack"""
    #     if parallel:
    #         df = dd.from_pandas(df, npartitions=num_partitions)
    #
    #     for preprocessor in self._stack:
    #         partial_fn = partial(preprocessor, col_idx=col_idx)
    #         if parallel:
    #             df.apply(partial_fn, axis=1, meta=df)
    #         else:
    #             df = df.apply(partial_fn, axis=1)
    #     return df
    #
    def __iter__(self):
        return self

    def __len__(self):
        return len(self.m_learners)

    def __getitem__(self, item: int) -> LearnerItem:
        if not isinstance(item, int):
            raise TypeError("Collection indices must be integers")
        return self.m_learners[item]

    def __setitem__(self, item: int, learner_item: LearnerItem) -> None:
        if not isinstance(item, int):
            raise TypeError("Collection indices must be integers")
        self.m_learners[item] = learner_item

    def __next__(self) -> LearnerItem:
        self.m_idx += 1
        try:
            return self.m_learners[self.m_idx - 1]
        except IndexError:
            self.m_idx = 0
            raise StopIteration

    def __repr__(self):
        print(self.m_learners)
