from __future__ import annotations

import pathlib
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, SupportsIndex, Union

# dask.config.set({"dataframe.query-planning": True})
# import dask.dataframe as dd  # noqa
import pandas as pd  # noqa

# import dask
from loguru import logger
from rich.console import Console  # noqa
from rich.table import Table  # noqa
from typing_extensions import TypeAlias

console = Console()


Preprocessor: TypeAlias = Union[
    Callable[[pd.Series, SupportsIndex], pd.Series], pathlib.Path
]
"""A function (pd.Series, position) -> pd.Series or a path to a file (CSV)"""


class PreprocessorStack:
    r"""The PreprocessorStack is an iterable of preprocessors designed to operate on a
    dataframe. Each preprocessor takes a single Pandas Series and a column index and
    returns a Pandas Series.
    """

    def __init__(self):
        self._idx = 0
        self._stack: list[Preprocessor] = []

    def get(self, item: str) -> Preprocessor:
        if len(self._stack) == 0:
            raise ValueError("The stack is empty!")

        for pr in self._stack:
            if pr.__name__ == item:
                return pr

        raise KeyError(f"Cannot find {item} in PreprocessorStack!")

    def add(
        self,
        preprocessor: Preprocessor,
        position: int = -1,
    ) -> None:
        r"""Adds a preprocessor to the stack of preprocessors. Pre-processors must have
        the following function signature: (pd.Series, int) -> pd.Series. The second
        argument of the function is the column index in the series to operate on.

        Args:
            preprocessor: A preprocessor function that operates on Pandas Series.
            position: Index position of the preprocessor in the stack

        Returns:
            None

        Raises:
            IndexError: If the preprocessor function is not callable or has the wrong
            signature.

        Example:
            >>> from language_toolkit.filters.preprocessor_stack import PreprocessorStack
            >>> def make_lower_case(ds: pd.Series, col_idx: int):
            >>>     s: str = ds.iat[col_idx]
            >>>     ds.iat[col_idx] = s.lower()
            >>>     return ds
            >>> stack = PreprocessorStack()
            >>> stack.add(make_lower_case)
            >>> # stack.add(make_lower_case, 0)

        """
        if not isinstance(position, SupportsIndex):
            raise IndexError(f"Position {position} is not valid for list indexing.")

        if position > len(self._stack) + 1:
            raise IndexError(
                f"Index {position} larger than number of preprocessor functions."
            )

        if position < -1:
            raise IndexError(f"Index {position} should be in range [0, len(stack)).")

        if position == -1 or position >= len(self._stack):
            self._stack.append(preprocessor)
        else:
            self._stack.insert(position, preprocessor)

    def append(self, preprocessor: Preprocessor) -> None:
        r"""Convenience function that calls add(fn, -1)"""
        self.add(preprocessor, -1)

    # TODO: needs to support adding without positions
    def add_multiple(
        self,
        preprocessors: (Iterable[tuple[Preprocessor, SupportsIndex] | Preprocessor]),
    ) -> None:
        r"""Adds multiple preprocessors to the stack. Takes in an iterable of tuples of
        indices and preprocessors, using the indices for insertion position.

        Args:
            preprocessors (Iterable[tuple[Callable[[pd.Series, int], pd.Series], int]]

        Returns:
            None

        Example:
            >>> from language_toolkit.filters.preprocessor_stack import PreprocessorStack
            >>> stack = PreprocessorStack()
            >>> processor0 = ... # func with signature (pd.Series, int) -> pd.Series
            >>> processor1 = ... # func with signature (pd.Series, int) -> pd.Series
            >>> stack.add_multiple([(processor0, 0), (processor1, 1)])
        """
        stack_copy = self._stack.copy()

        for preprocessor_tuple in preprocessors:
            try:
                if isinstance(preprocessor_tuple[0], Path):
                    if preprocessor_tuple[0].suffix == ".csv":
                        if len(preprocessor_tuple) == 1:
                            self.add_csv_preprocessor(preprocessor_tuple[0])
                        else:
                            self.add_csv_preprocessor(
                                preprocessor_tuple[0], order=preprocessor_tuple[1]
                            )
                else:
                    self.add(*preprocessor_tuple)
            except IndexError as e:
                self._stack = stack_copy
                raise e

    def add_csv_preprocessor(
        self,
        csv_path: Path,
        search_idx: int = 0,
        replace_idx: int = 1,
        order: int = None,
    ) -> None:
        r"""Registers a CSV file to be used for substring replacement preprocessing.

        Note:
            The CSV file will not be serialized when saving the StringFilter object.
            Internally we will store the search and replacement strings in a dictionary
            that will get pickled with the object. Thus, when loading the object the
            CSV file is not necessary.

        Args:
            csv_path (Path): Path to the CSV file.
            search_idx (int): Index of the column containing the string to be replaced
                (Defaults to 0).
            replace_idx (int): Index of the column containing the replacement string
                (Defaults to 1).
            order (int| None): The position in the call stack to place the preprocessor
                function. The default is None which places the caller at the end of the
                stack.

        Returns:
            None

        Raises:
            AssertionError: raised if the CSV file does not exist, or the indices are not
                integers.

        Example:
            >>> from language_toolkit.filters.preprocessor_stack import PreprocessorStack
            >>> stack = PreprocessorStack()
            >>> stack.add_csv_preprocessor(Path("replacement_text1.csv"), 0, 1)
            >>> stack.add_csv_preprocessor(Path("replacement_text2.csv"))
        """
        logger.info(f"Registering CSV for preprocessing: {csv_path.stem}")

        if not isinstance(csv_path, Path):
            raise ValueError("CSV path must be a pathlib.Path object.")

        assert csv_path.exists(), f"CSV file {csv_path.absolute()} does not exist!"

        csv_df = pd.read_csv(csv_path)
        num_cols = csv_df.columns.size

        if num_cols <= 1:
            raise ValueError("Pandas DataFrame must have at least two columns.")

        if search_idx < 0 or search_idx > num_cols:
            raise IndexError("Search index must be in [0, df.columns.size)")

        if replace_idx < 0 or replace_idx > num_cols:
            raise IndexError("Replacement index must be in [0, df.columns.size)")

        assert search_idx != replace_idx, "Search and replace must be different!"

        csv_name = csv_path.stem
        search_col_name = csv_df.columns[search_idx]
        replace_col_name = csv_df.columns[replace_idx]
        csv_df = csv_df.set_index(search_col_name)
        self.__dict__[f"{csv_name}_data"] = csv_df[replace_col_name].to_dict()

        # TODO: make this faster
        def preprocessor(ds: pd.Series, col_idx: int) -> pd.Series:
            _d: dict = self.__dict__[f"{csv_name}_data"]
            _s: str = ds.iat[col_idx]
            for key, val in _d.items():
                if _s.find(key) != -1:
                    _s = _s.replace(key, val)
            ds.iat[col_idx] = _s
            return ds

        preprocessor.__name__ = f"{csv_name}_{preprocessor.__name__}"
        processor_stack_size = len(self._stack)
        if order is None:
            self.add(preprocessor, processor_stack_size)
        else:
            self.add(preprocessor, order)
        logger.info(
            f"CSV registered successfully at callstack position: {processor_stack_size}!"
        )

    # TODO: Update example and check positions after removal
    def remove(self, preprocessor: Preprocessor | str):
        r"""Remove a preprocessor from the stack.

        Args:
            preprocessor (Callable[[pd.Series, int | str], pd.Series]): Preprocessor
            reference to be removed

        Returns:
            None

        Raises:
            ValueError: If the preprocessor is not callable or the preprocessor is not in
            the stack.

        Example:
            >>> from language_toolkit.filters.preprocessor_stack import PreprocessorStack
            >>> stack = PreprocessorStack()
            >>> # Define a preprocessor
            >>> def example_preprocessor(ds: pd.Series, position: int) -> pd.Series:
            >>>     # This function will test for string lengths greater than 10
            >>>     if len(ds.iat[position]) >= 10:
            >>>         return ds
            >>>     ds.iat[position] = ""
            >>>     return ds
            >>> stack.append(example_preprocessor)
            >>> # Remove the previously added preprocessor
            >>> stack.remove(example_preprocessor)
        """
        if len(self._stack) == 0:
            raise ValueError("Stack is empty!")

        if not callable(preprocessor) and not isinstance(preprocessor, str):
            raise ValueError(
                "Must pass a reference to a preprocessor or the function name"
            )

        try:
            if isinstance(preprocessor, str):
                for fn in self._stack:
                    if preprocessor == fn.__name__:
                        self._stack.remove(fn)
            else:
                self._stack.remove(preprocessor)
        except ValueError:
            logger.warning(f"Preprocessing function: {preprocessor} not in the stack.")

    def update(self, preprocessor: Preprocessor):
        r"""Update an existing preprocessor in the stack

        Args:
            preprocessor (Preprocessor): Updated preprocessor

        Returns:
            None

        Raises:
            ValueError: If the preprocessor is not in the stack.

        Example:
            >>> from language_toolkit.filters.preprocessor_stack import PreprocessorStack
            >>> stack = PreprocessorStack()
            >>> # Define a preprocessor
            >>> def example_preprocessor(ds: pd.Series, position: int) -> pd.Series:
            >>>     s: str = ds.iat[position]
            >>>     ds.iat[position] = s.lower()
            >>>     return ds
            >>> stack.append(example_preprocessor)
            >>> # event necessitates changing a preprocessor
            >>> def example_preprocessor(ds: pd.Series, position: int) -> pd.Series:
            >>>     s: str = ds.iat[position]
            >>>     ds.iat[position] = s.upper() # <-- change
            >>>     return ds
            >>> stack.update(example_preprocessor)
        """
        for idx, fn in enumerate(self._stack):
            if preprocessor.__name__ == fn.__name__:
                self.remove(preprocessor)
                self.add(preprocessor, idx - 1)

    def __call__(
        self,
        df: pd.DataFrame,
        col_idx: int = 0,
        parallel: bool = False,
        num_partitions: int = 2,
    ) -> pd.DataFrame:
        r"""Sequentially execute functions in the preprocessor stack"""
        if parallel:
            pass
            # df = dd.from_pandas(df, npartitions=num_partitions)

        for preprocessor in self._stack:
            partial_fn = partial(preprocessor, col_idx=col_idx)
            if parallel:
                df.apply(partial_fn, axis=1, meta=df)
            else:
                df = df.apply(partial_fn, axis=1)
        return df

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._stack)

    def __getitem__(self, item: SupportsIndex) -> Preprocessor:
        if not isinstance(item, SupportsIndex):
            raise TypeError("Preprocessor stack indices must be integers")
        return self._stack[item]

    def __setitem__(
        self,
        item: SupportsIndex,
        preprocessor: Preprocessor,
    ) -> None:
        if not isinstance(item, SupportsIndex):
            raise TypeError("Preprocessor stack indices must be integers")
        self._stack[item] = preprocessor

    def __delitem__(self, item: SupportsIndex) -> None:
        if len(self._stack) == 0:
            raise ValueError("Stack is empty!")
        if not isinstance(item, SupportsIndex):
            raise TypeError("Preprocessor stack indices must be integers")
        del self._stack[item]

    def __next__(self) -> Preprocessor | pathlib.Path:
        self._idx += 1
        try:
            return self._stack[self._idx - 1]
        except IndexError:
            self._idx = 0
            raise StopIteration

    def __repr__(self):
        table = Table(title="Preprocessor Callstack", show_lines=True)
        table.add_column("Index")
        table.add_column("Function", style="bold")
        table.add_column("Type")
        for idx, fn in enumerate(self._stack):
            table.add_row(str(idx), fn.__name__, str(type(fn)))
        console.print(table)
        return ""
