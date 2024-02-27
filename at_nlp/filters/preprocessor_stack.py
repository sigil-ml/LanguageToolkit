from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, TypeAlias

import dask

dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd  # noqa
import pandas as pd  # noqa
from rich.console import Console  # noqa
from rich.table import Table  # noqa

console = Console()

Preprocessor: TypeAlias = Callable[[pd.Series, int | str], pd.Series]


class PreprocessorStack:
    r"""The PreprocessorStack is an iterable of preprocessors designed to operate on a dataframe. Each preprocessor
    takes a single Pandas Series and a column index and returns a Pandas Series. Multiprocessing is available via the
    'multiprocessing' flag.
    """

    def __init__(self):
        self._idx = 0
        self._stack: list[Preprocessor] = []

    def add(
            self,
            preprocessor: Preprocessor,
            position: int = -1,
    ) -> None:
        r"""Adds a preprocessor to the filter. Pre-processors must have the following
        function signature: (pd.Series, int) -> pd.Series. The second argument of the
        function is column index or name of the item in the series to operate on.

        Args:
            preprocessor: A preprocessor function that operates on Pandas Series.
            position: Index position of the preprocessor in the stack

        Returns:
            None

        Raises:
            IndexError: If the preprocessor function is not callable or has the wrong signature.

        Example:
            >>> from at_nlp.filters.preprocessor_stack import PreprocessorStack
            >>> def make_lower_case(ds: pd.Series, col_idx: int):
            >>>     s: str = ds.iat[col_idx]
            >>>     ds.iat[col_idx] = s.lower()
            >>>     return ds
            >>> stack = PreprocessorStack()
            >>> stack.add(make_lower_case)
            >>> # stack.add(make_lower_case, 0) # a position can be given in the range [0, len(stack))

        """
        if position > len(self._stack):
            raise IndexError(f"Index {position} larger than the list length.")

        if position < -1:
            raise IndexError(f"Index {position} is negative.")

        if position == -1:
            self._stack.append(preprocessor)
        else:
            self._stack.insert(position, preprocessor)

    def append(self, preprocessor: Preprocessor) -> None:
        r"""convenience function that calls add(fn, -1)"""
        self.add(preprocessor, -1)

    def add_multiple(self, preprocessors: Iterable[tuple[int, Preprocessor]]):
        r"""Adds multiple preprocessors to the stack. Takes in a tuple of indices and preprocessors, using the indices
        for insertion position.

        Args:
            preprocessors (Iterable[tuple[int, Callable[[pd.Series, int | str], pd.Series]]]

        Returns:
            None

        Example:
            >>> from at_nlp.filters.preprocessor_stack import PreprocessorStack
            >>> stack = PreprocessorStack()
            >>> processor0 = ... # func with signature (pd.Series, int) -> pd.Series
            >>> processor1 = ... # func with signature (pd.Series, int) -> pd.Series
            >>> stack.add_multiple([(0, processor0), (1, processor1)])
        """
        for preprocessor_tuple in preprocessors:
            idx = preprocessor_tuple[0]
            preprocessor = preprocessor_tuple[1]
            self.add(preprocessor, idx)

    def add_csv_preprocessor(
            self,
            csv_path: Path,
            search_idx: int = 0,
            replace_idx: int = 1,
            order: int | None = None,
    ) -> None:
        r"""Registers a CSV file to be used for preprocessing. This is different from
        registering a CSV file for weak learning since we replace the strings
        before the weak learners are trained and applied. If you wish to use the CSV
        for weak learning then use the :meth:`register_csv_weak_learner` method instead.

        Note:
            The CSV file will not be serialized when saving the StringFilter object.
            Internally we will store the search and replacement strings in a dictionary
            that will get pickled with the object. Thus, when loading the object the
            CSV file is not necessary.

        Args:
            csv_path (Path): Path to the CSV file.
            search_idx (int): Index of the column containing the string to be replaced (Defaults to 0).
            replace_idx (int): Index of the column containing the replacement string (Defaults to 1).
            order (int| None): The position in the call stack to place the preprocessor function.
                The default is None which places the caller at the end of the stack.

        Returns:
            None

        Raises:
            AssertionError: raised if the CSV file does not exist, or the indices are not integers.

        Example:
            >>> from at_nlp.filters.preprocessor_stack import PreprocessorStack
            >>> stack = PreprocessorStack()
            >>> stack.add_csv_preprocessor(Path("replacement_text1.csv"), 0, 1)
            >>> stack.add_csv_preprocessor(Path("replacement_text2.csv"))
        """
        console.log(f"Registering CSV for preprocessing: {csv_path.stem}")

        if not isinstance(csv_path, Path):
            raise ValueError("CSV path must be a pathlib.Path object.")

        assert csv_path.exists(), f"CSV file {csv_path.absolute()} does not exist!"

        csv_df = pd.read_csv(csv_path)
        num_cols = csv_df.columns.size

        if num_cols <= 1:
            raise ValueError("Pandas DataFrame must have at least two columns.")

        if search_idx < 0 or search_idx > num_cols:
            raise IndexError(f"Search index must be in [0, df.columns.size)")

        if replace_idx < 0 or replace_idx > num_cols:
            raise IndexError(f"Replacement index must be in [0, df.columns.size)")

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
        console.log(
            f"CSV registered successfully at callstack position: {processor_stack_size}!"
        )

    def remove(self, preprocessor: Preprocessor):
        r"""Remove a preprocessor from the stack.

        Args:
            preprocessor (Callable[[pd.Series, int | str], pd.Series]): Preprocessor reference to be removed

        Returns:
            None

        Raises:
            ValueError: If the preprocessor is not callable or the preprocessor is not in the stack.

        Example:
            >>> from at_nlp.filters.preprocessor_stack import PreprocessorStack
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
        if not callable(preprocessor):
            raise ValueError("Must provide preprocessor reference")

        try:
            self._stack.remove(preprocessor)
        except ValueError:
            logging.warning(
                f"Preprocessing function: {preprocessor} not in the stack."
            )

    def update(self, preprocessor: Preprocessor):
        r"""Update an existing preprocessor in the stack

        Args:
            preprocessor (Preprocessor): Updated preprocessor

        Returns:
            None

        Raises:
            ValueError: If the preprocessor is not in the stack.

        Example:
            >>> from at_nlp.filters.preprocessor_stack import PreprocessorStack
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
            df = dd.from_pandas(df, npartitions=num_partitions)

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

    def __getitem__(self, item: int) -> Preprocessor:
        if not isinstance(item, int):
            raise TypeError("Preprocessor stack indices must be integers")
        return self._stack[item]

    def __setitem__(self, item: int, preprocessor: Preprocessor) -> None:
        if not isinstance(item, int):
            raise TypeError("Preprocessor stack indices must be integers")
        self._stack[item] = preprocessor

    def __next__(self) -> Preprocessor:
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
