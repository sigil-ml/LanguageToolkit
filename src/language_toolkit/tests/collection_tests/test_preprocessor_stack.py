from pathlib import Path

import pandas as pd
import pytest

from language_toolkit.filters.preprocessor_stack import PreprocessorStack

TEST_MSGS = ["1", "22", "333", "4444", "55555", "666666", "7777777" "hello", "hola"]


def t_preprocessor_fn0(ds: pd.Series, position: int) -> pd.Series:
    r"""Test function for testing CRUD operations"""
    s: str = ds.iat[position]
    ds.iat[position] = s.lower()
    return ds


def t_preprocessor_fn1(ds: pd.Series, position: int) -> pd.Series:
    r"""Test function for testing CRUD operations"""
    s: str = ds.iat[position]
    ds.iat[position] = s.upper()
    return ds


def t_preprocessor_fn2(ds: pd.Series, position: int) -> pd.Series:
    r"""Test function for testing CRUD operations"""
    s: str = ds.iat[position]
    ds.iat[position] = s.capitalize()
    return ds


# TODO: Add preprocess calls
class TestCSVPreprocessor:
    csv_path = Path("../data/test.csv").absolute()
    test_df = pd.DataFrame(
        [[0, "test"], [1, "test2"], [2, "csv"], [3, "test3"], [4, "APL"]],
        columns=["id", "text"],
    )
    true_df = pd.DataFrame(
        [
            [0, "test"],
            [1, "test2"],
            [2, "Comma-seperated Values"],
            [3, "test3"],
            [4, "A Programming Language"],
        ],
        columns=["id", "text"],
    )

    @pytest.fixture
    def empty_stack(self):
        ps = PreprocessorStack()
        yield ps

    @pytest.fixture
    def full_stack(self):
        ps = PreprocessorStack()
        ps.add_multiple(
            [(t_preprocessor_fn0, 0), (t_preprocessor_fn1, 1), (t_preprocessor_fn2, 2)]
        )
        yield ps

    def test_add(self, empty_stack):
        empty_stack.add(t_preprocessor_fn0)
        assert empty_stack[-1].__name__ == "t_preprocessor_fn0"
        assert callable(empty_stack[-1])

    def test_add2(self, empty_stack):
        empty_stack._stack.append(t_preprocessor_fn0)
        empty_stack._stack.append(t_preprocessor_fn1)
        empty_stack.add(t_preprocessor_fn2, position=2)
        assert empty_stack[1].__name__ == "t_preprocessor_fn1"
        assert callable(empty_stack[1])

    def test_append(self, empty_stack):
        empty_stack.append(t_preprocessor_fn0)
        assert empty_stack[-1].__name__ == "t_preprocessor_fn0"
        assert callable(empty_stack[-1])

    def test_append2(self, empty_stack):
        empty_stack._stack.append(t_preprocessor_fn0)
        empty_stack._stack.append(t_preprocessor_fn1)
        empty_stack.append(t_preprocessor_fn2)
        assert empty_stack[-1].__name__ == "t_preprocessor_fn2"
        assert callable(empty_stack[-1])

    def test_add_multiple(self, empty_stack):
        empty_stack.add_multiple(
            [(t_preprocessor_fn0, 0), (t_preprocessor_fn1, 1), (t_preprocessor_fn2, 2)]
        )
        assert len(empty_stack) == 3
        for i in range(3):
            assert empty_stack[i].__name__ == f"t_preprocessor_fn{i}"
            assert callable(empty_stack[i])

    def test_add_multiple_with_error_1(self, empty_stack):
        with pytest.raises(IndexError):
            empty_stack.add_multiple(
                [
                    (t_preprocessor_fn0, 0),
                    (t_preprocessor_fn1, -100),
                    (t_preprocessor_fn2, 1),
                ]
            )
        assert len(empty_stack) == 0

    def test_add_multiple_with_error_2(self, empty_stack):
        with pytest.raises(IndexError):
            empty_stack.add_multiple(
                [
                    (t_preprocessor_fn0, 0),
                    (t_preprocessor_fn1, 10000),
                    (t_preprocessor_fn2, 1),
                ]
            )
        assert len(empty_stack) == 0

    def test_add_multiple_with_error_3(self, empty_stack):
        with pytest.raises(IndexError):
            empty_stack.add_multiple(
                [
                    (t_preprocessor_fn0, 0),
                    (t_preprocessor_fn1, "Hello World"),
                    (t_preprocessor_fn2, 1),
                ]
            )
        assert len(empty_stack) == 0

    def test_add_multiple_with_error_4(self, full_stack):
        with pytest.raises(IndexError):
            full_stack.add_multiple(
                [
                    (t_preprocessor_fn0, 0),
                    (t_preprocessor_fn1, 10000),
                    (t_preprocessor_fn2, 1),
                ]
            )
        assert len(full_stack) == 3

    def test_csv_register_full(self, empty_stack):
        empty_stack.add_csv_preprocessor(self.csv_path, 0, 1)
        assert empty_stack.__dict__["test_data"] is not None
        assert isinstance(empty_stack.__dict__["test_data"], dict)
        assert empty_stack[-1].__name__ == "test_preprocessor"
        assert callable(empty_stack[-1])

    def test_csv_register_partial(self, empty_stack):
        empty_stack.add_csv_preprocessor(self.csv_path)
        assert empty_stack.__dict__["test_data"] is not None
        assert isinstance(empty_stack.__dict__["test_data"], dict)
        assert empty_stack[-1].__name__ == "test_preprocessor"
        assert callable(empty_stack[-1])

    def test_csv_partial_ordering_start(self, empty_stack):
        empty_stack.add_csv_preprocessor(self.csv_path, order=0)
        assert empty_stack[0].__name__ == "test_preprocessor"

    def test_csv_partial_ordering_end(self, full_stack):
        full_stack.add_csv_preprocessor(self.csv_path, order=-1)
        assert full_stack[-1].__name__ == "test_preprocessor"

    def test_csv_partial_ordering_end2(self, full_stack):
        full_stack.add_csv_preprocessor(self.csv_path)
        assert full_stack[-1].__name__ == "test_preprocessor"

    def test_csv_non_int_order(self, empty_stack):
        with pytest.raises(IndexError):
            empty_stack.add_csv_preprocessor(self.csv_path, order=0.1)  # noqa

    def test_csv_non_int_search_idx(self, empty_stack):
        with pytest.raises(IndexError):
            empty_stack.add_csv_preprocessor(self.csv_path, search_idx=0.1)  # noqa

    def test_csv_non_int_replace_idx(self, empty_stack):
        with pytest.raises(IndexError):
            empty_stack.add_csv_preprocessor(self.csv_path, replace_idx=0.1)  # noqa

    def test_csv_non_positive_search_idx(self, empty_stack):
        with pytest.raises(IndexError):
            empty_stack.add_csv_preprocessor(self.csv_path, search_idx=-1)

    def test_csv_non_positive_replace_idx(self, empty_stack):
        with pytest.raises(IndexError):
            empty_stack.add_csv_preprocessor(self.csv_path, replace_idx=-1)

    def test_csv_search_eq_replace_idx(self, empty_stack):
        with pytest.raises(AssertionError):
            empty_stack.add_csv_preprocessor(self.csv_path, search_idx=0, replace_idx=0)

    def test_csv_too_large_search_idx(self, empty_stack):
        with pytest.raises(IndexError):
            empty_stack.add_csv_preprocessor(self.csv_path, search_idx=100)

    def test_csv_too_large_replace_idx(self, empty_stack):
        with pytest.raises(IndexError):
            empty_stack.add_csv_preprocessor(self.csv_path, replace_idx=100)

    def test_remove_simple(self, empty_stack):
        empty_stack.add(t_preprocessor_fn0)
        empty_stack.remove(t_preprocessor_fn0)
        assert len(empty_stack) == 0

    def test_remove_simple2(self, full_stack):
        full_stack.remove(t_preprocessor_fn0)
        assert len(full_stack) == 2

    def test_remove_non_existent(self, empty_stack):
        with pytest.raises(ValueError):
            empty_stack.remove("t_preprocessor_fn0")

    def test_update_preprocessor(self, full_stack):
        old_code = full_stack[1].__code__.co_code  # noqa

        def t_preprocessor_fn1(ds: pd.Series, position: int) -> pd.Series:  # noqa
            return ds

        full_stack.update(t_preprocessor_fn1)
        assert old_code != full_stack[1].__code__.co_code  # noqa

    # TODO: Add Call Tests
