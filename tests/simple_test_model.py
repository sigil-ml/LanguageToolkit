from at_nlp.filters.string_filter import StringFilter

from at_nlp.logger import logger
import pandas as pd
from pathlib import Path
from test_data import data_factory


if __name__ == "__main__":
    test_data = data_factory(pull_data=True)
