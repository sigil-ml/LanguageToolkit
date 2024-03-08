from at_nlp.filters.string_filter import StringFilter

from at_nlp.logger import logger
import pandas as pd
from pathlib import Path
from test_data import data_factory


if __name__ == "__main__":
    # Define Data
    test_data = data_factory(pull_data=True, retain_data=True)

    # Define the filter
    sf = StringFilter(col_name="text")
    X_split, y_split = sf.train_test_split(test_data["text"], train_size=0.8)

    logger.info(f"X_split shape: {X_split.shape}")
    logger.info(f"y_split shape: {y_split.shape}")

