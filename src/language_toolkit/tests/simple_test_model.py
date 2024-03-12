from language_toolkit.filters.string_filter import StringFilter

from language_toolkit.logger import logger
import pandas as pd
from pathlib import Path
from test_data import data_factory


if __name__ == "__main__":
    # Define Data
    test_data = data_factory(pull_data=True, retain_data=True)

    # Define the filter
    sf = StringFilter(col_name="text")
    train_df, test_df = sf.train_test_split(test_data["text"], train_size=0.8)
    res = sf.fit(train_df, train_col="text", target_col="label", template_miner=True)
    for i, row in enumerate(res.results.items()):
        if row[1] == "Sorry, I'll call later <:*:> <:*:>":
            print(i)
