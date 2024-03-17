from language_toolkit.filters.string_filter import StringFilter

from language_toolkit.logger import logger
import pandas as pd
from pathlib import Path
from test_data import data_factory
from example_functions import rf
from sklearn.naive_bayes import MultinomialNB

if __name__ == "__main__":
    nb = MultinomialNB()

    # Define Data
    test_data = data_factory(pull_data=False, retain_data=True)

    # Define the filter
    sf = StringFilter()
    sf.add_labeling_function(rf)
    sf.add_labeling_function(nb)
    train_df, test_df = sf.train_test_split(test_data, train_size=0.8)
    res = sf.fit(train_df, train_col="text", target_col="label", template_miner=True)
