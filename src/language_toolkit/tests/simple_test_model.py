from language_toolkit.filters.string_filter import StringFilter

from language_toolkit.logger import logger
import pandas as pd
from pathlib import Path
from test_data import data_factory
from example_functions import rf
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from pprint import pprint

if __name__ == "__main__":
    nb = MultinomialNB()
    lr = LogisticRegression()

    # Define Data
    test_data = data_factory(pull_data=False, retain_data=True)

    # Define the filter
    sf = StringFilter("text")
    sf.add_labeling_function(rf)
    sf.add_labeling_function(nb)
    sf.add_labeling_function(lr)
    train_df, test_df = sf.train_test_split(test_data, train_size=0.8)
    res = sf.fit(train_df, train_col="text", target_col="label", template_miner=True)
    pprint(sf.eval(test_df, "text", "label"))
    sf.save(Path("./test_model"))

    print("============= NEW MODEL =============")
    new_filter = StringFilter.load(Path("./test_model"))
    pprint(new_filter.eval(test_df, "text", "label"))
