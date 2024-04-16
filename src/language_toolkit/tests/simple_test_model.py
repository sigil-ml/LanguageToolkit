from language_toolkit.filters.string_filter import StringFilter
from snorkel.labeling import labeling_function
from language_toolkit.logger import logger
import pandas as pd
from pathlib import Path
from test_data import data_factory
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pprint import pprint

if __name__ == "__main__":
    nb = MultinomialNB()
    # lr = LogisticRegression()
    rf = RandomForestClassifier()
    mlp = MLPClassifier()

    # Define Data
    # test_data = data_factory(pull_data=False, retain_data=True)
    test_data = pd.read_csv(
        "./src/language_toolkit/tests/data/(CUI) alexa_816th_file_1a1.csv"
    )

    # Define the filter
    sf = StringFilter("Message")
    sf.add_preprocessor(Path("./src/language_toolkit/tests/data/acronyms.csv"))
    filter_messages = [
        "joined the channel",
        "added to the channel",
        "hello",
        "hola",
        "good morning",
        "good evening",
        "good afternoon",
        "good night",
        "rgr",
        "roger",
        "lunch",
        "dinner",
        "breakfast",
        "food",
    ]

    sf.add_labeling_function(lambda x: 2 if any(i in x for i in filter_messages) else 0)
    sf.add_labeling_function(lambda x: 2 if len(x) > 6 else 0)

    sf.add_labeling_function(rf)
    sf.add_labeling_function(nb)
    # sf.add_labeling_function(rf)
    # sf.add_labeling_function(nb)
    # sf.add_labeling_function(mlp)
    train_df, test_df = sf.train_test_split(test_data, train_size=0.8)
    res = sf.fit(
        train_df, train_col="Message", target_col="labels", template_miner=False
    )
    pprint(sf.eval(test_df, "Message", "labels"))
    sf.save(Path("./test_model"))

    print("============= NEW MODEL =============")
    new_filter = StringFilter.load(Path("./test_model"))
    pprint(new_filter.eval(test_df, "Message", "labels"))
