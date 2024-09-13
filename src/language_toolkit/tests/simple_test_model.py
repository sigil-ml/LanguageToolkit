from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from snorkel.labeling import labeling_function
from test_data import data_factory
from wordcloud import WordCloud

from language_toolkit.filters.string_filter import StringFilter
from language_toolkit.logger import logger

if __name__ == "__main__":
    nb = MultinomialNB()
    # lr = LogisticRegression()
    rf = RandomForestClassifier()
    mlp = MLPClassifier()
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0)

    # pipe = make_pipeline(
    #     StandardScaler(with_mean=False),
    #     LinearSVC(dual=False, random_state=0, tol=1e-5, max_iter=5000),
    # )

    # Define Data
    # test_data = data_factory(pull_data=False, retain_data=True)
    data = pd.read_csv(
        "./src/language_toolkit/tests/data/corrected.csv"
    )
    
    prev_n_rows = len(data)
    
    # Remove duplicates
    data.drop_duplicates(subset="Message", inplace=True)
    
    print(f"Removed {prev_n_rows - len(data)} rows from the dataset.")
    # data = pd.read_csv(
    #     "./src/language_toolkit/tests/data/(CUI) alexa_816th_file_1a1.csv"
    # )
    
    # Define the filter
    sf = StringFilter("Message")
    sf.add_preprocessor(Path("./src/language_toolkit/tests/data/acronyms.csv"))
    spam_messages = [
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

    ham_messages = [
        "crew",
        "question",
        "etar",
        "paper",
        "cargo",
        "delayed",
        "OTBH",
        "OKAS",
        "OAIX",
        "mission",
        "answer",
        "delayed",
    ]

    sf.add_labeling_function(
        lambda x: 1 if any(i.lower() in x.lower() for i in spam_messages) else 0
    )
    sf.add_labeling_function(
        lambda x: 0 if any(i.lower() in x.lower() for i in ham_messages) else 1
    )
    sf.add_labeling_function(lambda x: 1 if len(x.split()) > 2 else 0)

    sf.add_labeling_function(rf)
    sf.add_labeling_function(gb)
    # sf.add_labeling_function(nb)
    # sf.add_labeling_function(mlp)

    use_template_miner = True
    train_df, test_df = sf.train_test_split(
        data, train_size=0.9, shuffle=True
    )
    res = sf.fit(
        train_df, train_col="Message", target_col="labels", template_miner=use_template_miner
    )
    pprint(sf.eval(test_df, "labels", use_template_miner=use_template_miner))
    sf.save(Path("./spam_model"))

    # print("============= NEW MODEL =============")
    # new_filter = StringFilter.load(Path("./test_model"))
    # pprint(new_filter.eval(test_df, "Message", "labels"))


