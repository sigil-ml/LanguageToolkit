from pathlib import Path
from pprint import pprint

import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from snorkel.labeling import labeling_function

from language_toolkit.filters.string_filter import StringFilter
from language_toolkit.logger import logger

def replace_acronyms(text, acronym_dict):
    words = text.split()
    replaced_text = ' '.join([acronym_dict.get(word, word) for word in words])
    return replaced_text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    # Load pretrained models
    rf = joblib.load('models/best_rf.pkl')
    lr = joblib.load('models/best_lr.pkl')
    nb = joblib.load('models/best_nb.pkl')

    # Define Data
    test_data = pd.read_csv("src/language_toolkit/tests/data/(CUI) alexa_816th_file_1a1.csv")
    acronym_data = pd.read_csv("src/language_toolkit/tests/data/acronyms.csv")

    # Create a dictionary for acronyms
    acronyms = acronym_data['Jargon'].tolist()
    acronym_meaning = acronym_data['Meaning'].tolist()
    acronym_dict = dict(zip(acronyms, acronym_meaning))

    # Clean the messages
    test_data['Message'] = test_data['Message'].apply(clean_text)

    # Replace acronyms in the messages
    test_data['Message'] = test_data['Message'].apply(lambda text: replace_acronyms(text, acronym_dict))

    # Ensure the target column is of integer type
    test_data['labels'] = test_data['labels'].astype(int)

    # Vectorize the text data
    vectorizer = CountVectorizer()
    vectorizer.fit(test_data['Message'])

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

    @labeling_function()
    def lf_spam(x):
        return 2 if any(i.lower() in x.Message.lower() for i in spam_messages) else 0

    @labeling_function()
    def lf_ham(x):
        return 0 if any(i.lower() in x.Message.lower() for i in ham_messages) else 1

    @labeling_function()
    def lf_word_count(x):
        return 0 if len(x.Message.split()) > 2 else 1

    @labeling_function()
    def lf_rf(x):
        vectorized_message = vectorizer.transform([x.Message])
        return rf.predict(vectorized_message)[0]

    @labeling_function()
    def lf_nb(x):
        vectorized_message = vectorizer.transform([x.Message]).toarray()
        return nb.predict(vectorized_message)[0]

    @labeling_function()
    def lf_lr(x):
        vectorized_message = vectorizer.transform([x.Message])
        return lr.predict(vectorized_message)[0]

    sf.add_labeling_function(lf_spam)
    sf.add_labeling_function(lf_ham)
    sf.add_labeling_function(lf_word_count)
    sf.add_labeling_function(lf_rf)
    sf.add_labeling_function(lf_nb)
    sf.add_labeling_function(lf_lr)

    train_df, test_df = sf.train_test_split(test_data, train_size=0.8, shuffle=True)
    res = sf.fit(train_df, train_col="Message", target_col="labels", template_miner=True)
    pprint(sf.eval(test_df, "labels", use_template_miner=True))
    sf.save(Path("./test_model1"))

    # print("============= NEW MODEL =============")
    # new_filter = StringFilter.load(Path("./test_model1"))
    # pprint(new_filter.eval(test_df, "labels", use_template_miner=True))