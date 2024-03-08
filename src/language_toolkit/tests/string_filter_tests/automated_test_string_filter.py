# r"""
#        ______      _             _____ ____            ______        __
#       / __/ /_____(_)__  ___ _  / __(_) / /____ ____  /_  __/__ ___ / /_
#      _\ \/ __/ __/ / _ \/ _ `/ / _// / / __/ -_) __/   / / / -_|_-</ __/
#     /___/\__/_/ /_/_//_/\_, / /_/ /_/_/\__/\__/_/     /_/  \__/___/\__/
#                        /___/
#
#
#     This file tests the various functions of the StringFilter. Specific tests for the
#     WeakLearners collection are in test_weak_learner_collection.py, and tests for the
#     PreprocessorStack are in test_preprocessor_stack.py.
#
#     We will test the following combination of functions:
#
#     D := 1 if using template miner, 0 otherwise
#
#     Preprocessors:
#     P1 := AcronymExpansion
#     P2 := Strip Non-ASCII
#     P3 := Link remover
#
#     Weak Learners:
#     W1 := Profanity Learner                 # Labeling function Test
#     W2 := sklearn RandomForestClassifier    # sklearn test
#     W3 := Length Learner                    # Primitive test
#
#
#     There are 2^4 * (2^3-1) = 112 possible combinations of tests. Testing the pre-processors
#     individually is un-necessary since they are already tested in test_preprocessor_stack.py.
#     Therefore, there are 2^4 possible configurations of pre-weak learner tests. We do not
#     consider the case when all the weak learners are turned off, so we subtract one.
#
#     We are interested in testing the interaction between different types of pre-processors
#     and the weak learners. There are three archetypal pre-processors: csv mappings (P1),
#     symbol-based (P2), and regex-based (P3). We will also test each of the accepted weak
#     learner types: Snorkel labeling function (W1), Sci-kit Learn (W2), and primitives (W3)
#     to ensure that they are functioning correctly.
#
#     +========================================================+
#     |                      TEST CHART                        |
#     +=========+================+==============+==============+
#     | TEST ID | Template Miner | Preprocessors| Weak Learners|
#     +---------+----------------+--------------+--------------+
#     |         |       D        | P1 | P2 | P3 | W1 | W2 | W3 |
#     +---------+----------------+----+----+----+----+----+----+
#     | 1       |       0        |  0 |  0 |  0 |  1 |  0 |  0 |
#     | 2       |       0        |  0 |  0 |  0 |  1 |  1 |  0 |
#     | 3       |       0        |  0 |  0 |  0 |  1 |  0 |  1 |
#     | 4       |       0        |  0 |  0 |  0 |  0 |  1 |  1 |
#     | 5       |       0        |  0 |  0 |  0 |  1 |  1 |  1 |
#     | 6       |       0        |  0 |  0 |  0 |  0 |  0 |  1 |
#     | 7       |       0        |  0 |  0 |  0 |  0 |  1 |  0 |
#     | 9       |       0        |  1 |  0 |  0 |  1 |  0 |  0 |
#     | 9       |       0        |  1 |  0 |  0 |  1 |  1 |  0 |
#     | 10      |       0        |  1 |  0 |  0 |  1 |  0 |  1 |
#     | 11      |       0        |  1 |  0 |  0 |  0 |  1 |  1 |
#     | 12      |       0        |  1 |  0 |  0 |  1 |  1 |  1 |
#     | 13      |       0        |  1 |  0 |  0 |  0 |  0 |  1 |
#     | 14      |       0        |  1 |  0 |  0 |  0 |  1 |  0 |
#     | 15      |       0        |  0 |  1 |  0 |  1 |  0 |  0 |
#     | 16      |       0        |  0 |  1 |  0 |  1 |  1 |  0 |
#     | 17      |       0        |  0 |  1 |  0 |  1 |  0 |  1 |
#     | 18      |       0        |  0 |  1 |  0 |  0 |  1 |  1 |
#     | 19      |       0        |  0 |  1 |  0 |  1 |  1 |  1 |
#     | 20      |       0        |  0 |  1 |  0 |  0 |  0 |  1 |
#     | 21      |       0        |  0 |  1 |  0 |  0 |  1 |  0 |
#     | 22      |       0        |  0 |  0 |  1 |  1 |  0 |  0 |
#     | 23      |       0        |  0 |  0 |  1 |  1 |  1 |  0 |
#     | 24      |       0        |  0 |  0 |  1 |  1 |  0 |  1 |
#     | 25      |       0        |  0 |  0 |  1 |  0 |  1 |  1 |
#     | 26      |       0        |  0 |  0 |  1 |  1 |  1 |  1 |
#     | 27      |       0        |  0 |  0 |  1 |  0 |  0 |  1 |
#     | 28      |       0        |  0 |  0 |  1 |  0 |  1 |  0 |
#     | 29      |       0        |  1 |  1 |  0 |  1 |  0 |  0 |
#     | 30      |       0        |  1 |  1 |  0 |  1 |  1 |  0 |
#     | 31      |       0        |  1 |  1 |  0 |  1 |  0 |  1 |
#     | 32      |       0        |  1 |  1 |  0 |  0 |  1 |  1 |
#     | 33      |       0        |  1 |  1 |  0 |  1 |  1 |  1 |
#     | 34      |       0        |  1 |  1 |  0 |  0 |  0 |  1 |
#     | 35      |       0        |  1 |  1 |  0 |  0 |  1 |  0 |
#     | 36      |       0        |  1 |  0 |  1 |  1 |  0 |  0 |
#     | 37      |       0        |  1 |  0 |  1 |  1 |  1 |  0 |
#     | 38      |       0        |  1 |  0 |  1 |  1 |  0 |  1 |
#     | 39      |       0        |  1 |  0 |  1 |  0 |  1 |  1 |
#     | 40      |       0        |  1 |  0 |  1 |  1 |  1 |  1 |
#     | 41      |       0        |  1 |  0 |  1 |  0 |  0 |  1 |
#     | 42      |       0        |  1 |  0 |  1 |  0 |  1 |  0 |
#     | 43      |       0        |  0 |  1 |  1 |  1 |  0 |  0 |
#     | 44      |       0        |  0 |  1 |  1 |  1 |  1 |  0 |
#     | 45      |       0        |  0 |  1 |  1 |  1 |  0 |  1 |
#     | 46      |       0        |  0 |  1 |  1 |  0 |  1 |  1 |
#     | 47      |       0        |  0 |  1 |  1 |  1 |  1 |  1 |
#     | 48      |       0        |  0 |  1 |  1 |  0 |  0 |  1 |
#     | 49      |       0        |  0 |  1 |  1 |  0 |  1 |  0 |
#     | 50      |       0        |  1 |  1 |  1 |  1 |  0 |  0 |
#     | 51      |       0        |  1 |  1 |  1 |  1 |  1 |  0 |
#     | 52      |       0        |  1 |  1 |  1 |  1 |  0 |  1 |
#     | 53      |       0        |  1 |  1 |  1 |  0 |  1 |  1 |
#     | 54      |       0        |  1 |  1 |  1 |  1 |  1 |  1 |
#     | 55      |       0        |  1 |  1 |  1 |  0 |  0 |  1 |
#     | 56      |       0        |  1 |  1 |  1 |  0 |  1 |  0 |
#     | 57      |       1        |  0 |  0 |  0 |  1 |  0 |  0 |
#     | 58      |       1        |  0 |  0 |  0 |  1 |  1 |  0 |
#     | 59      |       1        |  0 |  0 |  0 |  1 |  0 |  1 |
#     | 60      |       1        |  0 |  0 |  0 |  0 |  1 |  1 |
#     | 61      |       1        |  0 |  0 |  0 |  1 |  1 |  1 |
#     | 62      |       1        |  0 |  0 |  0 |  0 |  0 |  1 |
#     | 63      |       1        |  0 |  0 |  0 |  0 |  1 |  0 |
#     | 64      |       1        |  1 |  0 |  0 |  1 |  0 |  0 |
#     | 65      |       1        |  1 |  0 |  0 |  1 |  1 |  0 |
#     | 66      |       1        |  1 |  0 |  0 |  1 |  0 |  1 |
#     | 67      |       1        |  1 |  0 |  0 |  0 |  1 |  1 |
#     | 68      |       1        |  1 |  0 |  0 |  1 |  1 |  1 |
#     | 69      |       1        |  1 |  0 |  0 |  0 |  0 |  1 |
#     | 70      |       1        |  1 |  0 |  0 |  0 |  1 |  0 |
#     | 71      |       1        |  0 |  1 |  0 |  1 |  0 |  0 |
#     | 72      |       1        |  0 |  1 |  0 |  1 |  1 |  0 |
#     | 73      |       1        |  0 |  1 |  0 |  1 |  0 |  1 |
#     | 74      |       1        |  0 |  1 |  0 |  0 |  1 |  1 |
#     | 75      |       1        |  0 |  1 |  0 |  1 |  1 |  1 |
#     | 76      |       1        |  0 |  1 |  0 |  0 |  0 |  1 |
#     | 77      |       1        |  0 |  1 |  0 |  0 |  1 |  0 |
#     | 78      |       1        |  0 |  0 |  1 |  1 |  0 |  0 |
#     | 79      |       1        |  0 |  0 |  1 |  1 |  1 |  0 |
#     | 80      |       1        |  0 |  0 |  1 |  1 |  0 |  1 |
#     | 81      |       1        |  0 |  0 |  1 |  0 |  1 |  1 |
#     | 82      |       1        |  0 |  0 |  1 |  1 |  1 |  1 |
#     | 83      |       1        |  0 |  0 |  1 |  0 |  0 |  1 |
#     | 84      |       1        |  0 |  0 |  1 |  0 |  1 |  0 |
#     | 85      |       1        |  1 |  1 |  0 |  1 |  0 |  0 |
#     | 86      |       1        |  1 |  1 |  0 |  1 |  1 |  0 |
#     | 87      |       1        |  1 |  1 |  0 |  1 |  0 |  1 |
#     | 88      |       1        |  1 |  1 |  0 |  0 |  1 |  1 |
#     | 89      |       1        |  1 |  1 |  0 |  1 |  1 |  1 |
#     | 90      |       1        |  1 |  1 |  0 |  0 |  0 |  1 |
#     | 91      |       1        |  1 |  1 |  0 |  0 |  1 |  0 |
#     | 92      |       1        |  1 |  0 |  1 |  1 |  0 |  0 |
#     | 93      |       1        |  1 |  0 |  1 |  1 |  1 |  0 |
#     | 94      |       1        |  1 |  0 |  1 |  1 |  0 |  1 |
#     | 95      |       1        |  1 |  0 |  1 |  0 |  1 |  1 |
#     | 96      |       1        |  1 |  0 |  1 |  1 |  1 |  1 |
#     | 97      |       1        |  1 |  0 |  1 |  0 |  0 |  1 |
#     | 98      |       1        |  1 |  0 |  1 |  0 |  1 |  0 |
#     | 99      |       1        |  0 |  1 |  1 |  1 |  0 |  0 |
#     | 100     |       1        |  0 |  1 |  1 |  1 |  1 |  0 |
#     | 101     |       1        |  0 |  1 |  1 |  1 |  0 |  1 |
#     | 102     |       1        |  0 |  1 |  1 |  0 |  1 |  1 |
#     | 103     |       1        |  0 |  1 |  1 |  1 |  1 |  1 |
#     | 104     |       1        |  0 |  1 |  1 |  0 |  0 |  1 |
#     | 105     |       1        |  0 |  1 |  1 |  0 |  1 |  0 |
#     | 106     |       1        |  1 |  1 |  1 |  1 |  0 |  0 |
#     | 107     |       1        |  1 |  1 |  1 |  1 |  1 |  0 |
#     | 108     |       1        |  1 |  1 |  1 |  1 |  0 |  1 |
#     | 109     |       1        |  1 |  1 |  1 |  0 |  1 |  1 |
#     | 110     |       1        |  1 |  1 |  1 |  1 |  1 |  1 |
#     | 111     |       1        |  1 |  1 |  1 |  0 |  0 |  1 |
#     | 112     |       1        |  1 |  1 |  1 |  0 |  1 |  0 |
#     +=========+================+====+====+====+====+====+====+
#
#     """
#
#     import requests
#     import re
#     import zipfile
#     import pandas as pd
#     from snorkel.labeling import labeling_function
#     from sklearn.ensemble import RandomForestClassifier
#     from language_toolkit.filters.string_filter import StringFilter
#     from pathlib import Path
#     from drain3.template_miner import TemplateMiner
#
#     compressed_test_data_path = Path("./tests/test_data.zip")
#     assert compressed_test_data_path.exists(), "Cannot find test data!"
#
#     test_data_path = Path("../tests/spam.csv")
#     if not test_data_path.exists():
#         with zipfile.ZipFile(compressed_test_data_path, 'r') as z:
#             z.extractall(Path('../tests/'))
#
#     test_data = pd.read_csv(test_data_path.absolute(), encoding="ISO-8859-1")
#     test_data.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
#     test_data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
#
#     rsrcs = dict(col_name="text")
#     @labeling_function(name="profanity_learner", resources=rsrcs)
#     def check_profanity(text):
#         url = "https://www.purgomalum.com/service/containsprofanity"
#         params = {'text': text}
#         response = requests.get(url, params=params)
#
#         if response.text == "true":
#             return 2
#         else:
#             return 0
#
# def test_PROF_():
# 	"""Test id: 0 | (0, 0, 0, 0, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_PROF_FOREST_():
# 	"""Test id: 1 | (0, 0, 0, 0, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_PROF_LEN():
# 	"""Test id: 2 | (0, 0, 0, 0, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_FOREST_LEN():
# 	"""Test id: 3 | (0, 0, 0, 0, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_PROF_FOREST_LEN():
# 	"""Test id: 4 | (0, 0, 0, 0, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_LEN():
# 	"""Test id: 5 | (0, 0, 0, 0, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_FOREST_():
# 	"""Test id: 6 | (0, 0, 0, 0, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_PROF_():
# 	"""Test id: 7 | (0, 1, 0, 0, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_PROF_FOREST_():
# 	"""Test id: 8 | (0, 1, 0, 0, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_PROF_LEN():
# 	"""Test id: 9 | (0, 1, 0, 0, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_FOREST_LEN():
# 	"""Test id: 10 | (0, 1, 0, 0, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_PROF_FOREST_LEN():
# 	"""Test id: 11 | (0, 1, 0, 0, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_LEN():
# 	"""Test id: 12 | (0, 1, 0, 0, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_FOREST_():
# 	"""Test id: 13 | (0, 1, 0, 0, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_PROF_():
# 	"""Test id: 14 | (0, 0, 1, 0, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_PROF_FOREST_():
# 	"""Test id: 15 | (0, 0, 1, 0, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_PROF_LEN():
# 	"""Test id: 16 | (0, 0, 1, 0, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_FOREST_LEN():
# 	"""Test id: 17 | (0, 0, 1, 0, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_PROF_FOREST_LEN():
# 	"""Test id: 18 | (0, 0, 1, 0, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_LEN():
# 	"""Test id: 19 | (0, 0, 1, 0, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_FOREST_():
# 	"""Test id: 20 | (0, 0, 1, 0, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_LINK_PROF_():
# 	"""Test id: 21 | (0, 0, 0, 1, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_LINK_PROF_FOREST_():
# 	"""Test id: 22 | (0, 0, 0, 1, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_LINK_PROF_LEN():
# 	"""Test id: 23 | (0, 0, 0, 1, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_LINK_FOREST_LEN():
# 	"""Test id: 24 | (0, 0, 0, 1, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_LINK_PROF_FOREST_LEN():
# 	"""Test id: 25 | (0, 0, 0, 1, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_LINK_LEN():
# 	"""Test id: 26 | (0, 0, 0, 1, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_LINK_FOREST_():
# 	"""Test id: 27 | (0, 0, 0, 1, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_PROF_():
# 	"""Test id: 28 | (0, 1, 1, 0, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_PROF_FOREST_():
# 	"""Test id: 29 | (0, 1, 1, 0, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_PROF_LEN():
# 	"""Test id: 30 | (0, 1, 1, 0, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_FOREST_LEN():
# 	"""Test id: 31 | (0, 1, 1, 0, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_PROF_FOREST_LEN():
# 	"""Test id: 32 | (0, 1, 1, 0, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_LEN():
# 	"""Test id: 33 | (0, 1, 1, 0, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_FOREST_():
# 	"""Test id: 34 | (0, 1, 1, 0, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_LINK_PROF_():
# 	"""Test id: 35 | (0, 1, 0, 1, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_LINK_PROF_FOREST_():
# 	"""Test id: 36 | (0, 1, 0, 1, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_LINK_PROF_LEN():
# 	"""Test id: 37 | (0, 1, 0, 1, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_LINK_FOREST_LEN():
# 	"""Test id: 38 | (0, 1, 0, 1, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_LINK_PROF_FOREST_LEN():
# 	"""Test id: 39 | (0, 1, 0, 1, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_LINK_LEN():
# 	"""Test id: 40 | (0, 1, 0, 1, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_LINK_FOREST_():
# 	"""Test id: 41 | (0, 1, 0, 1, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_LINK_PROF_():
# 	"""Test id: 42 | (0, 0, 1, 1, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_LINK_PROF_FOREST_():
# 	"""Test id: 43 | (0, 0, 1, 1, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_LINK_PROF_LEN():
# 	"""Test id: 44 | (0, 0, 1, 1, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_LINK_FOREST_LEN():
# 	"""Test id: 45 | (0, 0, 1, 1, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_LINK_PROF_FOREST_LEN():
# 	"""Test id: 46 | (0, 0, 1, 1, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_LINK_LEN():
# 	"""Test id: 47 | (0, 0, 1, 1, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_STRIP_LINK_FOREST_():
# 	"""Test id: 48 | (0, 0, 1, 1, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_LINK_PROF_():
# 	"""Test id: 49 | (0, 1, 1, 1, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_LINK_PROF_FOREST_():
# 	"""Test id: 50 | (0, 1, 1, 1, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_LINK_PROF_LEN():
# 	"""Test id: 51 | (0, 1, 1, 1, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_LINK_FOREST_LEN():
# 	"""Test id: 52 | (0, 1, 1, 1, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_LINK_PROF_FOREST_LEN():
# 	"""Test id: 53 | (0, 1, 1, 1, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_LINK_LEN():
# 	"""Test id: 54 | (0, 1, 1, 1, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_AE_STRIP_LINK_FOREST_():
# 	"""Test id: 55 | (0, 1, 1, 1, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_PROF_(template_miner: TemplateMiner):
# 	"""Test id: 56 | (1, 0, 0, 0, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_PROF_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 57 | (1, 0, 0, 0, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_PROF_LEN(template_miner: TemplateMiner):
# 	"""Test id: 58 | (1, 0, 0, 0, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 59 | (1, 0, 0, 0, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_PROF_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 60 | (1, 0, 0, 0, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_LEN(template_miner: TemplateMiner):
# 	"""Test id: 61 | (1, 0, 0, 0, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 62 | (1, 0, 0, 0, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_PROF_(template_miner: TemplateMiner):
# 	"""Test id: 63 | (1, 1, 0, 0, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_PROF_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 64 | (1, 1, 0, 0, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_PROF_LEN(template_miner: TemplateMiner):
# 	"""Test id: 65 | (1, 1, 0, 0, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 66 | (1, 1, 0, 0, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_PROF_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 67 | (1, 1, 0, 0, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_LEN(template_miner: TemplateMiner):
# 	"""Test id: 68 | (1, 1, 0, 0, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 69 | (1, 1, 0, 0, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_PROF_(template_miner: TemplateMiner):
# 	"""Test id: 70 | (1, 0, 1, 0, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_PROF_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 71 | (1, 0, 1, 0, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_PROF_LEN(template_miner: TemplateMiner):
# 	"""Test id: 72 | (1, 0, 1, 0, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 73 | (1, 0, 1, 0, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_PROF_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 74 | (1, 0, 1, 0, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_LEN(template_miner: TemplateMiner):
# 	"""Test id: 75 | (1, 0, 1, 0, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 76 | (1, 0, 1, 0, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_LINK_PROF_(template_miner: TemplateMiner):
# 	"""Test id: 77 | (1, 0, 0, 1, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_LINK_PROF_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 78 | (1, 0, 0, 1, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_LINK_PROF_LEN(template_miner: TemplateMiner):
# 	"""Test id: 79 | (1, 0, 0, 1, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_LINK_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 80 | (1, 0, 0, 1, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_LINK_PROF_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 81 | (1, 0, 0, 1, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_LINK_LEN(template_miner: TemplateMiner):
# 	"""Test id: 82 | (1, 0, 0, 1, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_LINK_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 83 | (1, 0, 0, 1, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_PROF_(template_miner: TemplateMiner):
# 	"""Test id: 84 | (1, 1, 1, 0, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_PROF_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 85 | (1, 1, 1, 0, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_PROF_LEN(template_miner: TemplateMiner):
# 	"""Test id: 86 | (1, 1, 1, 0, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 87 | (1, 1, 1, 0, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_PROF_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 88 | (1, 1, 1, 0, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_LEN(template_miner: TemplateMiner):
# 	"""Test id: 89 | (1, 1, 1, 0, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 90 | (1, 1, 1, 0, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_LINK_PROF_(template_miner: TemplateMiner):
# 	"""Test id: 91 | (1, 1, 0, 1, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_LINK_PROF_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 92 | (1, 1, 0, 1, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_LINK_PROF_LEN(template_miner: TemplateMiner):
# 	"""Test id: 93 | (1, 1, 0, 1, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_LINK_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 94 | (1, 1, 0, 1, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_LINK_PROF_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 95 | (1, 1, 0, 1, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_LINK_LEN(template_miner: TemplateMiner):
# 	"""Test id: 96 | (1, 1, 0, 1, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_LINK_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 97 | (1, 1, 0, 1, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_LINK_PROF_(template_miner: TemplateMiner):
# 	"""Test id: 98 | (1, 0, 1, 1, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_LINK_PROF_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 99 | (1, 0, 1, 1, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_LINK_PROF_LEN(template_miner: TemplateMiner):
# 	"""Test id: 100 | (1, 0, 1, 1, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_LINK_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 101 | (1, 0, 1, 1, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_LINK_PROF_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 102 | (1, 0, 1, 1, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_LINK_LEN(template_miner: TemplateMiner):
# 	"""Test id: 103 | (1, 0, 1, 1, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_STRIP_LINK_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 104 | (1, 0, 1, 1, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_LINK_PROF_(template_miner: TemplateMiner):
# 	"""Test id: 105 | (1, 1, 1, 1, 1, 0, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_LINK_PROF_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 106 | (1, 1, 1, 1, 1, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_LINK_PROF_LEN(template_miner: TemplateMiner):
# 	"""Test id: 107 | (1, 1, 1, 1, 1, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_LINK_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 108 | (1, 1, 1, 1, 0, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_LINK_PROF_FOREST_LEN(template_miner: TemplateMiner):
# 	"""Test id: 109 | (1, 1, 1, 1, 1, 1, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(check_profanity)
# 	sf.add_learner(RandomForestClassifier())
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_LINK_LEN(template_miner: TemplateMiner):
# 	"""Test id: 110 | (1, 1, 1, 1, 0, 0, 1)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(lambda x: 0 if len(x) > 6 else 2)
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
# def test_TM_AE_STRIP_LINK_FOREST_(template_miner: TemplateMiner):
# 	"""Test id: 111 | (1, 1, 1, 1, 0, 1, 0)"""
# 	sf = StringFilter()
# 	sf.add_template_miner(tm)
# 	sf.add_csv_preprocessor(self.csv_path, 0, 1)
# 	sf.add_learner(RandomForestClassifier())
# 	train, test = sf.train_test_split(test_data)
# 	training_results = sf.fit(train)
#
