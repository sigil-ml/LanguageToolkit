import pandas as pd
from pathlib import Path

from loguru import logger

test_config_path = Path("../data/test_config.csv").absolute()
assert test_config_path.exists(), "Could not find test_config.csv"

test_df = pd.read_csv(test_config_path, encoding="utf-8")

module_string = ""

import requests
import re
from snorkel.labeling import labeling_function
from sklearn.ensemble import RandomForestClassifier
from language_toolkit.filters.string_filter import StringFilter


rsrcs = dict(col_name="text")


@labeling_function(name="profanity_learner", resources=rsrcs)
def check_profanity(text):
    url = "https://www.purgomalum.com/service/containsprofanity"
    params = {"text": text}
    response = requests.get(url, params=params)

    if response.text == "true":
        return 2
    else:
        return 0


def t_preprocessor_fn0(ds: pd.Series, position: int) -> pd.Series:
    r"""Test function for testing CRUD operations"""
    s: str = ds.iat[position]
    ds.iat[position] = s.lower()
    return ds


def remove_links(text):
    re.findall("(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+", text)


# def test(self):
#     sf = StringFilter()
#     sf.add_template_miner(tm)
#     sf.add_csv_preprocessor(self.csv_path, 0, 1) # P1
#     sf.add_learner(check_profanity) # W1
#     sf.add_learner(RandomForestClassifier()) # W2
#     sf.add_learner(lambda x: 0 if len(x) > 6 else 2) # W3
#     train, test = sf.train_test_split(test_data)
#     training_results = sf.fit(train)


def better_name(inclusion_tuple: tuple) -> str:
    name = ""
    if bool(inclusion_tuple[1]):
        name += "TM"
        name += "_"
    if bool(inclusion_tuple[2]):
        name += "AE"
        name += "_"
    if bool(inclusion_tuple[3]):
        name += "STRIP"
        name += "_"
    if bool(inclusion_tuple[4]):
        name += "LINK"
        name += "_"
    if bool(inclusion_tuple[5]):
        name += "PROF"
        name += "_"
    if bool(inclusion_tuple[6]):
        name += "FOREST"
        name += "_"
    if bool(inclusion_tuple[7]):
        name += "LEN"
    return name


def test_factory(inclusion_tuple: tuple) -> str:
    test_name = f"test_"
    test_name += better_name(inclusion_tuple)
    logger.info(f"Creating test: {test_name}")
    if bool(inclusion_tuple[1]):  # D
        test_string = f"def {test_name}(template_miner: TemplateMiner):\n"
        test_string += (
            f'\t"""Test id: {inclusion_tuple[0]} | {inclusion_tuple[1:]}"""\n'
        )
        test_string += f"\tsf = StringFilter()\n"
        test_string += f"\tsf.add_template_miner(template_miner)\n"
    else:
        test_string = f"def {test_name}():\n"
        test_string += (
            f'\t"""Test id: {inclusion_tuple[0]} | {inclusion_tuple[1:]}"""\n'
        )
        test_string += f"\tsf = StringFilter()\n"

    if bool(inclusion_tuple[2]):  # P1
        test_string += f"\tsf.add_csv_preprocessor(self.csv_path, 0, 1)\n"
    # if bool(inclusion_tuple[3]): # P2
    #     pass
    # if bool(inclusion_tuple[4]): # P3
    #     pass
    if bool(inclusion_tuple[5]):  # W1
        test_string += f"\tsf.add_learner(check_profanity)\n"
    if bool(inclusion_tuple[6]):  # W2
        test_string += f"\tsf.add_learner(RandomForestClassifier())\n"
    if bool(inclusion_tuple[7]):  # W3
        test_string += f"\tsf.add_learner(lambda x: 0 if len(x) > 6 else 2)\n"
    test_string += f"\ttrain, test = sf.train_test_split(test_data)\n"
    test_string += f"\ttraining_results = sf.fit(train)\n"
    return test_string


if __name__ == "__main__":
    header = r"""
       ______      _             _____ ____            ______        __
      / __/ /_____(_)__  ___ _  / __(_) / /____ ____  /_  __/__ ___ / /_
     _\ \/ __/ __/ / _ \/ _ `/ / _// / / __/ -_) __/   / / / -_|_-</ __/
    /___/\__/_/ /_/_//_/\_, / /_/ /_/_/\__/\__/_/     /_/  \__/___/\__/
                       /___/


    This file tests the various functions of the StringFilter. Specific tests for the
    WeakLearners collection are in test_weak_learner_collection.py, and tests for the
    PreprocessorStack are in test_preprocessor_stack.py.

    We will test the following combination of functions:

    D := 1 if using template miner, 0 otherwise

    Preprocessors:
    P1 := AcronymExpansion
    P2 := Strip Non-ASCII
    P3 := Link remover

    Weak Learners:
    W1 := Profanity Learner                 # Labeling function Test
    W2 := sklearn RandomForestClassifier    # sklearn test
    W3 := Length Learner                    # Primitive test


    There are 2^4 * (2^3-1) = 112 possible combinations of tests. Testing the pre-processors
    individually is un-necessary since they are already tested in test_preprocessor_stack.py.
    Therefore, there are 2^4 possible configurations of pre-weak learner tests. We do not
    consider the case when all the weak learners are turned off, so we subtract one.

    We are interested in testing the interaction between different types of pre-processors
    and the weak learners. There are three archetypal pre-processors: csv mappings (P1),
    symbol-based (P2), and regex-based (P3). We will also test each of the accepted weak
    learner types: Snorkel labeling function (W1), Sci-kit Learn (W2), and primitives (W3)
    to ensure that they are functioning correctly.
    
    +========================================================+
    |                      TEST CHART                        |
    +=========+================+==============+==============+
    | TEST ID | Template Miner | Preprocessors| Weak Learners|
    +---------+----------------+--------------+--------------+
    |         |       D        | P1 | P2 | P3 | W1 | W2 | W3 |
    +---------+----------------+----+----+----+----+----+----+
    | 1       |       0        |  0 |  0 |  0 |  1 |  0 |  0 |
    | 2       |       0        |  0 |  0 |  0 |  1 |  1 |  0 |
    | 3       |       0        |  0 |  0 |  0 |  1 |  0 |  1 |
    | 4       |       0        |  0 |  0 |  0 |  0 |  1 |  1 |
    | 5       |       0        |  0 |  0 |  0 |  1 |  1 |  1 |
    | 6       |       0        |  0 |  0 |  0 |  0 |  0 |  1 |
    | 7       |       0        |  0 |  0 |  0 |  0 |  1 |  0 |
    | 9       |       0        |  1 |  0 |  0 |  1 |  0 |  0 |
    | 9       |       0        |  1 |  0 |  0 |  1 |  1 |  0 |
    | 10      |       0        |  1 |  0 |  0 |  1 |  0 |  1 |
    | 11      |       0        |  1 |  0 |  0 |  0 |  1 |  1 |
    | 12      |       0        |  1 |  0 |  0 |  1 |  1 |  1 |
    | 13      |       0        |  1 |  0 |  0 |  0 |  0 |  1 |
    | 14      |       0        |  1 |  0 |  0 |  0 |  1 |  0 |
    | 15      |       0        |  0 |  1 |  0 |  1 |  0 |  0 |
    | 16      |       0        |  0 |  1 |  0 |  1 |  1 |  0 |
    | 17      |       0        |  0 |  1 |  0 |  1 |  0 |  1 |
    | 18      |       0        |  0 |  1 |  0 |  0 |  1 |  1 |
    | 19      |       0        |  0 |  1 |  0 |  1 |  1 |  1 |
    | 20      |       0        |  0 |  1 |  0 |  0 |  0 |  1 |
    | 21      |       0        |  0 |  1 |  0 |  0 |  1 |  0 |
    | 22      |       0        |  0 |  0 |  1 |  1 |  0 |  0 |
    | 23      |       0        |  0 |  0 |  1 |  1 |  1 |  0 |
    | 24      |       0        |  0 |  0 |  1 |  1 |  0 |  1 |
    | 25      |       0        |  0 |  0 |  1 |  0 |  1 |  1 |
    | 26      |       0        |  0 |  0 |  1 |  1 |  1 |  1 |
    | 27      |       0        |  0 |  0 |  1 |  0 |  0 |  1 |
    | 28      |       0        |  0 |  0 |  1 |  0 |  1 |  0 |
    | 29      |       0        |  1 |  1 |  0 |  1 |  0 |  0 |
    | 30      |       0        |  1 |  1 |  0 |  1 |  1 |  0 |
    | 31      |       0        |  1 |  1 |  0 |  1 |  0 |  1 |
    | 32      |       0        |  1 |  1 |  0 |  0 |  1 |  1 |
    | 33      |       0        |  1 |  1 |  0 |  1 |  1 |  1 |
    | 34      |       0        |  1 |  1 |  0 |  0 |  0 |  1 |
    | 35      |       0        |  1 |  1 |  0 |  0 |  1 |  0 |
    | 36      |       0        |  1 |  0 |  1 |  1 |  0 |  0 |
    | 37      |       0        |  1 |  0 |  1 |  1 |  1 |  0 |
    | 38      |       0        |  1 |  0 |  1 |  1 |  0 |  1 |
    | 39      |       0        |  1 |  0 |  1 |  0 |  1 |  1 |
    | 40      |       0        |  1 |  0 |  1 |  1 |  1 |  1 |
    | 41      |       0        |  1 |  0 |  1 |  0 |  0 |  1 |
    | 42      |       0        |  1 |  0 |  1 |  0 |  1 |  0 |
    | 43      |       0        |  0 |  1 |  1 |  1 |  0 |  0 |
    | 44      |       0        |  0 |  1 |  1 |  1 |  1 |  0 |
    | 45      |       0        |  0 |  1 |  1 |  1 |  0 |  1 |
    | 46      |       0        |  0 |  1 |  1 |  0 |  1 |  1 |
    | 47      |       0        |  0 |  1 |  1 |  1 |  1 |  1 |
    | 48      |       0        |  0 |  1 |  1 |  0 |  0 |  1 |
    | 49      |       0        |  0 |  1 |  1 |  0 |  1 |  0 |
    | 50      |       0        |  1 |  1 |  1 |  1 |  0 |  0 |
    | 51      |       0        |  1 |  1 |  1 |  1 |  1 |  0 |
    | 52      |       0        |  1 |  1 |  1 |  1 |  0 |  1 |
    | 53      |       0        |  1 |  1 |  1 |  0 |  1 |  1 |
    | 54      |       0        |  1 |  1 |  1 |  1 |  1 |  1 |
    | 55      |       0        |  1 |  1 |  1 |  0 |  0 |  1 |
    | 56      |       0        |  1 |  1 |  1 |  0 |  1 |  0 |
    | 57      |       1        |  0 |  0 |  0 |  1 |  0 |  0 |
    | 58      |       1        |  0 |  0 |  0 |  1 |  1 |  0 |
    | 59      |       1        |  0 |  0 |  0 |  1 |  0 |  1 |
    | 60      |       1        |  0 |  0 |  0 |  0 |  1 |  1 |
    | 61      |       1        |  0 |  0 |  0 |  1 |  1 |  1 |
    | 62      |       1        |  0 |  0 |  0 |  0 |  0 |  1 |
    | 63      |       1        |  0 |  0 |  0 |  0 |  1 |  0 |
    | 64      |       1        |  1 |  0 |  0 |  1 |  0 |  0 |
    | 65      |       1        |  1 |  0 |  0 |  1 |  1 |  0 |
    | 66      |       1        |  1 |  0 |  0 |  1 |  0 |  1 |
    | 67      |       1        |  1 |  0 |  0 |  0 |  1 |  1 |
    | 68      |       1        |  1 |  0 |  0 |  1 |  1 |  1 |
    | 69      |       1        |  1 |  0 |  0 |  0 |  0 |  1 |
    | 70      |       1        |  1 |  0 |  0 |  0 |  1 |  0 |
    | 71      |       1        |  0 |  1 |  0 |  1 |  0 |  0 |
    | 72      |       1        |  0 |  1 |  0 |  1 |  1 |  0 |
    | 73      |       1        |  0 |  1 |  0 |  1 |  0 |  1 |
    | 74      |       1        |  0 |  1 |  0 |  0 |  1 |  1 |
    | 75      |       1        |  0 |  1 |  0 |  1 |  1 |  1 |
    | 76      |       1        |  0 |  1 |  0 |  0 |  0 |  1 |
    | 77      |       1        |  0 |  1 |  0 |  0 |  1 |  0 |
    | 78      |       1        |  0 |  0 |  1 |  1 |  0 |  0 |
    | 79      |       1        |  0 |  0 |  1 |  1 |  1 |  0 |
    | 80      |       1        |  0 |  0 |  1 |  1 |  0 |  1 |
    | 81      |       1        |  0 |  0 |  1 |  0 |  1 |  1 |
    | 82      |       1        |  0 |  0 |  1 |  1 |  1 |  1 |
    | 83      |       1        |  0 |  0 |  1 |  0 |  0 |  1 |
    | 84      |       1        |  0 |  0 |  1 |  0 |  1 |  0 |
    | 85      |       1        |  1 |  1 |  0 |  1 |  0 |  0 |
    | 86      |       1        |  1 |  1 |  0 |  1 |  1 |  0 |
    | 87      |       1        |  1 |  1 |  0 |  1 |  0 |  1 |
    | 88      |       1        |  1 |  1 |  0 |  0 |  1 |  1 |
    | 89      |       1        |  1 |  1 |  0 |  1 |  1 |  1 |
    | 90      |       1        |  1 |  1 |  0 |  0 |  0 |  1 |
    | 91      |       1        |  1 |  1 |  0 |  0 |  1 |  0 |
    | 92      |       1        |  1 |  0 |  1 |  1 |  0 |  0 |
    | 93      |       1        |  1 |  0 |  1 |  1 |  1 |  0 |
    | 94      |       1        |  1 |  0 |  1 |  1 |  0 |  1 |
    | 95      |       1        |  1 |  0 |  1 |  0 |  1 |  1 |
    | 96      |       1        |  1 |  0 |  1 |  1 |  1 |  1 |
    | 97      |       1        |  1 |  0 |  1 |  0 |  0 |  1 |
    | 98      |       1        |  1 |  0 |  1 |  0 |  1 |  0 |
    | 99      |       1        |  0 |  1 |  1 |  1 |  0 |  0 |
    | 100     |       1        |  0 |  1 |  1 |  1 |  1 |  0 |
    | 101     |       1        |  0 |  1 |  1 |  1 |  0 |  1 |
    | 102     |       1        |  0 |  1 |  1 |  0 |  1 |  1 |
    | 103     |       1        |  0 |  1 |  1 |  1 |  1 |  1 |
    | 104     |       1        |  0 |  1 |  1 |  0 |  0 |  1 |
    | 105     |       1        |  0 |  1 |  1 |  0 |  1 |  0 |
    | 106     |       1        |  1 |  1 |  1 |  1 |  0 |  0 |
    | 107     |       1        |  1 |  1 |  1 |  1 |  1 |  0 |
    | 108     |       1        |  1 |  1 |  1 |  1 |  0 |  1 |
    | 109     |       1        |  1 |  1 |  1 |  0 |  1 |  1 |
    | 110     |       1        |  1 |  1 |  1 |  1 |  1 |  1 |
    | 111     |       1        |  1 |  1 |  1 |  0 |  0 |  1 |
    | 112     |       1        |  1 |  1 |  1 |  0 |  1 |  0 |
    +=========+================+====+====+====+====+====+====+

    """
    module = ""
    module += 'r"""' + header + '"""\n'

    module += r"""
    import requests
    import re
    import zipfile
    import pandas as pd
    from snorkel.labeling import labeling_function
    from sklearn.ensemble import RandomForestClassifier
    from language_toolkit.filters.string_filter import StringFilter
    from pathlib import Path
    from drain3.template_miner import TemplateMiner
    """

    module += r"""
    compressed_test_data_path = Path("./tests/test_data.zip")
    assert compressed_test_data_path.exists(), "Cannot find test data!"
    
    test_data_path = Path("./tests/spam.csv")
    if not test_data_path.exists():
        with zipfile.ZipFile(compressed_test_data_path, 'r') as z:
            z.extractall(Path('./tests/'))
    
    test_data = pd.read_csv(test_data_path.absolute(), encoding="ISO-8859-1")
    test_data.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
    test_data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    """

    module += r"""
    resources = dict(col_name="text")
    @labeling_function(name="profanity_learner", resources=resources)
    def check_profanity(text):
        url = "https://www.purgomalum.com/service/containsprofanity"
        params = {'text': text}
        response = requests.get(url, params=params)
    
        if response.text == "true":
            return 2
        else:
            return 0
    """

    module += "\n"
    for row in test_df.itertuples():
        module += test_factory(row)
        module += "\n"

    with open("automated_test_string_filter.py", "w") as f:
        f.write(module)
