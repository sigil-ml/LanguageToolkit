import shutil
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

from at_nlp.logger import logger


def data_factory(pull_data: bool, retain_data: bool = False) -> pd.DataFrame:
    r"""Processes and returns the test data as a Pandas DataFrame. This data will be used
    for functional and exhaustive testing of the StringFilter class. It is based on the
    spam dataset from Kaggle:

    https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

    Almeida, T.A., GÃ³mez Hidalgo, J.M., Yamakami, A.
    Contributions to the Study of SMS Spam Filtering: New Collection and Results.
    Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11),
    Mountain View, CA, USA, 2011.

    This dataframe has 5574 rows and 2 columns. The first column identifies if the
    corresponding text is spam (2) or not (0). The second column contains the text
    message. The columns are named "label" and "text" respectively.
    """

    tmp_dir = Path("./tmp")
    data_dir = Path("./data")

    if not data_dir.exists():
        data_dir.mkdir()

    # We are not using the Kaggle version of this link because it requires you to pass
    # in your Kaggle credentials to download the file. Instead, we are using the UCI
    # version from their website.
    if pull_data:
        url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        response = requests.get(url)
        logger.info("Downloading test data...")
        if response.status_code == 200:
            if not tmp_dir.exists():
                tmp_dir.mkdir()
            with open("./tmp/test_data.zip", "wb") as file:
                file.write(response.content)
        else:
            logger.warning("Failed to download the ZIP file.")

    # This is required because there is a readme file in the zip file, so we cannot
    # directly extract with Pandas
    logger.trace("Extracting test data...")
    with ZipFile(Path("./tmp/test_data.zip"), "r") as z:
        z.extract("SMSSpamCollection", data_dir)
    if not retain_data:
        logger.info(f"Removing {tmp_dir.absolute()}...")
        shutil.rmtree(tmp_dir)

    logger.trace("Loading test data into Pandas DataFrame...")
    test_data = pd.read_csv(
        Path("./data/SMSSpamCollection").absolute(),
        delimiter="\t",
        names=["label", "text"],
    )

    def preprocess(s: str) -> int:
        match s:
            case "ham":
                return 0
            case "spam":
                return 2
            case _:
                return -1

    logger.trace("Preprocessing label column...")
    test_data["label"] = test_data["label"].apply(preprocess)
    return test_data