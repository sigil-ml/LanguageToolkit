from string_filter import StringFilter
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    drain3_conf = Path("drain3.ini")
    assert drain3_conf.exists(), "Cannot find drain3.ini"

    test_data = Path("../../../nitmre/data/(CUI) alexa_816th_file_1a1.csv")
    assert test_data.exists(), "Cannot find test data"
    data = pd.read_csv(test_data)
    print(data.head())

    acronyms_path = Path("../../../nitmre/data/acronyms.csv")
    assert acronyms_path.exists(), "Cannot find acronyms data"

    sf = StringFilter()

    sf.train(
        data,
        {
            "stage-one": {"split": 0.9, "amt": 2000},
            "stage-two": {"split": 0.9, "amt": 1700},
        },
    )

    model_save_path = Path("./test_model/")
    sf.save_models(model_save_path)
