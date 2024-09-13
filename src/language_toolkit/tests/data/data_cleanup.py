from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    csv_path = Path("./(CUI) alexa_816th_file_1a1.csv")
    df = pd.read_csv(csv_path)
    
    df["labels"].replace(0, 1, inplace=True)
    df["labels"].replace(2, 0, inplace=True)
    df.to_csv("./corrected.csv")
    
    print(len(df["id"].unique()))
    print(df.head())
    
    