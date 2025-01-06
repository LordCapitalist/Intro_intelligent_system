import pandas as pd


def Import():
    file_path = "C:.\\spam_assassin.csv"
    data = pd.read_csv(file_path)
    print(data)
    return data


Import()
