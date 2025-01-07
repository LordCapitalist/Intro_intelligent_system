import pandas as pd


def Import():
    file_path = ".\\spam_assassin.csv"
    data = pd.read_csv(file_path)
    print(data)
    spam_data = data[data['target'] == 1]  
    not_spam_data = data[data['target'] == 0]

    print("Spam data:")
    print(spam_data)
        
    print("Not spam data:")
    print(not_spam_data)
    return data


    
Import()
