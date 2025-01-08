import openpyxl
import pandas as pd


def Import():
    file_path = ".\\Behandlet_enron_data.xlsx"
    data = pd.read_excel(file_path)
    print(data)
    spam_data = data[data['label_num'] == 1]  
    not_spam_data = data[data['label_num'] == 0]

    print("Spam data:")
    print(spam_data)
        
    print("Not spam data:")
    print(not_spam_data)
    return data


    
Import()

