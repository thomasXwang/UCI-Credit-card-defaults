import pandas as pd
from pathlib import Path


def load_raw_data(path):
    """Returns the raw CSV data from path (in the same folder as the notebook) as a pandas DataFrame"""
    root = Path('.')
    df = pd.read_csv(root / path)
    return df


def load_data(path):
    """Returns the CSV data from path (in the same folder as the notebook) as a (X,y) couple usable for training"""
    df = load_raw_data(path)

    # Dropping ID column
    df = df.drop(columns='ID')

    # Renaming a few misnamed columns
    df = df.rename(columns={'PAY_0': 'PAY_1',
                            'default.payment.next.month': 'default',
                            'SEX': 'GENDER'})
    # Lowercase our column names
    colonnes = list(df.columns)
    renaming_dict = {}
    new_colonnes = []
    for colonne in colonnes:
        new_colonne = colonne.lower()
        renaming_dict[colonne] = new_colonne
        new_colonnes.append(new_colonne)
    df = df.rename(columns=renaming_dict)

    # Cleaning our mislabelled Data
    # Cleaning our mislabelled pay_n data
    pay_columns = ['pay_' + str(i) for i in range(1, 7)]
    for pay_col in pay_columns:
        df.loc[df[pay_col] > 0, pay_col] = 1
        df.loc[df[pay_col] <= 0, pay_col] = 0
    # Cleaning our mislabelled education data
    df.loc[df['education'].isin([0, 4, 5, 6]), 'education'] = 4
    # Cleaning our mislabelled education data
    df.loc[df['marriage'].isin([0, 3, 4]), 'marriage'] = 3

    # Defining our (X, y)
    X = df.drop(['default'], axis=1)
    y = df[['default']]

    return X, y