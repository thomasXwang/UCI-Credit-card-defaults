import pandas as pd


def bin_age(df):
    """Returns df with new column 'age_bin' and without column 'age'"""
    bin_names = []
    for i in range(2, 8):
        bin_names.append(str(i) + '0s')
    df['age_bin'] = pd.cut(df['age'], 6, labels=bin_names)

    df['age_bin'] = df['age_bin'].cat.add_categories('60+')
    df.loc[(df['age_bin'] == '60s') | (df['age_bin'] == '70s'), 'age_bin'] = '60+'
    df['age_bin'] = df['age_bin'].cat.remove_unused_categories()
    # df = df.drop(columns='age')
    return df


def gender_marriage(df):
    """Returns df with a cross-categorical column gender x marriage"""




    return None