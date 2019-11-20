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


def gender_x_marriage(df):
    """Returns df with a cross-categorical column gender x marriage"""
    df['gen_mar'] = 0
    df = df.assign(gen_mar=list((zip(df.gender, df.marriage))))
    return df


def gender_x_agebin(df):
    """Returns df with a cross-categorical column gender x ageBin"""
    df['gen_ageBin'] = 0
    df = df.assign(gen_ageBin=list(zip(df.gender, df.age_bin)))
    return df
