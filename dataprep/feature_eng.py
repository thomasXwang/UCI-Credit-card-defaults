import numpy as np
import pandas as pd


def drop_column(df, column):
    """Return new DF without column"""
    df2 = df.copy(deep=True)
    df2 = df2.drop(columns=column)
    return df2


def bin_age(df):
    """Returns DF with new column 'age_bin'"""
    df2 = df.copy(deep=True)
    bin_names = []
    for i in range(2, 8):
        bin_names.append(str(i) + '0s')
    df2['age_bin'] = pd.cut(df2['age'], 6, labels=bin_names)

    df2['age_bin'] = df2['age_bin'].cat.add_categories('60+')
    df2.loc[(df2['age_bin'] == '60s') | (df2['age_bin'] == '70s'), 'age_bin'] = '60+'
    df2['age_bin'] = df2['age_bin'].cat.remove_unused_categories()
    return df2


def gender_x_marriage(df):
    """Returns DF with a cross-categorical column gender x marriage"""
    df2 = df.copy(deep=True)
    df2['gen_mar'] = 0
    df2 = df2.assign(gen_mar=list((zip(df.gender, df.marriage))))
    return df2


def gender_x_agebin(df):
    """Returns DF with a cross-categorical column gender x ageBin"""
    df2 = df.copy(deep=True)
    df2['gen_ageBin'] = 0
    df2 = df2.assign(gen_ageBin=list(zip(df.gender, df.age_bin)))
    return df2


def next_bill_amt(df):
    """Returns df with prediction of next bill amount using exponential smoothing, to know if client will default"""
    df2 = df.copy(deep=True)

    # Determining our best alpha
    alphas = np.linspace(0.9, 1, 100)

    median_errors = {}

    for alpha in alphas:
        bill_amts_df = df2[['bill_amt' + str(i) for i in range(6, 0, -1)]]
        pondered_bill_amts_df = bill_amts_df.copy(deep=True)
        for i in range(6, 1, -1):
            pondered_bill_amts_df['bill_amt' + str(i)] = (1 - alpha) ** (i - 2) * pondered_bill_amts_df[
                'bill_amt' + str(i)]
        pondered_bill_amts_df['pred_bill_amt1'] = alpha * pondered_bill_amts_df.drop('bill_amt1', axis=1).sum(axis=1)
        pondered_bill_amts_df['error'] = (
                pondered_bill_amts_df['bill_amt1'] - pondered_bill_amts_df['pred_bill_amt1']).abs()
        median_errors[alpha] = pondered_bill_amts_df['error'].median(axis=0)

    alpha_best = min(median_errors, key=median_errors.get)

    # Predicting our next bill using this value of alpha
    alpha = alpha_best

    pondered_bill_amts_df = df[['bill_amt' + str(i) for i in range(6, 0, -1)]].copy(deep=True)
    for i in range(6, 0, -1):
        pondered_bill_amts_df['bill_amt' + str(i)] = (1 - alpha) ** (i - 1) * pondered_bill_amts_df['bill_amt' + str(i)]
    pondered_bill_amts_df['pred_bill_amt0'] = alpha * pondered_bill_amts_df.sum(axis=1)
    df2['pred_bill_amt0'] = pondered_bill_amts_df['pred_bill_amt0']

    return df2
