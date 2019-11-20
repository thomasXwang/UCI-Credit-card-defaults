import pandas as pd
import numpy as np


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


def next_bill_amt(df):
    """Returns df with prediction of next bill amount using exponential smoothing, to know if client will default"""
    # Determining our best alpha
    alphas = np.linspace(0.9, 1, 100)

    median_errors = {}

    for alpha in alphas:
        bill_amts_df = df[['bill_amt' + str(i) for i in range(6, 0, -1)]]
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
    df['pred_bill_amt0'] = pondered_bill_amts_df['pred_bill_amt0']

    return df
