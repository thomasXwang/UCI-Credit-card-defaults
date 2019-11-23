import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from .feature_eng import *


# Column Dropper
class ColumnDropper(BaseEstimator, TransformerMixin):

    def __init__(self, column='age'):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return drop_column(X, column=self.column)


# Age Bin Adder
class AgeBinAdder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return bin_age(X)


# Gender X Marriage Variable Adder
class GenderXMarriageAdder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return gender_x_marriage(X)


# Gender X Marriage Variable Adder
class GenderXAgeBinAdder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return gender_x_agebin(X)


# Predicted Next Bill Amount Variable Adder
class NextBillAdder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return next_bill_amt(X)


# https://masongallo.github.io/machine/learning,/python/2017/10/07/machine-learning-pipelines.html
from pandas import Categorical, get_dummies
from sklearn.base import TransformerMixin, BaseEstimator


class CategoricalWarrior(BaseEstimator, TransformerMixin):
    """One hot encoder for all categorical features"""
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        cats = {}
        for column in self.attribute_names:
            cats[column] = X[column].unique().tolist()
        self.categoricals = cats
        return self

    def transform(self, X, y=None):
        df = X.copy()
        for column in self.attribute_names:
            df[column] = Categorical(df[column], categories=self.categoricals[column])
        new_df = get_dummies(df)
        # in case we need them later
        self.columns = new_df.columns
        return new_df