import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from .feature_eng import *


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