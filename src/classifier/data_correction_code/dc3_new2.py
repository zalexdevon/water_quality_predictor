import pandas as pd
from Mylib import myfuncs
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from classifier.data_correction_code import dc1_new2


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        df = X

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class TransformerOnTrain(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        df = X

        print("Tiến hành TransformerOnTrain !!!!!!")

        cols = [
            "Iron_num",
            "Nitrate_num",
            "Chloride_num",
            "Lead_num",
            "Zinc_num",
            "Turbidity_num",
            "Fluoride_num",
            "Copper_num",
            "Sulfate_num",
            "Manganese_num",
            "Water_Temperature_num",
        ]
        myfuncs.log_many_columns_57(df, cols)

        return df

    def transform(self, X, y=None):

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


dc = Pipeline(
    steps=[
        ("1", dc1_new2.dc),
        ("2", Transformer()),
        ("3", TransformerOnTrain()),
    ]
)

FEATURE_ORDINAL_DICT = dc1_new2.FEATURE_ORDINAL_DICT
