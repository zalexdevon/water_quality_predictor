import pandas as pd
from Mylib import myfuncs
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from classifier.data_correction_code import dc1_new2


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, weights) -> None:
        super().__init__()
        self.weights = weights

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        df = X

        df = df.drop(
            columns=["Conductivity_num", "Water_Temperature_num", "Source_nom"]
        )
        cols = ["Nitrate_num", "Chloride_num", "Sulfate_num"]
        myfuncs.log_many_columns_57(df, cols)

        self.cols = df.columns.tolist()

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


dc = Pipeline(
    steps=[
        ("1", dc1_new2.dc),
        ("2", Transformer()),
    ]
)

FEATURE_ORDINAL_DICT = dc1_new2.FEATURE_ORDINAL_DICT
