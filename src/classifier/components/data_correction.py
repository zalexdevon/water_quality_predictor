from classifier import logger
import pandas as pd
from classifier.entity.config_entity import DataCorrectionConfig
from Mylib import myfuncs
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from Mylib import stringToObjectConverter
import re
from sklearn.impute import SimpleImputer
from classifier.data_correction_code.dc_here_here import dc, FEATURE_ORDINAL_DICT
import os


class DataCorrection:
    def __init__(self, config: DataCorrectionConfig):
        self.config = config

    def load_data(self):
        self.df = myfuncs.load_python_object(self.config.train_data_path)

    def create_preprocessor_for_train_data(self):
        self.transformer = dc

    def transform_data(self):
        print(f"Đang correct dữ liệu cho {self.config.name}")

        df = self.transformer.fit_transform(self.df)

        print(f"Kích thước của tập {self.config.name} là {df.shape}")

        myfuncs.save_python_object(
            os.path.join(self.config.root_dir, "data.pkl"),
            df,
        )
        myfuncs.save_python_object(
            os.path.join(self.config.root_dir, "feature_ordinal_dict.pkl"),
            FEATURE_ORDINAL_DICT,
        )
        myfuncs.save_python_object(
            os.path.join(self.config.root_dir, "transformer.pkl"),
            self.transformer,
        )
