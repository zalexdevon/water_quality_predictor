from classifier import logger
import pandas as pd
from classifier.entity.config_entity import DataTransformationConfig
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
from Mylib import myclasses
import os


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_data(self):
        self.df_train = myfuncs.load_python_object(
            os.path.join(self.config.data_correction_path, "data.pkl")
        )
        self.feature_ordinal_dict = myfuncs.load_python_object(
            os.path.join(self.config.data_correction_path, "feature_ordinal_dict.pkl")
        )

        self.correction_transformer = myfuncs.load_python_object(
            os.path.join(self.config.data_correction_path, "transformer.pkl")
        )

        self.df_val = myfuncs.load_python_object(self.config.val_data_path)

        self.num_train_sample = len(self.df_train)

        self.feature_cols, self.target_col = (
            myfuncs.get_feature_cols_and_target_col_from_df_27(self.df_train)
        )

        # Load các transfomers
        self.list_after_feature_transformer = [
            stringToObjectConverter.convert_MLmodel_yaml_to_object(transformer)
            for transformer in self.config.list_after_feature_transformer
        ]

    def create_preprocessor_for_train_data(self):
        after_feature_pipeline = (
            Pipeline(
                steps=[
                    (str(index), transformer)
                    for index, transformer in enumerate(
                        self.list_after_feature_transformer
                    )
                ]
            )
            if len(self.list_after_feature_transformer) > 0
            else Pipeline(steps=[("passthrough", "passthrough")])
        )

        feature_pipeline = Pipeline(
            steps=[
                (
                    "during",
                    myclasses.DuringFeatureTransformer(self.feature_ordinal_dict),
                ),
                ("after", after_feature_pipeline),
            ]
        )

        column_transformer = ColumnTransformer(
            transformers=[
                ("feature", feature_pipeline, self.feature_cols),
                ("target", OrdinalEncoder(), [self.target_col]),
            ]
        )

        self.transformation_transformer = myclasses.NamedColumnTransformer(
            column_transformer
        )

    def transform_data(self):
        df_train_transformed = self.transformation_transformer.fit_transform(
            self.df_train
        )
        df_train_feature = df_train_transformed.drop(columns=[self.target_col]).astype(
            "float32"
        )
        df_train_target = df_train_transformed[self.target_col].astype("int8")

        if self.config.do_smote == "t":
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            df_train_feature, df_train_target = smote.fit_resample(
                df_train_feature, df_train_target
            )

        print(f"Kích thước tập {self.config.number}: {df_train_feature.shape}")

        df_val_corrected = self.correction_transformer.transform(self.df_val)
        df_val_transformed = self.transformation_transformer.transform(df_val_corrected)
        df_val_feature = df_val_transformed.drop(columns=[self.target_col]).astype(
            "float32"
        )
        df_val_target = df_val_transformed[self.target_col].astype("int8")

        class_names = list(
            self.transformation_transformer.column_transformer.named_transformers_[
                "target"
            ].categories_[0]
        )

        myfuncs.save_python_object(
            os.path.join(self.config.root_dir, "transformer.pkl"),
            self.transformation_transformer,
        )
        myfuncs.save_python_object(
            os.path.join(self.config.root_dir, "train_features.pkl"), df_train_feature
        )
        myfuncs.save_python_object(
            os.path.join(self.config.root_dir, "train_target.pkl"), df_train_target
        )
        myfuncs.save_python_object(
            os.path.join(self.config.root_dir, "val_features.pkl"), df_val_feature
        )
        myfuncs.save_python_object(
            os.path.join(self.config.root_dir, "val_target.pkl"), df_val_target
        )
        myfuncs.save_python_object(
            os.path.join(self.config.root_dir, "class_names.pkl"), class_names
        )
