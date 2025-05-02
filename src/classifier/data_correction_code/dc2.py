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


class DC:
    def __init__(self):
        pass

    def transform(self, df, data_type):
        # TODO: d
        print("Chạy vào DC2 đây nè nhen man !!!!!")
        # d

        # Xóa các cột không cần thiết
        df = df.drop(
            columns=[
                "Ethnicity",
                "EmploymentStatus",
                "MaritalStatus",
                "AlcoholConsumption",
                "Residence",
                "Diet",
                "PhysicalActivity",
            ]
        )

        #  Đổi tên cột
        rename_dict = {
            "Age": "Age_num",
            "Gender": "Gender_nom",
            "Cholesterol": "Cholesterol_num",
            "BloodPressure": "BloodPressure_num",
            "HeartRate": "HeartRate_num",
            "BMI": "BMI_num",
            "Smoker": "Smoker_bin",
            "Diabetes": "Diabetes_bin",
            "Hypertension": "Hypertension_bin",
            "FamilyHistory": "FamilyHistory_bin",
            "StressLevel": "StressLevel_numcat",
            "Income": "Income_num",
            "EducationLevel": "EducationLevel_ord",
            "Medication": "Medication_bin",
            "ChestPainType": "ChestPainType_nom",
            "ECGResults": "ECGResults_nom",
            "MaxHeartRate": "MaxHeartRate_num",
            "ST_Depression": "ST_Depression_num",
            "ExerciseInducedAngina": "ExerciseInducedAngina_bin",
            "Slope": "Slope_nom",
            "NumberOfMajorVessels": "NumberOfMajorVessels_numcat",
            "Thalassemia": "Thalassemia_nom",
            "PreviousHeartAttack": "PreviousHeartAttack_bin",
            "StrokeHistory": "StrokeHistory_bin",
            "Outcome": "Outcome_target",
        }

        df = df.rename(columns=rename_dict)

        # Sắp xếp các cột theo đúng thứ tự
        (
            numeric_cols,
            numericCat_cols,
            cat_cols,
            binary_cols,
            nominal_cols,
            ordinal_cols,
            target_col,
        ) = myfuncs.get_different_types_cols_from_df_4(df)

        df = df[
            numeric_cols
            + numericCat_cols
            + binary_cols
            + nominal_cols
            + ordinal_cols
            + [target_col]
        ]

        # Xử lí missing value
        if data_type == "train":
            self.missing_value_transformer = ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy="mean"), numeric_cols),
                    (
                        "numCat",
                        SimpleImputer(strategy="most_frequent"),
                        numericCat_cols,
                    ),
                    ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
                    ("target", SimpleImputer(strategy="most_frequent"), [target_col]),
                ]
            )
            self.missing_value_transformer.fit(df)

        df = self.missing_value_transformer.transform(df)
        df = pd.DataFrame(
            df, columns=numeric_cols + numericCat_cols + cat_cols + [target_col]
        )

        # Chuyển đổi về đúng kiểu dữ liệu
        df[numeric_cols] = df[numeric_cols].astype("float32")
        df[numericCat_cols] = df[numericCat_cols].astype("float32")
        df[cat_cols] = df[cat_cols].astype("category")
        df[target_col] = df[target_col].astype("category")

        # Loại bỏ duplicates
        df = df.drop_duplicates().reset_index(drop=True)

        return df


FEATURE_ORDINAL_DICT = {
    "Smoker_bin": [0, 1],
    "Diabetes_bin": [0, 1],
    "Hypertension_bin": [0, 1],
    "FamilyHistory_bin": [0, 1],
    "Medication_bin": ["No", "Yes"],
    "ExerciseInducedAngina_bin": ["No", "Yes"],
    "PreviousHeartAttack_bin": [0, 1],
    "StrokeHistory_bin": [0, 1],
    "EducationLevel_ord": ["High School", "College", "Postgraduate"],
}
