import pandas as pd
from Mylib import myfuncs
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np


class DC:
    def __init__(self):
        pass

    def transform(self, df, data_type):
        # Xóa các cột không cần thiết
        df = df.drop(
            columns=[
                "taken_time",
            ]
        )

        #  Đổi tên cột
        rename_dict = {
            "pH": "pH_num",
            "Iron": "Iron_num",
            "Nitrate": "Nitrate_num",
            "Chloride": "Chloride_num",
            "Lead": "Lead_num",
            "Zinc": "Zinc_num",
            "Color": "Color_ord",
            "Turbidity": "Turbidity_num",
            "Fluoride": "Fluoride_num",
            "Copper": "Copper_num",
            "Odor": "Odor_num",
            "Sulfate": "Sulfate_num",
            "Conductivity": "Conductivity_num",
            "Chlorine": "Chlorine_num",
            "Manganese": "Manganese_num",
            "Total Dissolved Solids": "Total_Dissolved_Solids_num",
            "Source": "Source_nom",
            "Water Temperature": "Water_Temperature_num",
            "Air Temperature": "Air_Temperature_num",
            "Target": "Target_target",
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

        # Kiểm tra nội dung các cột numeric và numericcat
        col_name = "Air_Temperature_num"
        df.loc[df.index[df[col_name] < 0], col_name] = np.nan

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
    "Color_ord": [
        "Colorless",
        "Near Colorless",
        "Faint Yellow",
        "Light Yellow",
        "Yellow",
    ],
}
