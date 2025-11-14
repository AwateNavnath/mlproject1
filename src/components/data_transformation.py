import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor(self, df: pd.DataFrame):
        try:
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            logging.info(f"Numeric columns: {numeric_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_cols),
                    ("cat", categorical_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = train_df.columns[-1]

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessor = self.get_preprocessor(pd.concat([X_train, X_test], ignore_index=True))

            X_train_trans = preprocessor.fit_transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_trans, y_train]
            test_arr = np.c_[X_test_trans, y_test]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Data transformation successful")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
