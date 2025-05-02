import pandas as pd
import os
from classifier import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, GridSearchCV
from classifier.entity.config_entity import ModelTrainerConfig
from Mylib import myfuncs
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from xgboost import XGBClassifier
from scipy.stats import randint
import random
from lightgbm import LGBMClassifier
from sklearn.model_selection import ParameterSampler
from sklearn import metrics
from sklearn.base import clone
import time
from Mylib import myclasses
from Mylib import stringToObjectConverter
import timeit


class ManyModelsAndBatchSizeTypeModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data_to_train(self):
        # Load các training data
        self.num_batch = myfuncs.load_python_object(
            os.path.join(self.config.data_transformation_path, "num_batch.pkl")
        )
        self.val_feature_data = myfuncs.load_python_object(self.config.val_feature_path)
        self.val_target_data = myfuncs.load_python_object(self.config.val_target_path)

        # Load models
        self.models, self.model_indices = myfuncs.get_models_from_yaml_52(
            self.config.models
        )

        self.num_models = len(self.models)

        # Lưu từng model vào file luôn
        for model, model_index in zip(self.models, self.model_indices):
            myfuncs.save_python_object(
                os.path.join(self.config.root_dir, f"{model_index}.pkl"), model
            )

    def train_on_batches(self, model):
        list_train_scoring = []
        for i in range(0, self.num_batch):
            feature_batch = myfuncs.load_python_object(
                os.path.join(
                    self.config.data_transformation_path, f"train_features_{i}.pkl"
                )
            )
            target_batch = myfuncs.load_python_object(
                os.path.join(
                    self.config.data_transformation_path, f"train_target_{i}.pkl"
                )
            )

            # TODO: d
            print(f"kích thước của train: {feature_batch.shape}")
            # d

            myfuncs.fit_model_incremental_53(model, i, feature_batch, target_batch)

            train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
                model,
                feature_batch,
                target_batch,
                self.config.scoring,
            )

            list_train_scoring.append(train_scoring)

        return list_train_scoring[-1]  # Lấy kết quả trên batch cuối cùng

    def train_model(self):
        print(
            f"\n========TIEN HANH TRAIN {self.num_models} MODELS, SỐ BATCH = {self.num_batch} !!!!!!================\n"
        )

        start_time = time.time()
        for model_index in self.model_indices:
            # Load model để train
            model = myfuncs.load_python_object(
                os.path.join(self.config.root_dir, f"{model_index}.pkl")
            )

            # TODO: d
            print("Đã load xong model")
            print(f"Model: {model}")
            # d

            print(f"Bắt đầu train 1 model")
            train_scoring = self.train_on_batches(model)
            print(f"Kết thúc train 1 model")

            val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
                model,
                self.val_feature_data,
                self.val_target_data,
                self.config.scoring,
            )

            # In kết quả
            print("Kết quả của model")
            print(
                f"Model index {model_index}\n -> Train {self.config.scoring}: {train_scoring}, Val {self.config.scoring}: {val_scoring}\n"
            )

            # Lưu model lại
            myfuncs.save_python_object(
                os.path.join(self.config.root_dir, f"{model_index}.pkl"), model
            )

            # Lưu để vẽ biểu đồ
            model_name = f"{self.config.model_name}_{model_index}"

            myfuncs.save_python_object(
                os.path.join(self.config.plot_dir, "components", f"{model_name}.pkl"),
                (model_name, train_scoring, val_scoring),
            )

        all_model_end_time = time.time()
        self.true_all_models_train_time = (all_model_end_time - start_time) / 60
        self.true_average_train_time = self.true_all_models_train_time / self.num_models

        print(f"Thời gian chạy tất cả: {self.true_all_models_train_time}")
        print(f"Thời gian chạy trung bình: {self.true_average_train_time}")

        print(
            f"\n========KET THUC TRAIN {self.num_models} MODELS !!!!!!================\n"
        )
