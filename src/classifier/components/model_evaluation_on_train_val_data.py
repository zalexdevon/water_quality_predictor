import pandas as pd
import os
from classifier import logger
from classifier.entity.config_entity import ModeEvaluationOnTrainValDataConfig
from Mylib import myfuncs
from sklearn import metrics
from Mylib import myclasses


class ModeEvaluationOnTrainValData:
    def __init__(self, config: ModeEvaluationOnTrainValDataConfig):
        self.config = config

    def evaluate_model(self):
        # Load data
        train_feature = myfuncs.load_python_object(
            os.path.join(self.config.data_transformation_path), "train_features.pkl"
        )
        train_target = myfuncs.load_python_object(
            os.path.join(self.config.data_transformation_path), "train_target.pkl"
        )
        val_feature = myfuncs.load_python_object(
            os.path.join(self.config.data_transformation_path), "val_features.pkl"
        )
        val_target = myfuncs.load_python_object(
            os.path.join(self.config.data_transformation_path), "val_target.pkl"
        )

        # Load model
        model = myfuncs.load_python_object(self.config.model_path)

        class_names = myfuncs.load_python_object(
            os.path.join(self.config.data_transformation_path, "class_names.pkl")
        )

        # Các chỉ số đánh giá của model
        self.model_results_text = "========KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH================\n"

        # Các chỉ số khác
        model_results_text, test_confusion_matrix = myclasses.ClassifierEvaluator(
            model=model,
            class_names=class_names,
            train_feature_data=train_feature,
            train_target_data=train_target,
            val_feature_data=val_feature,
            val_target_data=val_target,
        ).evaluate()

        self.model_results_text += model_results_text

        print(self.model_results_text)

        test_confusion_matrix_path = os.path.join(
            self.config.root_dir, "test_confusion_matrix.png"
        )
        test_confusion_matrix.savefig(
            test_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
        )

        # Lưu chỉ số đánh giá vào file results.txt
        with open(os.path.join(self.config.root_dir, "result.txt"), mode="w") as file:
            file.write(self.model_results_text)
