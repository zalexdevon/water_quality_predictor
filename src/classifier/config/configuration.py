from classifier.constants import *
from Mylib.myfuncs import read_yaml, create_directories
from classifier.entity.config_entity import (
    DataCorrectionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModeEvaluationOnTrainValDataConfig,
    ModelEvaluationConfig,
    MonitorPlotterConfig,
    TestDataCorrectionConfig,
)
import os


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        plot_components_path = os.path.join(self.config.plot_dir, "components")

        create_directories(
            [self.config.artifacts_root, self.config.plot_dir, plot_components_path]
        )

    def get_data_correction_config(self) -> DataCorrectionConfig:
        config = self.config.data_correction
        params = self.params.data_correction

        create_directories([config.root_dir])

        data_correction_config = DataCorrectionConfig(
            # config input
            train_data_path=config.train_data_path,
            # config output
            root_dir=config.root_dir,
            # param
            name=params.name,
        )

        return data_correction_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.data_transformation

        create_directories([config.root_dir])

        number = params.number
        batch_size = int(number.split("_")[-1]) if "batch" in number else None

        is_weighted = True if "weighted" in number else False

        data_transformation_config = DataTransformationConfig(
            # config input
            data_correction_path=config.data_correction_path,
            weights_path=config.weights_path,
            val_data_path=config.val_data_path,
            # config output
            root_dir=config.root_dir,
            # params
            number=params.number,
            do_smote=params.do_smote,
            list_after_feature_transformer=params.list_after_feature_transformer,
            # params được suy ra
            batch_size=batch_size,
            is_weighted=is_weighted,
        )

        return data_transformation_config

    def get_model_trainer_config(
        self,
    ) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer

        create_directories([config.root_dir])

        name = params.model_name
        do_run_on_batch = True if "batch" in name else False
        do_run_with_multithreading = True if "mt" in name else False

        model_trainer_config = ModelTrainerConfig(
            # config input
            data_transformation_path=config.data_transformation_path,
            # config output
            root_dir=config.root_dir,
            # config common
            plot_dir=self.config.plot_dir,
            # params
            model_name=params.model_name,
            model_training_type=params.model_training_type,
            # params to use GridSearch
            base_model=params.base_model,
            n_iter=params.n_iter,
            param_grid=params.param_grid,
            # params to train many models
            models=params.models,
            # common params
            scoring=self.params.scoring,
            target_score=self.params.target_score,
            # param được suy ra
            do_run_on_batch=do_run_on_batch,
            do_run_with_multithreading=do_run_with_multithreading,
        )

        return model_trainer_config

    # MODEL EVALUATION ON TRAIN VAL
    def get_model_evaluation_on_train_val_config(
        self,
    ) -> ModeEvaluationOnTrainValDataConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer

        create_directories([config.root_dir])

        obj = ModeEvaluationOnTrainValDataConfig(
            # config input
            data_transformation_path=config.data_transformation_path,
            model_path=config.model_path,
            # config output
            root_dir=config.root_dir,
        )

        return obj

    # TEST DATA CORRECTION
    def get_test_data_correction_config(self) -> TestDataCorrectionConfig:
        config = self.config.test_data_correction

        create_directories([config.root_dir])

        obj = TestDataCorrectionConfig(
            # input
            test_raw_data_path=config.test_raw_data_path,
            preprocessor_path=config.preprocessor_path,
            # output
            root_dir=config.root_dir,
        )

        return obj

    # MODEL_EVALUATION
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        obj = ModelEvaluationConfig(
            # input
            test_data_path=config.test_data_path,
            data_transformation_path=config.data_transformation_path,
            model_path=config.model_path,
            # output
            root_dir=config.root_dir,
            # common params
            scoring=self.params.scoring,
        )

        return obj

    def get_monitor_plot_config(self) -> MonitorPlotterConfig:
        params = self.params.monitor_plotter

        obj = MonitorPlotterConfig(
            plot_dir=self.config.plot_dir,
            target_val_value=params.target_val_value,
            max_val_value=params.max_val_value,
            dtick_y_value=params.dtick_y_value,
        )

        return obj
