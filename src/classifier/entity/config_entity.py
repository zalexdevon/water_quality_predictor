from dataclasses import dataclass


@dataclass(frozen=True)
class DataCorrectionConfig:
    # config input
    train_data_path: str

    # config output
    root_dir: str

    # param
    name: str


@dataclass(frozen=True)
class DataTransformationConfig:
    # config input
    data_correction_path: str
    weights_path: str
    val_data_path: str

    # config output
    root_dir: str

    # params
    number: str
    do_smote: str
    list_after_feature_transformer: list

    # params được suy ra
    batch_size: int
    is_weighted: bool


@dataclass(frozen=True)
class ModelTrainerConfig:
    # config input
    data_transformation_path: str

    # config output
    root_dir: str

    # config common
    plot_dir: str

    # params
    model_name: str
    model_training_type: str

    # params to use GridSearch
    base_model: str
    n_iter: int
    param_grid: dict

    # params to train many models
    models: list

    # common params
    scoring: str
    target_score: float

    # param được suy ra
    do_run_on_batch: bool
    do_run_with_multithreading: bool


@dataclass(frozen=True)
class ModeEvaluationOnTrainValDataConfig:
    # config input
    data_transformation_path: str
    model_path: str

    # config output
    root_dir: str


# TEST DATA CORRECTION
@dataclass(frozen=True)
class TestDataCorrectionConfig:
    # input
    test_raw_data_path: str
    preprocessor_path: str

    # output
    root_dir: str


# MODEL_EVALUATION
@dataclass(frozen=True)
class ModelEvaluationConfig:
    # input
    test_data_path: str
    data_transformation_path: str
    model_path: str

    # output
    root_dir: str

    # common params
    scoring: str


@dataclass(frozen=True)
class MonitorPlotterConfig:
    plot_dir: str
    target_val_value: float
    max_val_value: float
    dtick_y_value: float
