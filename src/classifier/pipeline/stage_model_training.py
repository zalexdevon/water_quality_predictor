from classifier.config.configuration import ConfigurationManager
from classifier.components.model_trainer import (
    ModelTrainer,
)
from classifier.components.many_models_type_model_trainer import (
    ManyModelsTypeModelTrainer,
)
from classifier.components.many_models_and_batch_size_type_model_trainer import (
    ManyModelsAndBatchSizeTypeModelTrainer,
)
from classifier.components.many_models_type_model_trainer_multithreading import (
    ManyModelsTypeModelTrainerMultithreading,
)
from classifier.components.many_models_batch_type_model_trainer_multithreading import (
    ManyModelsBatchTypeModelTrainerMultithreading,
)
from classifier import logger
from classifier.components.monitor_plotter import (
    MonitorPlotter,
)
import traceback

STAGE_NAME = "Model Trainer stage"


#
class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        model_trainer = None
        if model_trainer_config.model_training_type == "m":
            if (
                model_trainer_config.do_run_on_batch
                and model_trainer_config.do_run_with_multithreading == False
            ):
                model_trainer = ManyModelsAndBatchSizeTypeModelTrainer(
                    model_trainer_config
                )
            else:
                model_trainer = ManyModelsTypeModelTrainer(model_trainer_config)

        try:
            model_trainer.load_data_to_train()
            print("\n===== Load data thành công ====== \n")

            model_trainer.train_model()
            print("\n===== Train data thành công ====== \n")

            print("================ NO ERORR :)))))))))) ==========================")
        except Exception as e:
            print(f"==========ERROR: =============")
            print(f"Exception: {e}\n")
            print("=====Traceback========")
            traceback.print_exc()
            exit(1)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
