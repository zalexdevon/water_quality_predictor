from classifier.config.configuration import ConfigurationManager
from classifier.components.model_evaluation_on_train_val_data import (
    ModeEvaluationOnTrainValData,
)
from classifier import logger
import traceback

STAGE_NAME = "Model Evaluation On Train Val Data stage"


class ModeEvaluationOnTrainValDataPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_on_train_val_config()
        eval = ModeEvaluationOnTrainValData(config=model_evaluation_config)

        try:
            eval.evaluate_model()

            print(f"==== Đánh giá model thành công ===")
            print("================ NO ERORR :)))))))))) ==========================")
        except Exception as e:
            print(f"==========ERROR: =============")
            print(f"Exception: {e}\n")
            print("=====Traceback========\n")
            traceback.print_exc()
            exit(1)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModeEvaluationOnTrainValDataPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
