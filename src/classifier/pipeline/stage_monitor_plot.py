from classifier.config.configuration import ConfigurationManager
from classifier import logger
from classifier.components.monitor_plotter import (
    MonitorPlotter,
)
import traceback

STAGE_NAME = "Model Result Plot stage"


class MonitorPlotterPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        monitor_plotter_config = config.get_monitor_plot_config()

        monitor_plot = MonitorPlotter(config=monitor_plotter_config)

        try:
            monitor_plot.load_data()
            print("\n===== Load plot components thành công ====== \n")

            monitor_plot.plot()
            print("\n===== Plot thành công ====== \n")

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
        obj = MonitorPlotterPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
