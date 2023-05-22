from labml import experiment
from torch.utils.data import DataLoader

from deepr.data.configuration import DataConfiguration
from deepr.data.generator import DataGenerator
from deepr.model.configs import Configs
from deepr.utilities.logger import get_logger
from deepr.utilities.yml import read_yaml_file

logger = get_logger(__name__)


class MainPipeline:
    def __init__(self, configuration_file: str):
        """
        Initialize the MainPipeline class.

        Parameters
        ----------
        configuration_file : str
            Path to the configuration file.
        """
        self.configuration = read_yaml_file(configuration_file)

    def run_pipeline(self):
        """
        Run the pipeline and return the data generator.

        Returns
        -------
        DataGenerator
            The initialized DataGenerator object.
        """
        experiment.create(name="diffuse", writers={"tensorboard", "screen", "labml"})
        logger.info("Prepare DataLoader object for modeling")
        dataset, data_loader = self.prepare_dataloader()

        configs = Configs()
        configs.init()
        experiment.configs(configs, {"dataset": dataset, "data_loader": data_loader})
        experiment.add_pytorch_models({"eps_model": configs.eps_model})

        with experiment.start():
            configs.run()

    def prepare_dataloader(self):
        """
        Initialize the data_loader for the pipeline.

        Returns
        -------
        DataLoader
            The initialized DataLoader object.
        """
        data_configuration = DataConfiguration(self.configuration["data_configuration"])
        logger.info("Get features from data_configuration dictionary")
        features_collection = data_configuration.get_features()
        logger.info("Get label from data_configuration dictionary")
        label_collection = data_configuration.get_label()
        logger.info("Define the DataGenerator object")
        data_generator = DataGenerator(features_collection, label_collection)
        logger.info("Define the DataLoader object")
        data_loader = DataLoader(
            dataset=data_generator,
            batch_size=data_configuration.common_configuration["batch_size"],
        )
        return data_generator, data_loader


if __name__ == "__main__":
    MainPipeline("../resources/configuration.yml").run_pipeline()
