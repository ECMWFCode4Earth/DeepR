from labml import experiment
from torch.utils.data import Dataset, DataLoader

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
        dataset = self.get_dataset()
        self.train_model(dataset)

    def get_dataset(self):
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
        return data_generator


    def train_model(self, dataset: Dataset):
        configs = Configs()
        config_param = {
            **{"dataset": dataset}, **self.configuration["training_configuration"]
        }
        experiment.configs(configs, config_param)
        configs.init()
        experiment.add_pytorch_models({"eps_model": configs.eps_model})

        with experiment.start():
            configs.run()

if __name__ == "__main__":
    MainPipeline(
        "/home/santacruzm/Git/DeepR/resources/configuration.yml"
    ).run_pipeline()
