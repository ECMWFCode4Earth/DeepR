from deepr.data.configuration import DataConfiguration
from deepr.data.generator import DataGenerator
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
        data_generator = self.initialize_data()
        return data_generator

    def initialize_data(self):
        """
        Initialize the data for the pipeline.

        Returns
        -------
        DataGenerator
            The initialized DataGenerator object.
        """
        data_configuration = DataConfiguration(self.configuration["data_configuration"])
        features_collection = data_configuration.get_features()
        label_collection = data_configuration.get_label()
        data_generator = DataGenerator(
            features_collection,
            label_collection,
            data_configuration.common_configuration["batch_size"],
        )
        return data_generator


if __name__ == "__main__":
    MainPipeline("../resources/configuration.yml").run_pipeline()
