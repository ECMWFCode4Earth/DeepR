from torch import nn
from torch.utils.data import Dataset

from deepr.data.configuration import DataConfiguration
from deepr.data.generator import DataGenerator
from deepr.model.configs import DiffusionTrainingConfiguration
from deepr.utilities.logger import get_logger
from deepr.utilities.yml import read_yaml_file

logger = get_logger(__name__)


def get_neural_network(class_name: str, kwargs: dict):
    """Get neural network.

    Given a class name and a dictionary of keyword arguments, returns an instance of a
    neural network. Current options are: "UNet".

    Arguments
    ---------
        class_name : str
            The name of the neural network class to use.
        kwargs : dict
            Dictionary of keyword arguments to pass to the neural network constructor.

    Returns
    -------
        An instance of a neural network.

    Raises:
    ------
        NotImplementedError: If the specified neural network class is not implemented.
    """
    if class_name.lower() == "unet":
        from deepr.model.unet import UNet

        return UNet(**kwargs)
    else:
        raise NotImplementedError(f"{class_name} is not implemented")


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

    def train_diffusion(self, dataset: Dataset) -> nn.Module:
        configs = self.configuration["training_configuration"]["model_configuration"]
        eps_model = get_neural_network(**configs.pop("eps_model"))
        train_conf = DiffusionTrainingConfiguration(eps_model, dataset, **configs)
        return train_conf.run()

    def train_end2end_nn(self, dataset: Dataset) -> nn.Module:
        raise NotImplementedError("Not implemented yet")

    def train_model(self, dataset: Dataset) -> nn.Module:
        model_type = self.configuration["training_configuration"]["type"]
        if model_type == "diffusion":
            return self.train_diffusion(dataset)
        elif model_type == "end2end":
            return self.train_end2end_nn(dataset)
        else:
            raise NotImplementedError(
                f"The training procedure {model_type} is not supported."
            )

    def evaluate_model(self, model: nn.Module):
        raise NotImplementedError("Not implemented yet")

    def run_pipeline(self):
        """
        Run the pipeline and return the data generator.

        Returns
        -------
        DataGenerator
            The initialized DataGenerator object.
        """
        logger.info("Prepare DataLoader object for modeling")
        dataset = self.get_dataset()
        model = self.train_model(dataset)
        self.evaluate_model(model)


if __name__ == "__main__":
    main_pipeline = MainPipeline("./resources/configuration.yml")
    main_pipeline.run_pipeline()
