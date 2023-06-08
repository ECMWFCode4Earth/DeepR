from pathlib import Path
from typing import Dict, Tuple
import diffusers
from torch import nn
from torch.utils.data import IterableDataset

from deepr.data.configuration import DataConfiguration
from deepr.data.generator import DataGenerator
from deepr.data.scaler import XarrayStandardScaler
from deepr.model.configs import TrainingConfig
from deepr.model.diffusion_trainer import train_diffusion
from deepr.model.nn_trainer import train_nn
from deepr.utilities.logger import get_logger
from deepr.utilities.yml import read_yaml_file

logger = get_logger(__name__)


def get_neural_network(
    class_name: str, kwargs: dict, sample_size: Tuple[int] = None
) -> nn.Module:
    """Get neural network.

    Given a class name and a dictionary of keyword arguments, returns an instance of a
    neural network. Current options are: "UNet".

    Arguments
    ---------
    class_name : str
        The name of the neural network class to use.
    kwargs : dict
        Dictionary of keyword arguments to pass to the neural network constructor.
    sample_size : Optional[tuple]
        Sample size of the target samples.

    Returns
    -------
    model: nn.Module
        An instance of a neural network.

    Raises:
    ------
        NotImplementedError: If the specified neural network class is not implemented.
    """
    if "sample_size" in kwargs:
        kwargs["sample_size"] = tuple(kwargs["sample_size"])
    elif sample_size is None:
        raise ValueError(f"sample_size must be specified for {class_name}") 
    else:
        kwargs["sample_size"] = sample_size

    if class_name.lower() == "unet":
        from deepr.model.unet import UNet

        return UNet(**kwargs)
    elif class_name.split(".")[0].lower() == "diffusers":
        import diffusers

        return diffusers.__dict__[class_name.split(".")[1]](**kwargs)
    else:
        raise NotImplementedError(f"{class_name} is not implemented")


def get_hf_scheduler(class_name: str, kwargs: dict) -> diffusers.SchedulerMixin:
    logger.info(f"Loading scheduler {class_name}.")
    return getattr(diffusers, class_name)(**kwargs)


class MainPipeline:
    def __init__(self, configuration_file: Path):
        """
        Initialize the MainPipeline class.

        Parameters
        ----------
        configuration_file : Path
            Path to the configuration file.
        """
        logger.info(f"Reading experiment configuration from file {configuration_file}.")
        self.configuration = read_yaml_file(configuration_file)

    def _prepare_data_cfg_log(self) -> Dict:
        config = self.configuration["data_configuration"]
        for key, val in config.items():
            # Drop data dir
            if "data_dir" in val.keys():
                config[key].pop("data_dir")
            logger.info(f"{key.capitalize().replace('_', ' ')} configuration:")
            logger.info("\n".join([f"\t{k}: {v}" for k, v in val.items()]))
        
        return config

    def get_dataset(self) -> (IterableDataset, IterableDataset):
        """
        Initialize the data_loader for the pipeline.

        Returns
        -------
        data_generator : Dataset
            The initialized DataGenerator object.
        """
        logger.info("Loading configuration...")
        data_configuration = DataConfiguration(self.configuration["data_configuration"])
        logger.info("Get features from data_configuration dictionary.")
        features_collection = data_configuration.get_features()
        (
            features_collection_train,
            features_collection_val,
        ) = features_collection.split_data(
            self.configuration["data_configuration"]["common_configuration"][
                "data_split"
            ]["validation"]
        )
        if self.configuration["data_configuration"]["features_configuration"][
            "apply_standardization"
        ]:
            features_scaler = XarrayStandardScaler(features_collection_train)
        else:
            features_scaler = None
        logger.info("Get label from data_configuration dictionary.")
        label_collection = data_configuration.get_label()
        label_collection_train, label_collection_val = label_collection.split_data(
            self.configuration["data_configuration"]["common_configuration"][
                "data_split"
            ]["validation"]
        )
        if self.configuration["data_configuration"]["features_configuration"][
            "apply_standardization"
        ]:
            label_scaler = XarrayStandardScaler(label_collection_train)
        else:
            label_scaler = None
        logger.info("Define the DataGenerator object.")
        data_generator_train = DataGenerator(
            features_collection_train,
            label_collection_train,
            features_scaler,
            label_scaler,
        )
        data_generator_val = DataGenerator(
            features_collection_val, label_collection_val, features_scaler, label_scaler
        )
        return data_generator_train, data_generator_val

    def train_diffusion(self, dataset: IterableDataset, dataset_val: IterableDataset):
        """Train a Deep Diffusion model with the given dataset.

        Arguments
        ---------
        dataset : IterableDataset
            The dataset to train the model on.
        dataset_val: IterableDataset
            The dataset to validate the model on.

        Returns
        -------
        model : nn.Module
            The trained model.
        """
        logger.info("Train Deep Diffusion model for Super Resolution task.")
        train_configs = self.configuration["training_configuration"]
        model_cfg = train_configs["model_configuration"].pop("eps_model")
        scheduler_cfg = train_configs["model_configuration"].pop("scheduler")
        train_cfg = TrainingConfig(**train_configs["training_parameters"])

        # Instantiate objects
        eps_model = get_neural_network(**model_cfg, sample_size=dataset.output_shape)
        scheduler = get_hf_scheduler(**scheduler_cfg)

        # Train the diffusion model
        train_diffusion(
            train_cfg, 
            eps_model, 
            scheduler, 
            dataset, 
            dataset_val, 
            self._prepare_data_cfg_log(),
        )

    def train_end2end_nn(
        self, dataset: IterableDataset, val_dataset: IterableDataset
    ) -> nn.Module:
        """Train a end-to-end neural network with the given dataset.

        Arguments
        ---------
        dataset : IterableDataset
            The dataset to train the model on.
        dataset_val: IterableDataset
            The dataset to validate the model on.

        Returns
        -------
        model : nn.Module
            The trained neural network.
        """
        model_cfg = train_configs["model_configuration"].pop("neural_network")
        model = get_neural_network(**model_cfg, sample_size=dataset.output_shape)
        
        train_nn(config, model, dataset, val_dataset, self._prepare_data_cfg_log())

        raise NotImplementedError("Not implemented yet")

    def train_model(self, dataset: IterableDataset, dataset_val: IterableDataset):
        """Train a Super Resolution (SR) model with the given dataset.

        Arguments
        ---------
        dataset : IterableDataset
            The dataset to train the model on.
        dataset_val: IterableDataset

        Returns
        -------
        model : nn.Module | SuperResolutionDenoiseDiffusion
            The trained neural network.
        """
        model_type = self.configuration["training_configuration"]["type"]
        if model_type == "diffusion":
            self.train_diffusion(dataset, dataset_val)
        elif model_type == "end2end":
            self.train_end2end_nn(dataset, dataset_val)
        else:
            raise NotImplementedError(
                f"The training procedure {model_type} is not supported."
            )

    def run_pipeline(self):
        """Run the pipeline and return the data generator."""
        dataset_train, dataset_val = self.get_dataset()
        self.train_model(dataset_train, dataset_val)
