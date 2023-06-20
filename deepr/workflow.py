from pathlib import Path
from typing import Dict, Tuple

from torch import nn
from torch.utils.data import IterableDataset

from deepr.data.configuration import DataConfiguration
from deepr.data.generator import DataGenerator
from deepr.data.scaler import XarrayStandardScaler
from deepr.model.configs import TrainingConfig
from deepr.model.diffusion_trainer import train_diffusion
from deepr.model.models import get_hf_scheduler, get_neural_network
from deepr.model.nn_trainer import train_nn
from deepr.utilities.logger import get_logger
from deepr.utilities.yml import read_yaml_file
from deepr.validation.test_model import test_model

logger = get_logger(__name__)


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
        configuration = read_yaml_file(configuration_file)
        self.data_config = configuration["data_configuration"]
        train_config = configuration["training_configuration"]
        self.pipeline_type = train_config["type"]
        self.model_config = train_config["model_configuration"]
        self.train_config = TrainingConfig(**train_config["training_parameters"])

    def _prepare_data_cfg_log(self) -> Dict:
        config = self.data_config
        for key, val in config.items():
            # Drop data dir
            if "data_dir" in val.keys():
                config[key].pop("data_dir")
            logger.info(f"{key.capitalize().replace('_', ' ')} configuration:")
            logger.info("\n".join([f"\t{k}: {v}" for k, v in val.items()]))

        return config

    def get_dataset(self) -> Tuple[IterableDataset, IterableDataset, IterableDataset]:
        """
        Initialize the data_loader for the pipeline.

        Returns
        -------
        data_generator : Dataset
            The initialized DataGenerator object.
        """
        logger.info("Loading configuration...")
        data_configuration = DataConfiguration(self.data_config)
        data_splits = data_configuration.common_configuration["data_split"]

        test_split_size = data_splits["test"]
        val_split_size = data_splits["validation"] / (1 - test_split_size)

        logger.info("Get features from data_configuration dictionary.")
        features_collection = data_configuration.get_features()
        features_coll_train, features_coll_test = features_collection.split_data(
            test_split_size
        )
        features_coll_train, features_coll_val = features_coll_train.split_data(
            val_split_size
        )
        if data_configuration.features_configuration["apply_standardization"]:
            features_scaler = XarrayStandardScaler(
                features_coll_train,
                data_configuration.common_configuration["cache_dir"],
            )
        else:
            features_scaler = None

        logger.info("Get label from data_configuration dictionary.")
        label_collection = data_configuration.get_label()
        label_coll_train, label_coll_test = label_collection.split_data(test_split_size)
        label_coll_train, label_coll_val = label_coll_train.split_data(val_split_size)
        if data_configuration.features_configuration["apply_standardization"]:
            label_scaler = XarrayStandardScaler(
                label_coll_train, data_configuration.common_configuration["cache_dir"]
            )
        else:
            label_scaler = None

        # Define DataGenerators
        logger.info("Define the DataGenerator object.")
        data_generator_train = DataGenerator(
            features_coll_train,
            data_configuration.features_configuration["add_auxiliary"],
            label_coll_train,
            features_scaler,
            label_scaler,
        )
        data_generator_val = DataGenerator(
            features_coll_val,
            data_configuration.features_configuration["add_auxiliary"],
            label_coll_val,
            features_scaler,
            label_scaler,
        )
        data_generator_test = DataGenerator(
            features_coll_test,
            data_configuration.features_configuration["add_auxiliary"],
            label_coll_test,
            features_scaler,
            label_scaler,
        )
        return data_generator_train, data_generator_val, data_generator_test

    def train_diffusion(self, dataset: IterableDataset, dataset_val: IterableDataset):
        """
        Train a Deep Diffusion model with the given dataset.

        Parameters
        ----------
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
        model_cfg = self.model_config.pop("eps_model")
        scheduler_cfg = self.model_config.pop("scheduler")

        # Instantiate objects
        eps_model = get_neural_network(
            **model_cfg,
            sample_size=dataset.output_shape,
            out_channels=dataset.output_channels,
        )
        scheduler = get_hf_scheduler(**scheduler_cfg)

        # Train the diffusion model
        train_diffusion(
            self.train_config,
            eps_model,
            scheduler,
            dataset,
            dataset_val,
            self._prepare_data_cfg_log(),
        )

    def train_end2end_nn(
        self, dataset_train: IterableDataset, dataset_val: IterableDataset
    ) -> nn.Module:
        """
        Train an end-to-end neural network with the given dataset.

        Parameters
        ----------
        dataset_train : IterableDataset
            The dataset to train the model on.
        dataset_val: IterableDataset
            The dataset to validate the model on.

        Returns
        -------
        model : nn.Module
            The trained neural network.
        """
        model_cfg = self.model_config.pop("neural_network")

        model = get_neural_network(
            **model_cfg,
            out_channels=dataset_train.output_channels,
            sample_size=dataset_train.output_shape,
        )

        return train_nn(
            self.train_config,
            model,
            dataset_train,
            dataset_val,
            self._prepare_data_cfg_log(),
        )

    def train_model(
        self,
        dataset_train: IterableDataset,
        dataset_val: IterableDataset,
    ):
        """
        Train a Super Resolution (SR) model with the given dataset.

        Parameters
        ----------
        dataset_train : IterableDataset
            The dataset to train the model on.
        dataset_val: IterableDataset
            The dataset to validate the model on.

        Returns
        -------
        model : nn.Module | SuperResolutionDenoiseDiffusion
            The trained neural network.
        """
        if self.pipeline_type == "diffusion":
            return self.train_diffusion(dataset_train, dataset_val)
        elif self.pipeline_type == "end2end":
            return self.train_end2end_nn(dataset_train, dataset_val)
        else:
            raise NotImplementedError(
                f"The training procedure {self.pipeline_type} is not supported."
            )

    def test_model(self, model, dataset, hf_repo_name: str = None):
        push_to_hf = hf_repo_name is not None and self.train_config.push_to_hub
        hparams = self.data_config.get("data_split", None)
        return test_model(
            model,
            dataset,
            hparams=hparams,
            push_to_hub=push_to_hf,
            batch_size=self.train_config.batch_size,
        )

    def run_pipeline(self):
        """Run the pipeline and return the data generator."""
        dataset_train, dataset_val, dataset_test = self.get_dataset()
        model, repo_name = self.train_model(dataset_train, dataset_val)
        self.test_model(model, dataset_test, hf_repo_name=repo_name)
