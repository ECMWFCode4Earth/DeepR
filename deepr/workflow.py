import os
from pathlib import Path
from typing import Dict, Tuple

from torch import nn

from deepr.data.configuration import DataConfiguration
from deepr.data.generator import DataGenerator
from deepr.data.scaler import XarrayStandardScaler
from deepr.model.autoencoder_trainer import train_autoencoder
from deepr.model.configs import TrainingConfig
from deepr.model.diffusion_trainer import train_diffusion
from deepr.model.models import get_hf_scheduler, get_neural_network, load_trained_model
from deepr.model.nn_trainer import train_nn
from deepr.utilities.logger import get_logger
from deepr.utilities.yml import read_yaml_file
from deepr.validation import validation_diffusion, validation_nn

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
        train_config = configuration.get("training_configuration", {})
        self.pipeline_type = train_config.get("type", None)
        self.model_config = train_config.get("model_configuration", None)
        if "training_parameters" in train_config.keys():
            self.train_config = TrainingConfig(
                **train_config.get("training_parameters", {})
            )
        else:
            self.train_config = None
        self.validation_config = configuration.get("validation_configuration", {})
        self.features_scaler = None
        self.label_scaler = None

    def _prepare_data_cfg_log(self) -> Dict:
        """
        Prepare and log the data configuration.

        Returns
        -------
        config : Dict
            The prepared data configuration.
        """
        config = self.data_config
        for key, val in config.items():
            # Drop data dir
            if "data_location" in val.keys():
                config[key].pop("data_location")
            logger.info(f"{key.capitalize().replace('_', ' ')} configuration:")
            logger.info("\n".join([f"\t{k}: {v}" for k, v in val.items()]))

        return config

    def get_dataset(self) -> Tuple[DataGenerator, DataGenerator, DataGenerator]:
        """
        Initialize the data_loader for the pipeline.

        Returns
        -------
        data_generator : Tuple[DataGenerator, DataGenerator, DataGenerator]
            The initialized DataGenerator objects for training, validation, and testing.
        """
        logger.info("Loading configuration...")
        data_configuration = DataConfiguration(self.data_config)
        data_splits = data_configuration.common_configuration["data_split"]

        test_split_size = data_splits.get("test", 0.0)
        if test_split_size < 1.0:
            data_splits.get("validation", 0.0) / (1 - test_split_size)
        else:
            pass

        logger.info("Get features from data_configuration dictionary.")
        train_features, val_features, test_features = data_configuration.get_features()
        static_features = data_configuration.get_static_features()

        if (
            train_features is not None
            and data_configuration.features_configuration["apply_standardization"]
        ):
            features_scaler_file = (
                Path.home() / ".cache_reanalysis_scales" / "features_scale.pkl"
            )
            os.makedirs(features_scaler_file.parent, exist_ok=True)
            self.features_scaler = XarrayStandardScaler(
                train_features, features_scaler_file
            )

        logger.info("Get label from data_configuration dictionary.")
        train_label, val_label, test_label = data_configuration.get_labels()
        static_label = data_configuration.get_static_label()

        if (
            train_label is not None
            and data_configuration.label_configuration["apply_standardization"]
        ):
            label_scaler_file = (
                Path.home() / ".cache_reanalysis_scales" / "label_scale.pkl"
            )
            os.makedirs(label_scaler_file.parent, exist_ok=True)
            self.label_scaler = XarrayStandardScaler(train_label, label_scaler_file)

        # Define DataGenerators
        logger.info("Define the DataGenerator object.")
        add_aux = self.define_aux_data(
            data_configuration, static_features, static_label
        )

        data_generator_train = DataGenerator(
            train_features,
            train_label,
            add_aux,
            self.features_scaler,
            self.label_scaler,
        )
        data_generator_val = DataGenerator(
            val_features, val_label, add_aux, self.features_scaler, self.label_scaler
        )
        data_generator_test = DataGenerator(
            feature_files=test_features,
            label_files=test_label,
            add_auxiliary_features=add_aux,
            features_scaler=self.features_scaler,
            label_scaler=self.label_scaler,
            shuffle=False,
        )
        return data_generator_train, data_generator_val, data_generator_test

    @staticmethod
    def define_aux_data(data_configuration, static_features, static_label):
        """
        Define auxiliary data based on the provided configuration.

        Parameters
        ----------
        data_configuration : DataConfiguration
            Data configuration details.

        static_features : Tuple[xarray.DataArray or None, xarray.DataArray or None]
            Tuple containing land mask and orography data arrays used as
            static features.

        static_label : Tuple[xarray.DataArray or None, xarray.DataArray or None]
            Tuple containing land mask and orography data arrays used as
            static labels.

        Returns
        -------
        dict
            Dictionary containing the auxiliary data based on the data configuration.
        """
        f_cfg = data_configuration.features_configuration
        add_aux = f_cfg.get("add_auxiliary", False)
        for aux_type, aux_value in add_aux.items():
            if aux_type == "lsm-low" and aux_value:
                add_aux[aux_type] = static_features[0]
            elif aux_type == "orog-low" and aux_value:
                add_aux[aux_type] = static_features[1]
            elif aux_type == "lsm-high" and aux_value:
                add_aux[aux_type] = static_label[0]
            elif aux_type == "orog-high" and aux_value:
                add_aux[aux_type] = static_label[1]
        return add_aux

    def load_trained_model(self):
        if self.pipeline_type == "diffusion":
            cfg = self.model_config["neural_network"]
            model_name = cfg["trained_model_dir"]
            model = load_trained_model(cfg["class_name"], model_name)
        elif self.pipeline_type == "end2end":
            # If running validation on trained model, a "trained_model_dir" is required
            cfg = self.model_config["neural_network"]
            model_name = cfg["trained_model_dir"]
            model = load_trained_model(cfg["class_name"], model_name)
        else:
            raise NotImplementedError(
                f"Pipeline type {self.pipeline_type} not implemented!"
            )
        return model, model_name

    def train_diffusion(self, dataset: DataGenerator, dataset_val: DataGenerator):
        """
        Train a Deep Diffusion model with the given dataset.

        Parameters
        ----------
        dataset : DataGenerator
            The dataset to train the model on.
        dataset_val: DataGenerator
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

        obs_model_cfg = self.model_config.pop("trained_obs_model", {})
        obs_model = load_trained_model(**obs_model_cfg)

        # Train the diffusion model
        train_diffusion(
            self.train_config,
            eps_model,
            scheduler,
            dataset,
            dataset_val,
            obs_model=obs_model,
            dataset_info=self._prepare_data_cfg_log(),
        )

    def train_end2end_nn(
        self, dataset_train: DataGenerator, dataset_val: DataGenerator
    ) -> nn.Module:
        """
        Train an end-to-end neural network with the given dataset.

        Parameters
        ----------
        dataset_train : DataGenerator
            The dataset to train the model on.
        dataset_val: DataGenerator
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
            input_shape=dataset_train.input_shape,
        )

        return train_nn(
            self.train_config,
            model,
            dataset_train,
            dataset_val,
            self._prepare_data_cfg_log(),
        )

    def train_autoencoder(
        self, dataset_train: DataGenerator, dataset_val: DataGenerator
    ):
        model_cfg = self.model_config.pop("neural_network")
        model_cfg["kwargs"]["in_channels"] = dataset_train.input_channels
        model_cfg["kwargs"]["out_channels"] = dataset_train.output_channels
        model_cfg["kwargs"]["sample_size"] = dataset_train.output_shape

        # Instantiate objects
        model = get_neural_network(**model_cfg)
        return train_autoencoder(self.train_config, model, dataset_train, dataset_val)

    def train_model(
        self,
        dataset_train: DataGenerator,
        dataset_val: DataGenerator,
    ):
        """
        Train a Super Resolution (SR) model with the given dataset.

        Parameters
        ----------
        dataset_train : DataGenerator
            The dataset to train the model on.
        dataset_val: DataGenerator
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
        elif self.pipeline_type == "autoencoder":
            return self.train_autoencoder(dataset_train, dataset_val)
        else:
            raise NotImplementedError(
                f"The training procedure {self.pipeline_type} is not supported."
            )

    def validate_model(self, model, dataset, config: dict, hf_repo_name: str = None):
        """
        Test a trained model with the given dataset.

        Parameters
        ----------
        model : nn.Module | SuperResolutionDenoiseDiffusion
            The trained model to test.
        dataset : DataGenerator
            The dataset to test the model on.
        config: dict
            The configuration of the validation process.
        hf_repo_name : str, optional
            The name of the Hugging Face repository to push the model to,
            by default None.

        Returns
        -------
        test_results : Dict
            The test results of the model.
        """
        if self.pipeline_type == "diffusion":
            scheduler = get_hf_scheduler(**self.model_config.pop("scheduler"))
            return validation_diffusion.validate_model(
                model,
                scheduler,
                dataset,
                config,
                hf_repo_name=hf_repo_name,
                label_scaler=self.label_scaler,
            )
        elif self.pipeline_type == "end2end":
            return validation_nn.validate_model(
                model,
                dataset,
                config,
                hf_repo_name=hf_repo_name,
                label_scaler=self.label_scaler,
            )
        elif self.pipeline_type == "autoencoder":
            return validation_nn.validate_model(
                model,
                dataset,
                config,
                hf_repo_name=hf_repo_name,
                label_scaler=self.label_scaler,
            )
        else:
            raise NotImplementedError(
                f"The training procedure {self.pipeline_type} is not supported."
            )

    def run_validation(self):
        """Run the validation on a trained model."""
        *_, dataset_test = self.get_dataset()
        model, repo_name = self.load_trained_model()
        self.validate_model(
            model, dataset_test, self.validation_config, hf_repo_name=repo_name
        )

    def run_pipeline(self):
        """Run the pipeline."""
        dataset_train, dataset_val, dataset_test = self.get_dataset()
        model, repo_name = self.train_model(dataset_train, dataset_val)
        self.validate_model(
            model, dataset_test, self.validation_config, hf_repo_name=repo_name
        )
