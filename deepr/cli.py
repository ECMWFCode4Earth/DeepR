from pathlib import Path

import click

from deepr.utilities.logger import get_logger
from deepr.workflow import MainPipeline

logger = get_logger(__name__)


@click.group()
def launcher():
    pass


@launcher.command("train_model")
@click.option(
    "-c",
    "--configuration_yaml",
    help="File configuring the parameters for training the model",
    type=str,
)
def train_model(configuration_yaml: str) -> None:
    """
    Train a model.

    Parameters
    ----------
    configuration_yaml : str
        File configuring the parameters for training the model.
        Default is "{main_folder}/resources/configuration/config.yml".

    Returns
    -------
    None
        This function does not return any value.

    """
    logger.info(
        f"Starting the process of training a model with the "
        f"configuration specified at {configuration_yaml}"
    )
    MainPipeline(configuration_file=Path(configuration_yaml)).run_pipeline()
    logger.info("Process finished")
