from deepr.data.configuration import DataConfiguration
from deepr.utilities.logger import get_logger
from deepr.utilities.yml import read_yaml_file

logger = get_logger(__name__)


class MainPipeline:
    def __init__(self, configuration_file: str):
        configuration = read_yaml_file(configuration_file)
        DataConfiguration(configuration["data"])
