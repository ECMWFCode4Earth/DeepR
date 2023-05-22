import yaml


def read_yaml_file(yaml_file_path: str) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    yaml_file_path : str
        The path of the YAML file to be read.

    Returns
    -------
    configuration : dict
        The dictionary containing the YAML file's contents.

    Raises
    ------
    FileNotFoundError
        If the specified YAML file does not exist.

    yaml.YAMLError
        If there's an error while parsing the YAML file.
    """
    with open(yaml_file_path) as file:
        configuration = yaml.safe_load(file)
        configuration = replace_none(configuration)
    return configuration


def replace_none(dictionary: dict) -> dict:
    """
    Recursively replace 'None' string values in a dictionary with None type.

    Parameters
    ----------
    dictionary : dict
        The dictionary in which 'None' values should be replaced.

    Returns
    -------
    new_dictionary : dict
        The dictionary with 'None' values replaced by None.
    """
    new_dictionary = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            new_dictionary[key] = replace_none(value)
        elif isinstance(value, list):
            new_list = []
            for element in value:
                if isinstance(element, dict):
                    new_element = replace_none(element)
                else:
                    new_element = element
                new_list.append(new_element)
            new_dictionary[key] = new_list
        elif value == "None":
            new_dictionary[key] = None
        else:
            new_dictionary[key] = value
    return new_dictionary
