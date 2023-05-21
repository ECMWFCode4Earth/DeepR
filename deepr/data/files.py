import os


class DataPath:
    """
    Class for generating and manipulating data paths based on a specific structure.

    Attributes
    ----------
        base_dir (str): The base directory where the data files are stored.
        variable (str): The variable name.
        dataset (str): The dataset name.
        date (str): The date of the data.
        resolution (str): The resolution of the data.
    """

    def __init__(self, base_dir, variable, dataset, date, resolution):
        """
        Initialize a DataPath instance.

        Parameters
        ----------
        base_dir : str
            The base directory where the data files are stored.
        variable : str
            The variable name.
        dataset : str
            The dataset name.
        date : str
            The date of the data.
        resolution : str
            The resolution of the data.
        """
        self.base_dir = base_dir
        self.variable = variable
        self.dataset = dataset
        self.date = date
        self.resolution = resolution

    @classmethod
    def from_path(cls, file_path):
        """
        Create a DataPath instance from a file path.

        Parameters
        ----------
        file_path : str
            The file path.

        Returns
        -------
        DataPath
            The DataPath instance.
        """
        base_dir, filename = os.path.split(file_path)
        variable, dataset, date, resolution = filename[:-3].split("_")
        return cls(base_dir, variable, dataset, date, resolution)

    def to_path(self):
        """
        Generate the file path based on the class attributes.

        Returns
        -------
        str
            The complete file path.
        """
        filename = f"{self.variable}_{self.dataset}_{self.date}_{self.resolution}.nc"
        return os.path.join(self.base_dir, filename)
