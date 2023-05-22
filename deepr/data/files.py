import os
from typing import List


class DataFile:
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
        DataFile
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

    def exist(self) -> bool:
        """
        Indicate whether the file returned by to_path method already exists.

        Returns
        -------
        bool
            True or False indicating if the file returned by to_path exists.
        """
        return os.path.exists(self.to_path())


class DataFileCollection:
    def __init__(self, collection: List[DataFile] = None):
        self.collection = collection

    def __len__(self):
        """Get the length of the collection list."""
        return len(self.collection)

    def append_data(self, data: DataFile):
        """
        Append a new data object to the data list.

        Parameters
        ----------
        data: Data
            The data object to be appended.
        """
        if isinstance(data, DataFile):
            self.collection.append(data)
        else:
            raise ValueError("The input object is not a Data object.")

    def find_data(self, **kwargs):
        """
        Find a DataFile object in the data list that matches the specified attributes.

        Parameters
        ----------
        **kwargs: dict
            A dictionary with attributes to match the data objects.

        Returns
        -------
        found_data: Data
            The first data object that matches the specified attributes.
        """
        found_data = None
        for data in self.collection:
            match = True
            for key, value in kwargs.items():
                if not hasattr(data, key) or getattr(data, key) != value:
                    match = False
                    break
            if match:
                found_data = data
                break
        return found_data

    def sort_data(self, attribute: str):
        """
        Sort the collection list by the specified attribute of the DataFile objects.

        Parameters
        ----------
        attribute : str
            The attribute name to sort the DataFile objects by.
        """
        self.collection.sort(key=lambda x: getattr(x, attribute))
