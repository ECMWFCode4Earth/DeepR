import os
from typing import List, Optional


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

    def __init__(
        self,
        base_dir: str,
        variable: str,
        dataset: str,
        temporal_coverage: str,
        spatial_resolution: str,
        spatial_coverage: Optional[dict] = None,
    ):
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
        temporal_coverage : str
            The temporal coverage of the data.
        spatial_resolution : str
            The temporal resolution of the data.
        spatial_coverage: Optional[dict]
            The spatial coverage of the data to be selected.
        """
        self.base_dir = base_dir
        self.variable = variable
        self.dataset = dataset
        self.temporal_coverage = temporal_coverage
        self.spatial_resolution = spatial_resolution
        self.spatial_coverage = spatial_coverage

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
        filename = (
            f"{self.variable}_{self.dataset}_"
            f"{self.temporal_coverage}_{self.spatial_resolution}.nc"
        )
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
    def __init__(self, collection: List[DataFile]):
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
        found_data = []
        for data in self.collection:
            match = True
            for key, value in kwargs.items():
                if not hasattr(data, key) or getattr(data, key) != value:
                    match = False
                    break
            if match:
                found_data.append(data)

        if len(found_data) == 0:
            return None
        else:
            return DataFileCollection(collection=found_data)

    def sort_data(self, attribute: str):
        """
        Sort the collection list by the specified attribute of the DataFile objects.

        Parameters
        ----------
        attribute : str
            The attribute name to sort the DataFile objects by.
        """
        self.collection.sort(key=lambda x: getattr(x, attribute))

    def split_data(self, split_coefficient: float):
        """
        Split the data collection into two different data collections.

        Parameters
        ----------
        split_coefficient : float
            The coefficient by which the data is split.
        """
        idx = int((1 - split_coefficient) * len(self.collection))
        split1 = DataFileCollection(collection=self.collection[:idx])
        split2 = DataFileCollection(collection=self.collection[idx:])
        return split1, split2

    def get_variable_list(self) -> List[str]:
        """
        Get the list of variables in the data collection.

        Returns
        -------
        variables : list
            The list of variables that are available in the data collection
        """
        variables = {data_file.variable for data_file in self.collection}
        return list(variables)
