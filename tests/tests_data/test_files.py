import os

import pytest

from deepr.data.files import DataFile


@pytest.fixture
def base_dir():
    return "/path/to/data"


@pytest.fixture
def data_file(base_dir):
    return DataFile(base_dir, "t2m", "era5", "201801", "025deg", None)


def test_data_path_attributes(data_file, base_dir):
    """
    Test the attributes of the DataPath instance.

    Parameters
    ----------
    data_file : DataFile
        The DataPath instance to test.
    base_dir : str
        The expected base directory.

    Returns
    -------
    None
    """
    assert data_file.base_dir == base_dir
    assert data_file.variable == "t2m"
    assert data_file.dataset == "era5"
    assert data_file.temporal_coverage == "201801"
    assert data_file.spatial_resolution == "025deg"


def test_data_path_to_path(data_file, base_dir):
    """
    Test the to_path method of the DataPath instance.

    Parameters
    ----------
    data_file : DataFile
        The DataPath instance to test.
    base_dir : str
        The expected base directory.

    Returns
    -------
    None
    """
    expected_path = os.path.join(base_dir, "t2m_era5_201801_025deg.nc")
    assert data_file.to_path() == expected_path


def test_data_path_from_path(data_file, base_dir):
    """
    Test the from_path class method of the DataPath class.

    Parameters
    ----------
    data_file : DataFile
        The DataPath instance for comparison.
    base_dir : str
        The expected base directory.

    Returns
    -------
    None
    """
    file_path = os.path.join(base_dir, "t2m_era5_201801_025deg.nc")
    new_data_path = DataFile.from_path(file_path)
    assert new_data_path.base_dir == base_dir
    assert new_data_path.variable == "t2m"
    assert new_data_path.dataset == "era5"
    assert new_data_path.temporal_coverage == "201801"
    assert new_data_path.spatial_resolution == "025deg"
