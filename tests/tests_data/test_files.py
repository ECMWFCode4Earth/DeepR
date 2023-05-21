import os

import pytest

from deepr.data.files import DataPath


@pytest.fixture
def base_dir():
    return "/path/to/data"


@pytest.fixture
def data_path(base_dir):
    return DataPath(base_dir, "t2m", "era5", "201801", "025deg")


def test_data_path_attributes(data_path, base_dir):
    """
    Test the attributes of the DataPath instance.

    Parameters
    ----------
    data_path : DataPath
        The DataPath instance to test.
    base_dir : str
        The expected base directory.

    Returns
    -------
    None
    """
    assert data_path.base_dir == base_dir
    assert data_path.variable == "t2m"
    assert data_path.dataset == "era5"
    assert data_path.date == "201801"
    assert data_path.resolution == "025deg"


def test_data_path_to_path(data_path, base_dir):
    """
    Test the to_path method of the DataPath instance.

    Parameters
    ----------
    data_path : DataPath
        The DataPath instance to test.
    base_dir : str
        The expected base directory.

    Returns
    -------
    None
    """
    expected_path = os.path.join(base_dir, "t2m_era5_201801_025deg.nc")
    assert data_path.to_path() == expected_path


def test_data_path_from_path(data_path, base_dir):
    """
    Test the from_path class method of the DataPath class.

    Parameters
    ----------
    data_path : DataPath
        The DataPath instance for comparison.
    base_dir : str
        The expected base directory.

    Returns
    -------
    None
    """
    file_path = os.path.join(base_dir, "t2m_era5_201801_025deg.nc")
    new_data_path = DataPath.from_path(file_path)
    assert new_data_path.base_dir == base_dir
    assert new_data_path.variable == "t2m"
    assert new_data_path.dataset == "era5"
    assert new_data_path.date == "201801"
    assert new_data_path.resolution == "025deg"
