import deepr


def test_version() -> None:
    assert deepr.__version__ != "999"
