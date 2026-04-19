from sva_toolkit import __version__


def test_package_version() -> None:
    assert __version__ == "3.0.0a1"
