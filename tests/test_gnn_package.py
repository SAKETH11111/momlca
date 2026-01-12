def test_gnn_version_import() -> None:
    from gnn import __version__

    assert __version__ == "0.1.0"
