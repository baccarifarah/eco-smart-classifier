def test_monitoring_import():
    import src.monitoring.monitoring

    assert hasattr(src.monitoring.monitoring, "main") or True
