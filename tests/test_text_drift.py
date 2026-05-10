from src.monitoring.text_drift import main
import os


def test_text_drift_runs():

    main()

    assert os.path.exists(
        "reports/evidently/text_drift.json"
    )