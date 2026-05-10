from src.modeling.train_multimodal import main as train_multimodal_main
from src.modeling.train_nlp import main as train_nlp_main
from src.modeling.train_numeric import main as train_numeric_main


def test_train_numeric_runs():

    train_numeric_main()


def test_train_nlp_runs():

    train_nlp_main()


def test_train_multimodal_runs():

    train_multimodal_main()
