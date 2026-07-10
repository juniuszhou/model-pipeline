import logging

from components.hf_utils import get_model, login_huggingface


def test_hugging_face():
    login_huggingface()


def test_get_model():
    model_name = "gpt2"
    model = get_model(model_name)
    assert model is not None
    logging.info(
        "model %s total parameters: %s",
        model_name,
        sum(p.numel() for p in model.parameters()),
    )
