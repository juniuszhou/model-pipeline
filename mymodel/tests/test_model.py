from projects.config import LLMTrainingConfig
from projects.model import TransformerLM, load_model, load_model_safe, save_model, save_model_safe


def _test_config() -> LLMTrainingConfig:
    return LLMTrainingConfig(
        vocab_size=21128,
        context_length=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_heads=8,
        d_ff=64,
        rope_theta=10000.0,
    )


def test_save_and_load_model():
    config = _test_config()
    model = TransformerLM(config)
    save_model(model)

    loaded = load_model(model, "model.pth")
    assert loaded.vocab_size == model.vocab_size
    assert type(loaded) is TransformerLM


def test_save_and_load_model_safe():
    config = _test_config()
    model = TransformerLM(config)
    save_model_safe(model, name="test-checkpoint")

    loaded = load_model_safe("test-checkpoint")
    assert loaded.vocab_size == model.vocab_size
    assert loaded.d_model == model.d_model
    assert type(loaded) is TransformerLM
