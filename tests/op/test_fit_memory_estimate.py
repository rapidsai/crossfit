import pytest
from sklearn.linear_model import LinearRegression

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")
rmm_torch_allocator = pytest.importorskip("rmm.allocators.torch").rmm_torch_allocator
fit_memory_estimate_curve = pytest.importorskip(
    "crossfit.backend.torch.hf.memory_curve_utils"
).fit_memory_estimate_curve

MODEL_NAME = "microsoft/deberta-v3-base"

# Have to do it globally
# TODO: Long term figure out a better way
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)


def test_fit_memory_estimate_curve(tmp_path):
    # Setup
    mem_model_path = tmp_path / "test_memory_model.joblib"
    model = transformers.AutoModel.from_pretrained(MODEL_NAME).to("cuda")
    result = fit_memory_estimate_curve(
        model=model, path_or_name=MODEL_NAME, mem_model_path=str(mem_model_path)
    )
    # Assertions
    assert isinstance(result, LinearRegression)
    assert result.coef_.shape == (3,)  # [batch_size, seq_len, seq_len**2]
    assert isinstance(result.intercept_, float)
    assert mem_model_path.exists()
