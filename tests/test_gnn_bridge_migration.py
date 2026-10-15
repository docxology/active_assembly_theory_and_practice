import numpy as np

from src.active_inference_bridge import (
    infer_hidden_states,
    infer_hidden_states_reference,
)
from src.disassembly import disassemble_sample
from src.emulation import emulate_sample
from src.features import encode_observations
from src.generative_model import build_generative_model


def test_real_agent_preserves_static_reference_result():
    observations = encode_observations(disassemble_sample(), emulate_sample())
    model = build_generative_model()
    real = infer_hidden_states(observations, model)
    reference = infer_hidden_states_reference(observations, model)
    np.testing.assert_allclose(
        real.posterior_by_step, reference.posterior_by_step, atol=1e-5
    )
    np.testing.assert_allclose(
        real.free_energy_by_step, reference.free_energy_by_step, atol=1e-5
    )
    assert real.most_likely_states == reference.most_likely_states
