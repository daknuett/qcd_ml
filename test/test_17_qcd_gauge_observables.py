import pytest
import torch

from qcd_ml.qcd.gauge.observables import (
    plaquette_field,
    topological_charge_density_clover,
    topological_charge_density_plaquette,
)


def test_plaquette_field_output_shape(config_1500):
    """Test that plaquette_field returns the correct output shape."""
    result = plaquette_field(config_1500)
    expected_shape = config_1500.shape[1:5]

    assert result.shape == expected_shape


def test_plaquette_field_positive(config_1500):
    """Test that plaquette values are positive for typical configurations."""
    result = plaquette_field(config_1500)

    assert torch.all(result > 0)


def test_plaquette_field_identity(config_1500):
    """Test that plaquette values are 1 if the gauge field is the identity."""
    identity_gauge_field = torch.zeros_like(config_1500)
    identity_gauge_field[..., :, :] = torch.eye(3, dtype=torch.cdouble)
    result = plaquette_field(identity_gauge_field, _gpt_compat=True)

    assert torch.allclose(result, torch.ones_like(result), atol=1e-14)


def test_topological_charge_density_clover_output_shape(config_1500):
    """Test that topological_charge_density_clover returns the correct output shape."""
    result = topological_charge_density_clover(config_1500)
    expected_shape = config_1500.shape[1:5]

    assert result.shape == expected_shape


def test_topological_charge_density_clover_output_type(config_1500):
    """Test that topological_charge_density_clover returns a tensor with negligible imaginary part."""
    result = topological_charge_density_clover(config_1500)

    assert result.dtype == torch.cdouble
    assert torch.allclose(
        result.imag, torch.zeros_like(result.imag), atol=1e-14
    )


def test_topological_charge_density_plaquette_output_shape(config_1500):
    """Test that topological_charge_density_plaquette returns the correct output shape."""
    result = topological_charge_density_plaquette(config_1500)
    expected_shape = config_1500.shape[1:5]

    assert result.shape == expected_shape


def test_topological_charge_density_plaquette_output_type(config_1500):
    """Test that topological_charge_density_plaquette returns a tensor with negligible imaginary part."""
    result = topological_charge_density_plaquette(config_1500)

    assert result.dtype == torch.cdouble
    assert torch.allclose(
        result.imag, torch.zeros_like(result.imag), atol=1e-14
    )
