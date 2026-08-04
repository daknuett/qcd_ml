import torch
import pytest
from qcd_ml.nn.matrix_layers.loop_generator import (
    PolyakovLoopGenerator,
    PositiveOrientationPlaquetteGenerator,
    AbstractLoopGenerator,
)


def test_PolyakovLoopGenerator_output_shape(config_1500):
    """Test that PolyakovLoopGenerator returns the correct output shape."""
    generator = PolyakovLoopGenerator()
    result = generator(config_1500)
    
    expected_shape = (PolyakovLoopGenerator.nfeatures_out, *config_1500.shape[1:])
    assert result.shape == expected_shape


def test_PolyakovLoopGenerator_disable_cache(config_1500):
    """Test that PolyakovLoopGenerator cache can be disabled."""
    generator = PolyakovLoopGenerator(disable_cache=False)
    
    result1 = generator(config_1500)
    assert len(generator.cache) > 0
    
    result2 = generator(config_1500)
    assert torch.equal(result1, result2)
    
    generator.clear_cache()
    assert len(generator.cache) == 0

    result3 = generator(config_1500)
    assert torch.allclose(result1, result3, atol=1e-14)


def test_PolyakovLoopGenerator_cache_disabled(config_1500):
    """Test that cache is not used when disabled."""
    generator = PolyakovLoopGenerator(disable_cache=True)
    
    result1 = generator(config_1500)
    assert len(generator.cache) == 0


def test_PositiveOrientationPlaquetteGenerator_output_shape(config_1500):
    """Test that PositiveOrientationPlaquetteGenerator returns the correct output shape."""
    generator = PositiveOrientationPlaquetteGenerator()
    result = generator(config_1500)
    
    expected_shape = (PositiveOrientationPlaquetteGenerator.nfeatures_out, *config_1500.shape[1:])
    assert result.shape == expected_shape

    assert generator.nfeatures_out == PositiveOrientationPlaquetteGenerator.nfeatures_out


def test_PositiveOrientationPlaquetteGenerator_disable_cache(config_1500):
    """Test that PositiveOrientationPlaquetteGenerator cache can be disabled."""
    generator = PositiveOrientationPlaquetteGenerator(disable_cache=False)
    
    result1 = generator(config_1500)
    assert len(generator.cache) > 0
    
    result2 = generator(config_1500)
    assert torch.equal(result1, result2)
    
    generator.clear_cache()
    assert len(generator.cache) == 0

    result3 = generator(config_1500)
    assert torch.allclose(result1, result3, atol=1e-14)
