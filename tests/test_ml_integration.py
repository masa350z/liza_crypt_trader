#!/usr/bin/env python3
"""完全なMLモデル統合テストスクリプト。

TensorFlowを含む完全なMLモデル機能をテストし、互換性を確認。
"""

import numpy as np
import os
from ml_model import HybridTechnicalModel, DummyModel


def test_dummy_model():
    """Test the dummy model functionality."""
    print("[TEST] Testing DummyModel...")
    
    model = DummyModel(k=90, p=30)
    
    # Generate dummy price series
    prices = np.random.rand(100) * 1000 + 5000000  # Random prices around 5M
    
    # Test prediction
    prob = model.predict_up_probability(prices)
    
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
    print(f"[TEST] DummyModel prediction: {prob:.3f} ✓")
    

def test_ml_model_without_weights():
    """Test ML model when no weights file exists (should fallback to random)."""
    print("[TEST] Testing HybridTechnicalModel without weights...")
    
    model = HybridTechnicalModel(
        k=90, 
        p=30, 
        model_path="nonexistent_model.h5"
    )
    
    # Generate dummy price series
    prices = np.random.rand(100) * 1000 + 5000000
    
    # Test prediction (should fallback to random)
    prob = model.predict_up_probability(prices)
    
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
    assert not model.is_loaded, "Model should not be loaded"
    print(f"[TEST] HybridTechnicalModel fallback prediction: {prob:.3f} ✓")


def test_ml_model_insufficient_data():
    """Test ML model with insufficient price data."""
    print("[TEST] Testing HybridTechnicalModel with insufficient data...")
    
    model = HybridTechnicalModel(
        k=90, 
        p=30
    )
    
    # Generate insufficient price data
    prices = np.random.rand(50) * 1000 + 5000000  # Only 50 prices, need 90
    
    # Test prediction (should fallback to random)
    prob = model.predict_up_probability(prices)
    
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
    print(f"[TEST] HybridTechnicalModel with insufficient data: {prob:.3f} ✓")


def test_trading_model_integration():
    """Test TradingModel class integration."""
    print("[TEST] Testing TradingModel integration...")
    
    # Import here to avoid issues if dependencies not installed
    from trader import TradingModel
    
    # Test with dummy model (default)
    os.environ['USE_ML_MODEL'] = 'false'
    trading_model = TradingModel(k=90, p=30)
    
    # Generate price history
    for i in range(100):
        price = 5000000 + np.random.randn() * 1000
        trading_model.update(price)
    
    # Test if ready
    assert trading_model.ready(), "TradingModel should be ready"
    
    # Test position decision
    trading_model.decide_position()
    
    # Test signal
    signal = trading_model.get_signal()
    assert signal in [-1, 0, 1], f"Invalid signal: {signal}"
    
    print(f"[TEST] TradingModel signal: {signal} ✓")


def test_price_processing():
    """Test price data processing and feature generation."""
    print("[TEST] Testing price data processing...")
    
    from ml_modules.data_processing import make_prediction_features
    
    # Generate realistic price series
    base_price = 5000000
    prices = []
    for i in range(120):
        price = base_price + np.random.randn() * 1000
        prices.append(price)
        base_price = price * 0.999 + base_price * 0.001  # Slight trend
    
    # Test feature generation
    try:
        features = make_prediction_features(prices, k=90)
        
        assert features.shape == (1, 90, 12), f"Wrong feature shape: {features.shape}"
        assert not np.isnan(features).any(), "Features contain NaN values"
        assert np.isfinite(features).all(), "Features contain infinite values"
        
        print(f"[TEST] Feature generation successful: {features.shape} ✓")
        
    except Exception as e:
        print(f"[TEST] Feature generation failed: {e}")
        raise


def main():
    """Run all tests."""
    print("=" * 50)
    print("ML Model Integration Test")
    print("=" * 50)
    
    try:
        test_hybrid_model_basic()
        test_ml_model_without_weights()
        test_ml_model_insufficient_data()
        test_price_processing()
        test_trading_model_integration()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()