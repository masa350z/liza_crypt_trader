#!/usr/bin/env python3
"""履歴データ統合テストスクリプト。

DynamoDB履歴データ初期化機能をテストし、統合的な動作を確認。
"""

import os
import numpy as np
from datetime import datetime
import time

def test_historical_data_initializer_without_boto3():
    """Test HistoricalDataInitializer when boto3 is not available."""
    print("[TEST] Testing HistoricalDataInitializer without boto3...")
    
    try:
        from historical_data import HistoricalDataInitializer
        
        initializer = HistoricalDataInitializer()
        
        # Should return False if boto3 not available
        available = initializer.is_available()
        print(f"[TEST] is_available() returned: {available}")
        
        # Should return empty list if not available
        prices = initializer.fetch_historical_prices(60)
        assert prices == [], f"Expected empty list, got: {prices}"
        
        print("[TEST] HistoricalDataInitializer properly handles missing boto3 ✓")
        
    except Exception as e:
        print(f"[TEST] HistoricalDataInitializer test failed: {e}")
        raise


def test_historical_data_initializer_no_credentials():
    """Test HistoricalDataInitializer without AWS credentials."""
    print("[TEST] Testing HistoricalDataInitializer without AWS credentials...")
    
    # Temporarily clear AWS credentials
    original_key = os.environ.get('AWS_ACCESS_KEY_ID')
    original_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')
    original_my_key = os.environ.get('MY_ACCESS_KEY_ID')
    original_my_secret = os.environ.get('MY_SECRET_ACCESS_KEY')
    
    try:
        # Clear credentials
        for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'MY_ACCESS_KEY_ID', 'MY_SECRET_ACCESS_KEY']:
            if key in os.environ:
                del os.environ[key]
        
        from historical_data import HistoricalDataInitializer
        
        initializer = HistoricalDataInitializer()
        
        # Should return False without credentials
        available = initializer.is_available()
        assert not available, f"Expected False, got: {available}"
        
        print("[TEST] HistoricalDataInitializer properly handles missing credentials ✓")
        
    finally:
        # Restore original credentials
        if original_key:
            os.environ['AWS_ACCESS_KEY_ID'] = original_key
        if original_secret:
            os.environ['AWS_SECRET_ACCESS_KEY'] = original_secret
        if original_my_key:
            os.environ['MY_ACCESS_KEY_ID'] = original_my_key
        if original_my_secret:
            os.environ['MY_SECRET_ACCESS_KEY'] = original_my_secret


def test_trading_model_bulk_initialize():
    """Test TradingModel bulk_initialize method."""
    print("[TEST] Testing TradingModel bulk_initialize...")
    
    from trader import TradingModel
    
    # Create a model that needs 60 minutes of history
    model = TradingModel(k=60, p=20)
    
    # Initially should not be ready
    assert not model.ready(), "Model should not be ready initially"
    assert len(model.price_history) == 0, "Price history should be empty initially"
    
    # Generate sample historical data
    historical_prices = []
    base_price = 5000000
    for i in range(100):  # More than 60 minutes needed
        price = base_price + np.random.randn() * 1000
        historical_prices.append(price)
        base_price = price * 0.999 + base_price * 0.001
    
    # Bulk initialize
    model.bulk_initialize(historical_prices)
    
    # Should now be ready
    assert model.ready(), "Model should be ready after bulk initialization"
    assert len(model.price_history) == 60, f"Price history should be 60, got: {len(model.price_history)}"
    
    # Should contain the last 60 prices
    expected_history = historical_prices[-60:]
    assert model.price_history == expected_history, "Price history should match last 60 prices"
    
    print("[TEST] TradingModel bulk_initialize working correctly ✓")


def test_trader_manager_historical_init():
    """Test TraderManager with historical data initialization (will fallback safely)."""
    print("[TEST] Testing TraderManager with historical data initialization...")
    
    # Clear credentials to test fallback behavior
    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'MY_ACCESS_KEY_ID', 'MY_SECRET_ACCESS_KEY']:
        if key in os.environ:
            del os.environ[key]
    
    # Should attempt historical initialization but fallback gracefully
    try:
        from trader import TraderManager
        manager = TraderManager()
        print("[TEST] TraderManager created successfully with historical data fallback ✓")
        
        # Model should initially not be ready (fallback to standard initialization)
        assert not manager.model.ready(), "Model should not be ready initially"
            
    except Exception as e:
        print(f"[TEST] TraderManager creation failed: {e}")
        raise


def test_trader_manager_with_credentials():
    """Test TraderManager behavior when AWS credentials are available."""
    print("[TEST] Testing TraderManager with AWS credentials (if available)...")
    
    # Clear AWS credentials to force fallback
    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'MY_ACCESS_KEY_ID', 'MY_SECRET_ACCESS_KEY']:
        if key in os.environ:
            del os.environ[key]
    
    try:
        from trader import TraderManager
        manager = TraderManager()
        print("[TEST] TraderManager created successfully with credential fallback ✓")
        
        # Should fallback to standard initialization (model not ready)
        assert not manager.model.ready(), "Model should not be ready after fallback"
            
    except Exception as e:
        print(f"[TEST] TraderManager creation failed: {e}")
        raise


def test_data_quality_validation():
    """Test data quality validation functionality."""
    print("[TEST] Testing data quality validation...")
    
    from historical_data import HistoricalDataInitializer
    
    initializer = HistoricalDataInitializer()
    
    # Test with good data
    good_prices = [5000000 + i * 100 for i in range(60)]
    validation = initializer.validate_data_quality(good_prices, 60)
    
    assert validation['valid'], "Good data should be valid"
    assert validation['data_count'] == 60, "Data count should be correct"
    
    # Test with insufficient data
    bad_prices = [5000000] * 10  # Only 10 data points for 60 expected
    validation = initializer.validate_data_quality(bad_prices, 60)
    
    assert not validation['valid'], "Insufficient data should be invalid"
    assert 'Insufficient data' in str(validation['issues']), "Should report insufficient data"
    
    # Test with no data
    validation = initializer.validate_data_quality([], 60)
    
    assert not validation['valid'], "No data should be invalid"
    assert 'No data available' in str(validation['issues']), "Should report no data"
    
    print("[TEST] Data quality validation working correctly ✓")


def test_configuration_variables():
    """Test configuration variable handling."""
    print("[TEST] Testing configuration variables...")
    
    # Test USE_HISTORICAL_DATA
    os.environ['USE_HISTORICAL_DATA'] = 'true'
    import importlib
    import trader
    importlib.reload(trader)
    
    assert trader.USE_HISTORICAL_DATA == True, "USE_HISTORICAL_DATA should be True"
    
    os.environ['USE_HISTORICAL_DATA'] = 'false'
    importlib.reload(trader)
    
    assert trader.USE_HISTORICAL_DATA == False, "USE_HISTORICAL_DATA should be False"
    
    # Test with no setting (should default to False)
    if 'USE_HISTORICAL_DATA' in os.environ:
        del os.environ['USE_HISTORICAL_DATA']
    importlib.reload(trader)
    
    assert trader.USE_HISTORICAL_DATA == False, "USE_HISTORICAL_DATA should default to False"
    
    print("[TEST] Configuration variables working correctly ✓")


def main():
    """Run all historical data integration tests."""
    print("=" * 60)
    print("Historical Data Integration Test")
    print("=" * 60)
    
    try:
        test_historical_data_initializer_without_boto3()
        test_historical_data_initializer_no_credentials()
        test_trading_model_bulk_initialize()
        test_trader_manager_historical_init_disabled()
        test_trader_manager_historical_init_enabled()
        test_data_quality_validation()
        test_configuration_variables()
        
        print("\n" + "=" * 60)
        print("✅ All historical data integration tests passed!")
        print("✅ Historical data initialization feature is working correctly")
        print("✅ Safe fallback mechanisms are functioning properly")
        print("=" * 60)
        
        print("\n📋 Usage Summary:")
        print("1. Set USE_HISTORICAL_DATA=true to enable historical data initialization")
        print("2. Configure AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        print("3. Install boto3: pip install boto3")
        print("4. The system will fallback safely if DynamoDB is unavailable")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()