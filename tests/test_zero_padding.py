#!/usr/bin/env python3
"""ゼロパディング処理テストスクリプト。

技術指標計算時のゼロパディング回避機能と一括初期化をテスト。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from trader import TradingModel
from ml_modules.data_processing import calculate_valid_start_index

def test_valid_start_index_calculation():
    """valid_start_index計算のテスト"""
    print("=== Testing valid_start_index calculation ===")
    
    # デフォルトパラメータでのテスト
    model = TradingModel(k=120, p=45)
    
    # 手動計算での検証
    sma_start = max(5, 20, 60) - 1  # 59
    bollinger_start = 20 - 1  # 19  
    macd_start = 26 - 1 + 9 - 1  # 33
    rsi_start = 14  # 14
    expected_valid_start = max(59, 19, 33, 14)  # 59
    
    assert model.valid_start_index == expected_valid_start, \
        f"Expected {expected_valid_start}, got {model.valid_start_index}"
    
    print(f"✅ Valid start index calculation: {model.valid_start_index}")
    print(f"   SMA start: {sma_start}, Bollinger: {bollinger_start}")
    print(f"   MACD start: {macd_start}, RSI: {rsi_start}")

def test_bulk_initialize_with_zero_padding():
    """ゼロパディングスキップのテスト"""
    print("\n=== Testing bulk_initialize with zero-padding skip ===")
    
    model = TradingModel(k=60, p=20)
    
    # テストデータ作成（valid_start_index + history_minutes + 余裕）
    test_data_length = model.valid_start_index + model.history_minutes + 20
    test_prices = list(range(1000, 1000 + test_data_length))  # 1000, 1001, 1002, ...
    
    print(f"Model config: k={model.k}, valid_start={model.valid_start_index}")
    print(f"Test data length: {len(test_prices)}")
    
    # bulk_initialize実行
    model.bulk_initialize(test_prices)
    
    # 期待される結果の確認
    expected_start_idx = model.valid_start_index
    expected_end_idx = expected_start_idx + model.k
    expected_prices = test_prices[expected_start_idx:expected_end_idx]
    
    assert model.ready(), "Model should be ready after bulk initialization"
    assert len(model.price_history) == model.k, \
        f"Expected {model.k} prices, got {len(model.price_history)}"
    assert model.price_history == expected_prices, \
        f"Price history mismatch. Expected first price: {expected_prices[0]}, got: {model.price_history[0]}"
    
    print(f"✅ Bulk initialization successful")
    print(f"   Skipped {model.valid_start_index} zero-padded samples")
    print(f"   Price history range: {model.price_history[0]} - {model.price_history[-1]}")
    print(f"   Expected range: {test_prices[expected_start_idx]} - {test_prices[expected_end_idx-1]}")

def test_insufficient_data_handling():
    """データ不足時の処理テスト"""
    print("\n=== Testing insufficient data handling ===")
    
    model = TradingModel(k=120, p=45)
    
    # 不十分なデータでテスト
    insufficient_data = list(range(1000, 1050))  # 50個のデータ（不十分）
    
    print(f"Required data length: {model.k + model.valid_start_index}")
    print(f"Provided data length: {len(insufficient_data)}")
    
    model.bulk_initialize(insufficient_data)
    
    # モデルがreadyでないことを確認
    assert not model.ready(), "Model should not be ready with insufficient data"
    assert len(model.price_history) == 0, "Price history should be empty with insufficient data"
    
    print("✅ Insufficient data handled correctly")

def test_single_model_scenario():
    """単一モデルでのテストシナリオ"""
    print("\n=== Testing single model scenario ===")
    
    # デフォルトの単一モデル設定
    model = TradingModel(k=90, p=30)
    
    print(f"Single model: k={model.k}, valid_start={model.valid_start_index}")
    
    # 必要データ長を計算
    required_length = model.k + model.valid_start_index
    
    print(f"Required total data length: {required_length}")
    
    # 十分なテストデータを作成
    test_data = list(range(2000, 2000 + required_length + 50))
    
    # モデルを初期化
    model.bulk_initialize(test_data)
    assert model.ready(), "Model should be ready"
    print(f"Model initialized: {len(model.price_history)} prices")
    
    print("✅ Single model initialized successfully")

def main():
    """テスト実行"""
    # 本来ならここにテスト実行エラーハンドリングが必要
    test_valid_start_index_calculation()
    test_bulk_initialize_with_zero_padding() 
    test_insufficient_data_handling()
    test_single_model_scenario()
    
    print("\n🎉 All zero-padding tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())