"""技術指標の算出とML用特徴量生成を行うデータ処理モジュール。

* calculate_valid_start_index: ゼロパディングを回避するための開始インデックス計算
* make_prediction_features: 価格データからML予測用特徴量を生成
"""

import numpy as np
from .technical_indicators import calc_sma, calc_bollinger_bands, calc_macd, calc_rsi


def calculate_valid_start_index(sma_short_window, sma_mid_window, sma_long_window, 
                               bollinger_band_window, macd_long_window, macd_signal_window, 
                               rsi_window):
    """全ての技術指標が有効な値を持つ開始インデックスを計算する。
    
    各技術指標の計算に必要な最小データ数を考慮し、ゼロパディングされた
    無効なデータを回避するための開始インデックスを決定する。
    
    Args:
        sma_short_window (int): 短期移動平均の期間
        sma_mid_window (int): 中期移動平均の期間
        sma_long_window (int): 長期移動平均の期間
        bollinger_band_window (int): ボリンジャーバンドの期間
        macd_long_window (int): MACD長期の期間
        macd_signal_window (int): MACDシグナルラインの期間
        rsi_window (int): RSIの期間
        
    Returns:
        int: 有効なデータが開始するインデックス
    """
    sma_start = max(sma_short_window, sma_mid_window, sma_long_window) - 1
    bollinger_start = bollinger_band_window - 1
    macd_start = macd_long_window - 1 + macd_signal_window - 1
    rsi_start = rsi_window
    
    valid_start = max(sma_start, bollinger_start, macd_start, rsi_start)
    return valid_start


def make_prediction_features(prices,
                           k,
                           sma_short_window=5,
                           sma_mid_window=20,
                           sma_long_window=60,
                           bollinger_band_window=20,
                           bollinger_band_sigma=2.0,
                           macd_short_window=12,
                           macd_long_window=26,
                           macd_signal_window=9,
                           rsi_window=14):
    """価格履歴からML予測用の特徴ベクトルを生成する。
    
    価格データから各種技術指標を計算し、正規化したMLモデル用の
    特徴量を作成する。
    
    Args:
        prices (np.ndarray): 最近のk個の価格データ
        k (int): 入力系列長
        sma_short_window (int): 短期移動平均の期間
        sma_mid_window (int): 中期移動平均の期間
        sma_long_window (int): 長期移動平均の期間
        bollinger_band_window (int): ボリンジャーバンドの期間
        bollinger_band_sigma (float): ボリンジャーバンドの標準偏差倍数
        macd_short_window (int): MACD短期の期間
        macd_long_window (int): MACD長期の期間
        macd_signal_window (int): MACDシグナルラインの期間
        rsi_window (int): RSIの期間
        
    Returns:
        np.ndarray: 正規化された特徴量配列 (1, k, 12) MLモデル予測準備完了
    """
    # 本来ならここに価格データ長チェックが必要
    
    # 最新のk個の価格データのみを使用
    recent_prices = np.array(prices[-k:])
    
    # 技術指標を計算
    sma_short = calc_sma(recent_prices, sma_short_window)
    sma_mid = calc_sma(recent_prices, sma_mid_window)
    sma_long = calc_sma(recent_prices, sma_long_window)

    bollinger_band_center, bollinger_band_upper, bollinger_band_lower = calc_bollinger_bands(
        recent_prices, bollinger_band_window, bollinger_band_sigma)

    macd_line, signal_line = calc_macd(
        recent_prices, short_window=macd_short_window, 
        long_window=macd_long_window, signal_window=macd_signal_window)

    rsi = calc_rsi(recent_prices, window=rsi_window)

    # 訓練時と同様に特徴量配列を構築
    bollinger_bands = np.stack(
        [bollinger_band_center, bollinger_band_upper, bollinger_band_lower], axis=1)
    macds = np.stack([macd_line, signal_line], axis=1)

    # SMA差分
    short_mid = sma_short - sma_mid
    short_long = sma_short - sma_long
    mid_short = sma_mid - sma_short
    mid_long = sma_mid - sma_long
    long_short = sma_long - sma_short
    long_mid = sma_long - sma_mid
    smas = np.stack([short_mid, short_long, mid_short, mid_long, long_short, long_mid], axis=1)

    # ボリンジャーとMACD差分
    bollinger_bands_input = bollinger_bands - recent_prices.reshape(-1, 1)
    macds_input = macds[:, 0] - macds[:, 1]

    # 全特徴量を結合
    input_array = np.concatenate([
        recent_prices.reshape(-1, 1), 
        smas, 
        bollinger_bands_input, 
        macds_input.reshape(-1, 1), 
        rsi.reshape(-1, 1)
    ], axis=1)

    # 特徴量を正規化（訓練時と同様）
    prices_col = input_array[:, 0]
    mean_p = np.mean(prices_col)
    std_p = np.std(prices_col) + 1e-8

    # 価格を正規化
    input_array[:, 0] = (prices_col - mean_p) / std_p
    # 技術指標を正規化（RSI以外）
    input_array[:, 1:11] /= std_p
    # RSIを[-1, 1]範囲に正規化
    input_array[:, 11] = (input_array[:, 11] - 50.0) / 50.0

    # バッチ次元を追加しfloat16に変換
    return input_array.reshape(1, k, 12).astype(np.float16)