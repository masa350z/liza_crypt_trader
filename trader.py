"""暗号通貨自動取引ボットのメイン取引管理モジュール。

* TradingModel: 価格予測と売買判定を行う取引モデル
* TraderManager: 取引ループの統括とAPI連携を管理
"""

import os
import time
import numpy as np
from exchanges import BitFlyer
from model import HybridTechnicalModel

API_KEY = os.getenv('BITFLYER_API_KEY')
API_SECRET = os.getenv('BITFLYER_API_SECRET')

BASE_AMOUNT = 0.001
THRESHOLD_UP = 0.5
THRESHOLD_DOWN = 0.5
RIKAKU_STD = 2
SONKIRI_STD = 1

# MLモデル設定
ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', 'models/best_model.weights.h5')


class TradingModel:
    """価格履歴を管理し、ML予測に基づく売買判定を行う取引モデル。"""
    
    def __init__(self, k, p, 
                 sma_short_window=5, sma_mid_window=20, sma_long_window=60,
                 bollinger_band_window=20, macd_long_window=26, macd_signal_window=9, rsi_window=14):
        """取引モデルを初期化する。
        
        Args:
            k (int): 入力系列長（過去何分の価格データを使用するか）
            p (int): 予測期間（何分先の値動きを予測するか）
            sma_short_window (int): 短期移動平均の期間
            sma_mid_window (int): 中期移動平均の期間
            sma_long_window (int): 長期移動平均の期間
            bollinger_band_window (int): ボリンジャーバンドの期間
            macd_long_window (int): MACD長期の期間
            macd_signal_window (int): MACDシグナルラインの期間
            rsi_window (int): RSIの期間
        """
        self.k = k  # 入力系列長
        self.p = p  # 予測期間
        
        # テクニカル指標のパラメータ
        self.sma_short_window = sma_short_window
        self.sma_mid_window = sma_mid_window
        self.sma_long_window = sma_long_window
        self.bollinger_band_window = bollinger_band_window
        self.macd_long_window = macd_long_window
        self.macd_signal_window = macd_signal_window
        self.rsi_window = rsi_window
        
        # ゼロパディング期間の開始インデックスを計算
        from ml_modules.data_processing import calculate_valid_start_index
        self.valid_start_index = calculate_valid_start_index(
            sma_short_window, sma_mid_window, sma_long_window,
            bollinger_band_window, macd_long_window, macd_signal_window, rsi_window
        )
        
        print(f"[INIT] Using ML model: {ML_MODEL_PATH}")
        self.model = HybridTechnicalModel(k, p, ML_MODEL_PATH)
        self.price_history = []
        self.position = {'side': 0, 'count': 0, 'price': 0, 'std': 0}

    def update(self, price):
        """新しい価格データで価格履歴を更新する。
        
        Args:
            price (float): 新しい価格データ
        """
        self.price_history.append(price)
        if len(self.price_history) > self.k:
            self.price_history = self.price_history[-self.k:]

    def ready(self):
        """モデルが予測可能な状態かチェックする。
        
        Returns:
            bool: 十分な価格データが蓄積されている場合True
        """
        return len(self.price_history) >= self.k
    
    def bulk_initialize(self, historical_prices):
        """履歴データを使用してモデルを一括初期化する。
        
        ゼロパディング期間をスキップして、有効なデータのみを使用する。
        
        Args:
            historical_prices (list): 履歴価格データのリスト
        """
        required_total_length = self.k + self.valid_start_index
        
        # 本来ならここにデータ長不足のエラーハンドリングが必要
        valid_start = self.valid_start_index
        valid_end = valid_start + self.k
        self.price_history = historical_prices[valid_start:valid_end]
        
        print(f"[INIT] Model bulk initialized: skipped {self.valid_start_index} zero-padded samples")
        print(f"[INIT] Using historical data [{valid_start}:{valid_end}] = {len(self.price_history)} prices")

    def decide_position(self):
        """現在の価格状況に基づいてポジションを決定する。
        
        MLモデルの予測結果とリスク管理ルールに基づいて、
        新規建玉、継続保有、決済の判定を行う。
        """
        if not self.ready():
            return

        current_price = self.price_history[-1]
        std = np.std(self.price_history)
        prob = self.model.predict_up_probability(np.array(self.price_history))
        predicted_side = 1 if prob > THRESHOLD_UP else -1 if prob <= THRESHOLD_DOWN else 0

        log_prefix = f"[MODEL] k={self.k}, p={self.p}"

        # ポジション保有中の処理
        if self.position['side'] != 0:
            self.position['count'] -= 1
            delta = (current_price -
                     self.position['price']) * self.position['side']
            risk = delta / self.position['std']
            print(
                f"{log_prefix} 保有中 | count={self.position['count']}, risk={risk:.2f}")

            exit_flag = self.position['count'] == 0 or risk > RIKAKU_STD or risk < -SONKIRI_STD

            if exit_flag:
                prev = self.position['side']
                print(
                    f"{log_prefix} 決済 (理由: {'カウント終了' if self.position['count']==0 else '利確/損切'})")
                self.position = {'side': 0, 'count': 0, 'price': 0, 'std': 0}

                if predicted_side == prev:
                    print(f"{log_prefix} 同方向継続 → カウントリセット")
                    self.position = {
                        'side': prev, 'count': self.p, 'price': current_price, 'std': std}
                elif predicted_side == -prev:
                    print(f"{log_prefix} 反対方向予測 → ポジション転換")
                    self.position = {
                        'side': predicted_side, 'count': self.p, 'price': current_price, 'std': std}
        elif predicted_side != 0:
            print(f"{log_prefix} 建玉開始 ({'LONG' if predicted_side == 1 else 'SHORT'})")
            self.position = {'side': predicted_side,
                             'count': self.p, 'price': current_price, 'std': std}

    def get_signal(self):
        """現在のトレードシグナルを取得する。
        
        Returns:
            int: トレードシグナル（1: 買い、-1: 売り、0: 待機）
        """
        return self.position['side'] if self.position['count'] > 0 else 0


class TraderManager:
    """取引全体を統括管理するメインクラス。"""
    
    def __init__(self):
        """トレーダーマネージャーを初期化する。
        
        API接続、取引モデルの設定、履歴データ初期化を行う。
        """
        self.api = BitFlyer(leverage=True, order_type='MARKET',
                            api_key=API_KEY, api_secret=API_SECRET)

        # 単一モデル構成
        self.model = TradingModel(k=90, p=30)
        
        # 履歴データ初期化（単一モデル用）
        self._initialize_single_model()
    
    def _initialize_single_model(self):
        """単一の取引モデルを履歴データで初期化する。
        
        DynamoDBから履歴価格データを取得してモデルを即座に使用可能な
        状態にする。失敗時は標準初期化にフォールバックする。
        """
        try:
            from historical_data import HistoricalDataInitializer
            
            initializer = HistoricalDataInitializer()
            print("[INIT] Attempting historical data initialization...")
            success = initializer.initialize_single_model(self.model)
            if success:
                print("[INIT] Historical data initialization completed successfully")
            else:
                print("[INIT] Historical data initialization failed, using standard initialization")
        except Exception as e:
            print(f"[INIT] Historical data initialization error: {e}")
            print("[INIT] Falling back to standard initialization")

    def step(self):
        """取引の1ステップを実行する。
        
        価格取得、モデル更新、売買判定、注文実行の一連の処理を行う。
        """
        print("\n[STEP] Start new trading step")
        start_time = time.time()

        self.api.cancel_all()

        # 本来ならここに価格取得エラーハンドリングが必要
        price = self.api.get_price()
        print(f"[INFO] Current price: {price}")

        # 単一モデルの更新と意思決定
        self.model.update(price)
        if self.model.ready():
            self.model.decide_position()
            signal = self.model.get_signal()
        else:
            print(f"[MODEL] k={self.model.k} not ready ({len(self.model.price_history)}/{self.model.k})")
            signal = 0

        actual_position = self.api.get_position()
        print(f"[INFO] Current position: {actual_position}")

        delta = signal - actual_position/BASE_AMOUNT
        print(f"[INFO] Model signal: {signal} (delta = {delta})")

        if delta != 0:
            amount = abs(delta) * BASE_AMOUNT
            side = 'BUY' if delta > 0 else 'SELL'
            print(f"[TRADE] {side} {amount:.4f} BTC (signal delta = {delta})")
            # 本来ならここに注文実行エラーハンドリングが必要
            self.api.make_order(side, amount)

        elapsed = time.time() - start_time
        # wait = max(0, 60 - elapsed)
        wait = max(0, 1 - elapsed)
        print(f"[WAIT] {wait:.2f}秒スリープ")
        time.sleep(wait)

    def run(self):
        """取引ボットのメインループを開始する。
        
        無限ループで取引ステップを継続実行する。
        """
        while True:
            # 本来ならここに全体的なエラーハンドリングが必要
            self.step()


if __name__ == '__main__':
    trader = TraderManager()
    # 本来ならここにキーボード割り込みハンドリングが必要
    trader.run()
