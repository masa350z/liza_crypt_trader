"""簡易ビットコイントレードボット

    * TradingModel: トレーディングモデル (履歴に基づくポジション判定)
    * TraderManager: トレード管理者 (API操作とモデル統括)
"""

from dynamodb import HistoricalDB
from datetime import datetime
from exchanges import BitFlyer
from model import DummyModel
import numpy as np
import time
import os


# 環境変数からAPIキーとシークレットを取得
API_KEY = os.getenv('BITFLYER_API_KEY')
API_SECRET = os.getenv('BITFLYER_API_SECRET')

# トレード設定値
EXCHANGE = 'bitflyer-FX'
BASE_AMOUNT = 0.001
THRESHOLD_UP = 0.5
THRESHOLD_DOWN = 0.5
RIKAKU_STD = 2
SONKIRI_STD = 1


class TradingModel:
    """履歴データをもとに売買ポジションを判断するモデル

    Args:
        history_minutes (int): 使用する過去データの分数
        predict_minutes (int): 予測対象となる未来の分数

    Attributes:
        model (DummyModel): 上昇確率を出力するダミーモデル
        price_history (list): 価格履歴リスト
        position (dict): 現在の保有ポジション情報

    """

    def __init__(self, history_minutes, predict_minutes):
        self.history_minutes = history_minutes
        self.predict_minutes = predict_minutes
        self.model = DummyModel(history_minutes, predict_minutes)
        self.position = {'side': 0, 'count': 0, 'price': 0, 'std': 0}

        historicaldb = HistoricalDB()
        self.price_history, _ = historicaldb.get_historical_data(
            exchange=EXCHANGE, past_minutes=history_minutes)

    def update(self, price):
        """現在価格を履歴に追加し、長さを制御する"""
        self.price_history.append(price)
        if len(self.price_history) > self.history_minutes:
            self.price_history = self.price_history[-self.history_minutes:]

    def ready(self):
        """履歴が十分たまったか判定する"""
        return len(self.price_history) >= self.history_minutes

    def ret_exit_flsg(self, current_price):
        # 現在価格と建値の差に、ポジション方向（LONG:+1, SHORT:-1）を掛けた値を算出
        delta = (current_price -
                 self.position['price']) * self.position['side']

        # 価格変動をボラティリティ（標準偏差）で割ってリスクを正規化
        risk = delta / self.position['std']

        # 決済条件判定（カウント終了 or 利確/損切到達）
        exit_flag = self.position['count'] == 0 or risk > RIKAKU_STD or risk < -SONKIRI_STD

        return exit_flag

    def decide_position(self):
        """売買ポジションを決定する"""
        if not self.ready():
            # 履歴が足りない場合は何もしない
            return

        current_price = self.price_history[-1]
        std = np.std(self.price_history)
        prob = self.model.predict_up_probability(np.array(self.price_history))
        # 三項演算子を使って、確率に応じてポジションを決定（上昇予測なら1、下降予測なら-1、どちらでもなければ0）
        predicted_side = 1 if prob > THRESHOLD_UP else -1 if prob <= THRESHOLD_DOWN else 0

        log_prefix = f"[MODEL] M={self.history_minutes}, N={self.predict_minutes}"

        # ポジション保有中の処理
        if self.position['side'] != 0:
            self.position['count'] -= 1  # カウントを減らす

            exit_flag = self.ret_exit_flsg(current_price)  # 利確、損切、カウント終了の判定

            if exit_flag:  # ポジションを解消する場合
                prev = self.position['side']

                # 継続判断（同方向なら建て直し、反対なら転換）
                if predicted_side == prev:
                    self.position = {
                        'side': predicted_side, 'count': self.predict_minutes, 'price': current_price, 'std': std}
                elif predicted_side == -prev:
                    self.position = {
                        'side': predicted_side, 'count': self.predict_minutes, 'price': current_price, 'std': std}
                else:
                    self.position = {'side': 0,
                                     'count': 0, 'price': 0, 'std': 0}

        elif predicted_side != 0:  # ポジション未保有時の新規建玉判断
            self.position = {'side': predicted_side,
                             'count': self.predict_minutes, 'price': current_price, 'std': std}

        print(f"{log_prefix} {self.position}")

    def get_signal(self):
        """現在ポジションの売買シグナルを返す"""
        return self.position['side'] if self.position['count'] > 0 else 0


class TraderManager:
    """複数モデルを管理し、取引所APIを通じて売買を実施するクラス

    Attributes:
        api (BitFlyer): BitFlyer取引所APIインスタンス
        models (list): 使用するトレードモデルリスト
    """

    def __init__(self, debug_mode=False):
        self.api = BitFlyer(leverage=True, order_type='MARKET',
                            api_key=API_KEY, api_secret=API_SECRET)

        self.models = [
            TradingModel(history_minutes=90, predict_minutes=30),
            TradingModel(history_minutes=60, predict_minutes=20),
            TradingModel(history_minutes=15, predict_minutes=5),
            TradingModel(history_minutes=120, predict_minutes=45),
        ]

        self.debug_mode = debug_mode
        self.sleep_time = 3 if debug_mode else 60

        self.count = 0

    def step(self):
        """1ステップ分のトレード実行処理"""
        self.count += 1

        print(f"\n[STEP] {self.count} | {datetime.now()}")  # ステップ開始のログ
        start_time = time.time()

        self.api.cancel_all()  # 既存注文をキャンセル

        try:
            price = self.api.get_price()
            print(f"[INFO] Current price: {price}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch price: {e}")
            return

        for model in self.models:  # 各モデルに価格を渡し、ポジション決定
            model.update(price)  # 価格履歴を更新
            if model.ready():
                model.decide_position()  # ポジションを決定
            else:
                print(
                    f"[MODEL] M={model.history_minutes} not ready ({len(model.price_history)}/{model.history_minutes})")

        # モデル全体のシグナルを集約
        total_signal = sum(model.get_signal() for model in self.models)

        actual_position = self.api.get_position()  # 現在のポジションを取得
        print(f"[INFO] Current position: {actual_position}")

        delta = total_signal - actual_position/BASE_AMOUNT  # シグナルの変化量を計算
        print(f"[INFO] Aggregated signal: {total_signal} (delta = {delta})")

        if delta != 0:  # シグナルに変化があった場合
            # 売買注文を出す
            amount = abs(delta) * BASE_AMOUNT
            side = 'BUY' if delta > 0 else 'SELL'
            print(f"[TRADE] {side} {amount:.4f} BTC (signal delta = {delta})")
            try:
                if not self.debug_mode:
                    self.api.make_order(side, amount)
            except Exception as e:
                print(f"[ERROR] Order failed: {e}")

        elapsed = time.time() - start_time
        wait = max(0, self.sleep_time - elapsed)
        time.sleep(wait)

    def run(self):
        while True:
            try:
                self.step()
            except Exception as e:
                print(f'[ERROR] Unhandled Exception: {e}')
                time.sleep(60)


if __name__ == '__main__':
    trader = TraderManager(debug_mode=True)
    try:
        trader.run()
    except KeyboardInterrupt:
        print('[STOPPED]')
