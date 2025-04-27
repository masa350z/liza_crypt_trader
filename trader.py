import os
import time
import numpy as np
from exchanges import BitFlyer
from model import DummyModel

API_KEY = os.getenv('BITFLYER_API_KEY')
API_SECRET = os.getenv('BITFLYER_API_SECRET')

BASE_AMOUNT = 0.001
THRESHOLD_UP = 0.5
THRESHOLD_DOWN = 0.5
RIKAKU_STD = 2
SONKIRI_STD = 1


class TradingModel:
    def __init__(self, history_minutes, predict_minutes):
        self.history_minutes = history_minutes
        self.predict_minutes = predict_minutes
        self.model = DummyModel(history_minutes, predict_minutes)
        self.price_history = []
        self.position = {'side': 0, 'count': 0, 'price': 0, 'std': 0}

    def update(self, price):
        self.price_history.append(price)
        if len(self.price_history) > self.history_minutes:
            self.price_history = self.price_history[-self.history_minutes:]

    def ready(self):
        return len(self.price_history) >= self.history_minutes

    def decide_position(self):
        if not self.ready():
            return

        current_price = self.price_history[-1]
        std = np.std(self.price_history)
        prob = self.model.predict_up_probability(np.array(self.price_history))
        predicted_side = 1 if prob > THRESHOLD_UP else -1 if prob <= THRESHOLD_DOWN else 0

        log_prefix = f"[MODEL] M={self.history_minutes}, N={self.predict_minutes}"

        # 保有中
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
                        'side': prev, 'count': self.predict_minutes, 'price': current_price, 'std': std}
                elif predicted_side == -prev:
                    print(f"{log_prefix} 反対方向予測 → ポジション転換")
                    self.position = {
                        'side': predicted_side, 'count': self.predict_minutes, 'price': current_price, 'std': std}
        elif predicted_side != 0:
            print(f"{log_prefix} 建玉開始 ({'LONG' if predicted_side == 1 else 'SHORT'})")
            self.position = {'side': predicted_side,
                             'count': self.predict_minutes, 'price': current_price, 'std': std}

    def get_signal(self):
        return self.position['side'] if self.position['count'] > 0 else 0


class TraderManager:
    def __init__(self):
        self.api = BitFlyer(leverage=True, order_type='MARKET',
                            api_key=API_KEY, api_secret=API_SECRET)

        self.models = [
            TradingModel(history_minutes=90, predict_minutes=30),
            TradingModel(history_minutes=60, predict_minutes=20),
            TradingModel(history_minutes=15, predict_minutes=5),
            TradingModel(history_minutes=120, predict_minutes=45),
        ]

    def step(self):
        print("\n[STEP] Start new trading step")
        start_time = time.time()

        self.api.cancel_all()

        try:
            price = self.api.get_price()
            print(f"[INFO] Current price: {price}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch price: {e}")
            return

        for model in self.models:
            model.update(price)
            if model.ready():
                model.decide_position()
            else:
                print(
                    f"[MODEL] M={model.history_minutes} not ready ({len(model.price_history)}/{model.history_minutes})")

        total_signal = sum(model.get_signal() for model in self.models)

        actual_position = self.api.get_position()
        print(f"[INFO] Current position: {actual_position}")

        delta = total_signal - actual_position/BASE_AMOUNT
        print(f"[INFO] Aggregated signal: {total_signal} (delta = {delta})")

        if delta != 0:
            amount = abs(delta) * BASE_AMOUNT
            side = 'BUY' if delta > 0 else 'SELL'
            print(f"[TRADE] {side} {amount:.4f} BTC (signal delta = {delta})")
            try:
                self.api.make_order(side, amount)
            except Exception as e:
                print(f"[ERROR] Order failed: {e}")

        elapsed = time.time() - start_time
        # wait = max(0, 60 - elapsed)
        wait = max(0, 1 - elapsed)
        print(f"[WAIT] {wait:.2f}秒スリープ")
        time.sleep(wait)

    def run(self):
        while True:
            try:
                self.step()
            except Exception as e:
                print(f'[ERROR] Unhandled Exception: {e}')
                time.sleep(60)


if __name__ == '__main__':
    trader = TraderManager()
    try:
        trader.run()
    except KeyboardInterrupt:
        print('[STOPPED]')
