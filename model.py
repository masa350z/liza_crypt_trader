import random


class DummyModel:
    def __init__(self, history_minutes, predict_minutes):
        self.history_minutes = history_minutes
        self.predict_minutes = predict_minutes

    def predict_up_probability(self, price_series):
        return random.uniform(0, 1)
