import urllib.parse
import requests
import hashlib
import json
import time
import hmac


class BitFlyer:
    def __init__(self, leverage=False, advantage=0, order_type='MARKET', api_key=None, api_secret=None):
        self.api_url = "https://api.bitflyer.jp"
        self.product_code = 'FX_BTC_JPY' if leverage else 'BTC_JPY'
        self.advantage = int(advantage)
        self.order_type = order_type
        self.api_key = api_key
        self.api_secret = api_secret

    def request(self, endpoint, method="GET", params=None, allow_empty=False):
        try:
            if method == "POST":
                body = json.dumps(params)
            else:
                body = "?" + urllib.parse.urlencode(params) if params else ""

            access_timestamp = str(time.time())
            text = f"{access_timestamp}{method}{endpoint}{body}".encode()
            secret = self.api_secret.encode()
            access_sign = hmac.new(secret, text, hashlib.sha256).hexdigest()

            headers = {
                "ACCESS-KEY": self.api_key,
                "ACCESS-TIMESTAMP": access_timestamp,
                "ACCESS-SIGN": access_sign,
                "Content-Type": "application/json"
            }

            url = self.api_url + endpoint
            with requests.Session() as session:
                session.headers.update(headers)
                if method == "GET":
                    response = session.get(url, params=params)
                else:
                    response = session.post(url, data=json.dumps(params))

                print(f"[DEBUG] {method} {url}")
                print(f"[DEBUG] Status Code: {response.status_code}")
                print(f"[DEBUG] Headers: {response.headers}")
                print(f"[DEBUG] Content: {response.content}")

                response.raise_for_status()

                if not response.content:
                    if allow_empty:
                        return None  # or {} if you prefer
                    raise ValueError("Empty response received.")

                return response.json()

        except requests.RequestException as e:
            raise ConnectionError(f"API request failed: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid response: {e}")

    def ticker(self):
        params = {'product_code': self.product_code}
        endpoint = "/v1/ticker"
        res = self.request(endpoint, params=params)

        return res

    def get_price(self):
        res = self.ticker()

        return float(res['ltp'])

    def board(self):
        params = {'product_code': self.product_code}
        endpoint = "/v1/board"
        res = self.request(endpoint, params=params)

        return res

    def asset(self):
        if self.product_code == 'FX_BTC_JPY':
            endpoint = "/v1/me/getcollateral"
        else:
            endpoint = "/v1/me/getbalance"
        res = self.request(endpoint)

        return res

    def position(self):
        params = {'product_code': 'FX_BTC_JPY'}
        endpoint = "/v1/me/getpositions"
        res = self.request(endpoint, params=params)

        return res

    def order(self, params):
        endpoint = "/v1/me/sendchildorder"
        res = self.request(endpoint, "POST", params=params)

        return res

    def get_board(self):
        res = self.board()

        return res['asks'], res['bids']

    def get_asset(self):
        if self.product_code == 'BTC_JPY':
            res = self.asset()
            jpy = float(res[0]['amount'])
            btc = float(res[1]['amount'])
            btc_price = self.get_price()

            return jpy + btc*btc_price

        else:  # product_code=='FX_BTC_JPY'
            res = self.asset()
            asset = res['collateral']
            pnl = res['open_position_pnl']

            return float(asset + pnl)

    def get_position(self):
        if self.product_code == 'BTC_JPY':
            res = self.asset()
            return int(float(res[1]['amount'])*1000)/1000

        else:  # product_code=='FX_BTC_JPY'
            res = self.position()

            posi = 0
            for p in res:
                if p['side'] == 'BUY':
                    posi += p['size']
                else:
                    posi -= p['size']

            return int(float(posi)*1000)/1000

    def make_order(self, side, size):
        if size < 0.001:
            return

        params = {'product_code': self.product_code,
                  'child_order_type': self.order_type,
                  'side': side}

        if self.order_type == 'LIMIT':
            asks, bids = self.get_board()
            if side == 'BUY':
                price = int(float(asks[0]['price'])) - self.advantage
            else:  # side == 'SELL'
                price = int(float(bids[0]['price'])) + self.advantage
            params['price'] = price

        amount = int(float(size)*1000)/1000
        params['size'] = amount
        self.order(params)

    def zero_position(self, emergency=False):
        order_size = -self.get_position()
        if not order_size == 0:
            order_side = 'BUY' if order_size > 0 else 'SELL'
            if emergency:
                pr_order_type = self.order_type
                self.order_type = 'MARKET'
                if self.product_code == 'BTC_JPY':
                    if order_side == 'SELL':
                        self.make_order(order_side, abs(order_size))
                else:
                    self.make_order(order_side, abs(order_size))
                self.order_type = pr_order_type
            else:
                if self.product_code == 'BTC_JPY':
                    if order_side == 'SELL':
                        self.make_order(order_side, abs(order_size))
                else:
                    self.make_order(order_side, abs(order_size))

    def make_position(self, act_position, calc_position):
        if calc_position == 0:
            self.zero_position()
        else:
            order_size = calc_position - act_position
            if order_size == 0:
                pass
            else:
                order_side = 'BUY' if order_size > 0 else 'SELL'
                if self.product_code == 'BTC_JPY':
                    if calc_position > 0:
                        self.make_order(order_side, abs(order_size))
                    else:
                        self.zero_position()
                else:
                    self.make_order(order_side, abs(order_size))

    def cancel_all(self):
        endpoint = "/v1/me/cancelallchildorders"
        params = {'product_code': self.product_code}
        res = self.request(endpoint, params=params,
                           method='POST', allow_empty=True)
        print(str(res)[:100] + '.....')

        return res
