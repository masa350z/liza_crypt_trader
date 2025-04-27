# %%
from exchanges import BitFlyer
import os

api_key = os.getenv('BITFLYER_API_KEY')
api_secret = os.getenv('BITFLYER_API_SECRET')
bf = BitFlyer(leverage=True, api_key=api_key, api_secret=api_secret)
# %%
bf.ticker()
# %%
bf.make_order(side='SELL', size=0.001)
# %%
bf.make_order(side='BUY', size=0.001)
