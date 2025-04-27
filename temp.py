# %%
from dynamodb import HistoricalDB
import boto3
from decimal import Decimal
from exchanges import BitFlyer
import os
import time
from boto3.dynamodb.conditions import Key

table_name = 'crypto_currency_historicaldata'
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(table_name)

# %%
timestamp_threshold = int(time.time()) - 60 * 60 * 3

# クエリ実行
response = table.query(
    KeyConditionExpression=Key('exchange').eq(
        "bitflyer-FX") & Key('timestamp').gte(timestamp_threshold)
)

# 結果取得
items = response['Items']

prices = [int(i['price']) for i in items]
timestamps = [int(i['timestamp']) for i in items]
# %%
# %%
db = HistoricalDB()
# %%
db.get_historical_data("bitflyer-FX", 3)
