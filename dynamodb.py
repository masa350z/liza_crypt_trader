from boto3.dynamodb.conditions import Key
import boto3
import time


class HistoricalDB:
    def __init__(self):
        dynamodb = boto3.resource('dynamodb')
        table_name = 'crypto_currency_historicaldata'
        self.table = dynamodb.Table(table_name)

    def get_historical_data(self, exchange, past_minutes):
        """指定した取引所とタイムスタンプに基づいて履歴データを取得する"""

        timestamp_threshold = int(time.time()) - 60 * 60 * (past_minutes + 1)

        response = self.table.query(
            KeyConditionExpression=Key('exchange').eq(
                exchange) & Key('timestamp').gte(timestamp_threshold)
        )

        items = response['Items']

        prices = [int(i['price']) for i in items][-past_minutes:]
        timestamps = [int(i['timestamp']) for i in items][-past_minutes:]

        return prices, timestamps
