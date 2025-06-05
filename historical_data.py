"""履歴データをDynamoDBから取得して取引モデルを初期化するモジュール。

* HistoricalDataInitializer: DynamoDBから履歴価格データを取得して初期化
"""

import os
import time
from decimal import Decimal

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("[WARNING] boto3 not available. Historical data initialization disabled.")


class HistoricalDataInitializer:
    """履歴データによる取引モデル初期化を担当するクラス。
    
    DynamoDBから履歴価格データを取得し、取引モデルを即座に使用可能な
    状態に初期化する。既存の取引ロジックに影響を与えない。
    """
    
    def __init__(self):
        """環境変数からAWS設定を読み込んで初期化する。"""
        self.table_name = os.getenv('DYNAMODB_TABLE_NAME', 'crypto_currency_historicaldata')
        self.region_name = os.getenv('DYNAMODB_REGION', 'ap-northeast-1')
        self.exchange_name = os.getenv('DYNAMODB_EXCHANGE_NAME', 'bitflyer-FX')
        
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID') or os.getenv('MY_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY') or os.getenv('MY_SECRET_ACCESS_KEY')
        
        self.dynamodb = None
        self.table = None
        self._available = None
        
    def is_available(self):
        """履歴データ初期化機能が使用可能かチェックする。
        
        boto3の存在、AWS認証情報、DynamoDB接続を確認する。
        
        Returns:
            bool: DynamoDB接続と認証が正常な場合True
        """
        if self._available is not None:
            return self._available
            
        try:
            if not BOTO3_AVAILABLE:
                print("[INFO] boto3 not installed, historical data initialization disabled")
                self._available = False
                return False
                
            if not self.aws_access_key_id or not self.aws_secret_access_key:
                print("[INFO] AWS credentials not found, historical data initialization disabled")
                self._available = False
                return False
                
            # DynamoDB接続テスト
            self._connect_dynamodb()
            self._available = True
            print(f"[INFO] DynamoDB connection established: table={self.table_name}")
            return True
            
        except Exception as e:
            print(f"[INFO] DynamoDB connection failed: {e}")
            self._available = False
            return False
    
    def _connect_dynamodb(self):
        """指定された設定でDynamoDBに接続する。
        
        Raises:
            接続に失敗した場合の例外
        """
        if self.dynamodb is None:
            self.dynamodb = boto3.resource(
                'dynamodb',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name
            )
            self.table = self.dynamodb.Table(self.table_name)
            
            # 接続テスト（テーブル存在確認）
            self.table.load()
    
    def fetch_historical_prices(self, m):
        """指定された分数分の履歴価格データをDynamoDBから取得する。
        
        現在時刻から指定分数遡りの価格データを時系列順で取得する。
        
        Args:
            m (int): 取得したい履歴データの分数
            
        Returns:
            list: 価格データのリスト（時系列順）、取得失敗時は空リスト
        """
        if not self.is_available():
            return []
            
        # 本来ならここに時刻計算エラーハンドリングが必要
        current_time = int(time.time())
        start_time = current_time - (m * 60)
        
        print(f"[HIST] Fetching {m} minutes of historical data from DynamoDB...")
        
        # 本来ならここにDynamoDB接続エラーハンドリングが必要
        response = self.table.scan(
                FilterExpression='exchange = :exchange AND #ts BETWEEN :start AND :end',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={
                    ':exchange': self.exchange_name,
                    ':start': start_time,
                    ':end': current_time
                }
            )
            
            items = response['Items']
            
            # ページネーション対応
            while 'LastEvaluatedKey' in response:
                response = self.table.scan(
                    FilterExpression='exchange = :exchange AND #ts BETWEEN :start AND :end',
                    ExpressionAttributeNames={'#ts': 'timestamp'},
                    ExpressionAttributeValues={
                        ':exchange': self.exchange_name,
                        ':start': start_time,
                        ':end': current_time
                    },
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response['Items'])
            
            # 本来ならここにデータ不足のエラーハンドリングが必要
            
            # タイムスタンプでソート
            sorted_items = sorted(items, key=lambda x: int(x['timestamp']))
            
            # 価格データを抽出
            prices = []
            for item in sorted_items:
                # 本来ならここに価格データ変換エラーハンドリングが必要
                price = float(item['price'])
                prices.append(price)
            
            print(f"[HIST] Retrieved {len(prices)} price points from DynamoDB")
            
            # データ品質チェック
            if len(prices) < m * 0.5:  # 期待データの50%未満の場合は警告
                print(f"[HIST] Warning: Only {len(prices)} data points for {m} minutes (expected ~{m})")
            
            return prices
    
    def initialize_trading_models(self, models_list):
        """複数の取引モデルを履歴データで一括初期化する。
        
        各モデルの最大必要データ長を計算し、必要な履歴データを取得して
        各モデルを初期化する。
        
        Args:
            models_list (list): TradingModelオブジェクトのリスト
            
        Returns:
            bool: 初期化が成功した場合True
        """
        if not self.is_available():
            print("[HIST] DynamoDB not available, skipping historical initialization")
            return False
            
        if not models_list:
            print("[HIST] No models to initialize")
            return True
            
        try:
            max_k = max(model.k for model in models_list)
            max_valid_start_index = max(model.valid_start_index for model in models_list)
            
            # 実効的に必要なデータ長 = 履歴長 + ゼロパディング期間 + 20%マージン
            effective_minutes = max_k + max_valid_start_index
            fetch_minutes = int(effective_minutes * 1.2)
            
            print(f"[HIST] Initializing {len(models_list)} models with {max_k} minutes history")
            print(f"[HIST] Max valid start index: {max_valid_start_index}, effective minutes: {effective_minutes}")
            print(f"[HIST] Fetching {fetch_minutes} minutes of data from DynamoDB")
            
            # 履歴データ取得
            historical_prices = self.fetch_historical_prices(fetch_minutes)
            
            if not historical_prices:
                print("[HIST] No historical data available")
                return False
            
            # 各モデルを履歴データで初期化
            initialized_count = 0
            for model in models_list:
                try:
                    if hasattr(model, 'bulk_initialize'):
                        model.bulk_initialize(historical_prices)
                        initialized_count += 1
                        print(f"[HIST] Model initialized: k={model.k}min, ready={model.ready()}")
                    else:
                        print(f"[HIST] Model doesn't support bulk_initialize, skipping")
                except Exception as e:
                    print(f"[HIST] Model initialization failed: {e}")
            
            print(f"[HIST] Successfully initialized {initialized_count}/{len(models_list)} models")
            return initialized_count > 0
            
        except Exception as e:
            print(f"[HIST] Models initialization error: {e}")
            return False

    def initialize_single_model(self, model):
        """単一の取引モデルを履歴データで初期化する。
        
        指定されたモデルの要件に合わせて必要な履歴データを取得し、
        モデルを即座に使用可能な状態に初期化する。
        
        Args:
            model: TradingModelインスタンス
            
        Returns:
            bool: 初期化が成功したかどうか
        """
        if not self.is_available():
            print("[HIST] DynamoDB not available, skipping historical initialization")
            return False
            
        try:
            effective_minutes = model.k + model.valid_start_index
            fetch_minutes = int(effective_minutes * 1.2)
            
            print(f"[HIST] Initializing single model: k={model.k}min, valid_start={model.valid_start_index}")
            print(f"[HIST] Fetching {fetch_minutes} minutes of data from DynamoDB")
            
            # 履歴データ取得
            historical_prices = self.fetch_historical_prices(fetch_minutes)
            
            if not historical_prices:
                print("[HIST] No historical data available")
                return False
            
            # モデルを履歴データで初期化
            if not hasattr(model, 'bulk_initialize'):
                print("[HIST] Model doesn't support bulk_initialize")
                return False
                
            model.bulk_initialize(historical_prices)
            print(f"[HIST] Single model initialized successfully: ready={model.ready()}")
            return True
            
        except Exception as e:
            print(f"[HIST] Single model initialization error: {e}")
            return False
    
    def validate_data_quality(self, prices, expected_m):
        """取得した価格データの品質を検証する。
        
        データ量、価格の範囲、異常値の有無などをチェックし、
        データ品質の評価結果を返す。
        
        Args:
            prices (list): 価格データのリスト
            expected_m (int): 期待されるデータ数（分数）
            
        Returns:
            dict: 検証結果の詳細情報
        """
        if not prices:
            return {
                'valid': False,
                'issues': ['No data available'],
                'data_count': 0,
                'expected_count': expected_m
            }
        
        issues = []
        
        # データ量チェック
        if len(prices) < expected_m * 0.3:  # 30%未満は不十分
            issues.append(f"Insufficient data: {len(prices)}/{expected_m}")
        
        # 価格の妥当性チェック
        try:
            min_price = min(prices)
            max_price = max(prices)
            avg_price = sum(prices) / len(prices)
            
            # 異常な価格範囲チェック（基準値から大きく外れていないか）
            if min_price <= 0:
                issues.append("Invalid price: zero or negative values found")
            
            if max_price / min_price > 2.0:  # 2倍以上の変動は異常
                issues.append(f"Extreme price variation: {min_price:.0f} to {max_price:.0f}")
                
        except (ValueError, ZeroDivisionError):
            issues.append("Price calculation error")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'data_count': len(prices),
            'expected_count': expected_m,
            'min_price': min_price if prices else 0,
            'max_price': max_price if prices else 0,
            'avg_price': avg_price if prices else 0
        }