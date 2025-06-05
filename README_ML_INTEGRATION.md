# ML Model Integration Guide

## 概要

liza_crypt_traderは、liza_simple_predictorで訓練された機械学習モデルを統合して、実際の価格予測に基づく自動売買を行えるように拡張されました。

## 統合されたコンポーネント

### 新しいファイル構成

```
liza_crypt_trader/
├── ml_model.py                 # MLモデルのラッパークラス
├── ml_modules/                 # MLモジュール群
│   ├── __init__.py
│   ├── hybrid_model.py         # ハイブリッドテクニカルモデル
│   ├── data_processing.py      # データ前処理
│   └── technical_indicators.py # テクニカル指標計算
├── test_basic_integration.py   # 基本統合テスト
├── test_ml_integration.py      # 完全MLテスト（TensorFlow必要）
└── README_ML_INTEGRATION.md   # このファイル
```

### 主要クラス

1. **HybridTechnicalModel**: liza_simple_predictorのハイブリッドテクニカルモデルのラッパー
2. **TradingModel**: 既存のDummyModelまたは新しいHybridTechnicalModelを使用可能

## 使用方法

### 1. 基本設定（MLモデル）

```bash
# MLモデルを使用（モデルファイルがない場合はエラーハンドリング）
python3 trader.py
```

### 2. MLモデルの使用

#### 2.1 依存関係のインストール

```bash
pip install -r requirements.txt
```

#### 2.2 ベストモデルの配置

liza_simple_predictorから最高性能のモデル重みファイル（.h5）をコピー：

```bash
# 例：ベストモデルをコピー
mkdir -p models
cp ../liza_simple_predictor/results/BTCJPY/weights/best_model.weights.h5 models/
```

#### 2.3 環境変数の設定

```bash
# モデルパスを指定（オプション）
export ML_MODEL_PATH=models/best_model.weights.h5

# 自動売買開始
python3 trader.py
```

## 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `ML_MODEL_PATH` | `models/best_model.weights.h5` | モデル重みファイルのパス |
| `BITFLYER_API_KEY` | - | BitFlyer API キー |
| `BITFLYER_API_SECRET` | - | BitFlyer API シークレット |

## モデル仕様

### 入力仕様

- **入力長 (k)**: 設定可能（推奨：90-360分）
- **特徴量**: 12次元
  - 価格（正規化済み）
  - SMA差分（6次元）
  - ボリンジャーバンド差分（3次元）
  - MACD差分（1次元）
  - RSI（1次元、-1〜+1正規化）

### 出力仕様

- **形式**: 2クラス確率（上昇/下降）
- **範囲**: 0.0〜1.0（上昇確率）

## エラーハンドリング

システムは以下の場合にエラーを発生させます：

1. TensorFlowがインストールされていない
2. モデルファイルが見つからない
3. モデル読み込みに失敗
4. 予測中にエラーが発生
5. 価格データが不足

※ ダミーモデルは削除されました。適切なモデルファイルが必要です。

## テスト

### 基本統合テスト（TensorFlow不要）

```bash
python3 test_basic_integration.py
```

### 完全MLテスト（TensorFlow必要）

```bash
python3 test_ml_integration.py
```

## トラブルシューティング

### TensorFlowインストールエラー

```bash
# CPU版をインストール
pip install tensorflow-cpu

# または、GPU版
pip install tensorflow
```

### モデル読み込みエラー

1. モデルファイルパスを確認
2. モデルファイルの権限を確認
3. ログを確認してエラー詳細を把握

### 予測エラー

- システムは自動的にダミーモデルにフォールバック
- ログで詳細なエラー原因を確認可能

## パフォーマンス

- **メモリ使用量**: 約200-500MB（モデルサイズによる）
- **予測時間**: 1回あたり10-50ms (kに依存)
- **フォールバック時間**: 即座（ランダム予測）

## DynamoDB履歴データ統合

### 概要

起動時の待機時間を0にするため、DynamoDBから過去の価格データを取得して即座に取引を開始する機能を追加しました。

### 新しいファイル構成

```
liza_crypt_trader/
├── historical_data.py          # DynamoDB履歴データ管理
├── test_historical_integration.py # 履歴データ統合テスト
└── (既存ファイル...)
```

### 使用方法

#### 1. DynamoDB履歴データ機能の有効化

```bash
# DynamoDB履歴データ機能を有効化
export USE_HISTORICAL_DATA=true

# AWS認証情報設定
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# DynamoDBテーブル設定（オプション）
export DYNAMODB_TABLE_NAME=crypto_currency_historicaldata
export DYNAMODB_REGION=ap-northeast-1
export DYNAMODB_EXCHANGE_NAME=bitflyer-FX
```

#### 2. 必要な依存関係

```bash
pip install boto3
```

#### 3. 実行

```bash
# MLモデル + 履歴データ統合（両方常に有効）
python3 trader.py
```

### 動作フロー比較

**従来**：
```
起動 → 90分間価格蓄積 → 取引開始
```

**DynamoDB統合後**：
```
起動 → DynamoDB履歴取得（5-10秒） → 即座に取引開始
※ DynamoDB利用不可時は従来フローに自動フォールバック
```

### 拡張環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `AWS_ACCESS_KEY_ID` | - | AWS認証キー |
| `AWS_SECRET_ACCESS_KEY` | - | AWS認証シークレット |
| `DYNAMODB_TABLE_NAME` | `crypto_currency_historicaldata` | DynamoDBテーブル名 |
| `DYNAMODB_REGION` | `ap-northeast-1` | AWSリージョン |
| `DYNAMODB_EXCHANGE_NAME` | `bitflyer-FX` | 取引所識別名 |

### フォールバック機能

以下の場合、自動的に従来の段階的初期化にフォールバック：

1. boto3がインストールされていない
2. AWS認証情報が設定されていない
3. DynamoDBテーブルにアクセスできない
4. 履歴データが不足している
5. データ品質に問題がある

### 履歴データ統合テスト

```bash
# 履歴データ統合テスト
python3 test_historical_integration.py
```

### トラブルシューティング（履歴データ）

#### boto3インストールエラー
```bash
pip install boto3
```

#### AWS認証エラー
- AWS認証情報を確認
- IAMロールでDynamoDBアクセス権限を確認

#### データ不足エラー
- DynamoDBテーブルに十分な履歴データがあるか確認
- Lambdaによる価格データ収集が正常に動作しているか確認

## 今後の拡張

1. 複数モデルのアンサンブル
2. リアルタイムモデル更新
3. パフォーマンス監視
4. A/Bテスト機能
5. DynamoDBクエリの最適化
6. リアルタイム価格ストリーミング

## サポート

問題が発生した場合は、以下のログを確認してください：

- `[INIT]`: 初期化ログ
- `[ML]`: MLモデル関連ログ
- `[MODEL]`: 取引判定ログ
- `[HIST]`: 履歴データ関連ログ
- `[WARNING]`: 警告メッセージ