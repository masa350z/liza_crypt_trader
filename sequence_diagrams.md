# 取引ボット処理シーケンス図

VSCodeでMermaid図を表示するには：
1. `Mermaid Preview` 拡張機能をインストール
2. このファイルを開いてコマンドパレット（Ctrl+Shift+P）から `Mermaid: Preview` を実行

## 1. 基本初期化シーケンス（ダミーモデル）

```mermaid
sequenceDiagram
    participant Main as main()
    participant TM as TraderManager
    participant API as BitFlyer
    participant Model as TradingModel
    participant DM as DummyModel

    Main->>TM: TraderManager()
    TM->>API: BitFlyer(leverage=True)
    API-->>TM: API instance
    TM->>Model: TradingModel(90, 30)
    Model->>DM: DummyModel(90, 30)
    DM-->>Model: model instance
    Model-->>TM: trading model
    TM-->>Main: trader manager

    Main->>TM: trader.run()
    loop 取引ループ
        TM->>TM: step()
    end
```

## 2. ML統合初期化シーケンス

```mermaid
sequenceDiagram
    participant Main as main()
    participant TM as TraderManager
    participant Model as TradingModel
    participant HM as HybridTechnicalModel
    participant TF as TensorFlow
    participant FS as FileSystem

    Note over Main: USE_ML_MODEL=true

    Main->>TM: TraderManager()
    TM->>Model: TradingModel(90, 30)
    Model->>HM: HybridTechnicalModel(90, 30, model_path)
    
    alt モデルファイルが存在
        HM->>FS: os.path.exists(model_path)
        FS-->>HM: True
        HM->>HM: load_model()
        HM->>TF: build_hybrid_technical_model(k=90)
        TF-->>HM: model architecture
        HM->>TF: model.load_weights(model_path)
        TF-->>HM: weights loaded
        HM-->>Model: ML model ready
    else モデルファイルが存在しない
        Note over HM: Fallback to DummyModel
        HM-->>Model: model not loaded
    end
    
    Model-->>TM: trading model
```

## 3. 履歴データ初期化シーケンス

```mermaid
sequenceDiagram
    participant TM as TraderManager
    participant HDI as HistoricalDataInitializer
    participant DB as DynamoDB
    participant Model as TradingModel

    Note over TM: USE_HISTORICAL_DATA=true

    TM->>TM: _initialize_single_model()
    TM->>HDI: HistoricalDataInitializer()
    HDI-->>TM: initializer
    
    TM->>HDI: initialize_single_model(model)
    HDI->>HDI: is_available()
    
    alt DynamoDB利用可能
        HDI->>DB: connect & test
        DB-->>HDI: connection OK
        HDI->>HDI: fetch_historical_prices(fetch_minutes)
        HDI->>DB: scan(FilterExpression)
        DB-->>HDI: historical price data
        HDI->>Model: bulk_initialize(historical_prices)
        Model-->>HDI: initialization complete
        HDI-->>TM: success=True
    else DynamoDB利用不可
        HDI-->>TM: success=False
        Note over TM: 標準初期化にフォールバック
    end
```

## 4. メイン取引ループシーケンス

```mermaid
sequenceDiagram
    participant TM as TraderManager
    participant API as BitFlyer
    participant Model as TradingModel
    participant ML as MLModel/DummyModel

    loop 無限ループ
        TM->>TM: step()
        TM->>API: cancel_all()
        API-->>TM: orders cancelled
        
        TM->>API: get_price()
        API-->>TM: current_price
        
        TM->>Model: update(price)
        Model->>Model: price_history.append(price)
        
        alt モデル準備完了
            TM->>Model: decide_position()
            Model->>ML: predict_up_probability(price_history)
            ML-->>Model: probability
            Model->>Model: 売買判定ロジック
            Model-->>TM: position updated
            
            TM->>Model: get_signal()
            Model-->>TM: signal (-1, 0, 1)
        else モデル未準備
            Note over Model: 履歴データ不足
            TM->>TM: signal = 0
        end
        
        TM->>API: get_position()
        API-->>TM: actual_position
        
        alt 注文が必要
            TM->>API: make_order(side, amount)
            API-->>TM: order executed
        else 注文不要
            Note over TM: ポジション変更なし
        end
        
        TM->>TM: sleep(wait_time)
    end
```

## 5. ML予測詳細シーケンス

```mermaid
sequenceDiagram
    participant Model as TradingModel
    participant HM as HybridTechnicalModel
    participant DP as DataProcessing
    participant TF as TensorFlow

    Model->>HM: predict_up_probability(price_series)
    HM->>DP: make_prediction_features(prices, k)
    
    DP->>DP: calculate_technical_indicators()
    Note over DP: SMA, Bollinger, MACD, RSI計算
    DP->>DP: normalize_features()
    DP-->>HM: features (batch, k, 12)
    
    HM->>TF: model.predict(features)
    TF-->>HM: prediction [up_prob, down_prob]
    HM->>HM: extract up_probability
    HM-->>Model: up_probability (0.0-1.0)
```

## 6. ポジション管理シーケンス

```mermaid
sequenceDiagram
    participant Model as TradingModel
    participant Logic as PositionLogic

    Model->>Logic: decide_position()
    Logic->>Logic: calculate current metrics
    Note over Logic: std, risk, predicted_side
    
    alt 保有中の場合
        Logic->>Logic: update position count
        Logic->>Logic: calculate risk
        
        alt 決済条件満了
            Logic->>Logic: 決済処理
            alt 同方向継続
                Logic->>Logic: カウントリセット
            else 反対方向予測
                Logic->>Logic: ポジション転換
            end
        else 保有継続
            Note over Logic: count減算のみ
        end
    else 未保有の場合
        alt 予測シグナルあり
            Logic->>Logic: 新規建玉
        else シグナルなし
            Note over Logic: 待機
        end
    end
    
    Logic-->>Model: position updated
```

## 7. エラーハンドリング・フォールバックシーケンス

```mermaid
sequenceDiagram
    participant System as System
    participant ML as MLModel
    participant Dummy as DummyModel
    participant HDI as HistoricalDataInitializer

    Note over System: エラー発生時のフォールバック

    alt TensorFlow利用不可
        System->>ML: load ML model
        ML->>ML: import tensorflow
        ML-->>System: ImportError
        System->>Dummy: fallback to DummyModel
        Dummy-->>System: dummy model ready
    end

    alt モデルファイル不存在
        System->>ML: load weights
        ML->>ML: load_weights(path)
        ML-->>System: FileNotFoundError
        System->>Dummy: fallback to DummyModel
    end

    alt DynamoDB接続失敗
        System->>HDI: initialize historical data
        HDI->>HDI: connect to DynamoDB
        HDI-->>System: ConnectionError
        System->>System: use standard initialization
        Note over System: 段階的価格蓄積
    end
```