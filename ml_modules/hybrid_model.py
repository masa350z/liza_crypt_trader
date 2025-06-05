"""ハイブリッドテクニカル分析モデルのアーキテクチャ定義モジュール。

* build_hybrid_technical_model: 技術指標を統合したTensorFlowモデルを構築
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def build_hybrid_technical_model(k) -> tf.keras.Model:
    """ハイブリッドテクニカル分析モデルを構築する。
    
    liza_simple_predictorと同一のアーキテクチャでモデルを構築する。
    
    Args:
        k (int): 入力系列長（時間ステップ数）
        
    Returns:
        tf.keras.Model: コンパイル済みのハイブリッドモデル
        
    Note:
        入力形状: (None, k, 12) - 正規化済みデータ
          0   : 価格
          1-6 : SMA差分 6チャンネル
          7-9 : ボリンジャー差分 3チャンネル
         10   : MACD差分
         11   : RSI (-1〜+1)
        出力: 2クラスsoftmax (上昇/下落確率)
    """
    inp = Input(shape=(k, 12), name="input")

    # 特徴量の分割
    price = layers.Lambda(lambda x: x[...,  0: 1])(inp)  # (B, L, 1)
    sma = layers.Lambda(lambda x: x[...,  1: 7])(inp)  # (B, L, 6)
    boll = layers.Lambda(lambda x: x[...,  7:10])(inp)  # (B, L, 3)
    macd = layers.Lambda(lambda x: x[..., 10:11])(inp)  # (B, L, 1)
    rsi = layers.Lambda(lambda x: x[..., 11:12])(inp)  # (B, L, 1)

    # 価格ブランチ (Conv→GRU)
    x_p = layers.Conv1D(16, 10, padding="same", activation="relu")(price)
    x_p = layers.Conv1D(8, 5, padding="same", activation="relu")(x_p)
    x_p = layers.GRU(32, return_sequences=False)(x_p)
    x_p = layers.Dense(32, activation="relu")(x_p)

    # SMAブランチ (Conv×2→GAP)
    x_s = layers.Conv1D(32, 10, padding="same", activation="relu")(sma)
    x_s = layers.Conv1D(16, 5, padding="same", activation="relu")(x_s)
    x_s = layers.GlobalAveragePooling1D()(x_s)
    x_s = layers.Dense(32, activation="relu")(x_s)

    # ボリンジャーブランチ (軽量Conv→GAP)
    x_b = layers.Conv1D(16, 5, padding="same", activation="relu")(boll)
    x_b = layers.GlobalAveragePooling1D()(x_b)
    x_b = layers.Dense(32, activation="relu")(x_b)

    # MACDブランチ (GRU)
    x_m = layers.GRU(16, return_sequences=False)(macd)
    x_m = layers.Dense(32, activation="relu")(x_m)

    # RSIブランチ (Conv→GAP)
    x_r = layers.Conv1D(16, 5, padding="same", activation="relu")(rsi)
    x_r = layers.GlobalAveragePooling1D()(x_r)
    x_r = layers.Dense(32, activation="relu")(x_r)

    # 特徴量結合
    fused = layers.Add()([x_p, x_s, x_b, x_m, x_r])

    # 分類ヘッダ
    x = layers.BatchNormalization()(fused)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)

    out = layers.Dense(2, activation="softmax", name="direction")(x)

    model = Model(inputs=inp, outputs=out, name="Hybrid_TISplit_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model