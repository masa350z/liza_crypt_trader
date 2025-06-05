"""暗号通貨取引のためのMLモデル統合モジュール。

* HybridTechnicalModel: テクニカル指標を統合したMLモデルのラッパー
"""

import os
import numpy as np

# 本来ならここにTensorFlowインポートエラーハンドリングが必要
import tensorflow as tf
from ml_modules.hybrid_model import build_hybrid_technical_model
from ml_modules.data_processing import make_prediction_features
TENSORFLOW_AVAILABLE = True


class HybridTechnicalModel:
    """ハイブリッドテクニカル分析モデルのラッパークラス。
    
    TensorFlowモデルをラップし、シームレスな統合を提供する。
    """
    
    def __init__(self, k, p, model_path=None):
        """ハイブリッドテクニカルモデルを初期化する。
        
        Args:
            k (int): 入力系列長（使用する過去の分数）
            p (int): 予測期間（予測では使用されないが互換性のため保持）
            model_path (:obj:`str`, optional): 保存されたモデル重みファイルのパス(.h5ファイル)
        """
        self.k = k
        self.p = p
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
        # モデルパスが指定されている場合はモデルを読み込み
        if model_path and os.path.exists(model_path):
            self.load_model()
    
    def load_model(self):
        """事前訓練済みモデルの重みを読み込む。
        
        Raises:
            モデルファイルが存在しないか読み込みに失敗した場合の例外
        """
        # 本来ならここにTensorFlow可用性チェックが必要
        # 本来ならここにモデルファイル存在チェックが必要
        
        # モデルアーキテクチャを構築
        self.model = build_hybrid_technical_model(k=self.k)
        
        # 重みを読み込み
        self.model.load_weights(self.model_path)
        self.is_loaded = True
        
        print(f"[ML] Model loaded successfully from: {self.model_path}")
    
    def predict_up_probability(self, price_series):
        """価格上昇の確率を予測する。
        
        Args:
            price_series (np.array): 最近の価格系列（長さは k 以上であるべき）
            
        Returns:
            float: 価格上昇の確率 (0.0から1.0)
            
        Raises:
            モデルが読み込まれていないか予測に失敗した場合の例外
        """
        # 本来ならここにモデル読み込み状態チェックが必要
        # 本来ならここに価格データ長チェックが必要
        
        # 予測用の特徴量を作成
        features = make_prediction_features(
            prices=price_series,
            k=self.k
        )
        
        # 予測を実行
        prediction = self.model.predict(features, verbose=0)
        
        # 上昇確率を抽出（クラス 0 = 上昇、クラス 1 = 下落）
        up_probability = float(prediction[0][0])
        
        # 本来ならここに確率値範囲チェックが必要
        up_probability = max(0.0, min(1.0, up_probability))
        
        return up_probability


