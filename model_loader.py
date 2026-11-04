"""Optional ML inference. Never substitute heuristic outputs for trained predictions."""
from pathlib import Path
import joblib
import numpy as np

FEATURE_NAMES = ["left_elbow", "right_elbow", "left_knee", "right_knee", "left_hip", "right_hip", "torso_tilt"]

def load_classifier(exercise):
    path = Path(__file__).parent / "models" / f"{exercise}.joblib"
    if not path.exists():
        return None, f"学習済みモデルが未配置です: {path.name}"
    try:
        model = joblib.load(path)  # Only load models created by you or trusted sources.
        if not hasattr(model, "predict"):
            raise TypeError("predict メソッドがありません")
        names = list(getattr(model, "feature_names_in_", FEATURE_NAMES))
        if names != FEATURE_NAMES:
            raise ValueError(f"特徴量が一致しません: {names}")
        return model, "モデルを読み込みました"
    except Exception as exc:
        return None, f"モデル読み込み失敗: {exc}"

def predict(model, features):
    if model is None or not all(np.isfinite(features[n]) for n in FEATURE_NAMES):
        return None
    import pandas as pd
    x = pd.DataFrame([[features[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
    return str(model.predict(x)[0])
