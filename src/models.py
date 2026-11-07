from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from . import config as cfg
from .preprocessing import to_z
from .utils import log

def fit_models(sample, cols, mean, std):
    Z = to_z(sample, cols, mean, std)
    d = Z.shape[1]
    iso = IsolationForest(n_estimators=200, max_samples=256, random_state=cfg.SEED, n_jobs=-1).fit(Z)
    iso.set_params(n_jobs=1)
    h1, h2, b = max(d, 16), max(d // 2, 8), max(d // 4, 4)
    ae = MLPRegressor(hidden_layer_sizes=(h1, h2, b, h2, h1), activation="relu", solver="adam", alpha=1e-5,
                      batch_size=512, max_iter=80, early_stopping=True, n_iter_no_change=8, random_state=cfg.SEED).fit(Z, Z)
    log(f"Đã huấn luyện IF + AE trên mẫu {Z.shape[0]:,} x {d}; AE {(d, h1, h2, b, h2, h1, d)}")
    return iso, ae
