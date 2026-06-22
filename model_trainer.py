"""
Model Trainer - XGBoost + Random Forest + GB Ensemble
Fixes:
  - Self-contained 5y data fetch (no import from data_fetcher)
  - Minimum rows lowered to 60 (fallback mode)
  - RobustScaler, price-level error, clipped targets
  - 30+ engineered features with lag values
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ─── Feature columns ──────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    'sma_20_ratio', 'sma_50_ratio', 'ema_9_ratio', 'ema_21_ratio', 'ema_50_ratio',
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
    'rsi', 'rsi_lag1', 'rsi_lag3',
    'stoch_k', 'stoch_d',
    'roc', 'roc_10',
    'macd_norm', 'macd_hist_norm', 'macd_signal_norm',
    'bb_width', 'atr_norm', 'high_low_range',
    'volume_ratio', 'volume_ratio_lag1',
    'close_lag1_ratio', 'close_lag3_ratio', 'close_lag5_ratio',
    'rolling_std_10', 'rolling_std_20',
    'day_of_week', 'month', 'week_of_year',
]


def _safe_ratio(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    result = (a / b.replace(0, np.nan)) - 1
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Self-contained indicator calculation — no external imports needed."""
    d = df.copy()
    c = d['close']
    h = d['high']
    lo = d['low']
    v = d['volume']

    # MAs
    d['sma_20']  = c.rolling(20).mean()
    d['sma_50']  = c.rolling(50).mean()
    d['sma_200'] = c.rolling(200).mean()
    d['ema_9']   = c.ewm(span=9,  adjust=False).mean()
    d['ema_21']  = c.ewm(span=21, adjust=False).mean()
    d['ema_50']  = c.ewm(span=50, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d['macd']        = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist']   = d['macd'] - d['macd_signal']

    # Bollinger Bands
    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std()
    d['bb_upper'] = bb_mid + 2 * bb_std
    d['bb_lower'] = bb_mid - 2 * bb_std
    d['bb_mid']   = bb_mid
    d['bb_width'] = (d['bb_upper'] - d['bb_lower']) / bb_mid.replace(0, np.nan)

    # ATR
    tr = pd.concat([h - lo,
                    (h - c.shift()).abs(),
                    (lo - c.shift()).abs()], axis=1).max(axis=1)
    d['atr'] = tr.rolling(14).mean()

    # Stochastic
    low14  = lo.rolling(14).min()
    high14 = h.rolling(14).max()
    d['stoch_k'] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    d['stoch_d'] = d['stoch_k'].rolling(3).mean()

    # Volume MA
    d['vol_sma_20']   = v.rolling(20).mean()
    d['volume_ratio'] = v / d['vol_sma_20'].replace(0, np.nan)

    # Momentum / ROC
    d['momentum'] = c - c.shift(10)
    d['roc']      = c.pct_change(10) * 100

    return d


def _fetch_5y(symbol: str) -> pd.DataFrame:
    """Fetch 5 years of daily data with indicators — fully self-contained."""
    try:
        import yfinance as yf
        raw = yf.Ticker(symbol).history(period='5y', interval='1d')
        if raw.empty:
            return pd.DataFrame()
        raw.columns = [col.lower() for col in raw.columns]
        raw.index   = pd.to_datetime(raw.index)
        raw = raw.dropna()
        return _add_indicators(raw)
    except Exception:
        return pd.DataFrame()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer 30+ features from OHLCV + indicator columns."""
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    f = df.copy()
    c = f['close']

    # MA ratios
    for col, key in [('sma_20','sma_20_ratio'), ('sma_50','sma_50_ratio'),
                     ('ema_9','ema_9_ratio'), ('ema_21','ema_21_ratio'), ('ema_50','ema_50_ratio')]:
        f[key] = _safe_ratio(c, f[col]) if col in f.columns else 0.0

    # Returns
    f['ret_1d']  = c.pct_change(1)
    f['ret_3d']  = c.pct_change(3)
    f['ret_5d']  = c.pct_change(5)
    f['ret_10d'] = c.pct_change(10)
    f['ret_20d'] = c.pct_change(20)

    # RSI lags
    if 'rsi' not in f.columns: f['rsi'] = 50.0
    f['rsi_lag1'] = f['rsi'].shift(1)
    f['rsi_lag3'] = f['rsi'].shift(3)

    # Stochastic
    if 'stoch_k' not in f.columns: f['stoch_k'] = 50.0
    if 'stoch_d' not in f.columns: f['stoch_d'] = 50.0

    # ROC
    if 'roc' not in f.columns: f['roc'] = c.pct_change(10) * 100
    f['roc_10'] = c.pct_change(10) * 100

    # MACD normalised
    if 'macd' in f.columns:
        f['macd_norm']        = _safe_ratio(f['macd'],        c.abs())
        f['macd_hist_norm']   = _safe_ratio(f['macd_hist'],   c.abs()) if 'macd_hist'   in f.columns else 0.0
        f['macd_signal_norm'] = _safe_ratio(f['macd_signal'], c.abs()) if 'macd_signal' in f.columns else 0.0
    else:
        f['macd_norm'] = f['macd_hist_norm'] = f['macd_signal_norm'] = 0.0

    # Volatility
    if 'bb_width' not in f.columns: f['bb_width'] = 0.0
    f['atr_norm']       = _safe_ratio(f['atr'], c.abs()) if 'atr' in f.columns else 0.0
    f['high_low_range'] = (f['high'] - f['low']) / c.replace(0, np.nan)

    # Volume
    if 'volume_ratio' not in f.columns:
        vol_ma = f['volume'].rolling(20).mean()
        f['volume_ratio'] = (f['volume'] / vol_ma.replace(0, np.nan)).fillna(1.0)
    f['volume_ratio_lag1'] = f['volume_ratio'].shift(1)

    # Lagged close ratios
    f['close_lag1_ratio'] = _safe_ratio(c, c.shift(1))
    f['close_lag3_ratio'] = _safe_ratio(c, c.shift(3))
    f['close_lag5_ratio'] = _safe_ratio(c, c.shift(5))

    # Rolling std
    f['rolling_std_10'] = c.rolling(10).std() / c.replace(0, np.nan)
    f['rolling_std_20'] = c.rolling(20).std() / c.replace(0, np.nan)

    # Time
    f['day_of_week']  = f.index.dayofweek
    f['month']        = f.index.month
    f['week_of_year'] = f.index.isocalendar().week.astype(int)

    return f


def _price_level_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    abs_err = np.abs(y_true - y_pred)
    denom   = np.abs(1.0 + y_true) + 1e-6
    return float(np.mean(abs_err / denom) * 100)


def build_ensemble_model():
    estimators = []

    if XGBOOST_AVAILABLE:
        estimators.append(('xgboost', XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
            random_state=42, verbosity=0, n_jobs=-1,
        )))

    estimators.append(('random_forest', RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1,
    )))

    estimators.append(('gradient_boost', GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.75, min_samples_split=10, min_samples_leaf=5,
        max_features='sqrt', random_state=42,
    )))

    return VotingRegressor(estimators=estimators) if len(estimators) > 1 else estimators[0][1]


def train_and_predict(df: pd.DataFrame,
                      prediction_days: int = 5,
                      symbol: str = None) -> dict:
    """
    Train ensemble + generate price prediction.
    1. Always tries to fetch 5y data via symbol for 500-1250 rows.
    2. Falls back to whatever df was passed (even 60 rows).
    3. Minimum requirement is just 50 rows.
    """

    # ── Step 1: try to get 5y data ────────────────────────────────────────────
    working = df.copy()

    if symbol:
        extended = _fetch_5y(symbol)
        if not extended.empty and len(extended) > len(working):
            working = extended

    # ── Step 2: add indicators if missing (fallback for short df) ────────────
    if 'rsi' not in working.columns:
        working = _add_indicators(working)

    # ── Step 3: minimum check (very lenient — 50 rows) ───────────────────────
    if working.empty or len(working) < 50:
        return {
            "error": f"Insufficient data ({len(working)} rows). Need at least 50. "
                     f"Please select a longer period (1y or 2y) in the sidebar.",
            "predicted_price": None,
            "predicted_change_pct": None,
        }

    try:
        feat = prepare_features(working)
        if feat.empty:
            return {"error": "Feature preparation failed.", "predicted_price": None}

        avail = [col for col in FEATURE_COLUMNS if col in feat.columns]
        if len(avail) < 5:
            return {"error": f"Only {len(avail)} features available.", "predicted_price": None}

        # Target: clipped fractional return
        raw_target     = feat['close'].shift(-prediction_days) / feat['close'] - 1
        feat['target'] = raw_target.clip(-0.15, 0.15)

        model_df = feat[avail + ['target', 'close']].dropna()

        # Minimum 40 clean rows
        if len(model_df) < 40:
            return {
                "error": f"Only {len(model_df)} clean rows. Select 1y or 2y period in sidebar.",
                "predicted_price": None
            }

        X = np.nan_to_num(model_df[avail].values, nan=0.0, posinf=0.0, neginf=0.0)
        y = model_df['target'].values

        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Time-series CV (scale splits to data size)
        n_splits   = min(5, max(2, len(model_df) // 80))
        tscv       = TimeSeriesSplit(n_splits=n_splits)
        val_errors = []

        for tr_idx, val_idx in tscv.split(X_scaled):
            if len(tr_idx) < 30:
                continue
            m = build_ensemble_model()
            m.fit(X_scaled[tr_idx], y[tr_idx])
            preds = m.predict(X_scaled[val_idx])
            val_errors.append(_price_level_error(y[val_idx], preds))

        # Final model on all data
        final = build_ensemble_model()
        final.fit(X_scaled, y)

        # Predict current
        last_x          = np.nan_to_num(feat[avail].iloc[-1:].values, nan=0.0, posinf=0.0, neginf=0.0)
        pred_return     = float(final.predict(scaler.transform(last_x))[0])
        current_price   = float(df['close'].iloc[-1])
        predicted_price = current_price * (1 + pred_return)

        # Multi-horizon forecasts
        predictions = {}
        for h in [1, 3, 5, 10, 20]:
            try:
                fh = prepare_features(working)
                fh['target'] = (fh['close'].shift(-h) / fh['close'] - 1).clip(-0.15, 0.15)
                mh = fh[avail + ['target']].dropna()
                if len(mh) >= 40:
                    Xh   = np.nan_to_num(mh[avail].values, nan=0.0, posinf=0.0, neginf=0.0)
                    sh   = RobustScaler()
                    Xh_s = sh.fit_transform(Xh)
                    mhm  = build_ensemble_model()
                    mhm.fit(Xh_s, mh['target'].values)
                    lh   = np.nan_to_num(fh[avail].iloc[-1:].values, nan=0.0, posinf=0.0, neginf=0.0)
                    pr   = float(mhm.predict(sh.transform(lh))[0])
                    predictions[h] = {"return_pct": round(pr * 100, 2),
                                      "price":      round(current_price * (1 + pr), 2)}
                else:
                    predictions[h] = {"return_pct": round(pred_return * 100, 2),
                                      "price":      round(predicted_price, 2)}
            except Exception:
                predictions[h] = {"return_pct": round(pred_return * 100, 2),
                                  "price":      round(predicted_price, 2)}

        # Confidence score
        avg_err    = float(np.mean(val_errors)) if val_errors else 5.0
        confidence = max(35.0, min(92.0, 95.0 - avg_err * 3.5))

        # Feature importance
        feature_importance = {}
        try:
            if hasattr(final, 'estimators_'):
                imps = np.zeros(len(avail))
                cnt  = 0
                for _, est in final.estimators_:
                    if hasattr(est, 'feature_importances_'):
                        imps += est.feature_importances_
                        cnt  += 1
                if cnt:
                    imps /= cnt
                    feature_importance = dict(
                        sorted(zip(avail, imps.tolist()),
                               key=lambda x: x[1], reverse=True)[:10]
                    )
            elif hasattr(final, 'feature_importances_'):
                feature_importance = dict(
                    sorted(zip(avail, final.feature_importances_.tolist()),
                           key=lambda x: x[1], reverse=True)[:10]
                )
        except Exception:
            pass

        return {
            "predicted_price":         round(predicted_price, 2),
            "predicted_change_pct":    round(pred_return * 100, 2),
            "current_price":           round(current_price, 2),
            "confidence":              round(confidence, 1),
            "prediction_horizon_days": prediction_days,
            "predictions":             predictions,
            "val_error_pct":           round(avg_err, 2),
            "feature_importance":      feature_importance,
            "models_used":             (["XGBoost", "Random Forest", "Gradient Boosting"]
                                        if XGBOOST_AVAILABLE else
                                        ["Random Forest", "Gradient Boosting"]),
            "training_samples":        len(X_scaled),
            "error":                   None,
        }

    except Exception as e:
        return {
            "error": str(e),
            "predicted_price": None,
            "predicted_change_pct": None,
        }
    
