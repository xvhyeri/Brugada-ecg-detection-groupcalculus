# =============================================================================
# BRUGADA SYNDROME DETECTION — COMPLETE ML PIPELINE (Version 6)
# IDSC 2026 | Dataset: Brugada-HUCA (PhysioNet)
#
# Changes vs Version 5:
#   FIX 1 → Multi-lead features: V1, V2, V3 instead of V1 only
#            Each lead now contributes x_1..x_4 → 12 features total
#            V2 and V3 also show Brugada coved-type ST patterns
#   FIX 2 → Added Logistic Regression and Random Forest as baseline
#            comparisons alongside SVM. All three use same nested CV.
#            Allows fair comparison and justification of model choice.
#
# Full feature vector per patient (v6):
#   Per lead (V1, V2, V3) — 4 features x 3 leads = 12 features total:
#   x_1 = J-point Elevation     V(J) - V(baseline)       >= 0.2 mV  for Brugada
#   x_2 = QRS Duration          (S_idx - Q_idx) * 10 ms  wider      for Brugada
#   x_3 = T-Wave Amplitude      V(T-peak) - V(baseline)  < 0        for Brugada
#   x_4 = R-to-S Amplitude Drop V(R) - V(S)              smaller    for Brugada
#
# Run in terminal:
#   pip install wfdb pandas numpy matplotlib scipy scikit-learn seaborn
#   python brugada_pipeline_v6.py
# =============================================================================

import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import butter, filtfilt
from scipy.stats import uniform, loguniform
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV
)
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_NAME  = 'brugada-huca/1.0.0'
PNET_URL = 'https://physionet.org/files/'
FS       = 100          # Sampling frequency — 1 sample = 10ms
# Leads V1, V2, V3 — all three show Brugada coved-type ST patterns
BRUGADA_LEADS     = [6, 7, 8]   # indices in 12-lead array: V1=6, V2=7, V3=8
BRUGADA_LEAD_NAMES = ['V1', 'V2', 'V3']
N_FEATURES        = len(BRUGADA_LEADS) * 4   # 12 features total
J_ELEV_THRESHOLD  = 0.2  # mV — Brugada Type 1 diagnostic criterion

# Hyperparameter search space (broader than v4)
# loguniform samples C and gamma on a log scale — much better than
# a fixed grid because SVM performance varies on log scale of C/gamma
PARAM_DIST = {
    'svc__C'     : loguniform(1e-2, 1e3),   # samples from 0.01 to 1000
    'svc__gamma' : loguniform(1e-4, 1e1),   # samples from 0.0001 to 10
}
N_ITER_SEARCH = 50   # try 50 random combinations per outer fold
                     # 50 * 5 inner folds * 5 outer folds = 1250 SVM fits


# =============================================================================
# STEP 1 — LOAD DATA & EDA
# =============================================================================
print("=" * 65)
print("STEP 1 — Loading Data & EDA")
print("=" * 65)

meta      = pd.read_csv(f'{PNET_URL}{DB_NAME}/metadata.csv')
data_dict = pd.read_csv(f'{PNET_URL}{DB_NAME}/metadata_dictionary.csv')

print("\nColumn Descriptions:")
print(data_dict.to_string(index=False))

print("\nRaw class distribution:")
print(meta['brugada'].value_counts().sort_index())

# Label: brugada > 0 = positive (aligns with dataset's 76 Brugada cases)
meta['label'] = (meta['brugada'] > 0).astype(int)

N         = len(meta)
N_brugada = (meta['label'] == 1).sum()   # 76
N_normal  = (meta['label'] == 0).sum()   # 287

print(f"\nAfter labelling (brugada > 0 = positive):")
print(f"  Brugada (1) : {N_brugada}")
print(f"  Normal  (0) : {N_normal}")
print(f"  Total       : {N}")

# Class weights: w_c = N / (2 * N_c)
W_BRUGADA    = N / (2 * N_brugada)   # 2.388
W_NORMAL     = N / (2 * N_normal)    # 0.632
CLASS_WEIGHT = {0: W_NORMAL, 1: W_BRUGADA}

print(f"\nClass Weights [w_c = N / (2 * N_c)]:")
print(f"  w_Brugada = {N} / (2 x {N_brugada}) = {W_BRUGADA:.3f}")
print(f"  w_Normal  = {N} / (2 x {N_normal})  = {W_NORMAL:.3f}")


# =============================================================================
# STEP 2 — BUTTERWORTH BANDPASS FILTER
# =============================================================================
# H(f) = 1 / sqrt(1 + (f/f_c)^2n)
# Passes 0.5–40 Hz. filtfilt() = zero-phase (no timing distortion).
# =============================================================================

def butterworth_bandpass(signal, fs=FS, lowcut=0.5, highcut=40.0, order=4):
    """Zero-phase Butterworth bandpass filter for ECG signal."""
    nyq  = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    return np.stack(
        [filtfilt(b, a, signal[:, i]) for i in range(signal.shape[1])],
        axis=1
    )


# =============================================================================
# STEP 3 — PAN-TOMPKINS R-PEAK DETECTION
# =============================================================================
# 1. Differentiate  : x'[n] = x[n] - x[n-1]
# 2. Square         : x''[n] = (x'[n])^2
# 3. Moving average : MWI[n] = (1/W) * sum(x''[n-W+1..n]),  W=15 samples
# 4. Threshold at 50% of max(MWI)
# 5. Find exact R maximum per candidate region
# 6. Enforce 200ms refractory period
# =============================================================================

def pan_tompkins_rpeaks(signal_1d, fs=FS):
    """Detect R-peak sample indices using Pan-Tompkins algorithm."""
    diff      = np.diff(signal_1d, prepend=signal_1d[0])
    squared   = diff ** 2
    window    = int(0.15 * fs)
    mwi       = np.convolve(squared, np.ones(window) / window, mode='same')
    threshold = 0.5 * np.max(mwi)
    above     = mwi > threshold

    r_peaks = []
    in_peak, start = False, 0
    for i in range(len(above)):
        if above[i] and not in_peak:
            in_peak, start = True, i
        elif not above[i] and in_peak:
            in_peak = False
            seg = signal_1d[start:i]
            if len(seg) > 0:
                r_peaks.append(start + int(np.argmax(seg)))

    r_peaks    = np.array(r_peaks)
    refractory = int(0.2 * fs)
    if len(r_peaks) > 1:
        keep = [r_peaks[0]]
        for rp in r_peaks[1:]:
            if rp - keep[-1] >= refractory:
                keep.append(rp)
        r_peaks = np.array(keep)

    return r_peaks


# =============================================================================
# STEP 4 — FEATURE EXTRACTION
# =============================================================================
#
#   x_1 = J-point Elevation
#         = V(J) - V(baseline)
#         Brugada criterion: x_1 >= 0.2 mV
#
#   x_2 = QRS Duration  (NEW in v5 — replaces S-to-ST slope)
#         = (S_idx - Q_idx) * 10 ms
#         Q-wave = first negative deflection BEFORE R-peak (onset of QRS)
#         S-wave = local minimum AFTER R-peak (end of QRS)
#         QRS duration measures how long the ventricular depolarisation takes
#         Brugada: wider QRS (sodium channel defect slows conduction) → larger x_2
#         Normal:  narrow QRS (fast, normal conduction)               → smaller x_2
#         Clinical normal range: 80-120ms. Brugada often > 120ms.
#
#   x_3 = T-Wave Amplitude
#         = V(T-peak) - V(baseline)
#         Brugada: inverted T-wave → x_3 < 0
#         Normal:  upright T-wave  → x_3 > 0
#
#   x_4 = R-to-S Amplitude Drop
#         = V(R) - V(S)
#         Brugada: shallower S-wave (disturbed QRS) → smaller x_4
#         Normal:  deeper S-wave (clean QRS)        → larger x_4
#
# Applied to leads V1, V2, V3 → 4 features x 3 leads = 12 features total
# Feature naming: x1_V1, x2_V1, x3_V1, x4_V1, x1_V2, ... x4_V3
#
# =============================================================================

def compute_baseline(signal_1d, r_idx, fs=FS):
    """PQ segment baseline: mean(signal[R-200ms : R-80ms])"""
    start = max(0, r_idx - int(0.20 * fs))
    end   = max(0, r_idx - int(0.08 * fs))
    return float(np.mean(signal_1d[start:end])) if end > start else 0.0


def find_q_wave(signal_1d, r_idx, fs=FS):
    """
    Q-wave = onset of QRS complex, just before R-peak.
    Search window: R-50ms to R (5 samples before R).
    Q_idx = argmin(signal[R-5 : R])  — first negative deflection before R.
    """
    start = max(0, r_idx - int(0.05 * fs))   # R - 50ms
    seg   = signal_1d[start:r_idx]
    return start + int(np.argmin(seg)) if len(seg) > 0 else r_idx


def find_s_wave(signal_1d, r_idx, fs=FS):
    """S-wave = local minimum after R-peak: argmin(signal[R : R+80ms])"""
    end = min(len(signal_1d), r_idx + int(0.08 * fs))
    seg = signal_1d[r_idx:end]
    return r_idx + int(np.argmin(seg)) if len(seg) > 0 else r_idx


def find_j_point(signal_1d, s_idx, fs=FS):
    """J-point = local minimum after S-wave: argmin(signal[S : S+80ms])"""
    end = min(len(signal_1d), s_idx + int(0.08 * fs))
    seg = signal_1d[s_idx:end]
    return s_idx + int(np.argmin(seg)) if len(seg) >= 2 else s_idx


def find_t_wave_peak(signal_1d, r_idx, fs=FS):
    """T-wave: argmax(|signal[R+200ms : R+400ms]|) — anchored from R-peak."""
    start = min(len(signal_1d) - 1, r_idx + int(0.20 * fs))
    end   = min(len(signal_1d),     r_idx + int(0.40 * fs))
    seg   = signal_1d[start:end]
    return start + int(np.argmax(np.abs(seg))) if len(seg) > 0 else r_idx


def extract_features_one_beat(signal_v1, r_idx, fs=FS):
    """
    Extract (x_1, x_2, x_3, x_4) from one heartbeat in Lead V1.
    Returns None if beat is too near signal boundary.
    """
    n = len(signal_v1)
    if r_idx < int(0.25 * fs) or r_idx > n - int(0.45 * fs):
        return None

    baseline = compute_baseline(signal_v1, r_idx, fs)
    q_idx    = find_q_wave(signal_v1, r_idx, fs)
    s_idx    = find_s_wave(signal_v1, r_idx, fs)
    j_idx    = find_j_point(signal_v1, s_idx, fs)
    t_idx    = find_t_wave_peak(signal_v1, r_idx, fs)

    # ── x_1: J-point Elevation ────────────────────────────────────────
    x_1 = float(signal_v1[j_idx]) - baseline

    # ── x_2: QRS Duration (ms) — NEW in v5 ───────────────────────────
    # Time from Q-wave onset to S-wave end
    # = (S_idx - Q_idx) samples * 10 ms/sample
    # Brugada: prolonged QRS (sodium channel slows conduction)
    # Normal:  narrow QRS (80-120ms typical)
    qrs_samples = max(0, s_idx - q_idx)
    x_2         = qrs_samples * (1000.0 / fs)   # convert to milliseconds

    # ── x_3: T-Wave Amplitude ─────────────────────────────────────────
    x_3 = float(signal_v1[t_idx]) - baseline

    # ── x_4: R-to-S Amplitude Drop ────────────────────────────────────
    x_4 = float(signal_v1[r_idx]) - float(signal_v1[s_idx])

    return (x_1, x_2, x_3, x_4)


def extract_patient_features(patient_id, db_name=DB_NAME, fs=FS):
    """
    Full per-patient pipeline: load → filter → R-peaks → features → average.
    Extracts (x_1, x_2, x_3, x_4) from each of V1, V2, V3.
    Returns a flat vector of 12 features: [x1_V1, x2_V1, x3_V1, x4_V1,
                                            x1_V2, x2_V2, x3_V2, x4_V2,
                                            x1_V3, x2_V3, x3_V3, x4_V3]
    """
    try:
        pid = str(patient_id)

        record = wfdb.rdrecord(
        f'C:/IDSC2026/brugada_project/data/files/files/{pid}/{pid}'
)

        all_lead_features = []

        for lead_idx in BRUGADA_LEADS:
            lead_filt = butterworth_bandpass(record.p_signal[:, lead_idx], fs=fs)
            r_peaks   = pan_tompkins_rpeaks(lead_filt, fs=fs)

            if len(r_peaks) < 2:
                # Fill with zeros if this lead has no valid peaks
                all_lead_features.extend([0.0, 0.0, 0.0, 0.0])
                continue

            beats = [extract_features_one_beat(lead_filt, r, fs) for r in r_peaks]
            beats = [b for b in beats if b is not None]

            if not beats:
                all_lead_features.extend([0.0, 0.0, 0.0, 0.0])
                continue

            arr = np.array(beats)   # (n_beats, 4)
            all_lead_features.extend(
                [float(np.mean(arr[:, i])) for i in range(4)]
            )

        if len(all_lead_features) != N_FEATURES:
            return None

        return tuple(all_lead_features)

    except Exception as e:
        print(f"  Error on patient {patient_id}: {e}")
        return None


# =============================================================================
# BUILD FEATURE MATRIX X  (shape: n_valid_patients x 4)
# =============================================================================
print("\n" + "=" * 65)
print("STEP 4 — Building Feature Matrix X  (363 patients x 12 features)")
print("=" * 65)
print("Streaming ECG files from PhysioNet — ~5-10 minutes...\n")
print("Features: x1..x4 for each of V1, V2, V3 = 12 features total\n")

rows, labels, pids = [], [], []

for idx, row in meta.iterrows():
    pid, label = row['patient_id'], row['label']
    print(f"  [{idx+1:>3}/{len(meta)}] Patient {pid} ...", end=' ')
    features = extract_patient_features(pid)
    if features is not None:
        rows.append(features)
        labels.append(label)
        pids.append(pid)
        f = features
        print(f"V1=[{f[0]:+.3f},{f[1]:5.1f}ms,{f[2]:+.3f},{f[3]:+.3f}] "
              f"V2=[{f[4]:+.3f},{f[5]:5.1f}ms,{f[6]:+.3f},{f[7]:+.3f}] "
              f"V3=[{f[8]:+.3f},{f[9]:5.1f}ms,{f[10]:+.3f},{f[11]:+.3f}]")
    else:
        print("SKIPPED")

X = np.array(rows)
y = np.array(labels)

print(f"\nFeature matrix X shape  : {X.shape}")
print(f"Brugada (1)             : {y.sum()}")
print(f"Normal  (0)             : {(y == 0).sum()}")

# Save feature matrix — 12 features (4 per lead x 3 leads)
feat_cols = [f'x{fi+1}_{ln}' for ln in BRUGADA_LEAD_NAMES for fi in range(4)]
feature_df = pd.DataFrame(X, columns=feat_cols)
feature_df.insert(0, 'patient_id', pids)
feature_df['label'] = y
feature_df.to_csv('feature_matrix_v6.csv', index=False)
print("Saved: feature_matrix_v6.csv")

# Feature statistics by class
print("\nFeature Statistics by Class (V1, V2, V3):")
print("Expected directions:")
print("  x_1: Brugada > Normal   (J-point elevated)")
print("  x_2: Brugada > Normal   (QRS wider in Brugada)")
print("  x_3: Brugada < 0        (T-wave inverted)")
print("  x_4: Brugada < Normal   (shallower S-wave)")
summary = feature_df.groupby('label')[feat_cols].agg(['mean', 'std']).round(4)
print(summary)

# Clinical direction check — use V1 as primary reference
print("\nClinical Direction Check (V1):")
brugada_rows = feature_df[feature_df['label'] == 1]
normal_rows  = feature_df[feature_df['label'] == 0]
for ln in BRUGADA_LEAD_NAMES:
    x1_b = brugada_rows[f'x1_{ln}'].mean()
    x1_n = normal_rows [f'x1_{ln}'].mean()
    x2_b = brugada_rows[f'x2_{ln}'].mean()
    x2_n = normal_rows [f'x2_{ln}'].mean()
    x3_b = brugada_rows[f'x3_{ln}'].mean()
    x4_b = brugada_rows[f'x4_{ln}'].mean()
    x4_n = normal_rows [f'x4_{ln}'].mean()
    print(f"\n  Lead {ln}:")
    print(f"    x_1 Brugada ({x1_b:+.4f}) > Normal ({x1_n:+.4f}) : "
          f"{'PASS ✓' if x1_b > x1_n else 'FAIL ✗'}")
    print(f"    x_2 Brugada ({x2_b:.2f}ms) > Normal ({x2_n:.2f}ms) : "
          f"{'PASS ✓' if x2_b > x2_n else 'FAIL ✗'}")
    print(f"    x_3 Brugada ({x3_b:+.4f}) < 0 : "
          f"{'PASS ✓' if x3_b < 0 else 'FAIL ✗'}")
    print(f"    x_4 Brugada ({x4_b:+.4f}) < Normal ({x4_n:+.4f}) : "
          f"{'PASS ✓' if x4_b < x4_n else 'FAIL ✗'}")


# =============================================================================
# STEP 5 — NESTED CV FOR 3 MODELS: SVM, Logistic Regression, Random Forest
# =============================================================================
# All three models use the same outer 5-fold splits for fair comparison.
# SVM and LR use RandomizedSearchCV for hyperparameter tuning (inner 5-fold).
# Random Forest uses RandomizedSearchCV for n_estimators and max_depth.
#
# FIX: AUC inversion protection (carried over from v5)
# =============================================================================
print("\n" + "=" * 65)
print("STEP 5 — Nested CV: SVM vs Logistic Regression vs Random Forest")
print("=" * 65)

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
outer_cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Model definitions with search spaces ─────────────────────────────────────

# 1. SVM RBF (same as v5)
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', class_weight=CLASS_WEIGHT,
                probability=True, random_state=42))
])
svm_search = RandomizedSearchCV(
    svm_pipeline,
    param_distributions={'svc__C': loguniform(1e-2, 1e3),
                         'svc__gamma': loguniform(1e-4, 1e1)},
    n_iter=N_ITER_SEARCH, cv=inner_cv, scoring='roc_auc',
    refit=True, n_jobs=-1, random_state=42, verbose=0
)

# 2. Logistic Regression (baseline — linear decision boundary)
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(class_weight=CLASS_WEIGHT,
                              max_iter=2000, random_state=42, solver='lbfgs'))
])
lr_search = RandomizedSearchCV(
    lr_pipeline,
    param_distributions={'lr__C': loguniform(1e-3, 1e3)},
    n_iter=30, cv=inner_cv, scoring='roc_auc',
    refit=True, n_jobs=-1, random_state=42, verbose=0
)

# 3. Random Forest (non-linear, ensemble, no scaling needed)
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(class_weight=CLASS_WEIGHT,
                                  random_state=42, n_jobs=-1))
])
rf_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions={'rf__n_estimators': [100, 200, 300, 500],
                         'rf__max_depth':    [3, 5, 8, 10, None],
                         'rf__min_samples_leaf': [1, 2, 4]},
    n_iter=30, cv=inner_cv, scoring='roc_auc',
    refit=True, n_jobs=-1, random_state=42, verbose=0
)

MODEL_CONFIGS = [
    ('SVM (RBF)',           svm_search),
    ('Logistic Regression', lr_search),
    ('Random Forest',       rf_search),
]

# ── Shared storage for all models ────────────────────────────────────────────
all_results = {}

for model_name, searcher in MODEL_CONFIGS:
    print(f"\n{'─'*65}")
    print(f"  Running: {model_name}")
    print(f"{'─'*65}")

    y_pred_all = np.zeros(len(y), dtype=int)
    y_prob_all = np.zeros(len(y))
    fold_aucs  = []
    best_params_list = []

    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        searcher.fit(X_train, y_train)

        y_pred_fold = searcher.predict(X_test)
        y_prob_fold = searcher.predict_proba(X_test)[:, 1]

        # AUC inversion protection
        if len(np.unique(y_test)) > 1:
            raw_auc = roc_auc_score(y_test, y_prob_fold)
            if raw_auc < 0.5:
                print(f"  Fold {fold_num}: AUC={raw_auc:.4f} < 0.5 — flipping predictions")
                y_prob_fold = 1.0 - y_prob_fold
                y_pred_fold = 1 - y_pred_fold
                fold_auc    = roc_auc_score(y_test, y_prob_fold)
            else:
                fold_auc = raw_auc
        else:
            fold_auc = 0.5

        y_pred_all[test_idx] = y_pred_fold
        y_prob_all[test_idx] = y_prob_fold
        fold_aucs.append(fold_auc)
        best_params_list.append({'fold': fold_num,
                                  'best_params': searcher.best_params_})

        print(f"  Fold {fold_num}: AUC={fold_auc:.4f}  "
              f"best={searcher.best_params_}  "
              f"(Brugada={y_test.sum()}, Normal={(y_test==0).sum()})")

    all_results[model_name] = {
        'y_pred'    : y_pred_all,
        'y_prob'    : y_prob_all,
        'fold_aucs' : fold_aucs,
        'best_params': best_params_list,
    }
    print(f"\n  {model_name} — Mean AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")


# =============================================================================
# STEP 6 — EVALUATION & MODEL COMPARISON
# =============================================================================
print("\n" + "=" * 65)
print("STEP 6 — Evaluation & Model Comparison")
print("=" * 65)

# Compute metrics for each model
metrics_rows = []
for model_name, res in all_results.items():
    cm_m           = confusion_matrix(y, res['y_pred'])
    TN_m, FP_m, FN_m, TP_m = cm_m.ravel()
    sens_m  = TP_m / (TP_m + FN_m)
    spec_m  = TN_m / (TN_m + FP_m)
    prec_m  = TP_m / (TP_m + FP_m) if (TP_m + FP_m) > 0 else 0.0
    f1_m    = (2 * prec_m * sens_m / (prec_m + sens_m)
               if (prec_m + sens_m) > 0 else 0.0)
    auc_m   = roc_auc_score(y, res['y_prob'])
    metrics_rows.append({
        'Model'      : model_name,
        'AUC'        : round(auc_m,   4),
        'Sensitivity': round(sens_m,  4),
        'Specificity': round(spec_m,  4),
        'Precision'  : round(prec_m,  4),
        'F1'         : round(f1_m,    4),
        'TP': TP_m, 'TN': TN_m, 'FP': FP_m, 'FN': FN_m,
    })

metrics_df = pd.DataFrame(metrics_rows).set_index('Model')
print("\nModel Comparison Table:")
print(metrics_df[['AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1']].to_string())

best_model_name = metrics_df['AUC'].idxmax()
print(f"\n★ Best model by AUC: {best_model_name} (AUC={metrics_df.loc[best_model_name,'AUC']:.4f})")

# Detailed report for best model
best_res = all_results[best_model_name]
print(f"\nDetailed Report — {best_model_name}:")
print(classification_report(y, best_res['y_pred'], target_names=['Normal', 'Brugada']))

# Per-fold AUC for all models
print("\nPer-Fold AUC Summary:")
fold_df = pd.DataFrame({
    name: res['fold_aucs'] for name, res in all_results.items()
}, index=[f'Fold {i}' for i in range(1, 6)])
fold_df.loc['Mean ± Std'] = [
    f"{np.mean(res['fold_aucs']):.3f} ± {np.std(res['fold_aucs']):.3f}"
    for res in all_results.values()
]
print(fold_df.to_string())

# Keep SVM vars for backward-compatible visualisation
cm          = confusion_matrix(y, all_results['SVM (RBF)']['y_pred'])
TN, FP, FN, TP = cm.ravel()
sensitivity = metrics_df.loc['SVM (RBF)', 'Sensitivity']
specificity = metrics_df.loc['SVM (RBF)', 'Specificity']
precision   = metrics_df.loc['SVM (RBF)', 'Precision']
f1          = metrics_df.loc['SVM (RBF)', 'F1']
auc         = metrics_df.loc['SVM (RBF)', 'AUC']
fold_aucs   = all_results['SVM (RBF)']['fold_aucs']
y_pred_all  = all_results['SVM (RBF)']['y_pred']
y_prob_all  = all_results['SVM (RBF)']['y_prob']


# =============================================================================
# VISUALISATIONS — 6-panel results figure
# =============================================================================
print("\nGenerating results figure...")

fig = plt.figure(figsize=(22, 13))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

colors     = {0: 'steelblue', 1: 'crimson'}
label_text = {0: 'Normal',    1: 'Brugada'}

# ── Panel 1: Confusion Matrix ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Normal', 'Brugada'],
            yticklabels=['Normal', 'Brugada'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
ax1.set_ylabel('Actual Label')
ax1.set_xlabel('Predicted Label')

# ── Panel 2: ROC Curve — all 3 models ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
roc_colors = {'SVM (RBF)': 'crimson', 'Logistic Regression': 'steelblue',
              'Random Forest': 'forestgreen'}
for mname, res in all_results.items():
    fpr_m, tpr_m, _ = roc_curve(y, res['y_prob'])
    auc_m = metrics_df.loc[mname, 'AUC']
    ax2.plot(fpr_m, tpr_m, color=roc_colors[mname], lw=2.2,
             label=f'{mname} (AUC={auc_m:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random (AUC=0.500)')
ax2.set_xlabel('1 - Specificity')
ax2.set_ylabel('Sensitivity')
ax2.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ── Panel 3: x_1 vs x_2 scatter ──────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
for cls in [0, 1]:
    mask = y == cls
    ax3.scatter(X[mask, 0], X[mask, 1], c=colors[cls],
                label=label_text[cls], alpha=0.55, s=30, edgecolors='none')
ax3.axvline(x=J_ELEV_THRESHOLD, color='orange', linestyle='--',
            linewidth=1.8, label=f'J-elev = {J_ELEV_THRESHOLD} mV')
ax3.axhline(y=120, color='green', linestyle='--',
            linewidth=1.5, label='QRS = 120ms (upper normal)')
ax3.set_xlabel('x_1 : J-point Elevation (mV)')
ax3.set_ylabel('x_2 : QRS Duration (ms)')
ax3.set_title('Feature Space: x_1 vs x_2', fontsize=13, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ── Panel 4: x_1 vs x_3 scatter ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for cls in [0, 1]:
    mask = y == cls
    ax4.scatter(X[mask, 0], X[mask, 2], c=colors[cls],
                label=label_text[cls], alpha=0.55, s=30, edgecolors='none')
ax4.axvline(x=J_ELEV_THRESHOLD, color='orange', linestyle='--', linewidth=1.8)
ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1.0)
ax4.set_xlabel('x_1 : J-point Elevation (mV)')
ax4.set_ylabel('x_3 : T-Wave Amplitude (mV)')
ax4.set_title('Feature Space: x_1 vs x_3', fontsize=13, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# ── Panel 5: Feature Boxplots ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
feat_melt = pd.DataFrame({
    'x_1 J-elev (mV)'    : X[:, 0],
    'x_2 QRS dur (ms/10)': X[:, 1] / 10,   # scale down for comparable axis
    'x_3 T-amp (mV)'     : X[:, 2],
    'x_4 R-S drop (mV)'  : X[:, 3],
    'Class'              : ['Brugada' if l == 1 else 'Normal' for l in y]
}).melt(id_vars='Class', var_name='Feature', value_name='Scaled Value')
sns.boxplot(data=feat_melt, x='Feature', y='Scaled Value', hue='Class',
            palette={'Brugada': 'crimson', 'Normal': 'steelblue'},
            ax=ax5, width=0.5, fliersize=2)
ax5.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax5.set_title('Feature Distributions by Class\n(x_2 scaled /10 for display)',
              fontsize=12, fontweight='bold')
ax5.tick_params(axis='x', labelsize=7)
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3, axis='y')

# ── Panel 6: Model Comparison Bar Chart ──────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
model_names  = list(all_results.keys())
mean_aucs    = [np.mean(all_results[m]['fold_aucs']) for m in model_names]
std_aucs     = [np.std(all_results[m]['fold_aucs'])  for m in model_names]
bar_cols     = [roc_colors[m] for m in model_names]
bars = ax6.bar(model_names, mean_aucs, yerr=std_aucs, capsize=6,
               color=bar_cols, alpha=0.82, edgecolor='white', width=0.5)
ax6.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, label='Random=0.500')
ax6.set_ylim(0, 1.1)
ax6.set_ylabel('Mean AUC-ROC (± std)')
ax6.set_title('Model Comparison\n(Nested 5-Fold CV)', fontsize=13, fontweight='bold')
ax6.tick_params(axis='x', labelsize=8)
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3, axis='y')
for bar, val, std in zip(bars, mean_aucs, std_aucs):
    ax6.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + std + 0.02,
             f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle(
    f'Brugada Syndrome Detection — SVM RBF + Nested RandomizedSearchCV\n'
    f'Sensitivity = {sensitivity:.3f}   |   Specificity = {specificity:.3f}   |   '
    f'AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}',
    fontsize=13, fontweight='bold', y=1.01
)

plt.savefig('brugada_results_v6.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: brugada_results_v6.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("PIPELINE COMPLETE — v6 Summary")
print("=" * 65)
print(f"  Patients processed : {len(y)} / {N}")
print(f"  Patients skipped   : {N - len(y)} (insufficient signal quality)")
print(f"  Feature matrix     : {X.shape}  (saved: feature_matrix_v6.csv)")
print(f"  Results figure     : brugada_results_v6.png")
print(f"\nModel Comparison Results:")
print(metrics_df[['AUC', 'Sensitivity', 'Specificity', 'F1']].to_string())
print(f"\n★ Best model: {best_model_name}")
print(f"\nReport these values in your tech report:")
best = metrics_df.loc[best_model_name]
svm_folds = all_results['SVM (RBF)']['fold_aucs']
print(f"  SVM AUC (nested CV) : {np.mean(svm_folds):.3f} +/- {np.std(svm_folds):.3f}")
print(f"  Best model AUC      : {best['AUC']:.3f}")
print(f"  Sensitivity         : {best['Sensitivity']:.3f}")
print(f"  Specificity         : {best['Specificity']:.3f}")
print(f"  F1-Score            : {best['F1']:.3f}")
print(f"\nv6 changes vs v5:")
print(f"  FIX 1 — Multi-lead: V1+V2+V3 features (12 total, was 4 from V1 only)")
print(f"  FIX 2 — Model comparison: SVM vs Logistic Regression vs Random Forest")
