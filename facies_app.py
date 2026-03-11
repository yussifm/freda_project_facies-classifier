"""
Facies Classification from Wireline Logs
=========================================
A fully open-source, free-to-host Streamlit application for automated
lithofacies classification using machine learning.

Deployment options (all free):
  - Streamlit Community Cloud: https://streamlit.io/cloud
  - Hugging Face Spaces:       https://huggingface.co/spaces
  - Render:                    https://render.com  (free tier)

Run locally:
  pip install -r requirements.txt
  streamlit run facies_app.py

Author: [Your Name]
Date:   2025
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import io
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Facies Classifier",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants & Metadata
# ---------------------------------------------------------------------------
FEATURES = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]
WELL_LOGS = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
IMPUTER_FEATURES = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "NM_M", "RELPOS"]

FACIES_NAMES = ["SS", "CSiS", "FSiS", "SiSh", "MS", "WS", "D", "PS", "BS"]
FACIES_DESCRIPTIONS = {
    "SS":   "Nonmarine Sandstone",
    "CSiS": "Nonmarine Coarse Siltstone",
    "FSiS": "Nonmarine Fine Siltstone",
    "SiSh": "Marine Siltstone and Shale",
    "MS":   "Mudstone (Limestone)",
    "WS":   "Wackestone (Limestone)",
    "D":    "Dolomite",
    "PS":   "Packstone-Grainstone",
    "BS":   "Phylloid-Algal Bafflestone",
}
FACIES_COLORS = {
    1: "#F4D03F", 2: "#F5B041", 3: "#DC7633", 4: "#6E2C00",
    5: "#1B4F72", 6: "#2E86C1", 7: "#AED6F1", 8: "#A569BD", 9: "#196F3D",
}
EXCLUDE_WELL = "Recruit F9"          # Known problematic well


# ===========================================================================
# DATA LOADING & CLEANING
# ===========================================================================

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse the uploaded CSV, standardise column names, and remove the
    known bad well (Recruit F9) that contains unreliable measurements.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip().str.replace(" ", ".")
    df = df[df["Well.Name"] != EXCLUDE_WELL].copy()
    df = df.reset_index(drop=True)
    return df


def check_required_columns(df: pd.DataFrame) -> list[str]:
    """Return list of any required columns missing from the dataframe."""
    required = ["Well.Name", "Depth", "Facies"] + FEATURES
    return [c for c in required if c not in df.columns]


# ===========================================================================
# MISSING VALUE IMPUTATION
# ===========================================================================

@st.cache_data(show_spinner=False)
def impute_pe(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, list[str]]:
    """
    Impute missing PE values using a Random Forest regressor trained on
    wells that have complete PE measurements.

    Returns (cleaned_df, was_imputed, list_of_wells_imputed).
    """
    df = df.copy()
    missing_mask = df["PE"].isna()
    if not missing_mask.any():
        return df, False, []

    wells_missing = df.loc[missing_mask, "Well.Name"].unique().tolist()
    wells_complete = df.loc[~df["Well.Name"].isin(wells_missing), "Well.Name"].unique()

    train_rows = df[df["Well.Name"].isin(wells_complete)].dropna(subset=IMPUTER_FEATURES + ["PE"])
    X_tr = train_rows[IMPUTER_FEATURES].values
    y_tr = train_rows["PE"].values

    regressor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    regressor.fit(X_tr, y_tr)

    for well in wells_missing:
        mask = (df["Well.Name"] == well) & df["PE"].isna()
        X_pred = df.loc[mask, IMPUTER_FEATURES].values
        if len(X_pred):
            df.loc[mask, "PE"] = regressor.predict(X_pred)

    return df, True, wells_missing


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================

@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Create additional predictive features:
      - Pairwise polar coordinates (rho, phi) between log pairs
      - Depth-wise gradients per well per log
      - K-Means cluster label on raw logs
    """
    df = df.copy()
    new_cols: list[str] = []

    # --- Polar coordinates ---
    for i in range(len(WELL_LOGS)):
        for j in range(i + 1, len(WELL_LOGS)):
            f1, f2 = WELL_LOGS[i], WELL_LOGS[j]
            x = df[f1] - df[f1].mean()
            y = df[f2] - df[f2].mean()
            rho_col = f"{f1}_{f2}_rho"
            phi_col = f"{f1}_{f2}_phi"
            df[rho_col] = np.sqrt(x**2 + y**2)
            df[phi_col] = np.arctan2(y, x)
            new_cols += [rho_col, phi_col]

    # --- Per-well, per-log depth gradients ---
    for well in df["Well.Name"].unique():
        w_mask = df["Well.Name"] == well
        w_depth = df.loc[w_mask, "Depth"].values
        for feat in WELL_LOGS:
            grad_col = f"{feat}_grad"
            if grad_col not in df.columns:
                df[grad_col] = np.nan
                new_cols.append(grad_col)
            values = df.loc[w_mask, feat].values
            # np.gradient handles uneven spacing natively
            df.loc[w_mask, grad_col] = np.gradient(values, w_depth)

    # Deduplicate new_cols list while preserving order
    seen: set[str] = set()
    new_cols = [c for c in new_cols if not (c in seen or seen.add(c))]

    # --- K-Means cluster (unsupervised log grouping) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[WELL_LOGS].fillna(df[WELL_LOGS].mean()))
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    new_cols.append("cluster")

    return df, new_cols


def build_feature_cols(df: pd.DataFrame) -> list[str]:
    """Collect all original + engineered columns available in the dataframe."""
    extra = [
        c for c in df.columns
        if any(tag in c for tag in ("_rho", "_phi", "_grad", "cluster"))
    ]
    return FEATURES + extra


# ===========================================================================
# MODEL TRAINING & EVALUATION
# ===========================================================================

def train_model(
    df: pd.DataFrame,
    test_well: str,
    feature_cols: list[str],
    model_type: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> dict:
    """
    Split data into train / test by well, train classifier, return results dict.
    """
    train_df = df[df["Well.Name"] != test_well].copy()
    test_df  = df[df["Well.Name"] == test_well].copy()

    # Encode facies labels 0-indexed
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].mean())
    y_train = train_df["Facies"].values - 1
    X_test  = test_df[feature_cols].fillna(train_df[feature_cols].mean())
    y_test  = test_df["Facies"].values - 1

    if model_type == "XGBoost":
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    else:  # Random Forest
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Cross-validation on training data (5-fold stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)

    return {
        "model":       clf,
        "X_train":     X_train,
        "y_train":     y_train,
        "X_test":      X_test,
        "y_test":      y_test,
        "y_pred":      y_pred,
        "test_df":     test_df,
        "feature_cols": feature_cols,
        "accuracy":    accuracy_score(y_test, y_pred),
        "f1":          f1_score(y_test, y_pred, average="weighted"),
        "cv_scores":   cv_scores,
    }


# ===========================================================================
# PLOTTING HELPERS
# ===========================================================================

def plot_well_logs(well_df: pd.DataFrame, y_pred: np.ndarray | None = None,
                   well_name: str = "") -> plt.Figure:
    """
    Plot depth tracks for a single well: log curves + actual facies +
    (optionally) predicted facies side by side.
    """
    well_df = well_df.sort_values("Depth").reset_index(drop=True)
    n_extra = 2 if y_pred is not None else 1
    n_cols  = len(WELL_LOGS) + n_extra
    fig, axes = plt.subplots(1, n_cols, figsize=(2.8 * n_cols, 12), sharey=True)
    fig.patch.set_facecolor("#F8F9FA")

    # Log curves
    for i, log in enumerate(WELL_LOGS):
        ax = axes[i]
        ax.plot(well_df[log], well_df["Depth"], "k-", linewidth=0.6)
        ax.set_xlabel(log, fontsize=9)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.set_facecolor("white")
        if i == 0:
            ax.set_ylabel("Depth (ft)", fontsize=9)

    depth_diff = well_df["Depth"].diff().median() or 0.5

    def _facies_track(ax, facies_series, label):
        colors = [FACIES_COLORS.get(int(f), "#CCCCCC") for f in facies_series]
        ax.barh(facies_series.index.map(well_df["Depth"]),
                [1] * len(facies_series),
                color=colors, height=depth_diff, align="center")
        # Re-plot using actual depth values
        ax.cla()
        ax.barh(well_df["Depth"], [1] * len(well_df),
                color=colors, height=depth_diff, align="center")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_xlabel(label, fontsize=9)
        ax.set_facecolor("#F8F9FA")

    _facies_track(axes[len(WELL_LOGS)], well_df["Facies"], "Actual\nFacies")

    if y_pred is not None:
        pred_series = pd.Series(y_pred + 1, index=well_df.index)
        _facies_track(axes[len(WELL_LOGS) + 1], pred_series, "Predicted\nFacies")

    # Legend
    legend_els = [
        mpatches.Patch(facecolor=FACIES_COLORS[i + 1],
                       label=f"{FACIES_NAMES[i]} – {FACIES_DESCRIPTIONS[FACIES_NAMES[i]]}")
        for i in range(len(FACIES_NAMES))
    ]
    fig.legend(handles=legend_els, loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 1.01), fontsize=7.5,
               framealpha=0.95)

    plt.suptitle(f"Well: {well_name}", fontsize=13, y=1.04, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_test, y_pred, test_well: str, accuracy: float) -> plt.Figure:
    present = sorted(np.unique(y_test))
    labels  = [FACIES_NAMES[i] for i in present]
    cm      = confusion_matrix(y_test, y_pred, labels=present)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.4, linecolor="#E5E7EB", ax=ax)
    ax.set_title(f"Confusion Matrix — {test_well}\nAccuracy: {accuracy:.2%}",
                 fontsize=13, pad=14)
    ax.set_xlabel("Predicted Facies", fontsize=11)
    ax.set_ylabel("True Facies", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_cols: list[str], top_n: int = 20) -> plt.Figure:
    importances = model.feature_importances_
    imp_df = (
        pd.DataFrame({"Feature": feature_cols, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(imp_df)))
    ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Top {top_n} Most Important Features", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# ===========================================================================
# STREAMLIT UI
# ===========================================================================

def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/rocks.png", width=64)
        st.title("⚙️ Configuration")

        uploaded_file = st.file_uploader(
            "📁 Upload CSV (facies_vectors.csv)",
            type=["csv"],
            help="Standard SEG 2016 ML contest format.",
        )

        st.divider()
        st.subheader("🎯 Model Settings")
        model_type   = st.selectbox("Algorithm", ["XGBoost", "Random Forest"])
        n_estimators = st.slider("Number of Trees", 50, 500, 150, 50)
        max_depth    = st.slider("Max Tree Depth",  3,  15,   7,  1)
        learning_rate = 0.1
        if model_type == "XGBoost":
            learning_rate = st.slider("Learning Rate", 0.01, 0.30, 0.10, 0.01)

        st.divider()
        st.caption("**Free hosting options:**")
        st.caption("• [Streamlit Community Cloud](https://streamlit.io/cloud)")
        st.caption("• [Hugging Face Spaces](https://huggingface.co/spaces)")
        st.caption("• [Render Free Tier](https://render.com)")

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("🪨 Facies Classification from Wireline Logs")
    st.markdown(
        "Automated lithofacies classification using machine learning on wireline log data. "
        "Upload the **facies_vectors.csv** file to begin."
    )

    if uploaded_file is None:
        _welcome_screen()
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    raw_bytes = uploaded_file.read()
    with st.spinner("Loading data…"):
        df_raw = load_and_clean(raw_bytes)

    missing_cols = check_required_columns(df_raw)
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "🔧 Preprocessing",
        "🤖 Train & Evaluate",
        "📈 Results & Metrics",
        "🎨 Visualisations",
    ])

    # ── TAB 1: Data Overview ─────────────────────────────────────────────────
    with tab1:
        st.header("Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",   df_raw.shape[0])
        c2.metric("Input Features", len(FEATURES))
        c3.metric("Wells",         df_raw["Well.Name"].nunique())
        c4.metric("Facies Classes", df_raw["Facies"].nunique())

        st.subheader("Preview")
        st.dataframe(df_raw.head(20), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Wells & Sample Counts")
            wc = (
                df_raw.groupby("Well.Name")["Facies"]
                .count()
                .reset_index()
                .rename(columns={"Facies": "Samples"})
                .sort_values("Samples", ascending=False)
            )
            st.dataframe(wc, use_container_width=True)

        with col_b:
            st.subheader("Facies Legend")
            legend_df = pd.DataFrame([
                {"#": i + 1, "Code": FACIES_NAMES[i],
                 "Description": FACIES_DESCRIPTIONS[FACIES_NAMES[i]]}
                for i in range(len(FACIES_NAMES))
            ])
            st.dataframe(legend_df, use_container_width=True)

        # Basic EDA plots
        st.subheader("Exploratory Visualisations")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # Facies distribution
        facies_counts = df_raw["Facies"].value_counts().sort_index()
        axes[0].bar(facies_counts.index, facies_counts.values,
                    color=[FACIES_COLORS.get(i, "#888") for i in facies_counts.index])
        axes[0].set_title("Facies Distribution")
        axes[0].set_xlabel("Facies Code")
        axes[0].set_ylabel("Count")
        axes[0].set_xticks(facies_counts.index)
        axes[0].set_xticklabels([FACIES_NAMES[i - 1] for i in facies_counts.index], rotation=45)

        # Samples per well
        wc_sorted = wc.sort_values("Samples")
        axes[1].barh(wc_sorted["Well.Name"], wc_sorted["Samples"], color="steelblue")
        axes[1].set_title("Samples per Well")
        axes[1].set_xlabel("Count")

        # Feature correlation heatmap
        corr = df_raw[FEATURES].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=axes[2], annot_kws={"size": 7})
        axes[2].set_title("Feature Correlations")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── TAB 2: Preprocessing ─────────────────────────────────────────────────
    with tab2:
        st.header("Preprocessing Pipeline")

        # Missing values table
        st.subheader("Missing Value Analysis")
        mv = pd.DataFrame({
            "Feature":      FEATURES,
            "Missing (n)":  [df_raw[f].isna().sum() for f in FEATURES],
            "Missing (%)":  [f"{df_raw[f].isna().sum()/len(df_raw)*100:.2f}%" for f in FEATURES],
        })
        st.dataframe(mv, use_container_width=True)

        # Imputation
        with st.spinner("Imputing missing PE values (if any)…"):
            df_imp, was_imputed, imp_wells = impute_pe(df_raw)

        if was_imputed:
            st.success(f"✅ PE imputed for: {', '.join(imp_wells)} using Random Forest regressor.")
        else:
            st.info("ℹ️ No missing PE values detected — imputation step skipped.")

        # Feature engineering
        st.subheader("Feature Engineering")
        with st.spinner("Building engineered features…"):
            df_eng, new_feat_cols = engineer_features(df_imp)

        st.success(
            f"✅ Feature engineering complete.  "
            f"Dataset: **{df_eng.shape[0]}** rows × **{df_eng.shape[1]}** columns.  "
            f"New features added: **{len(new_feat_cols)}**"
        )

        with st.expander("New feature list"):
            st.write(new_feat_cols)

        # Store preprocessed df in session state for subsequent tabs
        st.session_state["df_eng"]       = df_eng
        st.session_state["new_feat_cols"] = new_feat_cols

    # ── TAB 3: Train & Evaluate ───────────────────────────────────────────────
    with tab3:
        st.header("Model Training")

        if "df_eng" not in st.session_state:
            st.warning("⚠️ Please complete the Preprocessing step first (Tab 2).")
            st.stop()

        df_eng = st.session_state["df_eng"]
        available_wells = sorted(df_eng["Well.Name"].unique())
        default_idx = available_wells.index("SHANKLE") if "SHANKLE" in available_wells else 0
        test_well = st.selectbox("Hold-out Test Well", available_wells, index=default_idx)

        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            feature_cols = build_feature_cols(df_eng)

            with st.spinner(f"Training {model_type} model…"):
                results = train_model(
                    df_eng, test_well, feature_cols,
                    model_type, n_estimators, max_depth, learning_rate,
                )

            st.session_state["results"]   = results
            st.session_state["test_well"] = test_well
            st.success("✅ Training complete!")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",          f"{results['accuracy']:.2%}")
            c2.metric("Weighted F1",        f"{results['f1']:.3f}")
            c3.metric("CV F1 (mean±std)",
                      f"{results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}")
            c4.metric("Training Samples",  len(results["X_train"]))

    # ── TAB 4: Results & Metrics ──────────────────────────────────────────────
    with tab4:
        st.header("Evaluation Results")

        if "results" not in st.session_state:
            st.info("👈 Train the model first (Tab 3).")
            st.stop()

        results   = st.session_state["results"]
        test_well = st.session_state["test_well"]
        y_test    = results["y_test"]
        y_pred    = results["y_pred"]
        accuracy  = results["accuracy"]

        # Classification report
        st.subheader("Classification Report")
        present_labels = sorted(np.unique(y_test))
        label_names    = [FACIES_NAMES[i] for i in present_labels]
        report_dict = classification_report(
            y_test, y_pred,
            labels=present_labels,
            target_names=label_names,
            output_dict=True,
        )
        report_df = pd.DataFrame(report_dict).transpose().round(3)
        st.dataframe(
            report_df.style.background_gradient(cmap="RdYlGn", subset=["f1-score"]),
            use_container_width=True,
        )

        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_test, y_pred, test_well, accuracy)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        # Cross-validation scores
        st.subheader("Cross-Validation Results (Training Data)")
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(results['cv_scores']))],
            "Weighted F1": results["cv_scores"],
        })
        fig_cv, ax_cv = plt.subplots(figsize=(7, 3))
        ax_cv.bar(cv_df["Fold"], cv_df["Weighted F1"], color="steelblue", alpha=0.8)
        ax_cv.axhline(results["cv_scores"].mean(), color="red", linestyle="--",
                      label=f"Mean = {results['cv_scores'].mean():.3f}")
        ax_cv.set_ylim(0, 1)
        ax_cv.set_ylabel("Weighted F1")
        ax_cv.legend()
        ax_cv.set_title("5-Fold Stratified Cross-Validation")
        plt.tight_layout()
        st.pyplot(fig_cv)
        plt.close(fig_cv)

        # Download predictions as CSV
        pred_export = results["test_df"][["Well.Name", "Depth", "Facies"]].copy()
        pred_export["Predicted_Facies"]    = y_pred + 1
        pred_export["Predicted_Facies_Name"] = [FACIES_NAMES[p] for p in y_pred]
        csv_bytes = pred_export.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv_bytes,
            file_name=f"predictions_{test_well}.csv",
            mime="text/csv",
        )

    # ── TAB 5: Visualisations ─────────────────────────────────────────────────
    with tab5:
        st.header("Visualisations")

        if "results" not in st.session_state or "df_eng" not in st.session_state:
            st.info("👈 Train the model first (Tab 3).")
            st.stop()

        results   = st.session_state["results"]
        df_eng    = st.session_state["df_eng"]
        test_well = st.session_state["test_well"]
        y_pred    = results["y_pred"]

        # Well log tracks
        st.subheader("Well Log Tracks")
        all_wells = sorted(df_eng["Well.Name"].unique())
        well_choice = st.selectbox(
            "Select Well",
            options=[f"🔍 {test_well} (test, with predictions)"] + all_wells,
        )

        if well_choice.startswith("🔍"):
            plot_well_name = test_well
            well_df        = results["test_df"].copy()
            show_pred      = y_pred
        else:
            plot_well_name = well_choice
            well_df        = df_eng[df_eng["Well.Name"] == well_choice].copy()
            show_pred      = None

        with st.spinner("Rendering log tracks…"):
            fig_well = plot_well_logs(well_df, show_pred, plot_well_name)
        st.pyplot(fig_well)
        plt.close(fig_well)

        # Feature importance
        st.subheader("Feature Importance")
        top_n = st.slider("Show top N features", 5, 40, 20, 5)
        fig_imp = plot_feature_importance(results["model"], results["feature_cols"], top_n)
        st.pyplot(fig_imp)
        plt.close(fig_imp)

        # GR vs Depth coloured by predicted facies (test well)
        st.subheader("GR vs Depth — Predicted Facies (Test Well)")
        tw_sorted = results["test_df"].sort_values("Depth").reset_index(drop=True)
        fig_gr, ax_gr = plt.subplots(figsize=(5, 10))
        for facies_code in sorted(tw_sorted["Facies"].unique()):
            mask = tw_sorted["Facies"] == facies_code
            ax_gr.scatter(
                tw_sorted.loc[mask, "GR"],
                tw_sorted.loc[mask, "Depth"],
                c=FACIES_COLORS.get(facies_code, "#888"),
                s=4, label=FACIES_NAMES[facies_code - 1], alpha=0.7,
            )
        ax_gr.invert_yaxis()
        ax_gr.set_xlabel("GR (API)")
        ax_gr.set_ylabel("Depth (ft)")
        ax_gr.set_title(f"GR Profile — {test_well}")
        ax_gr.legend(loc="lower right", fontsize=7, markerscale=3)
        ax_gr.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig_gr)
        plt.close(fig_gr)


# ===========================================================================
# WELCOME SCREEN
# ===========================================================================

def _welcome_screen():
    st.info("👈 Upload your **facies_vectors.csv** file from the sidebar to begin.")

    with st.expander("📖 About this App"):
        st.markdown(
            """
            This application performs **automated lithofacies classification** from
            wireline well log data using gradient-boosted trees (XGBoost) or
            Random Forests.

            **Pipeline summary**
            1. Upload CSV → clean data (remove Recruit F9)
            2. Impute missing PE values via Random Forest regression
            3. Engineer pairwise polar coordinates, depth gradients, and cluster labels
            4. Train classifier on all wells except the held-out test well
            5. Evaluate: accuracy, weighted F1, confusion matrix, 5-fold CV
            6. Visualise log tracks with actual vs. predicted facies

            **Required CSV columns**
            `Well.Name`, `Depth`, `Facies`, `GR`, `ILD_log10`,
            `DeltaPHI`, `PHIND`, `PE`, `NM_M`, `RELPOS`

            **Data source:** SEG 2016 Machine Learning Contest  
            (Brendon Hall, Enthought — Kansas subsurface)
            """
        )

    st.subheader("Facies Reference")
    legend_df = pd.DataFrame([
        {"Code": i + 1, "Name": FACIES_NAMES[i],
         "Description": FACIES_DESCRIPTIONS[FACIES_NAMES[i]]}
        for i in range(len(FACIES_NAMES))
    ])
    st.dataframe(legend_df, use_container_width=True)


# ===========================================================================
if __name__ == "__main__":
    main()