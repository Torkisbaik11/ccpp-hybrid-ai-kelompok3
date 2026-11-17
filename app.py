import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import skfuzzy as fuzz
from skfuzzy import control as ctrl

st.set_page_config(
    page_title="CCPP Hybrid AI System - Kelompok 3",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    return joblib.load("ccpp_hybrid_artifacts.joblib")

@st.cache_resource
def build_fuzzy_system():
    pred_power = ctrl.Antecedent(np.arange(420, 501, 1), "predicted_power")
    ambient_temp = ctrl.Antecedent(np.arange(0, 41, 1), "ambient_temp")
    perf = ctrl.Consequent(np.arange(0, 101, 1), "operating_performance")

    pred_power["Low"] = fuzz.trapmf(pred_power.universe, [420, 420, 442, 455])
    pred_power["Medium"] = fuzz.trimf(pred_power.universe, [445, 460, 475])
    pred_power["High"] = fuzz.trapmf(pred_power.universe, [465, 480, 500, 500])

    ambient_temp["Cool"] = fuzz.trapmf(ambient_temp.universe, [0, 0, 12, 18])
    ambient_temp["Moderate"] = fuzz.trimf(ambient_temp.universe, [12, 21, 30])
    ambient_temp["Hot"] = fuzz.trapmf(ambient_temp.universe, [24, 32, 40, 40])

    perf["Low"] = fuzz.trapmf(perf.universe, [0, 0, 20, 35])
    perf["Normal"] = fuzz.trimf(perf.universe, [25, 42, 58])
    perf["Good"] = fuzz.trimf(perf.universe, [50, 68, 82])
    perf["Optimal"] = fuzz.trapmf(perf.universe, [75, 88, 100, 100])

    rules = [
        ctrl.Rule(pred_power["Low"] & ambient_temp["Cool"], perf["Low"]),
        ctrl.Rule(pred_power["Low"] & ambient_temp["Moderate"], perf["Low"]),
        ctrl.Rule(pred_power["Low"] & ambient_temp["Hot"], perf["Normal"]),
        ctrl.Rule(pred_power["Medium"] & ambient_temp["Cool"], perf["Normal"]),
        ctrl.Rule(pred_power["Medium"] & ambient_temp["Moderate"], perf["Good"]),
        ctrl.Rule(pred_power["Medium"] & ambient_temp["Hot"], perf["Good"]),
        ctrl.Rule(pred_power["High"] & ambient_temp["Cool"], perf["Good"]),
        ctrl.Rule(pred_power["High"] & ambient_temp["Moderate"], perf["Optimal"]),
        ctrl.Rule(pred_power["High"] & ambient_temp["Hot"], perf["Optimal"]),
    ]
    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system), pred_power, ambient_temp, perf

def evaluate_fuzzy(predicted_pe, raw_at):
    sim, _, _, _ = build_fuzzy_system()
    sim.input["predicted_power"] = float(np.clip(predicted_pe, 420, 500))
    sim.input["ambient_temp"] = float(np.clip(raw_at, 0, 40))
    sim.compute()
    score = float(sim.output["operating_performance"])
    if score < 30:
        label = "Low"
    elif score < 55:
        label = "Normal"
    elif score < 78:
        label = "Good"
    else:
        label = "Optimal"
    return score, label

art = load_artifacts()

st.sidebar.title("⚡ Hybrid AI Dashboard")
st.sidebar.caption("Kelompok 3 — Kecerdasan Buatan ITK")
menu = st.sidebar.radio(
    "Navigasi Halaman (8 Modul Wajib):",
    [
        "1. Project Overview",
        "2. Dataset Explorer",
        "3. GA Optimization",
        "4. PSO Optimization",
        "5. Neural Network",
        "6. Fuzzy Inference",
        "7. Prediction / Simulation",
        "8. Model Comparison",
    ]
)

# ------------------------------------------------------------------------------
# 1. PROJECT OVERVIEW
# ------------------------------------------------------------------------------
if menu == "1. Project Overview":
    st.title("⚡ Combined Cycle Power Plant Output Prediction")
    st.subheader("Hybrid Intelligent System: GA - PSO - MLP Regression - Fuzzy Logic")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### Anggota Kelompok 3")
        st.table(pd.DataFrame({
            "NIM": ["04231084", "04231085", "04231092", "04231088"],
            "Nama Mahasiswa": [
                "Syamsul Ma'arif Usman",
                "Tam Clement Bryan Kurniawan",
                "Zahrah Salsabila",
                "Torkis J Simanjuntak"
            ]
        }))
        st.markdown("### Deskripsi Masalah & Dataset")
        st.write(
            "Pembangkit listrik tenaga gas-uap (*Combined Cycle Power Plant* / CCPP) memiliki "
            "output daya listrik bersih (*Net Hourly Electrical Energy Output*, PE) yang dipengaruhi "
            "oleh kondisi lingkungan sekitar: **Ambient Temperature (AT)**, **Exhaust Vacuum (V)**, "
            "**Ambient Pressure (AP)**, dan **Relative Humidity (RH)** (Dataset UCI ID 294)."
        )

    with col2:
        st.markdown("### Arsitektur Pipeline Hybrid")
        st.info(
            "**1. Dataset UCI 294** (9.568 sampel, 4 fitur)\n\n"
            "⬇️ *Train/Val/Test Split & StandardScaler (No Leakage)*\n\n"
            "**2. Genetic Algorithm (GA)**: Seleksi variabel operasi paling informatif\n\n"
            "⬇️ *Subset Fitur Terpilih*\n\n"
            "**3. Particle Swarm Optimization (PSO)**: Optimasi *hidden units*, *learning rate*, dan *batch size*\n\n"
            "⬇️ *Konfigurasi Optimal*\n\n"
            "**4. Neural Network (MLP Regression)**: Prediksi daya listrik bersih (MW)\n\n"
            "⬇️ *Predicted Power + Ambient Temp*\n\n"
            "**5. Fuzzy Inference System**: Klasifikasi status operasi (*Low - Normal - Good - Optimal*)"
        )

# ------------------------------------------------------------------------------
# 2. DATASET EXPLORER
# ------------------------------------------------------------------------------
elif menu == "2. Dataset Explorer":
    st.title("📊 Dataset Explorer (UCI ID 294)")
    X_df = art["X_df"]
    y_series = art["y_series"]
    full_df = X_df.copy()
    full_df["PE (Target)"] = y_series

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Sampel", f"{len(full_df):,}")
    c2.metric("Jumlah Fitur Input", len(art["feature_names"]))
    c3.metric("Missing Values", int(full_df.isnull().sum().sum()))
    c4.metric("Rata-rata Output (MW)", f"{y_series.mean():.2f}")

    st.markdown("### Statistik Deskriptif")
    st.dataframe(full_df.describe().T, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribusi Target Daya Listrik (PE)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(y_series, bins=35, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Net Electrical Power Output (MW)")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("### Matriks Korelasi")
        fig, ax = plt.subplots(figsize=(6, 4))
        corr = full_df.corr()
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=20)
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------------------
# 3. GA OPTIMIZATION
# ------------------------------------------------------------------------------
elif menu == "3. GA Optimization":
    st.title("🧬 Genetic Algorithm (Feature Selection)")
    ga_runs = art["ga_runs"]
    fits = [r["fitness"] for r in ga_runs]

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Fitness (5 Runs)", f"{np.min(fits):.4f}")
    c2.metric("Mean Fitness", f"{np.mean(fits):.4f}")
    c3.metric("Std Dev Fitness", f"{np.std(fits):.4f}")

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown("### Parameter GA")
        st.json(art["ga_params"])
        st.success(f"**Fitur Terpilih:** {', '.join(art['selected_features'])}")
        st.warning(f"**Fitur Dieliminasi:** {', '.join(art['eliminated_features']) if art['eliminated_features'] else 'Tidak ada (semua fitur informatif)'}")

    with col2:
        st.markdown("### GA Convergence Curve (Multi-Run)")
        fig, ax = plt.subplots(figsize=(7, 4))
        for r in ga_runs:
            ax.plot(range(1, len(r["curve"]) + 1), r["curve"], marker="o", label=f"Run {r['run']}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (Val RMSE + Penalty)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------------------
# 4. PSO OPTIMIZATION
# ------------------------------------------------------------------------------
elif menu == "4. PSO Optimization":
    st.title("🐝 Particle Swarm Optimization (Hyperparameter Tuning)")
    pso_runs = art["pso_runs"]
    vals = [r["gbest_val"] for r in pso_runs]

    c1, c2, c3 = st.columns(3)
    c1.metric("Global Best Val RMSE", f"{np.min(vals):.4f} MW")
    c2.metric("Mean Val RMSE", f"{np.mean(vals):.4f} MW")
    c3.metric("Std Dev Val RMSE", f"{np.std(vals):.4f} MW")

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown("### Parameter Swarm & Search Space")
        st.json(art["pso_params"])
        st.write("**Search Space:**")
        st.write("- `hidden_units`: 16 – 128")
        st.write("- `learning_rate`: $10^{-4}$ – $10^{-1.3}$")
        st.write("- `batch_size`: 16 – 256")
        st.success(f"**Hybrid Optimal Params:** {art['hybrid_best_params']}")

    with col2:
        st.markdown("### PSO Convergence Curve (Multi-Run)")
        fig, ax = plt.subplots(figsize=(7, 4))
        for r in pso_runs:
            ax.plot(range(1, len(r["curve"]) + 1), r["curve"], marker="s", label=f"Run {r['run']}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Global Best Validation RMSE")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------------------
# 5. NEURAL NETWORK
# ------------------------------------------------------------------------------
elif menu == "5. Neural Network":
    st.title("🧠 Neural Network (MLP Regression)")
    model_D = art["model_D"]
    m_D = art["metrics_D"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test MAE", f"{m_D['MAE']:.4f} MW")
    c2.metric("Test RMSE", f"{m_D['RMSE']:.4f} MW")
    c3.metric("Test R² Score", f"{m_D['R2']:.4f}")
    c4.metric("Jumlah Parameter", f"{art['n_params_D']:,}")

    st.info(
        f"**Arsitektur Hybrid MLP:** Input({len(art['selected_features'])}) → "
        f"Dense({model_D.hidden_layer_sizes[0]}, ReLU) → Dense({model_D.hidden_layer_sizes[1]}, ReLU) → Output(1, Linear). "
        "MLP dipilih karena mampu memodelkan hubungan non-linear antar variabel termodinamika tabular secara presisi."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Training Loss Curve (Model D)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(model_D.loss_curve_, color="#16a085", lw=2)
        ax.set_xlabel("Epoch / Iteration")
        ax.set_ylabel("Loss (MSE)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("### Actual vs Predicted (Test Set)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(art["y_test"], art["pred_D_test"], alpha=0.35, color="#16a085", edgecolors="k", linewidth=0.2)
        ax.plot([art["y_test"].min(), art["y_test"].max()], [art["y_test"].min(), art["y_test"].max()], "r--", lw=2)
        ax.set_xlabel("Actual Power Output (MW)")
        ax.set_ylabel("Predicted Power Output (MW)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------------------
# 6. FUZZY INFERENCE
# ------------------------------------------------------------------------------
elif menu == "6. Fuzzy Inference":
    st.title("🎛️ Fuzzy Inference System (Decision Support Layer)")
    _, mf_power, mf_temp, mf_perf = build_fuzzy_system()

    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for t in mf_power.terms:
            ax.plot(mf_power.universe, mf_power[t].mf, lw=2, label=t)
        ax.set_title("Input 1: Predicted Power (MW)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for t in mf_temp.terms:
            ax.plot(mf_temp.universe, mf_temp[t].mf, lw=2, label=t)
        ax.set_title("Input 2: Ambient Temp (°C)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for t in mf_perf.terms:
            ax.plot(mf_perf.universe, mf_perf[t].mf, lw=2, label=t)
        ax.set_title("Output: Operating Performance")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Fuzzy Rule Base (Mamdani Inference + Centroid Defuzzification)")
    rules_table = pd.DataFrame([
        ["Low", "Cool", "Low"], ["Low", "Moderate", "Low"], ["Low", "Hot", "Normal"],
        ["Medium", "Cool", "Normal"], ["Medium", "Moderate", "Good"], ["Medium", "Hot", "Good"],
        ["High", "Cool", "Good"], ["High", "Moderate", "Optimal"], ["High", "Hot", "Optimal"],
    ], columns=["IF Predicted Power", "AND Ambient Temp", "THEN Operating Performance"])
    st.table(rules_table)

# ------------------------------------------------------------------------------
# 7. PREDICTION / SIMULATION
# ------------------------------------------------------------------------------
elif menu == "7. Prediction / Simulation":
    st.title("🚀 Prediksi Daya, Sensitivitas Input & Keputusan Fuzzy")
    tab1, tab2 = st.tabs(["Simulasi Input Manual & Sensitivitas", "Batch Prediction (Upload CSV)"])

    with tab1:
        col_in, col_out = st.columns([1, 1.3])
        with col_in:
            st.markdown("### Parameter Kondisi Operasi")
            at_val = st.slider("Ambient Temperature - AT (°C)", 1.8, 37.1, 19.6, 0.1)
            v_val = st.slider("Exhaust Vacuum - V (cm Hg)", 25.3, 81.6, 54.3, 0.1)
            ap_val = st.slider("Ambient Pressure - AP (milibar)", 992.9, 1033.3, 1013.3, 0.1)
            rh_val = st.slider("Relative Humidity - RH (%)", 25.5, 100.1, 73.3, 0.1)

        raw_input = np.array([[at_val, v_val, ap_val, rh_val]])
        scaled_input = art["scaler"].transform(raw_input)
        pred_pe = float(art["model_D"].predict(scaled_input[:, art["best_ga_mask"]])[0])
        fuzzy_score, fuzzy_label = evaluate_fuzzy(pred_pe, at_val)

        with col_out:
            st.markdown("### Hasil Prediksi & Keputusan Sistem")
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediksi Daya (PE)", f"{pred_pe:.2f} MW")
            m2.metric("Skor Fuzzy", f"{fuzzy_score:.2f} / 100")
            m3.metric("Status Operasi", fuzzy_label)

            # Analisis Sensitivitas Input (One-at-a-Time Sensitivity)
            st.markdown("### Analisis Sensitivitas Input terhadap Prediksi Daya")
            ranges = {
                "AT": np.linspace(1.8, 37.1, 40),
                "V": np.linspace(25.3, 81.6, 40),
                "AP": np.linspace(992.9, 1033.3, 40),
                "RH": np.linspace(25.5, 100.1, 40),
            }
            fig, axes = plt.subplots(2, 2, figsize=(8, 5))
            for idx, (fname, grid) in enumerate(ranges.items()):
                ax = axes[idx // 2, idx % 2]
                sweep = np.repeat(raw_input, len(grid), axis=0)
                sweep[:, idx] = grid
                sweep_scaled = art["scaler"].transform(sweep)
                preds = art["model_D"].predict(sweep_scaled[:, art["best_ga_mask"]])
                ax.plot(grid, preds, color="#2980b9", lw=2)
                ax.axvline(raw_input[0, idx], color="red", linestyle="--", lw=1)
                ax.set_title(f"Sensitivitas {fname}")
                ax.set_ylabel("Prediksi PE (MW)")
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        st.markdown("Unggah file CSV yang memiliki kolom: `AT`, `V`, `AP`, `RH`.")
        uploaded = st.file_uploader("Pilih file CSV", type=["csv"])
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            req_cols = ["AT", "V", "AP", "RH"]
            if all(c in df_up.columns for c in req_cols):
                X_up_scaled = art["scaler"].transform(df_up[req_cols].values)
                preds_up = art["model_D"].predict(X_up_scaled[:, art["best_ga_mask"]])
                decisions = [evaluate_fuzzy(p, a) for p, a in zip(preds_up, df_up["AT"].values)]
                df_up["Predicted_PE_MW"] = np.round(preds_up, 2)
                df_up["Fuzzy_Score"] = [round(d[0], 2) for d in decisions]
                df_up["Operating_Status"] = [d[1] for d in decisions]
                st.dataframe(df_up, use_container_width=True)
                st.download_button(
                    "Unduh Hasil Prediksi (CSV)",
                    df_up.to_csv(index=False).encode("utf-8"),
                    "hasil_prediksi_ccpp.csv",
                    "text/csv"
                )
            else:
                st.error(f"Kolom wajib tidak lengkap. Pastikan ada kolom: {req_cols}")

# ------------------------------------------------------------------------------
# 8. MODEL COMPARISON
# ------------------------------------------------------------------------------
elif menu == "8. Model Comparison":
    st.title("📈 Perbandingan Eksperimen (Model A vs B vs C vs D)")
    comp_df = art["comparison_df"]
    st.dataframe(comp_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(comp_df["Model"], comp_df["RMSE"], color=["#7f8c8d", "#2980b9", "#8e44ad", "#16a085"])
        ax.set_title("Perbandingan Test RMSE (MW)")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(comp_df["Model"], comp_df["R2"], color=["#7f8c8d", "#2980b9", "#8e44ad", "#16a085"])
        ax.set_ylim(max(0, comp_df["R2"].min() - 0.02), 1.0)
        ax.set_title("Perbandingan Test R² Score")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
