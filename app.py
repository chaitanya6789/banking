"""
Fraud Drift Guard  -  Streamlit deployment of the project
"An Event-Driven Feature-Partitioned Heterogeneous Sub-Ensemble Framework for
Selective Concept Drift Mitigation in Financial Fraud Detection"

Run locally :  streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import fraud_core as fc

# ----------------------------------------------------------------------------
# Page setup + small CSS polish
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Fraud Drift Guard", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {padding: 1.2rem 1.5rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(120deg,#1f2a44 0%,#3b1f3f 100%);}
.hero h1 {margin: 0; font-size: 1.9rem;}
.hero p {margin: .3rem 0 0 0; opacity: .85;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600;
       font-size:.8rem;}
.pill-ok {background:#14532d; color:#bbf7d0;}
.pill-warn {background:#713f12; color:#fde68a;}
.pill-bad {background:#7f1d1d; color:#fecaca;}
div[data-testid="stMetric"] {background:#1a1f2b; border:1px solid #2a3142;
       padding:.7rem .9rem; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

COLORS = {"P1": "#8e44ad", "P2": "#16a085", "P3": "#f39c12"}


def pill(state: str) -> str:
    cls = {"STABLE": "pill-ok", "WARNING": "pill-warn", "DRIFT": "pill-bad"}[state]
    return f'<span class="pill {cls}">{state}</span>'


def need(key: str, msg: str) -> bool:
    if key not in st.session_state:
        st.info(msg)
        return False
    return True


# ----------------------------------------------------------------------------
# Sidebar: data source, settings, train button
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Control panel")
    st.caption("Step 1 - load data and train the three sub-models")

    local_path = fc.find_local_dataset()
    options = (["Dataset in repo"] if local_path else []) + ["Upload CSV / ZIP"]
    src = st.radio("Data source", options)
    uploaded = None
    if src == "Upload CSV / ZIP":
        uploaded = st.file_uploader("creditcard.csv (or .zip)", type=["csv", "zip"])

    rows_mode = st.radio(
        "Rows to use",
        ["Notebook subset (first 186,283)", "Full dataset (284,807)", "Custom"],
        help="The Colab notebook used the first 186,283 rows. Pick 'Full' to use everything.")
    custom_rows = None
    if rows_mode == "Custom":
        custom_rows = st.number_input("Number of rows", 20_000, 284_807, 100_000, 5_000)
    nrows = {"Notebook subset (first 186,283)": fc.NOTEBOOK_ROWS,
             "Full dataset (284,807)": None}.get(rows_mode, custom_rows)

    with st.expander("⚙️ Advanced settings"):
        cfg = dict(fc.DEFAULT_CFG)
        cfg["seed"] = st.number_input("Random seed", 0, 9999, 42)
        c1, c2 = st.columns(2)
        cfg["rf_trees"] = c1.number_input("RF trees", 10, 300, 60, 10)
        cfg["rf_depth"] = c2.number_input("RF depth", 2, 20, 6)
        cfg["xgb_trees"] = c1.number_input("XGB trees", 10, 500, 60, 10)
        cfg["xgb_depth"] = c2.number_input("XGB depth", 2, 12, 4)
        cfg["dt_depth"] = c1.number_input("DT depth", 2, 20, 6)
        cfg["global_trees"] = c2.number_input("Global XGB trees", 20, 600, 180, 20)
        st.markdown("**Vote weights** (auto-normalised to 1.0)")
        cfg["w1"] = st.slider("w1  (P1 - RF)", 0.0, 1.0, 0.30, 0.05)
        cfg["w2"] = st.slider("w2  (P2 - XGB)", 0.0, 1.0, 0.45, 0.05)
        cfg["w3"] = st.slider("w3  (P3 - DT)", 0.0, 1.0, 0.25, 0.05)
        cfg["threshold"] = st.slider("Fraud threshold τ", 0.05, 0.95, 0.50, 0.05)
        cfg["psi_threshold"] = st.slider("PSI drift threshold", 0.05, 1.0, 0.25, 0.05)

    train_clicked = st.button("🚀 Load data & train models", type="primary",
                              width="stretch")
    st.divider()
    st.caption("SRM IST · CSE (AI/ML) · Team Fraud Detection Banking")

if train_clicked:
    try:
        with st.spinner("Loading dataset..."):
            if src == "Dataset in repo":
                df_raw = fc.load_dataset(local_path, nrows)
                data_name = local_path.split("/")[-1]
            elif uploaded is not None:
                df_raw = fc.load_dataset(uploaded, nrows, filename=uploaded.name)
                data_name = uploaded.name
            else:
                st.sidebar.error("Please upload a file first.")
                st.stop()
        with st.spinner("Preparing data (split, scale, partition)..."):
            prep = fc.prepare_data(df_raw, seed=cfg["seed"])
        with st.spinner("Training Random Forest, XGBoost and Decision Tree..."):
            models, times = fc.train_submodels(prep, cfg)
        for k in ("drift", "strat", "live"):
            st.session_state.pop(k, None)
        st.session_state["bundle"] = dict(prep=prep, models=models, times=times,
                                          cfg=cfg, data_name=data_name,
                                          base=fc.baseline_table(models, prep, cfg))
        st.sidebar.success("Models trained ✔")
    except Exception as e:  # show friendly error in the UI
        st.sidebar.error(f"Could not load/train: {e}")
        st.stop()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🛡️ Fraud Drift Guard</h1>
  <p>Feature-partitioned heterogeneous sub-ensemble · PSI drift monitor · selective retraining</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Overview", "📊 Data", "🧠 Models", "🔍 Predict", "📡 Drift Monitor",
                "⚔️ Strategies", "🔴 Live Stream", "🏁 Conclusion"])

B = st.session_state.get("bundle")

# ----------------------------------------------------------------------------
# TAB 1: Overview
# ----------------------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("The idea in one line")
        st.success("When customer behaviour drifts, **don't retrain everything** - "
                   "find *which feature group* drifted and retrain **only that sub-model**.")
        st.markdown("""
**The problem**
1. Fraud is **extremely rare** (~0.2 %), so accuracy is meaningless.
2. Behaviour **keeps changing** (concept drift) and models go stale.
3. Normal MLOps **retrains all 29 features** even if only one group changed.

**Our solution**
- Split the 29 features into **3 partitions**, one specialised sub-model each.
- Combine them with **weighted soft voting**: `0.30·p1 + 0.45·p2 + 0.25·p3 ≥ 0.50 → FRAUD`.
- Watch every partition with **PSI**. If `PSI ≥ 0.25`, retrain **only that** sub-model.
        """)
    with c2:
        st.subheader("Architecture")
        st.graphviz_chart("""
        digraph G { rankdir=TB; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fontname="Helvetica", fontcolor=white, color="#444"];
          edge [color="#999"];
          D [label="creditcard.csv", fillcolor="#34495e"];
          S [label="Drop Time · 60/40 split · RobustScaler", fillcolor="#34495e"];
          P1 [label="P1: V1-V9\\nRandom Forest (0.30)", fillcolor="#8e44ad"];
          P2 [label="P2: V10-V19\\nXGBoost (0.45)", fillcolor="#16a085"];
          P3 [label="P3: V20-V28+Amt\\nDecision Tree (0.25)", fillcolor="#d68910"];
          V [label="Weighted soft vote ≥ 0.50", fillcolor="#c0392b"];
          M [label="PSI monitor per partition", fillcolor="#2c3e50"];
          R [label="Retrain ONLY flagged sub-model", fillcolor="#27ae60"];
          D->S; S->P1; S->P2; S->P3; P1->V; P2->V; P3->V; V->M; M->R; R->P3 [style=dashed];
        }
        """)
    st.markdown("**How to use this app:** set options in the sidebar → **Load data & train** → "
                "explore the tabs from left to right.")
    if B:
        st.success(f"Ready: `{B['data_name']}` loaded, {len(B['prep']['df_raw']):,} rows, "
                   f"models trained.")

# ----------------------------------------------------------------------------
# TAB 2: Data
# ----------------------------------------------------------------------------
with tabs[1]:
    if need("bundle", "👈 Load the data and train the models from the sidebar first."):
        prep = B["prep"]
        df = prep["df_raw"]
        n, f = len(df), int(df["Class"].sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transactions", f"{n:,}")
        m2.metric("Genuine", f"{n - f:,}")
        m3.metric("Fraud", f"{f:,}")
        m4.metric("Fraud rate", f"{f / n * 100:.3f} %")

        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Bar(x=["Total", "Genuine (0)", "Fraud (1)"], y=[n, n - f, f],
                                   marker_color=["#34495e", "#2ecc71", "#e74c3c"],
                                   text=[f"{v:,}" for v in (n, n - f, f)],
                                   textposition="outside"))
            fig.update_yaxes(type="log", title="Transactions (log scale)")
            fig.update_layout(title="Class imbalance", height=380, margin=dict(t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        with b:
            fig = px.pie(names=["Genuine", "Fraud"], values=[n - f, f], hole=.5,
                         color_discrete_sequence=["#27ae60", "#c0392b"],
                         title="Share of each class")
            fig.update_layout(height=380, margin=dict(t=50, b=10))
            st.plotly_chart(fig, width="stretch")

        st.subheader("Amount: genuine vs fraud")
        samp = df[["Amount", "Class"]].copy()
        samp["Class"] = samp["Class"].map({0: "Genuine", 1: "Fraud"})
        fig = px.histogram(samp[samp["Amount"] < 500], x="Amount", color="Class", nbins=60,
                           barmode="overlay", histnorm="percent", opacity=.65,
                           color_discrete_map={"Genuine": "#2ecc71", "Fraud": "#e74c3c"})
        fig.update_layout(height=340, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Preprocessing & split")
        ref, prod = prep["df_ref"], prep["df_prod"]
        st.dataframe(pd.DataFrame({
            "Set": ["Reference (60 %) - training", "Production (40 %) - never seen"],
            "Rows": [f"{len(ref):,}", f"{len(prod):,}"],
            "Frauds": [int(ref['Class'].sum()), int(prod['Class'].sum())]}),
            hide_index=True, width="stretch")
        st.caption(f"`Time` dropped · `Amount` → `Scaled_Amount` using RobustScaler (fit on "
                   f"reference only) · class-weight ratio **{prep['pos_weight']:.2f}** "
                   f"(= genuine / fraud in reference).")
        with st.expander("Peek at the raw data"):
            st.dataframe(df.head(20), width="stretch")

# ----------------------------------------------------------------------------
# TAB 3: Models
# ----------------------------------------------------------------------------
with tabs[2]:
    if need("bundle", "👈 Load the data and train the models from the sidebar first."):
        cfg_b, models, times = B["cfg"], B["models"], B["times"]
        w = fc.normalise_weights(cfg_b)
        specs = [
            ("P1", "Sub-Model 1", "Random Forest", "V1-V9 (9 features)",
             f"{cfg_b['rf_trees']} trees, depth {cfg_b['rf_depth']}, balanced",
             "Many shallow trees averaged → stable"),
            ("P2", "Sub-Model 2", "XGBoost", "V10-V19 (10 features)",
             f"{cfg_b['xgb_trees']} trees, depth {cfg_b['xgb_depth']}, "
             f"scale_pos_weight {B['prep']['pos_weight']:.0f}",
             "Each tree fixes previous errors → strongest"),
            ("P3", "Sub-Model 3", "Decision Tree", "V20-V28 + Scaled Amount (10)",
             f"depth {cfg_b['dt_depth']}, balanced",
             "One small tree → retrains very fast"),
        ]
        cols = st.columns(3)
        for col, (k, title, algo, feats, params, why), wi in zip(cols, specs, w):
            with col:
                st.markdown(f"### {title}")
                st.markdown(f"**{algo}** on **{feats}**")
                st.caption(params)
                st.caption(f"💡 {why}")
                x, y = st.columns(2)
                x.metric("Vote weight", f"{wi:.2f}")
                y.metric("Train time", f"{times[k]:.2f} s")

        st.subheader("Baseline performance (production data, before drift)")
        base = B["base"]
        st.dataframe(base.style.format("{:.4f}").highlight_max(axis=0, color="#14532d"),
                     width="stretch")
        plot = base.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
        fig = px.bar(plot, x="index", y="Score", color="Metric", barmode="group", text_auto=".2f",
                     color_discrete_sequence=["#8e44ad", "#16a085", "#f1c40f"])
        fig.update_layout(height=400, xaxis_title="", yaxis_range=[0, 1.1])
        st.plotly_chart(fig, width="stretch")

        st.subheader("What each model looks at (feature importance)")
        imp_cols = st.columns(3)
        for col, k in zip(imp_cols, ("P1", "P2", "P3")):
            imp = pd.Series(models[k].feature_importances_, index=fc.PARTITIONS[k]
                            ).sort_values().tail(6)
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                                   marker_color=COLORS[k]))
            fig.update_layout(title=f"{k} · {fc.MODEL_NAMES[k]}", height=300,
                              margin=dict(l=10, r=10, t=40, b=10))
            col.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------
# TAB 4: Predict
# ----------------------------------------------------------------------------
with tabs[3]:
    if need("bundle", "👈 Load the data and train the models from the sidebar first."):
        prep, models, cfg_b = B["prep"], B["models"], B["cfg"]
        w, thr = fc.normalise_weights(cfg_b), cfg_b["threshold"]
        mode = st.radio("Mode", ["Single transaction", "Batch (upload CSV)"], horizontal=True)

        if mode == "Single transaction":
            prod = prep["df_prod"]
            c1, c2, c3 = st.columns([2, 1, 1])
            kind = c1.radio("Pick a transaction from the unseen production set",
                            ["Random fraud", "Random genuine", "Specific row"], horizontal=True)
            if c2.button("🎲 New sample"):
                st.session_state["seed_pick"] = int(time.time())
            rng = np.random.default_rng(st.session_state.get("seed_pick", 1))
            if kind == "Specific row":
                idx = int(c3.number_input("Row index", 0, len(prod) - 1, 0))
            else:
                pool = prod.index[prod["Class"] == (1 if kind == "Random fraud" else 0)]
                idx = int(rng.choice(pool))
            row = prod.loc[idx]
            amount_raw = float(prep["scaler"].inverse_transform([[row["Scaled_Amount"]]])[0][0])

            st.caption(f"Row **{idx}** · true label: "
                       f"**{'FRAUD' if row['Class'] == 1 else 'Genuine'}** · "
                       f"Amount ≈ {amount_raw:,.2f}. Edit any value below to test what-ifs.")
            vals = {}
            ptabs = st.tabs(["P1 · V1-V9", "P2 · V10-V19", "P3 · V20-V28 + Amount"])
            for tab, k in zip(ptabs, ("P1", "P2", "P3")):
                with tab:
                    grid = st.columns(5)
                    for i, c in enumerate(fc.PARTITIONS[k]):
                        if c == "Scaled_Amount":
                            vals["Amount"] = grid[i % 5].number_input(
                                "Amount", 0.0, 1e6, round(amount_raw, 2), key=f"in_{idx}_amt")
                        else:
                            vals[c] = grid[i % 5].number_input(
                                c, value=round(float(row[c]), 4), format="%.4f",
                                key=f"in_{idx}_{c}")
            x = pd.DataFrame([{c: vals[c] for c in fc.RAW_REQUIRED}])
            x["Scaled_Amount"] = prep["scaler"].transform(x[["Amount"]])
            p1, p2, p3, fin, pred = fc.predict_ensemble(models, x, w, thr)

            g1, g2 = st.columns([1, 1])
            with g1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=float(fin[0]) * 100,
                    number={"suffix": " %"}, title={"text": "Fraud risk score"},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": "#e74c3c" if pred[0] else "#2ecc71"},
                           "threshold": {"line": {"color": "white", "width": 4},
                                         "value": thr * 100}}))
                fig.update_layout(height=320, margin=dict(t=60, b=10))
                st.plotly_chart(fig, width="stretch")
                if pred[0]:
                    st.error("🚨 **FRAUD ALERT** - block / review this transaction")
                else:
                    st.success("✅ **Genuine** - transaction cleared")
            with g2:
                st.markdown("**How the three sub-models voted**")
                tbl = pd.DataFrame({
                    "Sub-model": ["RF (P1)", "XGB (P2)", "DT (P3)"],
                    "Fraud prob.": [p1[0], p2[0], p3[0]], "Weight": w,
                    "Contribution": [p1[0] * w[0], p2[0] * w[1], p3[0] * w[2]]})
                fig = px.bar(tbl, x="Contribution", y="Sub-model", orientation="h",
                             color="Sub-model", text_auto=".3f",
                             color_discrete_sequence=[COLORS["P1"], COLORS["P2"], COLORS["P3"]])
                fig.update_layout(height=220, showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig, width="stretch")
                st.dataframe(tbl.style.format({"Fraud prob.": "{:.3f}", "Weight": "{:.2f}",
                                               "Contribution": "{:.3f}"}),
                             hide_index=True, width="stretch")
                st.caption(f"Final = {w[0]:.2f}·{p1[0]:.3f} + {w[1]:.2f}·{p2[0]:.3f} + "
                           f"{w[2]:.2f}·{p3[0]:.3f} = **{fin[0]:.3f}**  (threshold {thr:.2f})")
        else:
            st.markdown("Upload a CSV with columns `V1`…`V28` and `Amount` "
                        "(`Time` / `Class` optional). Template: same format as `creditcard.csv`.")
            up = st.file_uploader("Transactions to score", type=["csv"], key="batch")
            if up is not None:
                try:
                    bd = pd.read_csv(up)
                    miss = [c for c in fc.RAW_REQUIRED if c not in bd.columns]
                    if miss:
                        st.error(f"Missing columns: {miss}")
                    else:
                        bd = bd.copy()
                        bd["Scaled_Amount"] = prep["scaler"].transform(bd[["Amount"]])
                        p1, p2, p3, fin, pred = fc.predict_ensemble(models, bd, w, thr)
                        out = bd.copy()
                        out["p1_RF"], out["p2_XGB"], out["p3_DT"] = p1, p2, p3
                        out["fraud_score"], out["prediction"] = fin, pred
                        m1, m2 = st.columns(2)
                        m1.metric("Rows scored", f"{len(out):,}")
                        m2.metric("Flagged as fraud", f"{int(pred.sum()):,}")
                        if "Class" in bd.columns:
                            pr, rc, f1 = fc.get_scores(bd["Class"].astype(int), pred)
                            a, b, c = st.columns(3)
                            a.metric("Precision", f"{pr:.4f}")
                            b.metric("Recall", f"{rc:.4f}")
                            c.metric("F1", f"{f1:.4f}")
                        show = out.sort_values("fraud_score", ascending=False)
                        st.dataframe(show.head(200), width="stretch")
                        st.download_button("⬇️ Download scored CSV",
                                           out.to_csv(index=False).encode(),
                                           "scored_transactions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Could not score file: {e}")

# ----------------------------------------------------------------------------
# TAB 5: Drift Monitor
# ----------------------------------------------------------------------------
with tabs[4]:
    if need("bundle", "👈 Load the data and train the models from the sidebar first."):
        prep, cfg_b = B["prep"], B["cfg"]
        st.markdown("**PSI** compares each feature's distribution in old (reference) vs new "
                    "(production) data. `< 0.10` stable · `0.10-0.25` warning · "
                    "`≥ threshold` drift → retrain that sub-model.")
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
        parts = c1.multiselect("Partitions to drift", ["P1", "P2", "P3"], default=["P3"])
        scale = c2.number_input("Scale ×", 0.5, 5.0, 1.6, 0.1)
        shift = c3.number_input("Shift μ", -5.0, 5.0, 2.0, 0.5)
        noise = c4.number_input("Noise σ", 0.0, 3.0, 0.5, 0.1)
        st.caption("Drift formula:  x' = scale · x + N(shift, σ²)  (default = notebook: 1.6·x + N(2.0, 0.5²))")

        if st.button("📡 Inject drift & run PSI audit", type="primary"):
            with st.spinner("Injecting drift and computing PSI..."):
                drifted = fc.apply_drift(prep["df_prod"], parts, scale, shift, noise,
                                         cfg_b["seed"])
                audit = fc.audit_all(prep["df_ref"], drifted, cfg_b["psi_threshold"])
            st.session_state["drift"] = dict(drifted=drifted, audit=audit, parts=parts)
            st.session_state.pop("strat", None)

        if need("drift", "Press the button above to simulate drift and audit it."):
            audit = st.session_state["drift"]["audit"]
            thr_psi = cfg_b["psi_threshold"]
            cols = st.columns(3)
            labels = {"P1": "Partition 1 (V1-V9)", "P2": "Partition 2 (V10-V19)",
                      "P3": "Partition 3 (V20-V28 + Amt)"}
            for col, k in zip(cols, ("P1", "P2", "P3")):
                with col:
                    st.metric(labels[k], f"{audit[k]['psi']:.4f}")
                    st.markdown(pill(fc.psi_state(audit[k]["psi"], thr_psi)),
                                unsafe_allow_html=True)

            psis = [audit[k]["psi"] for k in ("P1", "P2", "P3")]
            fig = go.Figure(go.Bar(
                x=["P1 (V1-V9)", "P2 (V10-V19)", "P3 (V20-V28+Amt)"], y=psis,
                marker_color=["#e74c3c" if p >= thr_psi else "#27ae60" for p in psis],
                text=[f"{p:.4f}" for p in psis], textposition="outside"))
            fig.add_hline(y=thr_psi, line_dash="dash", line_color="white",
                          annotation_text=f"Drift threshold {thr_psi}")
            fig.update_layout(title="Partition-level drift audit", height=420,
                              yaxis_title="Average PSI")
            st.plotly_chart(fig, width="stretch")

            flagged = [k for k in audit if audit[k]["flag"]]
            if flagged:
                st.warning("🔔 Drift detected in " + ", ".join(flagged) +
                           " → only these sub-models will be retrained "
                           f"({', '.join(fc.MODEL_NAMES[k] for k in flagged)}).")
            else:
                st.success("No partition crossed the threshold - nothing to retrain.")

            with st.expander("Per-feature PSI"):
                rows = [(k, c, v) for k in audit for c, v in audit[k]["per_feature"].items()]
                pf = pd.DataFrame(rows, columns=["Partition", "Feature", "PSI"])
                fig = px.bar(pf, x="Feature", y="PSI", color="Partition",
                             color_discrete_map=COLORS)
                fig.update_xaxes(categoryorder="array", categoryarray=fc.ALL_COLS)
                st.plotly_chart(fig, width="stretch")

            with st.expander("Before vs after distribution of one feature"):
                feat = st.selectbox("Feature", fc.ALL_COLS, index=fc.ALL_COLS.index("V20"))
                comp = pd.concat([
                    pd.DataFrame({"value": prep["df_ref"][feat], "set": "Reference (old)"}),
                    pd.DataFrame({"value": st.session_state["drift"]["drifted"][feat],
                                  "set": "Production (new)"})])
                fig = px.histogram(comp, x="value", color="set", nbins=80, barmode="overlay",
                                   opacity=.6, histnorm="percent")
                fig.update_layout(height=350)
                st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------
# TAB 6: Strategies
# ----------------------------------------------------------------------------
with tabs[5]:
    if need("bundle", "👈 Load the data and train the models from the sidebar first."):
        if need("drift", "Run the drift audit in the **Drift Monitor** tab first."):
            D = st.session_state["drift"]
            st.markdown("""
| Strategy | What it does |
|---|---|
| **A - Static** | Do nothing, keep using the old ensemble |
| **B - Global** | Throw everything away, train one big XGBoost on all 29 features |
| **C - Selective (ours)** | Retrain **only** the sub-model(s) whose partition drifted |
""")
            if st.button("⚔️ Run the three strategies", type="primary"):
                with st.spinner("Running A, B and C on the unseen test half..."):
                    st.session_state["strat"] = fc.run_strategies(
                        B["models"], B["prep"], D["drifted"], D["audit"], B["cfg"])
            if need("strat", "Press the button to compare the strategies."):
                S = st.session_state["strat"]
                sp = S["split"]
                st.caption(f"Adaptation half: {sp['adapt_rows']:,} rows ({sp['adapt_frauds']} frauds) - "
                           f"allowed for retraining · Test half: {sp['test_rows']:,} rows "
                           f"({sp['test_frauds']} frauds) - used only for scoring.")
                res = fc.results_frame(S)
                st.dataframe(res.style.format("{:.4f}").highlight_max(
                    subset=["Precision", "Recall", "F1-Score", "ROC-AUC"], color="#14532d"),
                    width="stretch")

                m1, m2, m3 = st.columns(3)
                m1.metric("F1: Selective (C)", f"{S['C']['scores'][2]:.4f}",
                          f"{S['C']['scores'][2] - S['A']['scores'][2]:+.4f} vs static")
                m2.metric("Retrain time: C vs B",
                          f"{S['C']['time']:.2f}s vs {S['B']['time']:.2f}s")
                m3.metric("Time saved by selective", f"{S['saved_pct']:.1f} %")

                a, b = st.columns(2)
                with a:
                    fig = go.Figure(go.Bar(x=["Global (B)", "Selective (C)"],
                                           y=[S["B"]["time"], S["C"]["time"]],
                                           marker_color=["#e67e22", "#2980b9"],
                                           text=[f"{S['B']['time']:.2f}s", f"{S['C']['time']:.2f}s"],
                                           textposition="outside"))
                    fig.update_layout(title=f"Retraining time ({S['saved_pct']:.1f}% saved)",
                                      height=380, yaxis_title="seconds")
                    st.plotly_chart(fig, width="stretch")
                with b:
                    pl = res[["Precision", "Recall", "F1-Score"]].reset_index().melt(
                        id_vars="index", var_name="Metric", value_name="Score")
                    fig = px.bar(pl, x="Metric", y="Score", color="index", barmode="group",
                                 text_auto=".3f",
                                 color_discrete_sequence=["#95a5a6", "#f39c12", "#27ae60"])
                    fig.update_layout(title="Performance on unseen test data", height=380,
                                      yaxis_range=[0, 1.1], legend_title="")
                    st.plotly_chart(fig, width="stretch")

                st.subheader("Confusion matrices")
                cm_cols = st.columns(3)
                for col, k, nm in zip(cm_cols, "ABC", ("A: Static", "B: Global", "C: Selective")):
                    cm = S[k]["cm"]
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                    x=["Genuine", "Fraud"], y=["Genuine", "Fraud"],
                                    labels=dict(x="Predicted", y="True"))
                    fig.update_layout(title=nm, height=320, coloraxis_showscale=False,
                                      margin=dict(t=40, b=10))
                    col.plotly_chart(fig, width="stretch")
                    col.caption(f"Caught {cm[1, 1]} · missed {cm[1, 0]} · false alarms {cm[0, 1]}")

                st.info("⚠️ Honest note: test half has few frauds and timings are single-run, so "
                        "gaps are indicative. Part of the time saving is because C trains one "
                        "small tree while B trains a large XGBoost.")

# ----------------------------------------------------------------------------
# TAB 7: Live stream simulation
# ----------------------------------------------------------------------------
with tabs[6]:
    if need("bundle", "👈 Load the data and train the models from the sidebar first."):
        prep, models, cfg_b = B["prep"], B["models"], B["cfg"]
        st.markdown("Simulates the **event-driven** deployment: the production stream arrives in "
                    "windows, PSI is checked on every window, and an alert fires the moment a "
                    "partition drifts.")
        c1, c2, c3 = st.columns(3)
        n_win = c1.slider("Number of windows", 6, 20, 10)
        start = c2.slider("Drift starts at window", 2, n_win, min(6, n_win))
        live_parts = c3.multiselect("Drifting partitions", ["P1", "P2", "P3"], default=["P3"],
                                    key="live_parts")
        if st.button("▶️ Run live simulation", type="primary"):
            prod = prep["df_prod"].reset_index(drop=True)
            chunks = np.array_split(np.arange(len(prod)), n_win)
            w, thr = fc.normalise_weights(cfg_b), cfg_b["threshold"]
            log, prog = [], st.progress(0.0, text="Streaming...")
            for i, idx in enumerate(chunks, start=1):
                win = prod.iloc[idx].copy()
                if i >= start:
                    win = fc.apply_drift(win, live_parts, seed=cfg_b["seed"] + i)
                aud = fc.audit_all(prep["df_ref"], win, cfg_b["psi_threshold"])
                X, y = win.drop(columns=["Class"]), win["Class"].values
                *_, pred = fc.predict_ensemble(models, X, w, thr)
                f1 = fc.get_scores(y, pred)[2] if y.sum() > 0 else np.nan
                log.append(dict(Window=i, Rows=len(win), Frauds=int(y.sum()),
                                P1=aud["P1"]["psi"], P2=aud["P2"]["psi"], P3=aud["P3"]["psi"],
                                F1_static=f1,
                                Alert=", ".join(k for k in aud if aud[k]["flag"]) or "-"))
                prog.progress(i / n_win, text=f"Window {i}/{n_win}")
                time.sleep(0.15)
            prog.empty()
            st.session_state["live"] = pd.DataFrame(log)

        if need("live", "Press **Run live simulation**."):
            L = st.session_state["live"]
            fig = go.Figure()
            for k in ("P1", "P2", "P3"):
                fig.add_trace(go.Scatter(x=L["Window"], y=L[k], mode="lines+markers", name=k,
                                         line=dict(color=COLORS[k])))
            fig.add_hline(y=cfg_b["psi_threshold"], line_dash="dash", line_color="white",
                          annotation_text="Drift threshold")
            fig.update_yaxes(type="log", title="PSI (log scale)")
            fig.update_layout(title="PSI per partition over time", height=400)
            st.plotly_chart(fig, width="stretch")

            alerts = L[L["Alert"] != "-"]
            if len(alerts):
                a0 = alerts.iloc[0]
                st.error(f"🚨 First alert at **window {int(a0['Window'])}** → drift in "
                         f"**{a0['Alert']}** → trigger selective retraining of that sub-model only.")
            else:
                st.success("No drift alerts in this run.")
            st.dataframe(L.style.format({"P1": "{:.4f}", "P2": "{:.4f}", "P3": "{:.4f}",
                                         "F1_static": "{:.3f}"}),
                         hide_index=True, width="stretch")
            st.caption("F1 per window is noisy (each window holds only a handful of frauds).")

# ----------------------------------------------------------------------------
# TAB 8: Conclusion
# ----------------------------------------------------------------------------
with tabs[7]:
    st.subheader("What happened - final summary")
    S = st.session_state.get("strat")
    if S:
        a, c, bb = S["A"]["scores"][2], S["C"]["scores"][2], S["B"]["scores"][2]
        st.success(
            f"PSI isolated the drift to **{', '.join(S['C']['flagged']) or 'no partition'}**. "
            f"Selective retraining reached **F1 = {c:.4f}** "
            f"(static {a:.4f}, global {bb:.4f}) using **{S['saved_pct']:.1f}% less retraining "
            f"time** than global retraining.")
    st.markdown("""
1. The monitor **localised** the drift to the partition we corrupted (others stayed ≈ 0).
2. **Selective retraining** matched or beat both static and global strategies.
3. The other sub-models **stayed online** - no full-pipeline downtime.
4. **Impact:** cheaper, targeted MLOps for fraud pipelines.

**Limits & next steps**
- Drift is *simulated*; validate on real temporal data.
- Few frauds in the test half and single-run timings → repeat over several splits.
- Try drifting P2 (more informative) and tune the thresholds.
    """)
    st.caption("Tip: the 'Advanced settings' in the sidebar let you change weights, depths and "
               "thresholds live during the demo.")
