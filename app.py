"""AI ValuePulse: an AI ROI dashboard for the BharatMart Supply Co. case.

Run with:   streamlit run app.py
Put the CSV files in a folder called data/ next to this file.
"""

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AI ValuePulse", layout="wide")

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
IDS = ["DF", "DR", "PM", "CS", "RA"]
DEFAULT_NAMES = {
    "DF": "Demand forecasting",
    "DR": "Late-delivery risk",
    "PM": "Predictive maintenance",
    "CS": "Support copilot",
    "RA": "Reorder agent",
}
SCENARIOS = {
    "Base": "portfolio_base.csv",
    "CS adoption fixed": "portfolio_cs_fix.csv",
    "Budget cut 20%": "portfolio_cut.csv",
}
FALLBACK_PORTFOLIO = "portfolio_monthly.csv"

PORTFOLIO_COLS = [
    "initiative_id", "month", "cum_cost_inr", "cum_value_inr", "realised_roi",
    "adoption_share", "risk_adj_roi", "p10_roi", "p90_roi", "priority_score",
    "recommendation", "reason",
]
PORTFOLIO_NUMBERS = [
    "month", "cum_cost_inr", "cum_value_inr", "realised_roi", "adoption_share",
    "risk_adj_roi", "p10_roi", "p90_roi", "priority_score",
]
METRIC_COLS = ["initiative_id", "month", "metric", "value", "baseline", "higher_is_better"]

# Recommendation label colours: (text, background)
STATUS = {
    "Scale": ("#0f6b3a", "#dcf3e5"),
    "Continue": ("#17509e", "#e0ecfa"),
    "Pivot": ("#8a5300", "#fdefc9"),
    "Stop": ("#9c1f1f", "#fbe0e0"),
}
LINE_COLORS = {
    "DF": "#0f766e", "DR": "#7c3aed", "PM": "#475569", "CS": "#be185d", "RA": "#0369a1",
}
BASELINE_COLOR = "#9ca3af"

# The one metric that best summarises each initiative's model
MAIN_METRIC = {
    "DF": "wape", "DR": "recall", "PM": "recall",
    "CS": "resolution_rate", "RA": "autonomous_rate",
}
METRIC_LABELS = {
    "wape": "Forecast error (WAPE)",
    "recall": "Recall",
    "precision": "Precision",
    "late_orders_caught": "Late orders caught",
    "false_alarm_rate": "False alarm rate",
    "failures_caught": "Failures caught",
    "drift_score": "Drift score",
    "resolution_rate": "Resolution rate",
    "hallucination_rate": "Hallucination rate",
    "human_intervention_rate": "Human intervention rate",
    "cost_per_interaction_inr": "Cost per interaction",
    "avg_tokens": "Average tokens per reply",
    "autonomous_rate": "Autonomous rate",
    "exception_rate": "Exception rate",
    "cost_per_txn_inr": "Cost per order",
}
PERCENT_METRICS = {
    "wape", "recall", "precision", "false_alarm_rate", "resolution_rate",
    "hallucination_rate", "human_intervention_rate", "autonomous_rate", "exception_rate",
}
RUPEE_METRICS = {"cost_per_interaction_inr", "cost_per_txn_inr"}


# ----------------------------------------------------------------------------
# Formatting helpers (Indian style: lakh and crore)
# ----------------------------------------------------------------------------
def indian_group(n):
    """1234567 -> '12,34,567' (Indian digit grouping)."""
    s = str(abs(int(n)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if n < 0 else "") + s


def inr(x):
    """Rupees in lakh or crore, chosen by size."""
    if pd.isna(x):
        return "n/a"
    sign, a = ("-" if x < 0 else ""), abs(x)
    if a >= 1e7:
        return f"{sign}Rs {a / 1e7:,.2f} Cr"
    if a >= 1e5:
        return f"{sign}Rs {a / 1e5:,.2f} lakh"
    return f"{sign}Rs {indian_group(round(a))}"


def crore(x):
    """Rupees always in crore."""
    if pd.isna(x):
        return "n/a"
    return f"{'-' if x < 0 else ''}Rs {abs(x) / 1e7:,.2f} Cr"


def pct(x, digits=0):
    return "n/a" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def metric_label(name):
    return METRIC_LABELS.get(name, name.replace("_", " ").capitalize())


def metric_kind(name):
    if name in PERCENT_METRICS or name.endswith("_rate") or name.endswith("_share"):
        return "pct"
    if name in RUPEE_METRICS or name.endswith("_inr"):
        return "inr"
    return "num"


def fmt_metric(v, kind):
    if pd.isna(v):
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "inr":
        return f"Rs {v:,.3f}" if abs(v) < 10 else f"Rs {indian_group(round(v))}"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


def fmt_delta(v, base, kind):
    d = v - base
    if kind == "pct":
        return f"{d * 100:+.1f} pts"
    return f"{d:+,.2f}"


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def pill(rec):
    fg, bg = STATUS.get(rec, ("#444444", "#eeeeee"))
    return f'<span class="vp-pill" style="color:{fg};background:{bg}">{escape(str(rec))}</span>'


# ----------------------------------------------------------------------------
# Loading data (never crashes: problems are collected and shown as messages)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_csv(path, mtime):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def load_csv(name, required):
    """Return (dataframe, None) or (None, friendly problem text)."""
    path = DATA_DIR / name
    if not path.exists():
        return None, f"{name} is missing."
    try:
        df = _read_csv(str(path), path.stat().st_mtime)
    except Exception as exc:  # unreadable or malformed file
        return None, f"{name} could not be read ({exc.__class__.__name__})."
    absent = [c for c in required if c not in df.columns]
    if absent:
        return None, f"{name} is missing these columns: {', '.join(absent)}."
    if df.empty:
        return None, f"{name} has no rows."
    return df.copy(), None


def load_portfolio(name):
    df, problem = load_csv(name, PORTFOLIO_COLS)
    if df is None:
        return None, problem
    df["initiative_id"] = df["initiative_id"].astype(str).str.strip()
    df["recommendation"] = df["recommendation"].astype(str).str.strip().str.capitalize()
    df["reason"] = df["reason"].fillna("").astype(str)
    for c in PORTFOLIO_NUMBERS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, None


def load_metrics(initiative_id):
    df, problem = load_csv(f"metrics_{initiative_id}.csv", METRIC_COLS)
    if df is None:
        return None, problem
    df["metric"] = df["metric"].astype(str).str.strip()
    df["higher_is_better"] = df["higher_is_better"].astype(str).str.strip().str.lower()
    for c in ("month", "value", "baseline"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, None


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """<style>
.vp-pill{display:inline-block;padding:2px 12px;border-radius:999px;font-weight:600;font-size:0.85rem;white-space:nowrap}
.vp-table{width:100%;border-collapse:collapse;font-size:0.95rem}
.vp-table th{text-align:left;font-weight:600;padding:8px 10px;border-bottom:2px solid rgba(128,128,128,.4)}
.vp-table td{vertical-align:top;padding:10px;border-bottom:1px solid rgba(128,128,128,.2)}
.vp-muted{opacity:.65;font-size:.85rem}
</style>""",
        unsafe_allow_html=True,
    )


def tile(col, label, value, caption=None, delta=None, delta_color="normal"):
    with col.container(border=True):
        st.metric(label, value, delta=delta, delta_color=delta_color)
        if caption:
            st.caption(caption)


def base_layout(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Month", dtick=1, range=[0.5, 12.5])
    return fig


# ----------------------------------------------------------------------------
# Page 1: Executive dashboard
# ----------------------------------------------------------------------------
def page_executive(pf, month, names, stages, scenario):
    st.title("Executive dashboard")
    st.caption(f"Month {month} of 12. Scenario: {scenario}.")

    snap = pf[pf["month"] == month]
    if snap.empty:
        st.info(f"The portfolio file has no rows for month {month}.")
        return

    cost, value = snap["cum_cost_inr"].sum(), snap["cum_value_inr"].sum()
    roi = (value - cost) / cost if cost else float("nan")
    n_scale = int((snap["recommendation"] == "Scale").sum())
    n_action = int(snap["recommendation"].isin(["Stop", "Pivot"]).sum())

    cols = st.columns(5)
    tile(cols[0], "Total investment", crore(cost))
    tile(cols[1], "Realised value", crore(value))
    tile(cols[2], "Portfolio ROI", pct(roi))
    tile(cols[3], "Initiatives to scale", str(n_scale))
    tile(cols[4], "To stop or pivot", str(n_action))

    st.subheader("Where each initiative stands")
    rows = []
    for _, r in snap.sort_values("priority_score", ascending=False).iterrows():
        iid = r["initiative_id"]
        priority = "n/a" if pd.isna(r["priority_score"]) else f"{r['priority_score']:.0f}"
        rows.append(
            "<tr>"
            f"<td><strong>{escape(names.get(iid, iid))}</strong><br><span class='vp-muted'>{escape(iid)}</span></td>"
            f"<td>{escape(stages.get(iid, ''))}</td>"
            f"<td>{inr(r['cum_cost_inr'])}</td>"
            f"<td>{pct(r['realised_roi'])}</td>"
            f"<td>{priority}</td>"
            f"<td>{pill(r['recommendation'])}</td>"
            f"<td>{escape(r['reason'])}</td>"
            "</tr>"
        )
    st.markdown(
        "<table class='vp-table'><thead><tr>"
        "<th>Initiative</th><th>Stage</th><th>Investment</th><th>Realised ROI</th>"
        "<th>Priority score</th><th>Recommendation</th><th>Reason</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
        unsafe_allow_html=True,
    )

    st.subheader("Realised ROI by month")
    fig = go.Figure()
    for iid in [i for i in IDS if i in set(pf["initiative_id"])] or sorted(pf["initiative_id"].unique()):
        s = pf[pf["initiative_id"] == iid].sort_values("month")
        color = LINE_COLORS.get(iid, "#334155")
        label = names.get(iid, iid)
        fig.add_trace(go.Scatter(
            x=s["month"], y=s["realised_roi"] * 100, mode="lines", name=label,
            line=dict(color=color, width=2.5), legendgroup=iid,
            hovertemplate="%{y:.0f}%<extra>" + label + "</extra>",
        ))
        at = s[s["month"] == month]
        fig.add_trace(go.Scatter(
            x=at["month"], y=at["realised_roi"] * 100, mode="markers",
            marker=dict(color=color, size=11, line=dict(color="white", width=2)),
            legendgroup=iid, showlegend=False, hoverinfo="skip",
        ))
    fig.add_hline(y=0, line_color=BASELINE_COLOR, line_width=1)
    fig.add_vline(x=month, line_dash="dot", line_color=BASELINE_COLOR)
    base_layout(fig, 380)
    fig.update_yaxes(title_text="Realised ROI", ticksuffix="%")
    st.plotly_chart(fig, width="stretch", key="exec_roi")


# ----------------------------------------------------------------------------
# Page 2: Initiative scorecard
# ----------------------------------------------------------------------------
def roi_outlook_chart(series, month, color):
    m = series["month"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m, y=series["p90_roi"] * 100, mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=m, y=series["p10_roi"] * 100, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=hex_to_rgba(color, 0.15),
                             name="P10 to P90 range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=m, y=series["realised_roi"] * 100, mode="lines", name="Realised ROI",
                             line=dict(color=color, width=2.5), hovertemplate="%{y:.0f}%<extra>Realised</extra>"))
    fig.add_trace(go.Scatter(x=m, y=series["risk_adj_roi"] * 100, mode="lines", name="Risk-adjusted ROI",
                             line=dict(color=color, width=2, dash="dash"),
                             hovertemplate="%{y:.0f}%<extra>Risk-adjusted</extra>"))
    at = series[series["month"] == month]
    fig.add_trace(go.Scatter(x=at["month"], y=at["realised_roi"] * 100, mode="markers",
                             marker=dict(color=color, size=11, line=dict(color="white", width=2)),
                             showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=0, line_color=BASELINE_COLOR, line_width=1)
    fig.add_vline(x=month, line_dash="dot", line_color=BASELINE_COLOR)
    base_layout(fig, 330)
    fig.update_yaxes(title_text="ROI", ticksuffix="%")
    return fig


def metric_chart(sub, month, color, kind):
    fmt = {"pct": ":.1%", "inr": ":,.3f", "num": ":,.2f"}[kind]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["month"], y=sub["value"], mode="lines+markers", name="Value",
        line=dict(color=color, width=2.5), marker=dict(size=5),
        hovertemplate="%{y" + fmt + "}<extra>Value</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sub["month"], y=sub["baseline"], mode="lines", name="Baseline",
        line=dict(color=BASELINE_COLOR, width=2, dash="dash"),
        hovertemplate="%{y" + fmt + "}<extra>Baseline</extra>",
    ))
    at = sub[sub["month"] == month]
    fig.add_trace(go.Scatter(x=at["month"], y=at["value"], mode="markers",
                             marker=dict(color=color, size=11, line=dict(color="white", width=2)),
                             showlegend=False, hoverinfo="skip"))
    fig.add_vline(x=month, line_dash="dot", line_color=BASELINE_COLOR)
    base_layout(fig, 300)
    if kind == "pct":
        fig.update_yaxes(tickformat=".0%")
    elif kind == "inr":
        fig.update_yaxes(tickprefix="Rs ", tickformat=",.3f")
    else:
        fig.update_yaxes(tickformat=",.2~f")
    return fig


def page_scorecard(pf, metrics, month, names, info):
    st.title("Initiative scorecard")
    ids = [i for i in IDS if i in set(pf["initiative_id"])] or sorted(pf["initiative_id"].unique())
    iid = st.selectbox("Initiative", ids, format_func=lambda i: f"{i}: {names.get(i, i)}")
    color = LINE_COLORS.get(iid, "#334155")

    detail = []
    if info.get(iid, {}).get("stage"):
        detail.append(f"{info[iid]['stage']} stage")
    if info.get(iid, {}).get("ai_type"):
        detail.append(f"{info[iid]['ai_type']} AI")
    st.caption(". ".join(detail + [f"Month {month} of 12"]) + ".")

    series = pf[pf["initiative_id"] == iid].sort_values("month")
    cur = series[series["month"] == month]
    if cur.empty:
        st.info(f"There is no data for {iid} in month {month}.")
        return
    r = cur.iloc[0]

    budget = info.get(iid, {}).get("budget_inr")
    spent_note = None
    if budget and not pd.isna(budget) and budget > 0:
        spent_note = f"{pct(r['cum_cost_inr'] / budget)} of the {inr(budget)} budget"

    row1 = st.columns(4)
    tile(row1[0], "Investment so far", inr(r["cum_cost_inr"]), spent_note)
    tile(row1[1], "Realised benefit", inr(r["cum_value_inr"]))
    tile(row1[2], "Current ROI", pct(r["realised_roi"]))
    tile(row1[3], "Adoption", pct(r["adoption_share"]))

    mdf = metrics.get(iid)
    row2 = st.columns([1, 1, 1.4])
    tile(row2[0], "Risk-adjusted forecast ROI", pct(r["risk_adj_roi"]),
         f"P10 {pct(r['p10_roi'])} to P90 {pct(r['p90_roi'])}")

    if mdf is not None:
        names_in_file = list(dict.fromkeys(mdf["metric"]))
        main = MAIN_METRIC.get(iid) if MAIN_METRIC.get(iid) in names_in_file else names_in_file[0]
        mrow = mdf[(mdf["metric"] == main) & (mdf["month"] == month)]
        if mrow.empty:
            tile(row2[1], metric_label(main), "n/a", "No value for this month")
        else:
            mr = mrow.iloc[0]
            kind = metric_kind(main)
            better = "normal" if mr["higher_is_better"] == "yes" else "inverse"
            base_text = f"Baseline {fmt_metric(mr['baseline'], kind)}. "
            base_text += "Higher is better." if mr["higher_is_better"] == "yes" else "Lower is better."
            has_base = not pd.isna(mr["baseline"]) and mr["baseline"] != 0
            tile(row2[1], metric_label(main), fmt_metric(mr["value"], kind), base_text,
                 delta=fmt_delta(mr["value"], mr["baseline"], kind) if has_base else None,
                 delta_color=better)
    else:
        tile(row2[1], "Main model metric", "n/a", f"metrics_{iid}.csv is missing")

    with row2[2].container(border=True):
        st.markdown("**Recommendation**")
        st.markdown(pill(r["recommendation"]) + f"<p style='margin-top:.6rem'>{escape(r['reason'])}</p>",
                    unsafe_allow_html=True)

    st.subheader("Return outlook")
    st.plotly_chart(roi_outlook_chart(series, month, color), width="stretch", key=f"outlook_{iid}")

    st.subheader("Metrics against baseline")
    if mdf is None:
        st.info(f"metrics_{iid}.csv is missing, so the metric charts cannot be shown. "
                "Add the file to the data folder and refresh.")
        return
    names_in_file = list(dict.fromkeys(mdf["metric"]))
    for start in range(0, len(names_in_file), 2):
        cols = st.columns(2)
        for col, mname in zip(cols, names_in_file[start:start + 2]):
            sub = mdf[mdf["metric"] == mname].sort_values("month")
            direction = "Higher is better" if (sub["higher_is_better"] == "yes").all() else "Lower is better"
            with col:
                st.markdown(f"**{metric_label(mname)}**")
                st.caption(direction)
                st.plotly_chart(metric_chart(sub, month, color, metric_kind(mname)),
                                width="stretch", key=f"m_{iid}_{mname}")


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main():
    inject_css()

    st.sidebar.title("AI ValuePulse")
    page = st.sidebar.radio("Page", ["Executive dashboard", "Initiative scorecard"])
    month = st.sidebar.slider("Month", 1, 12, 12)
    scenario = st.sidebar.selectbox("Scenario", list(SCENARIOS))

    if not DATA_DIR.exists():
        st.title("AI ValuePulse")
        st.info("The data folder was not found. Create a folder named data next to app.py "
                "and put the CSV files in it, then refresh this page.")
        st.stop()

    wanted = SCENARIOS[scenario]
    portfolio_name = wanted if (DATA_DIR / wanted).exists() else FALLBACK_PORTFOLIO
    if portfolio_name != wanted and scenario != "Base":
        st.sidebar.warning(f"{wanted} was not found, so this view shows {FALLBACK_PORTFOLIO}.")
    st.sidebar.caption(f"Portfolio file: {portfolio_name}")

    problems = []
    pf, problem = load_portfolio(portfolio_name)
    if problem:
        problems.append(problem)

    ini, problem = load_csv("initiatives.csv", ["initiative_id", "name"])
    if problem:
        problems.append(problem + " Default initiative names are used.")
    names, stages, info = dict(DEFAULT_NAMES), {}, {}
    if ini is not None:
        ini["initiative_id"] = ini["initiative_id"].astype(str).str.strip()
        for _, row in ini.iterrows():
            iid = row["initiative_id"]
            names[iid] = str(row["name"])
            stage = str(row["stage"]) if "stage" in ini.columns and not pd.isna(row["stage"]) else ""
            ai_type = str(row["ai_type"]) if "ai_type" in ini.columns and not pd.isna(row["ai_type"]) else ""
            budget = pd.to_numeric(row["budget_inr"], errors="coerce") if "budget_inr" in ini.columns else None
            stages[iid] = stage
            info[iid] = {"stage": stage, "ai_type": ai_type.capitalize(), "budget_inr": budget}

    metrics = {}
    for iid in IDS:
        mdf, problem = load_metrics(iid)
        if problem:
            problems.append(problem)
        else:
            metrics[iid] = mdf

    if problems:
        st.warning("Some data files need attention. The app shows what it can.\n\n"
                   + "\n".join(f"- {p}" for p in problems)
                   + "\n\nFiles belong in the data folder next to app.py.")

    if pf is None:
        st.info("This page needs the portfolio file. Add it to the data folder and refresh.")
        st.stop()

    if page == "Executive dashboard":
        page_executive(pf, month, names, stages, scenario)
    else:
        page_scorecard(pf, metrics, month, names, info)


main()
