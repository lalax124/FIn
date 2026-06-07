import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go

from database import (
    initialize_database, get_or_create_user,
    save_income, save_expenses, save_assets, save_liabilities,
    save_financial_goals, save_investment_portfolio, save_ai_insight,
    get_user_income, get_user_expenses, get_user_assets, get_user_liabilities,
    get_user_financial_goals, get_user_portfolio, get_user_insights,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Artha — AI Financial Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────────
def apply_custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
        color: #e8eaed;
    }
    .gradient-text {
        background: linear-gradient(90deg, #4169E1, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 14px 18px;
        transition: border-color 0.2s;
    }
    [data-testid="metric-container"]:hover { border-color: #4169E1; }

    /* ── CHAT BUBBLES ── */
    .chat-wrap { display: flex; flex-direction: column; gap: 18px; padding: 8px 0; }

    .bubble-user-row {
        display: flex; justify-content: flex-end;
        align-items: flex-end; gap: 10px;
    }
    .bubble-user {
        background: linear-gradient(135deg, #4169E1, #5a7ff5);
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 20px 20px 4px 20px;
        max-width: 68%; font-size: 14px; line-height: 1.6;
        box-shadow: 0 4px 20px rgba(65,105,225,0.35);
        word-break: break-word;
    }
    .bubble-user-avatar {
        width: 32px; height: 32px; border-radius: 50%;
        background: linear-gradient(135deg, #4169E1, #8A2BE2);
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; flex-shrink: 0;
    }
    .bubble-bot-row {
        display: flex; justify-content: flex-start;
        align-items: flex-start; gap: 10px;
    }
    .bubble-bot-avatar {
        width: 32px; height: 32px; border-radius: 50%;
        background: linear-gradient(135deg, #8A2BE2, #4169E1);
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; flex-shrink: 0; margin-top: 2px;
    }
    .bubble-bot {
        background: #1e2130; border: 1px solid #2d3348;
        color: #e8eaed; padding: 14px 18px;
        border-radius: 4px 20px 20px 20px;
        max-width: 72%; font-size: 14px; line-height: 1.7;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        word-break: break-word;
    }
    .bubble-bot-name {
        font-size: 11px; color: #8A2BE2; font-weight: 600;
        letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 6px;
    }
    .bubble-ts { font-size: 10px; color: #555d70; margin-top: 4px; padding: 0 4px; }
    .bubble-ts.right { text-align: right; }

    .stButton > button {
        background: #1e2130 !important; border: 1px solid #30363d !important;
        color: #c9d1d9 !important; border-radius: 20px !important;
        font-size: 13px !important; padding: 6px 14px !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        border-color: #4169E1 !important; color: #fff !important;
        background: #161b22 !important;
    }
    .ai-card {
        background: #1e2130; border-left: 3px solid #4169E1;
        border-radius: 0 12px 12px 0; padding: 16px 20px; margin: 12px 0;
        font-size: 14px; line-height: 1.75; color: #e8eaed;
    }
    .ai-card-label {
        font-size: 10px; font-weight: 700; color: #4169E1;
        letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;
    }
    .predict-badge {
        display: inline-block;
        background: #1e2130; border: 1px solid #4169E1;
        color: #e8eaed; border-radius: 20px;
        padding: 4px 14px; font-size: 13px; font-weight: 500;
        margin: 4px 4px 4px 0;
    }
    hr { border-color: #21262d !important; }
    </style>
    """, unsafe_allow_html=True)


# ── DATABASE INIT ─────────────────────────────────────────────────────────────
initialize_database()
apply_custom_style()

# ── AUTO-TRAIN CLASSIFIER ON FIRST LAUNCH ────────────────────────────────────
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artha_classifier.pkl")
if not os.path.exists(MODEL_FILE):
    with st.spinner("🧠 Training expense classifier for the first time... (~15 seconds)"):
        from expense_classifier import train_and_evaluate
        train_and_evaluate()

# ── IMPORTS ───────────────────────────────────────────────────────────────────
from groq_ai import generate_financial_adivice, investement_advise, analyze_budget
from expense_classifier import predict_category, load_model
from data_processing import (
    format_currency, calculate_budget_summary,
    calculate_investment_returns, calculate_loan_payment,
    generate_amortization_schedule, categorize_expenses,
)
from frontend import (
    create_expense_pie_chart, create_income_expense_bar_chart,
    create_investment_growth_chart, create_expense_trend_chart,
    create_savings_goal_progress_chart,
    create_portfolio_allocation_chart, create_portfolio_pnl_chart,
    create_net_worth_gauge,
)
from moneyanalyser import (
    calculate_net_worth, calculate_emergency_fund_ratio,
    calculate_portfolio_metrics, calculate_debt_to_income_ratio,
    analyze_retirement_readiness, analyze_mortgage_affordability,
    get_stock_metrics,
)

# Load classifier once into session state
if "classifier" not in st.session_state:
    st.session_state.classifier = load_model()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "income": 0.0, "expenses": {}, "assets": {}, "liabilities": {},
    "financial_goals": [], "chat_history": [], "portfolio": [],
    "user_id": None, "username": None, "authenticated": False,
    "watchlist": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="margin-bottom:2px"><span class="gradient-text">Artha</span> '
    '<span style="color:#555;font-weight:300;font-size:0.85em">/ AI Financial Assistant</span></h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#666;font-size:0.95em;margin-top:0">'
    'Track · Invest · Analyse · Chat — powered by Groq LLaMA 3 & FMP API</p>',
    unsafe_allow_html=True,
)

# ── AUTH ──────────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    st.sidebar.title("🔐 Authentication")
    auth_option = st.sidebar.radio("", ["Login", "Register"], horizontal=True)
    username = st.sidebar.text_input("Username")
    email    = st.sidebar.text_input("Email (optional)") if auth_option == "Register" else None

    if st.sidebar.button(auth_option, type="primary", use_container_width=True):
        if username.strip():
            uid = get_or_create_user(username.strip(), email)
            st.session_state.update({
                "user_id": uid, "username": username.strip(), "authenticated": True,
                "income":         get_user_income(uid),
                "expenses":       get_user_expenses(uid),
                "assets":         get_user_assets(uid),
                "liabilities":    get_user_liabilities(uid),
                "financial_goals":get_user_financial_goals(uid),
                "portfolio":      get_user_portfolio(uid),
            })
            st.rerun()
        else:
            st.sidebar.error("Please enter a username")

    st.markdown("""
    <div style="text-align:center;padding:60px 20px">
      <div style="font-size:64px">💰</div>
      <h2 class="gradient-text">Welcome to Artha</h2>
      <p style="color:#666;max-width:420px;margin:0 auto">
        Track income &amp; expenses · Monitor net worth<br>
        Invest in real stocks · Auto-classify transactions<br>
        Get AI-powered financial advice
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── SIDEBAR NAV ───────────────────────────────────────────────────────────────
st.sidebar.markdown(f"### 👋 {st.session_state.username}")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "💰 Budget Analyzer", "📈 Stocks & Portfolio", "🤖 AI Advisor"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

if st.sidebar.button("💾 Save Data", use_container_width=True):
    save_income(st.session_state.user_id, st.session_state.income)
    save_expenses(st.session_state.user_id, st.session_state.expenses)
    save_assets(st.session_state.user_id, st.session_state.assets)
    save_liabilities(st.session_state.user_id, st.session_state.liabilities)
    if st.session_state.financial_goals:
        save_financial_goals(st.session_state.user_id, st.session_state.financial_goals)
    if st.session_state.portfolio:
        save_investment_portfolio(st.session_state.user_id, st.session_state.portfolio)
    st.sidebar.success("✅ Saved!")

if st.sidebar.button("🚪 Logout", use_container_width=True):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def ai_card(label, text):
    safe = str(text).replace("\n", "<br>").replace("**", "")
    st.markdown(
        f'<div class="ai-card"><div class="ai-card-label">🤖 {label}</div>{safe}</div>',
        unsafe_allow_html=True,
    )


def chat_bubble_user(msg, ts):
    st.markdown(f"""
    <div class="bubble-user-row">
      <div>
        <div class="bubble-user">{msg}</div>
        <div class="bubble-ts right">{ts}</div>
      </div>
      <div class="bubble-user-avatar">👤</div>
    </div>""", unsafe_allow_html=True)


def chat_bubble_bot(msg, ts):
    safe = str(msg).replace("\n", "<br>")
    st.markdown(f"""
    <div class="bubble-bot-row">
      <div class="bubble-bot-avatar">💰</div>
      <div>
        <div class="bubble-bot">
          <div class="bubble-bot-name">Artha</div>
          {safe}
        </div>
        <div class="bubble-ts">{ts}</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.header("📊 Dashboard")

    net_worth, total_assets, total_liabilities = calculate_net_worth(
        st.session_state.assets, st.session_state.liabilities
    )
    total_expenses = sum(st.session_state.expenses.values()) if st.session_state.expenses else 0.0
    savings = st.session_state.income - total_expenses
    sr      = (savings / st.session_state.income * 100) if st.session_state.income > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Net Worth",        format_currency(net_worth))
    with k2: st.metric("Monthly Income",   format_currency(st.session_state.income))
    with k3: st.metric("Monthly Expenses", format_currency(total_expenses))
    with k4: st.metric("Savings Rate",     f"{sr:.1f}%",
                        delta="Good ✅" if sr >= 20 else "Improve ⚠️")

    st.markdown("---")
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.subheader("Net Worth Gauge")
        st.plotly_chart(create_net_worth_gauge(net_worth), use_container_width=True)
    with c2:
        if st.session_state.expenses:
            st.subheader("Expense Breakdown")
            st.plotly_chart(create_expense_pie_chart(st.session_state.expenses), use_container_width=True)
        else:
            st.info("Add expenses in Budget Analyzer to see breakdown.")

    if st.session_state.expenses and "Emergency Fund" in st.session_state.assets:
        ef     = st.session_state.assets.get("Emergency Fund", 0)
        months = calculate_emergency_fund_ratio(ef, total_expenses)
        if months:
            st.markdown("---")
            st.markdown(f"**🛡️ Emergency Fund** — covers **{months:.1f} months** of expenses")
            st.progress(int(min(months / 6 * 100, 100)))
            if months < 3:   st.warning("Critical: aim for at least 3 months.")
            elif months < 6: st.warning("Good start — try to reach 6 months.")
            else:            st.success("✅ Emergency fund is solid!")

    if st.session_state.portfolio:
        st.markdown("---")
        st.subheader("Portfolio Snapshot")
        with st.spinner("Loading live prices..."):
            pm = calculate_portfolio_metrics(st.session_state.portfolio)
        if "error" not in pm and pm["holdings_data"]:
            p1, p2, p3 = st.columns(3)
            with p1: st.metric("Portfolio Value", format_currency(pm["total_value"]))
            with p2: st.metric("Total Cost",      format_currency(pm["total_cost"]))
            with p3:
                dc = "normal" if pm["total_gain_loss"] >= 0 else "inverse"
                st.metric("Unrealised P&L", format_currency(pm["total_gain_loss"]),
                          delta=f"{pm['total_gain_loss_pct']:+.2f}%", delta_color=dc)
            pa1, pa2 = st.columns(2)
            with pa1: st.plotly_chart(create_portfolio_allocation_chart(pm["holdings_data"]), use_container_width=True)
            with pa2: st.plotly_chart(create_portfolio_pnl_chart(pm["holdings_data"]),        use_container_width=True)

    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ══════════════════════════════════════════════════════════════════════════════
#  BUDGET ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Budget Analyzer":
    st.header("💰 Budget Analyzer")

    income = st.number_input("Monthly Income (₹)", min_value=0.0,
                             value=float(st.session_state.income),
                             step=1000.0, format="%.2f")
    st.session_state.income = income

    # ── Smart transaction classifier ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧠 Smart Transaction Classifier")
    st.markdown(
        '<p style="color:#666;font-size:13px">Type any transaction and Artha will auto-predict its category '
        '(96.87% F1 Score)</p>', unsafe_allow_html=True
    )

    txn_col, btn_col = st.columns([4, 1])
    with txn_col:
        txn_desc = st.text_input(
            "", placeholder="e.g.  Zomato biryani   /   Airtel bill   /   BigBasket order",
            label_visibility="collapsed", key="txn_input"
        )
    with btn_col:
        classify_btn = st.button("Classify →", use_container_width=True)

    if classify_btn and txn_desc.strip():
        result = predict_category(txn_desc.strip(), st.session_state.classifier)
        cat    = result["category"]
        conf   = result["confidence"]
        top3   = result["top3"]

        conf_color = "#00C896" if conf >= 0.7 else "#FFA502" if conf >= 0.5 else "#FF4757"
        st.markdown(f"""
        <div style="background:#1e2130;border:1px solid #30363d;border-radius:12px;padding:16px 20px;margin:8px 0">
          <div style="font-size:11px;color:#666;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px">
            Prediction for: <b style="color:#e8eaed">{txn_desc}</b>
          </div>
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
            <span style="font-size:22px;font-weight:700;color:#4169E1">{cat}</span>
            <span style="font-size:14px;color:{conf_color};font-weight:600">{conf:.0%} confidence</span>
          </div>
          <div style="margin-top:10px;font-size:12px;color:#666">
            Also consider:
            {''.join([f'<span class="predict-badge">{c} <span style="color:#4169E1">{p:.0%}</span></span>'
                      for c, p in top3[1:]])}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Offer to add to expenses
        if st.button(f"➕ Add ₹ amount to '{cat}'"):
            st.session_state["pending_category"] = cat

    if st.session_state.get("pending_category"):
        pcat = st.session_state["pending_category"]
        amt  = st.number_input(f"Amount for '{pcat}' (₹)", min_value=0.0, step=50.0, format="%.2f")
        if st.button("Confirm Add"):
            existing = st.session_state.expenses.get(pcat, 0.0)
            st.session_state.expenses[pcat] = existing + amt
            st.session_state["pending_category"] = None
            st.success(f"Added ₹{amt:,.0f} to {pcat}")
            st.rerun()

    st.markdown("---")
    st.subheader("Monthly Expenses")

    expense_categories = [
        "Housing (Rent/Mortgage)", "Utilities", "Groceries", "Transportation",
        "Health Care", "Entertainment", "Dining Out", "Shopping",
        "Education", "Insurance", "Savings", "Other",
    ]

    updated = {}
    cl, cr  = st.columns(2)
    for i, cat in enumerate(expense_categories):
        col = cl if i < 6 else cr
        with col:
            v = st.number_input(
                f"{cat} (₹)", min_value=0.0,
                value=float(st.session_state.expenses.get(cat, 0.0)),
                step=100.0, format="%.2f", key=f"exp_{cat}",
            )
            if v > 0:
                updated[cat] = v

    st.markdown("---")
    custom = st.text_input("➕ Custom expense category")
    if custom:
        cv = st.number_input(f"{custom} (₹)", min_value=0.0,
                             value=float(st.session_state.expenses.get(custom, 0.0)),
                             step=100.0, format="%.2f")
        if cv > 0:
            updated[custom] = cv

    st.session_state.expenses = updated

    bs        = calculate_budget_summary(income, updated)
    total_exp = bs["total_expenses"]
    remaining = bs["remaining"]
    sr        = bs["savings_rate"]

    st.markdown("---")
    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Income",       format_currency(income))
    with m2: st.metric("Expenses",     format_currency(total_exp))
    with m3: st.metric("Remaining",    format_currency(remaining),
                        delta="surplus" if remaining >= 0 else "deficit")
    with m4: st.metric("Savings Rate", f"{sr:.1f}%",
                        delta="✅ Good" if sr >= 20 else "⚠️ Low")

    if updated:
        v1, v2 = st.columns(2)
        with v1: st.plotly_chart(create_income_expense_bar_chart(income, total_exp, remaining), use_container_width=True)
        with v2: st.plotly_chart(create_expense_pie_chart(updated), use_container_width=True)

        # 50/30/20 rule
        if income > 0:
            st.subheader("50 / 30 / 20 Rule")
            needs_cats   = ["Housing (Rent/Mortgage)", "Utilities", "Groceries", "Health Care", "Transportation", "Insurance"]
            wants_cats   = ["Entertainment", "Dining Out", "Shopping"]
            savings_cats = ["Savings"]
            needs   = sum(updated.get(c, 0) for c in needs_cats)
            wants   = sum(updated.get(c, 0) for c in wants_cats)
            sav_amt = sum(updated.get(c, 0) for c in savings_cats)

            fig_rule = go.Figure()
            for label, actual, target, color in [
                ("Needs (50%)",   needs,   income * 0.50, "#4169E1"),
                ("Wants (30%)",   wants,   income * 0.30, "#8A2BE2"),
                ("Savings (20%)", sav_amt, income * 0.20, "#00C896"),
            ]:
                pct = actual / income * 100
                fig_rule.add_trace(go.Bar(
                    x=[label], y=[actual], marker_color=color,
                    text=f"{pct:.0f}%", textposition="outside",
                    textfont=dict(color="#e8eaed"),
                    hovertemplate=f"<b>{label}</b><br>₹%{{y:,.0f}} ({pct:.1f}%)<extra></extra>",
                ))
                fig_rule.add_trace(go.Bar(
                    x=[label], y=[target],
                    marker=dict(color="rgba(0,0,0,0)", line=dict(color=color, width=2)),
                    hovertemplate=f"Target: ₹{target:,.0f}<extra></extra>",
                    showlegend=False,
                ))

            fig_rule.update_layout(
                barmode="overlay", height=280,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8eaed"),
                yaxis=dict(tickprefix="₹", tickformat=",", gridcolor="rgba(255,255,255,0.08)"),
                xaxis=dict(color="#e8eaed"),
                showlegend=False,
                margin=dict(t=16, b=16, l=8, r=8),
            )
            st.plotly_chart(fig_rule, use_container_width=True)

        # Expense table
        st.subheader("Expense Details")
        df_exp = pd.DataFrame([{
            "Category":    k,
            "Amount":      format_currency(v),
            "% of Income": f"{v/income*100:.1f}%" if income > 0 else "—",
            "Status":      "✅" if (v/income*100 < 15) else ("⚠️" if v/income*100 < 30 else "🔴"),
        } for k, v in updated.items()])
        st.dataframe(df_exp, use_container_width=True, hide_index=True)

    if updated and income > 0:
        st.markdown("---")
        if st.button("🤖 Get AI Budget Analysis", type="primary"):
            with st.spinner("Artha is analysing your budget..."):
                result = analyze_budget(income, updated)
                if st.session_state.user_id:
                    save_ai_insight(st.session_state.user_id, "budget_analysis", str(result))
            ai_card("Budget Analysis", result if isinstance(result, str) else str(result))


# ══════════════════════════════════════════════════════════════════════════════
#  STOCKS & PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Stocks & Portfolio":
    st.header("📈 Stocks & Portfolio")

    tab1, tab2, tab3 = st.tabs(["🕯️ Candlestick Chart", "💼 My Portfolio", "🔍 Stock Lookup"])

    # ── CANDLESTICK ───────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Search any stock and read its candles")
        ci1, ci2 = st.columns([2, 1])
        with ci1:
            ticker_input = st.text_input(
                "Stock ticker symbol",
                placeholder="e.g. AAPL  TSLA  RELIANCE.NS  INFY.NS",
                help="For Indian stocks add .NS (NSE) or .BO (BSE)",
            )
        with ci2:
            period = st.selectbox("Period", [30, 60, 90, 180, 365],
                                  format_func=lambda x: f"{x} days", index=2)

        if ticker_input:
            with st.spinner(f"Loading {ticker_input.upper()} candles..."):
                fig_candle = create_candlestick_chart(ticker_input.strip(), period_days=period)
            st.plotly_chart(fig_candle, use_container_width=True)

            with st.spinner("Fetching metrics..."):
                metrics = get_stock_metrics(ticker_input.strip())

            if "error" not in metrics:
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1: st.metric("Price",     f"${metrics['current_price']:.2f}"   if metrics["current_price"]    else "—")
                with m2: st.metric("52W High",  f"${metrics['price_52wk_high']:.2f}" if metrics["price_52wk_high"]  else "—")
                with m3: st.metric("52W Low",   f"${metrics['price_52wk_low']:.2f}"  if metrics["price_52wk_low"]   else "—")
                with m4: st.metric("P/E",       f"{metrics['pe_ratio']:.1f}"         if metrics["pe_ratio"]         else "—")
                with m5:
                    r1 = metrics["returns"].get("1mo")
                    st.metric("1M Return", f"{r1:+.2f}%" if r1 else "—")

            with st.expander("📖 How to read this candlestick chart"):
                st.markdown("""
| Element | Meaning |
|---|---|
| 🟢 Green candle | Closed **higher** than open — buyers in control |
| 🔴 Red candle | Closed **lower** than open — sellers in control |
| **Body** | Distance between open and close |
| **Upper wick** | Highest price reached in the period |
| **Lower wick** | Lowest price reached in the period |
| **EMA 20** (dotted yellow) | Short-term trend line |
| **EMA 50** (dashed purple) | Medium-term support / resistance |
| **Bollinger Bands** (blue fill) | Volatility envelope — breakouts are significant |
| **Volume bars** | High volume confirms a move; low = weak signal |

**Common patterns:** Doji (indecision) · Hammer (reversal) · Engulfing (strong signal)
                """)
        else:
            st.info("Enter a ticker symbol above to load the candlestick chart.")

    # ── PORTFOLIO ─────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Add stocks you own and track P&L in real-time")

        with st.form("add_holding"):
            fc1, fc2, fc3 = st.columns([2, 1, 1])
            with fc1: new_ticker = st.text_input("Ticker", placeholder="AAPL")
            with fc2: new_shares = st.number_input("Shares", min_value=0.001, value=1.0, step=0.1, format="%.3f")
            with fc3: new_cost   = st.number_input("Buy Price", min_value=0.01, value=100.0, step=0.5, format="%.2f")
            if st.form_submit_button("➕ Add to Portfolio", use_container_width=True) and new_ticker.strip():
                st.session_state.portfolio.append({
                    "ticker": new_ticker.strip().upper(),
                    "shares": new_shares,
                    "cost_basis": new_cost,
                })
                st.success(f"Added {new_ticker.upper()} × {new_shares}")

        if st.session_state.portfolio:
            st.markdown("---")
            tickers_in_portfolio = [h["ticker"] for h in st.session_state.portfolio]
            to_remove = st.selectbox("Remove a holding", ["— select —"] + tickers_in_portfolio)
            if to_remove != "— select —":
                if st.button(f"🗑️ Remove {to_remove}"):
                    st.session_state.portfolio = [h for h in st.session_state.portfolio if h["ticker"] != to_remove]
                    st.rerun()

            st.markdown("---")
            with st.spinner("Fetching live prices from FMP..."):
                pm = calculate_portfolio_metrics(st.session_state.portfolio)

            if "error" not in pm and pm["holdings_data"]:
                t1, t2, t3 = st.columns(3)
                with t1: st.metric("Total Value", format_currency(pm["total_value"]))
                with t2: st.metric("Invested",    format_currency(pm["total_cost"]))
                with t3:
                    dc = "normal" if pm["total_gain_loss"] >= 0 else "inverse"
                    st.metric("P&L", format_currency(pm["total_gain_loss"]),
                              delta=f"{pm['total_gain_loss_pct']:+.2f}%", delta_color=dc)

                g1, g2 = st.columns(2)
                with g1: st.plotly_chart(create_portfolio_allocation_chart(pm["holdings_data"]), use_container_width=True)
                with g2: st.plotly_chart(create_portfolio_pnl_chart(pm["holdings_data"]),        use_container_width=True)

                st.subheader("Holdings Detail")
                df_h = pd.DataFrame([{
                    "Ticker":     h["ticker"],
                    "Shares":     h["shares"],
                    "Buy Price":  format_currency(h["cost_basis"]),
                    "Current":    format_currency(h["latest_price"]),
                    "Value":      format_currency(h["current_value"]),
                    "P&L":        format_currency(h["gain_loss"]),
                    "P&L %":      f"{h['gain_loss_pct']:+.2f}%",
                } for h in pm["holdings_data"]])
                st.dataframe(df_h, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("Individual Candle Charts")
                for h in pm["holdings_data"]:
                    with st.expander(f"🕯️ {h['ticker']} — 60-day Candlestick"):
                        st.plotly_chart(create_candlestick_chart(h["ticker"], 60), use_container_width=True)

                st.markdown("---")
                if st.button("🤖 Get AI Portfolio Review", type="primary"):
                    with st.spinner("Artha is reviewing your portfolio..."):
                        advice = investement_advise(
                            risk_involved="moderate",
                            investment_horizon="medium-term",
                            current_investments={h["ticker"]: h["current_value"] for h in pm["holdings_data"]},
                        )
                    ai_card("Portfolio Review", advice)
            else:
                st.info("Add holdings above to see your portfolio analysis.")
        else:
            st.info("Your portfolio is empty. Add your first holding above.")

    # ── STOCK LOOKUP ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Research any stock — fundamentals + candle")
        lookup = st.text_input("Enter ticker to research", placeholder="MSFT  RELIANCE.NS  ...")

        if lookup:
            with st.spinner(f"Researching {lookup.upper()}..."):
                m      = get_stock_metrics(lookup.strip())
                fig_l  = create_candlestick_chart(lookup.strip(), period_days=90)

            if "error" not in m:
                st.markdown(f"### {m.get('name', lookup.upper())}")
                la, lb, lc, ld = st.columns(4)
                with la: st.metric("Price",    f"${m['current_price']:.2f}"   if m["current_price"]    else "—")
                with lb: st.metric("52W High", f"${m['price_52wk_high']:.2f}" if m["price_52wk_high"]  else "—")
                with lc: st.metric("52W Low",  f"${m['price_52wk_low']:.2f}"  if m["price_52wk_low"]   else "—")
                with ld: st.metric("Dividend", f"{m['dividend_yield']:.2f}"   if m["dividend_yield"]   else "—")

                r1, r2, r3 = st.columns(3)
                with r1: st.metric("1M Return", f"{m['returns']['1mo']:+.2f}%" if m["returns"].get("1mo") else "—")
                with r2: st.metric("3M Return", f"{m['returns']['3mo']:+.2f}%" if m["returns"].get("3mo") else "—")
                with r3: st.metric("1Y Return", f"{m['returns']['1yr']:+.2f}%" if m["returns"].get("1yr") else "—")

                st.plotly_chart(fig_l, use_container_width=True)

                if st.button("🤖 AI Stock Analysis", type="primary"):
                    with st.spinner("Artha is analysing..."):
                        advice = generate_financial_adivice(
                            f"Give me a brief analysis of {lookup.upper()} based on these metrics: {m}. "
                            "Is it worth investing? Key risks and opportunities?",
                            {"ticker": lookup.upper(), "metrics": m},
                        )
                    ai_card(f"AI Analysis: {lookup.upper()}", advice)
            else:
                st.error(m["error"])


# ══════════════════════════════════════════════════════════════════════════════
#  AI ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Advisor":
    st.header("🤖 AI Financial Advisor")
    st.markdown(
        '<p style="color:#666;font-size:0.9em">'
        'Personalised advice based on your financial data. Powered by Groq LLaMA 3.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("**💡 Quick topics**")
    topics = [
        "How can I improve my savings rate?",
        "Best way to pay off my debt?",
        "How should I prioritise my financial goals?",
        "How much should I save for retirement?",
        "How do I build an emergency fund faster?",
        "Where should I invest my surplus income?",
    ]
    tc = st.columns(3)
    for i, topic in enumerate(topics):
        with tc[i % 3]:
            if st.button(topic, key=f"t_{i}", use_container_width=True):
                ctx = {
                    "income": st.session_state.income,
                    "expenses": st.session_state.expenses,
                    "assets": st.session_state.assets,
                    "liabilities": st.session_state.liabilities,
                    "goals": st.session_state.financial_goals,
                    "portfolio": st.session_state.portfolio,
                }
                with st.spinner("Thinking..."):
                    resp = generate_financial_adivice(topic, ctx)
                st.session_state.chat_history += [
                    {"role": "user",      "content": topic, "ts": datetime.now().strftime("%H:%M")},
                    {"role": "assistant", "content": resp,  "ts": datetime.now().strftime("%H:%M")},
                ]
                st.rerun()

    st.markdown("---")

    with st.form("chat_form", clear_on_submit=True):
        fi1, fi2 = st.columns([5, 1])
        with fi1:
            user_q = st.text_input(
                "", placeholder="Ask Artha anything about your finances…",
                label_visibility="collapsed",
            )
        with fi2:
            send = st.form_submit_button("Send →", use_container_width=True)

    if send and user_q.strip():
        ctx = {
            "income": st.session_state.income,
            "expenses": st.session_state.expenses,
            "assets": st.session_state.assets,
            "liabilities": st.session_state.liabilities,
            "goals": st.session_state.financial_goals,
            "portfolio": st.session_state.portfolio,
        }
        with st.spinner("Artha is thinking..."):
            resp = generate_financial_adivice(user_q.strip(), ctx)
        st.session_state.chat_history += [
            {"role": "user",      "content": user_q.strip(), "ts": datetime.now().strftime("%H:%M")},
            {"role": "assistant", "content": resp,            "ts": datetime.now().strftime("%H:%M")},
        ]
        if st.session_state.user_id:
            save_ai_insight(st.session_state.user_id, "chat", resp)
        st.rerun()

    if st.session_state.chat_history:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_bubble_user(msg["content"], msg.get("ts", ""))
            else:
                chat_bubble_bot(msg["content"], msg.get("ts", ""))
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center;padding:40px;color:#444">
          <div style="font-size:40px">💬</div>
          <p>Ask a question or tap a topic chip to get started</p>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#444;font-size:12px;text-align:center'>"
    "⚠️ Artha provides general financial information only — not a substitute for professional advice."
    "</p>",
    unsafe_allow_html=True,
)
