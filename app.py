import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

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
    page_icon="💰", layout="wide",
    initial_sidebar_state="expanded",
)

# ── STYLE ─────────────────────────────────────────────────────────────────────
def apply_custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif;background-color:#0d1117;color:#e8eaed}
    .gradient-text{background:linear-gradient(90deg,#4169E1,#8A2BE2);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;display:inline-block;font-weight:700}
    section[data-testid="stSidebar"]{background:#161b22!important;border-right:1px solid #30363d}
    [data-testid="metric-container"]{background:#161b22;border:1px solid #30363d;border-radius:12px;
      padding:14px 18px;transition:border-color .2s}
    [data-testid="metric-container"]:hover{border-color:#4169E1}
    .chat-wrap{display:flex;flex-direction:column;gap:18px;padding:8px 0}
    .bubble-user-row{display:flex;justify-content:flex-end;align-items:flex-end;gap:10px}
    .bubble-user{background:linear-gradient(135deg,#4169E1,#5a7ff5);color:#fff;padding:12px 18px;
      border-radius:20px 20px 4px 20px;max-width:68%;font-size:14px;line-height:1.6;
      box-shadow:0 4px 20px rgba(65,105,225,.35);word-break:break-word}
    .bubble-user-avatar{width:32px;height:32px;border-radius:50%;
      background:linear-gradient(135deg,#4169E1,#8A2BE2);display:flex;align-items:center;
      justify-content:center;font-size:14px;flex-shrink:0}
    .bubble-bot-row{display:flex;justify-content:flex-start;align-items:flex-start;gap:10px}
    .bubble-bot-avatar{width:32px;height:32px;border-radius:50%;
      background:linear-gradient(135deg,#8A2BE2,#4169E1);display:flex;align-items:center;
      justify-content:center;font-size:14px;flex-shrink:0;margin-top:2px}
    .bubble-bot{background:#1e2130;border:1px solid #2d3348;color:#e8eaed;padding:14px 18px;
      border-radius:4px 20px 20px 20px;max-width:72%;font-size:14px;line-height:1.7;
      box-shadow:0 2px 12px rgba(0,0,0,.3);word-break:break-word}
    .bubble-bot-name{font-size:11px;color:#8A2BE2;font-weight:600;letter-spacing:.04em;
      text-transform:uppercase;margin-bottom:6px}
    .bubble-ts{font-size:10px;color:#555d70;margin-top:4px;padding:0 4px}
    .bubble-ts.right{text-align:right}
    .stButton>button{background:#1e2130!important;border:1px solid #30363d!important;
      color:#c9d1d9!important;border-radius:20px!important;font-size:13px!important;
      padding:6px 14px!important;transition:all .2s!important}
    .stButton>button:hover{border-color:#4169E1!important;color:#fff!important;background:#161b22!important}
    .ai-card{background:#1e2130;border-left:3px solid #4169E1;border-radius:0 12px 12px 0;
      padding:16px 20px;margin:12px 0;font-size:14px;line-height:1.75;color:#e8eaed}
    .ai-card-label{font-size:10px;font-weight:700;color:#4169E1;letter-spacing:.08em;
      text-transform:uppercase;margin-bottom:8px}
    .predict-badge{display:inline-block;background:#1e2130;border:1px solid #4169E1;color:#e8eaed;
      border-radius:20px;padding:4px 14px;font-size:13px;font-weight:500;margin:4px 4px 4px 0}
    hr{border-color:#21262d!important}
    </style>""", unsafe_allow_html=True)

# ── INIT ──────────────────────────────────────────────────────────────────────
initialize_database()
apply_custom_style()

# ── AUTO-TRAIN CLASSIFIER ─────────────────────────────────────────────────────
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
)
from moneyanalyser import (
    calculate_net_worth, calculate_emergency_fund_ratio,
    calculate_portfolio_metrics, calculate_debt_to_income_ratio,
    analyze_retirement_readiness, analyze_mortgage_affordability,
    get_stock_metrics,
)

# ── INLINE CHART FUNCTIONS (no dependency on new frontend.py) ─────────────────
_C = {"primary":"#4169E1","secondary":"#8A2BE2","success":"#00C896",
      "danger":"#FF4757","warning":"#FFA502","bg":"rgba(0,0,0,0)",
      "grid":"rgba(255,255,255,0.08)","text":"#E8EAED"}
_PAL = ["#4169E1","#8A2BE2","#00C896","#FFA502","#FF4757","#00B4D8","#F72585","#3A86FF"]
_LAY = dict(plot_bgcolor=_C["bg"],paper_bgcolor=_C["bg"],
            font=dict(family="Inter,sans-serif",color=_C["text"],size=13),
            margin=dict(t=48,b=16,l=8,r=8),
            hoverlabel=dict(bgcolor="#1E2130",font_color="#fff",bordercolor="#4169E1"))

def _fmp(endpoint, params=None):
    key = os.getenv("FMP_API_KEY")
    if not key: return None
    p = params or {}; p["apikey"] = key
    try:
        r = requests.get(f"https://financialmodelingprep.com/api/v3/{endpoint}",params=p,timeout=12)
        r.raise_for_status(); return r.json()
    except: return None

def create_candlestick_chart(ticker, period_days=90):
    ticker = ticker.upper().strip()
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today()-timedelta(days=period_days+30)).strftime("%Y-%m-%d")
    data  = _fmp(f"historical-price-full/{ticker}",{"from":start,"to":end})
    if not data or "historical" not in data or not data["historical"]:
        fig=go.Figure()
        fig.add_annotation(text=f"No data for {ticker}. Check ticker symbol.",
                           showarrow=False,font=dict(size=14,color=_C["danger"]))
        fig.update_layout(**_LAY,title=f"{ticker} — Not Found"); return fig
    df=pd.DataFrame(data["historical"]).sort_values("date")
    df["date"]=pd.to_datetime(df["date"]); df=df.tail(period_days).reset_index(drop=True)
    df["ema20"]=df["close"].ewm(span=20,adjust=False).mean()
    df["ema50"]=df["close"].ewm(span=50,adjust=False).mean()
    df["bb_mid"]=df["close"].rolling(20).mean(); df["bb_std"]=df["close"].rolling(20).std()
    df["bb_upper"]=df["bb_mid"]+2*df["bb_std"]; df["bb_lower"]=df["bb_mid"]-2*df["bb_std"]
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.72,0.28],vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df["date"],open=df["open"],high=df["high"],
        low=df["low"],close=df["close"],name=ticker,
        increasing_line_color=_C["success"],decreasing_line_color=_C["danger"],
        increasing_fillcolor=_C["success"],decreasing_fillcolor=_C["danger"]),row=1,col=1)
    fig.add_trace(go.Scatter(x=pd.concat([df["date"],df["date"][::-1]]),
        y=pd.concat([df["bb_upper"],df["bb_lower"][::-1]]),fill="toself",
        fillcolor="rgba(65,105,225,0.07)",line=dict(color="rgba(0,0,0,0)"),
        name="Bollinger Band",hoverinfo="skip"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df["date"],y=df["ema20"],name="EMA 20",
        line=dict(color=_C["warning"],width=1.5,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df["date"],y=df["ema50"],name="EMA 50",
        line=dict(color=_C["secondary"],width=1.5,dash="dash")),row=1,col=1)
    vol_colors=[_C["success"] if c>=o else _C["danger"] for c,o in zip(df["close"],df["open"])]
    fig.add_trace(go.Bar(x=df["date"],y=df["volume"],name="Volume",
        marker_color=vol_colors,opacity=0.6),row=2,col=1)
    if len(df)>=2:
        lat=df.iloc[-1]; prv=df.iloc[-2]; chg=lat["close"]-prv["close"]
        pct=chg/prv["close"]*100; sign="▲" if chg>=0 else "▼"
        col=_C["success"] if chg>=0 else _C["danger"]
        fig.add_annotation(x=lat["date"],y=lat["high"],
            text=f"<b>{sign} {lat['close']:.2f} ({pct:+.2f}%)</b>",
            showarrow=True,arrowhead=2,arrowcolor=col,
            font=dict(color=col,size=13),bgcolor="#1E2130",bordercolor=col,borderwidth=1,row=1,col=1)
    fig.update_layout(**_LAY,
        title=dict(text=f"<b>{ticker}</b> — Candlestick + Volume + Indicators",
                   font=dict(size=16,color=_C["text"])),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.01,x=0,font=dict(size=11,color=_C["text"])),
        yaxis=dict(color=_C["text"],gridcolor=_C["grid"],side="right"),
        yaxis2=dict(color=_C["text"],gridcolor=_C["grid"],title="Volume",side="right"),
        xaxis2=dict(color=_C["text"],gridcolor=_C["grid"]),hovermode="x unified",height=580)
    return fig

def create_portfolio_allocation_chart(holdings_data):
    if not holdings_data:
        fig=go.Figure(); fig.update_layout(**_LAY); return fig
    labels=[h["ticker"] for h in holdings_data]; values=[h["current_value"] for h in holdings_data]
    fig=go.Figure(go.Pie(labels=labels,values=values,hole=0.55,
        marker=dict(colors=_PAL,line=dict(color="#0d1117",width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>"))
    fig.add_annotation(text=f"<b>₹{sum(values):,.0f}</b><br>Portfolio",
        x=0.5,y=0.5,showarrow=False,font=dict(size=14,color=_C["text"]),xanchor="center")
    fig.update_layout(**_LAY,title=dict(text="Portfolio Allocation",
        font=dict(size=15,color=_C["text"])),showlegend=True,
        legend=dict(orientation="h",y=-0.2,x=0.5,xanchor="center",font=dict(size=11,color=_C["text"])))
    return fig

def create_portfolio_pnl_chart(holdings_data):
    if not holdings_data:
        fig=go.Figure(); fig.update_layout(**_LAY); return fig
    tickers=[h["ticker"] for h in holdings_data]; pnl=[h["gain_loss"] for h in holdings_data]
    pnl_pct=[h["gain_loss_pct"] for h in holdings_data]
    colors=[_C["success"] if v>=0 else _C["danger"] for v in pnl]
    fig=go.Figure(go.Bar(x=tickers,y=pnl,marker=dict(color=colors),
        text=[f"{p:+.1f}%" for p in pnl_pct],textposition="outside",
        textfont=dict(color=_C["text"],size=12),
        hovertemplate="<b>%{x}</b><br>P&L: ₹%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0,line_color="rgba(255,255,255,0.3)",line_width=1)
    fig.update_layout(**_LAY,title=dict(text="Unrealised P&L by Holding",
        font=dict(size=15,color=_C["text"])),
        yaxis=dict(tickprefix="₹",tickformat=",",color=_C["text"],gridcolor=_C["grid"]),
        xaxis=dict(color=_C["text"]),showlegend=False)
    return fig

def create_net_worth_gauge(net_worth, target=None):
    tgt=target or max(abs(net_worth)*2,1_000_000)
    fig=go.Figure(go.Indicator(mode="gauge+number+delta",value=net_worth,
        number=dict(prefix="₹",valueformat=",.0f",font=dict(color=_C["text"],size=28)),
        delta=dict(reference=0,valueformat=",.0f",prefix="₹"),
        gauge=dict(axis=dict(range=[0,tgt],tickformat=".2s",tickprefix="₹",
            tickcolor=_C["text"],tickwidth=1),
            bar=dict(color=_C["primary"],thickness=0.25),
            bgcolor="rgba(255,255,255,0.05)",borderwidth=0,
            steps=[dict(range=[0,tgt*0.33],color="rgba(255,71,87,0.15)"),
                   dict(range=[tgt*0.33,tgt*0.66],color="rgba(255,165,2,0.15)"),
                   dict(range=[tgt*0.66,tgt],color="rgba(0,200,150,0.15)")],
            threshold=dict(line=dict(color=_C["success"],width=3),value=tgt*0.66)),
        title=dict(text="Net Worth",font=dict(size=15,color=_C["text"]))))
    fig.update_layout(**_LAY,height=280)
    return fig

# Load classifier
if "classifier" not in st.session_state:
    st.session_state.classifier = load_model()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "income":0.0,"expenses":{},"assets":{},"liabilities":{},
    "financial_goals":[],"chat_history":[],"portfolio":[],
    "user_id":None,"username":None,"authenticated":False,"watchlist":[],
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="margin-bottom:2px"><span class="gradient-text">Artha</span> '
            '<span style="color:#555;font-weight:300;font-size:0.85em">/ AI Financial Assistant</span></h1>',
            unsafe_allow_html=True)
st.markdown('<p style="color:#666;font-size:0.95em;margin-top:0">'
            'Track · Invest · Analyse · Chat — powered by Groq & FMP API</p>',unsafe_allow_html=True)

# ── AUTH ──────────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    st.sidebar.title("🔐 Authentication")
    auth_option=st.sidebar.radio("",["Login","Register"],horizontal=True)
    username=st.sidebar.text_input("Username")
    email=st.sidebar.text_input("Email (optional)") if auth_option=="Register" else None
    if st.sidebar.button(auth_option,type="primary",use_container_width=True):
        if username.strip():
            uid=get_or_create_user(username.strip(),email)
            st.session_state.update({
                "user_id":uid,"username":username.strip(),"authenticated":True,
                "income":get_user_income(uid),"expenses":get_user_expenses(uid),
                "assets":get_user_assets(uid),"liabilities":get_user_liabilities(uid),
                "financial_goals":get_user_financial_goals(uid),"portfolio":get_user_portfolio(uid),
            })
            st.rerun()
        else: st.sidebar.error("Please enter a username")
    st.markdown("""<div style="text-align:center;padding:60px 20px">
      <div style="font-size:64px">💰</div>
      <h2 class="gradient-text">Welcome to Artha</h2>
      <p style="color:#666;max-width:420px;margin:0 auto">
        Track income &amp; expenses · Monitor net worth<br>
        Invest in real stocks · Auto-classify transactions<br>
        Get AI-powered financial advice</p></div>""",unsafe_allow_html=True)
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(f"### 👋 {st.session_state.username}")
st.sidebar.markdown("---")
page=st.sidebar.radio("Navigation",
    ["📊 Dashboard","💰 Budget Analyzer","📈 Stocks & Portfolio","🤖 AI Advisor"],
    label_visibility="collapsed")
st.sidebar.markdown("---")
if st.sidebar.button("💾 Save Data",use_container_width=True):
    save_income(st.session_state.user_id,st.session_state.income)
    save_expenses(st.session_state.user_id,st.session_state.expenses)
    save_assets(st.session_state.user_id,st.session_state.assets)
    save_liabilities(st.session_state.user_id,st.session_state.liabilities)
    if st.session_state.financial_goals:
        save_financial_goals(st.session_state.user_id,st.session_state.financial_goals)
    if st.session_state.portfolio:
        save_investment_portfolio(st.session_state.user_id,st.session_state.portfolio)
    st.sidebar.success("✅ Saved!")
if st.sidebar.button("🚪 Logout",use_container_width=True):
    for k in defaults: st.session_state[k]=defaults[k]
    st.rerun()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def ai_card(label,text):
    safe=str(text).replace("\n","<br>").replace("**","")
    st.markdown(f'<div class="ai-card"><div class="ai-card-label">🤖 {label}</div>{safe}</div>',
                unsafe_allow_html=True)

def chat_bubble_user(msg,ts):
    st.markdown(f"""<div class="bubble-user-row">
      <div><div class="bubble-user">{msg}</div><div class="bubble-ts right">{ts}</div></div>
      <div class="bubble-user-avatar">👤</div></div>""",unsafe_allow_html=True)

def chat_bubble_bot(msg,ts):
    safe=str(msg).replace("\n","<br>")
    st.markdown(f"""<div class="bubble-bot-row">
      <div class="bubble-bot-avatar">💰</div>
      <div><div class="bubble-bot"><div class="bubble-bot-name">Artha</div>{safe}</div>
      <div class="bubble-ts">{ts}</div></div></div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page=="📊 Dashboard":
    st.header("📊 Dashboard")
    net_worth,total_assets,total_liabilities=calculate_net_worth(
        st.session_state.assets,st.session_state.liabilities)
    total_expenses=sum(st.session_state.expenses.values()) if st.session_state.expenses else 0.0
    savings=st.session_state.income-total_expenses
    sr=(savings/st.session_state.income*100) if st.session_state.income>0 else 0
    k1,k2,k3,k4=st.columns(4)
    with k1: st.metric("Net Worth",format_currency(net_worth))
    with k2: st.metric("Monthly Income",format_currency(st.session_state.income))
    with k3: st.metric("Monthly Expenses",format_currency(total_expenses))
    with k4: st.metric("Savings Rate",f"{sr:.1f}%",delta="Good ✅" if sr>=20 else "Improve ⚠️")
    st.markdown("---")
    c1,c2=st.columns([1.1,0.9])
    with c1:
        st.subheader("Net Worth Gauge")
        st.plotly_chart(create_net_worth_gauge(net_worth),use_container_width=True)
    with c2:
        if st.session_state.expenses:
            st.subheader("Expense Breakdown")
            st.plotly_chart(create_expense_pie_chart(st.session_state.expenses),use_container_width=True)
        else: st.info("Add expenses in Budget Analyzer to see breakdown.")
    if st.session_state.expenses and "Emergency Fund" in st.session_state.assets:
        ef=st.session_state.assets.get("Emergency Fund",0)
        months=calculate_emergency_fund_ratio(ef,total_expenses)
        if months:
            st.markdown("---")
            st.markdown(f"**🛡️ Emergency Fund** — covers **{months:.1f} months** of expenses")
            st.progress(int(min(months/6*100,100)))
            if months<3: st.warning("Critical: aim for at least 3 months.")
            elif months<6: st.warning("Good start — try to reach 6 months.")
            else: st.success("✅ Emergency fund is solid!")
    if st.session_state.portfolio:
        st.markdown("---"); st.subheader("Portfolio Snapshot")
        with st.spinner("Loading live prices..."):
            pm=calculate_portfolio_metrics(st.session_state.portfolio)
        if "error" not in pm and pm["holdings_data"]:
            p1,p2,p3=st.columns(3)
            with p1: st.metric("Portfolio Value",format_currency(pm["total_value"]))
            with p2: st.metric("Total Cost",format_currency(pm["total_cost"]))
            with p3:
                dc="normal" if pm["total_gain_loss"]>=0 else "inverse"
                st.metric("Unrealised P&L",format_currency(pm["total_gain_loss"]),
                          delta=f"{pm['total_gain_loss_pct']:+.2f}%",delta_color=dc)
            pa1,pa2=st.columns(2)
            with pa1: st.plotly_chart(create_portfolio_allocation_chart(pm["holdings_data"]),use_container_width=True)
            with pa2: st.plotly_chart(create_portfolio_pnl_chart(pm["holdings_data"]),use_container_width=True)
    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ══════════════════════════════════════════════════════════════════════════════
#  BUDGET ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
elif page=="💰 Budget Analyzer":
    st.header("💰 Budget Analyzer")
    income=st.number_input("Monthly Income (₹)",min_value=0.0,
                            value=float(st.session_state.income),step=1000.0,format="%.2f")
    st.session_state.income=income
    st.markdown("---"); st.subheader("🧠 Smart Transaction Classifier")
    st.markdown('<p style="color:#666;font-size:13px">Type any transaction — Artha auto-predicts its category (96.87% F1)</p>',unsafe_allow_html=True)
    txn_col,btn_col=st.columns([4,1])
    with txn_col:
        txn_desc=st.text_input("",placeholder="e.g. Zomato biryani  /  Airtel bill  /  BigBasket order",
                               label_visibility="collapsed",key="txn_input")
    with btn_col:
        classify_btn=st.button("Classify →",use_container_width=True)
    if classify_btn and txn_desc.strip():
        result=predict_category(txn_desc.strip(),st.session_state.classifier)
        cat=result["category"]; conf=result["confidence"]; top3=result["top3"]
        conf_color="#00C896" if conf>=0.7 else "#FFA502" if conf>=0.5 else "#FF4757"
        st.markdown(f"""<div style="background:#1e2130;border:1px solid #30363d;border-radius:12px;padding:16px 20px;margin:8px 0">
          <div style="font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">
            Prediction for: <b style="color:#e8eaed">{txn_desc}</b></div>
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
            <span style="font-size:22px;font-weight:700;color:#4169E1">{cat}</span>
            <span style="font-size:14px;color:{conf_color};font-weight:600">{conf:.0%} confidence</span></div>
          <div style="margin-top:10px;font-size:12px;color:#666">Also consider:
            {''.join([f'<span class="predict-badge">{c} <span style="color:#4169E1">{p:.0%}</span></span>' for c,p in top3[1:]])}</div></div>""",
                    unsafe_allow_html=True)
        if st.button(f"➕ Add ₹ amount to '{cat}'"):
            st.session_state["pending_category"]=cat
    if st.session_state.get("pending_category"):
        pcat=st.session_state["pending_category"]
        amt=st.number_input(f"Amount for '{pcat}' (₹)",min_value=0.0,step=50.0,format="%.2f")
        if st.button("Confirm Add"):
            st.session_state.expenses[pcat]=st.session_state.expenses.get(pcat,0.0)+amt
            st.session_state["pending_category"]=None
            st.success(f"Added ₹{amt:,.0f} to {pcat}"); st.rerun()
    st.markdown("---"); st.subheader("Monthly Expenses")
    expense_categories=["Housing (Rent/Mortgage)","Utilities","Groceries","Transportation",
        "Health Care","Entertainment","Dining Out","Shopping","Education","Insurance","Savings","Other"]
    updated={}; cl,cr=st.columns(2)
    for i,cat in enumerate(expense_categories):
        col=cl if i<6 else cr
        with col:
            v=st.number_input(f"{cat} (₹)",min_value=0.0,
                              value=float(st.session_state.expenses.get(cat,0.0)),
                              step=100.0,format="%.2f",key=f"exp_{cat}")
            if v>0: updated[cat]=v
    st.markdown("---")
    custom=st.text_input("➕ Custom expense category")
    if custom:
        cv=st.number_input(f"{custom} (₹)",min_value=0.0,
                           value=float(st.session_state.expenses.get(custom,0.0)),
                           step=100.0,format="%.2f")
        if cv>0: updated[custom]=cv
    st.session_state.expenses=updated
    bs=calculate_budget_summary(income,updated)
    total_exp=bs["total_expenses"]; remaining=bs["remaining"]; sr=bs["savings_rate"]
    st.markdown("---"); st.subheader("Summary")
    m1,m2,m3,m4=st.columns(4)
    with m1: st.metric("Income",format_currency(income))
    with m2: st.metric("Expenses",format_currency(total_exp))
    with m3: st.metric("Remaining",format_currency(remaining),delta="surplus" if remaining>=0 else "deficit")
    with m4: st.metric("Savings Rate",f"{sr:.1f}%",delta="✅ Good" if sr>=20 else "⚠️ Low")
    if updated:
        v1,v2=st.columns(2)
        with v1: st.plotly_chart(create_income_expense_bar_chart(income,total_exp,remaining),use_container_width=True)
        with v2: st.plotly_chart(create_expense_pie_chart(updated),use_container_width=True)
        st.subheader("Expense Details")
        df_exp=pd.DataFrame([{"Category":k,"Amount":format_currency(v),
            "% of Income":f"{v/income*100:.1f}%" if income>0 else "—",
            "Status":"✅" if (v/income*100<15) else ("⚠️" if v/income*100<30 else "🔴")}
            for k,v in updated.items()])
        st.dataframe(df_exp,use_container_width=True,hide_index=True)
    if updated and income>0:
        st.markdown("---")
        if st.button("🤖 Get AI Budget Analysis",type="primary"):
            with st.spinner("Artha is analysing your budget..."):
                result=analyze_budget(income,updated)
                if st.session_state.user_id:
                    save_ai_insight(st.session_state.user_id,"budget_analysis",str(result))
            ai_card("Budget Analysis",result if isinstance(result,str) else str(result))

# ══════════════════════════════════════════════════════════════════════════════
#  STOCKS & PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
elif page=="📈 Stocks & Portfolio":
    st.header("📈 Stocks & Portfolio")
    tab1,tab2,tab3=st.tabs(["🕯️ Candlestick Chart","💼 My Portfolio","🔍 Stock Lookup"])

    with tab1:
        st.markdown("#### Search any stock and read its candles")
        ci1,ci2=st.columns([2,1])
        with ci1:
            ticker_input=st.text_input("Stock ticker symbol",
                placeholder="e.g. AAPL  TSLA  RELIANCE.NS  INFY.NS",
                help="For Indian stocks add .NS (NSE) or .BO (BSE)")
        with ci2:
            period=st.selectbox("Period",[30,60,90,180,365],
                                format_func=lambda x:f"{x} days",index=2)
        if ticker_input:
            with st.spinner(f"Loading {ticker_input.upper()} candles..."):
                fig_candle=create_candlestick_chart(ticker_input.strip(),period_days=period)
            st.plotly_chart(fig_candle,use_container_width=True)
            with st.spinner("Fetching metrics..."):
                metrics=get_stock_metrics(ticker_input.strip())
            if "error" not in metrics:
                m1,m2,m3,m4,m5=st.columns(5)
                with m1: st.metric("Price",f"${metrics['current_price']:.2f}" if metrics["current_price"] else "—")
                with m2: st.metric("52W High",f"${metrics['price_52wk_high']:.2f}" if metrics["price_52wk_high"] else "—")
                with m3: st.metric("52W Low",f"${metrics['price_52wk_low']:.2f}" if metrics["price_52wk_low"] else "—")
                with m4: st.metric("P/E",f"{metrics['pe_ratio']:.1f}" if metrics["pe_ratio"] else "—")
                with m5:
                    r1=metrics["returns"].get("1mo")
                    st.metric("1M Return",f"{r1:+.2f}%" if r1 else "—")
            with st.expander("📖 How to read this candlestick chart"):
                st.markdown("""| Element | Meaning |
|---|---|
|🟢 Green candle|Closed higher than open — buyers in control|
|🔴 Red candle|Closed lower than open — sellers in control|
|**Body**|Distance between open and close|
|**Upper wick**|Highest price in the period|
|**Lower wick**|Lowest price in the period|
|**EMA 20** (dotted yellow)|Short-term trend|
|**EMA 50** (dashed purple)|Medium-term support/resistance|
|**Bollinger Bands** (blue fill)|Volatility — breakouts are significant|
|**Volume bars**|High volume confirms a move|""")
        else: st.info("Enter a ticker symbol above to load the candlestick chart.")

    with tab2:
        st.markdown("#### Add stocks you own and track P&L in real-time")
        with st.form("add_holding"):
            fc1,fc2,fc3=st.columns([2,1,1])
            with fc1: new_ticker=st.text_input("Ticker",placeholder="AAPL")
            with fc2: new_shares=st.number_input("Shares",min_value=0.001,value=1.0,step=0.1,format="%.3f")
            with fc3: new_cost=st.number_input("Buy Price",min_value=0.01,value=100.0,step=0.5,format="%.2f")
            if st.form_submit_button("➕ Add to Portfolio",use_container_width=True) and new_ticker.strip():
                st.session_state.portfolio.append({"ticker":new_ticker.strip().upper(),"shares":new_shares,"cost_basis":new_cost})
                st.success(f"Added {new_ticker.upper()} × {new_shares}")
        if st.session_state.portfolio:
            st.markdown("---")
            tickers_in_portfolio=[h["ticker"] for h in st.session_state.portfolio]
            to_remove=st.selectbox("Remove a holding",["— select —"]+tickers_in_portfolio)
            if to_remove!="— select —":
                if st.button(f"🗑️ Remove {to_remove}"):
                    st.session_state.portfolio=[h for h in st.session_state.portfolio if h["ticker"]!=to_remove]
                    st.rerun()
            st.markdown("---")
            with st.spinner("Fetching live prices from FMP..."):
                pm=calculate_portfolio_metrics(st.session_state.portfolio)
            if "error" not in pm and pm["holdings_data"]:
                t1,t2,t3=st.columns(3)
                with t1: st.metric("Total Value",format_currency(pm["total_value"]))
                with t2: st.metric("Invested",format_currency(pm["total_cost"]))
                with t3:
                    dc="normal" if pm["total_gain_loss"]>=0 else "inverse"
                    st.metric("P&L",format_currency(pm["total_gain_loss"]),
                              delta=f"{pm['total_gain_loss_pct']:+.2f}%",delta_color=dc)
                g1,g2=st.columns(2)
                with g1: st.plotly_chart(create_portfolio_allocation_chart(pm["holdings_data"]),use_container_width=True)
                with g2: st.plotly_chart(create_portfolio_pnl_chart(pm["holdings_data"]),use_container_width=True)
                st.subheader("Holdings Detail")
                df_h=pd.DataFrame([{"Ticker":h["ticker"],"Shares":h["shares"],
                    "Buy Price":format_currency(h["cost_basis"]),"Current":format_currency(h["latest_price"]),
                    "Value":format_currency(h["current_value"]),"P&L":format_currency(h["gain_loss"]),
                    "P&L %":f"{h['gain_loss_pct']:+.2f}%"} for h in pm["holdings_data"]])
                st.dataframe(df_h,use_container_width=True,hide_index=True)
                st.markdown("---"); st.subheader("Individual Candle Charts")
                for h in pm["holdings_data"]:
                    with st.expander(f"🕯️ {h['ticker']} — 60-day Candlestick"):
                        st.plotly_chart(create_candlestick_chart(h["ticker"],60),use_container_width=True)
                st.markdown("---")
                if st.button("🤖 Get AI Portfolio Review",type="primary"):
                    with st.spinner("Artha is reviewing your portfolio..."):
                        advice=investement_advise("moderate","medium-term",
                            {h["ticker"]:h["current_value"] for h in pm["holdings_data"]})
                    ai_card("Portfolio Review",advice)
            else: st.info("Add holdings above to see your portfolio analysis.")
        else: st.info("Your portfolio is empty. Add your first holding above.")

    with tab3:
        st.markdown("#### Research any stock — fundamentals + candle")
        lookup=st.text_input("Enter ticker to research",placeholder="MSFT  RELIANCE.NS  ...")
        if lookup:
            with st.spinner(f"Researching {lookup.upper()}..."):
                m=get_stock_metrics(lookup.strip())
                fig_l=create_candlestick_chart(lookup.strip(),period_days=90)
            if "error" not in m:
                st.markdown(f"### {m.get('name',lookup.upper())}")
                la,lb,lc,ld=st.columns(4)
                with la: st.metric("Price",f"${m['current_price']:.2f}" if m["current_price"] else "—")
                with lb: st.metric("52W High",f"${m['price_52wk_high']:.2f}" if m["price_52wk_high"] else "—")
                with lc: st.metric("52W Low",f"${m['price_52wk_low']:.2f}" if m["price_52wk_low"] else "—")
                with ld: st.metric("Dividend",f"{m['dividend_yield']:.2f}" if m["dividend_yield"] else "—")
                r1,r2,r3=st.columns(3)
                with r1: st.metric("1M Return",f"{m['returns']['1mo']:+.2f}%" if m["returns"].get("1mo") else "—")
                with r2: st.metric("3M Return",f"{m['returns']['3mo']:+.2f}%" if m["returns"].get("3mo") else "—")
                with r3: st.metric("1Y Return",f"{m['returns']['1yr']:+.2f}%" if m["returns"].get("1yr") else "—")
                st.plotly_chart(fig_l,use_container_width=True)
                if st.button("🤖 AI Stock Analysis",type="primary"):
                    with st.spinner("Artha is analysing..."):
                        advice=generate_financial_adivice(
                            f"Give me a brief analysis of {lookup.upper()} based on these metrics: {m}. "
                            "Is it worth investing? Key risks and opportunities?",
                            {"ticker":lookup.upper(),"metrics":m})
                    ai_card(f"AI Analysis: {lookup.upper()}",advice)
            else: st.error(m["error"])

# ══════════════════════════════════════════════════════════════════════════════
#  AI ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
elif page=="🤖 AI Advisor":
    st.header("🤖 AI Financial Advisor")
    st.markdown('<p style="color:#666;font-size:0.9em">Personalised advice based on your financial data.</p>',unsafe_allow_html=True)
    st.markdown("**💡 Quick topics**")
    topics=["How can I improve my savings rate?","Best way to pay off my debt?",
            "How should I prioritise my financial goals?","How much should I save for retirement?",
            "How do I build an emergency fund faster?","Where should I invest my surplus income?"]
    tc=st.columns(3)
    for i,topic in enumerate(topics):
        with tc[i%3]:
            if st.button(topic,key=f"t_{i}",use_container_width=True):
                ctx={"income":st.session_state.income,"expenses":st.session_state.expenses,
                     "assets":st.session_state.assets,"liabilities":st.session_state.liabilities,
                     "goals":st.session_state.financial_goals,"portfolio":st.session_state.portfolio}
                with st.spinner("Thinking..."):
                    resp=generate_financial_adivice(topic,ctx)
                st.session_state.chat_history+=[
                    {"role":"user","content":topic,"ts":datetime.now().strftime("%H:%M")},
                    {"role":"assistant","content":resp,"ts":datetime.now().strftime("%H:%M")}]
                st.rerun()
    st.markdown("---")
    with st.form("chat_form",clear_on_submit=True):
        fi1,fi2=st.columns([5,1])
        with fi1:
            user_q=st.text_input("",placeholder="Ask Artha anything about your finances…",
                                 label_visibility="collapsed")
        with fi2: send=st.form_submit_button("Send →",use_container_width=True)
    if send and user_q.strip():
        ctx={"income":st.session_state.income,"expenses":st.session_state.expenses,
             "assets":st.session_state.assets,"liabilities":st.session_state.liabilities,
             "goals":st.session_state.financial_goals,"portfolio":st.session_state.portfolio}
        with st.spinner("Artha is thinking..."):
            resp=generate_financial_adivice(user_q.strip(),ctx)
        st.session_state.chat_history+=[
            {"role":"user","content":user_q.strip(),"ts":datetime.now().strftime("%H:%M")},
            {"role":"assistant","content":resp,"ts":datetime.now().strftime("%H:%M")}]
        if st.session_state.user_id:
            save_ai_insight(st.session_state.user_id,"chat",resp)
        st.rerun()
    if st.session_state.chat_history:
        st.markdown('<div class="chat-wrap">',unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"]=="user": chat_bubble_user(msg["content"],msg.get("ts",""))
            else: chat_bubble_bot(msg["content"],msg.get("ts",""))
        st.markdown("</div>",unsafe_allow_html=True)
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history=[]; st.rerun()
    else:
        st.markdown("""<div style="text-align:center;padding:40px;color:#444">
          <div style="font-size:40px">💬</div>
          <p>Ask a question or tap a topic chip to get started</p></div>""",unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='color:#444;font-size:12px;text-align:center'>"
            "⚠️ Artha provides general financial information only — not a substitute for professional advice."
            "</p>",unsafe_allow_html=True)
