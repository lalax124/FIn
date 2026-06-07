import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

FMP_API_KEY  = os.getenv("FMP_API_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# ── THEME ─────────────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#4169E1",
    "secondary": "#8A2BE2",
    "success":   "#00C896",
    "danger":    "#FF4757",
    "warning":   "#FFA502",
    "bg":        "rgba(0,0,0,0)",
    "grid":      "rgba(255,255,255,0.08)",
    "text":      "#E8EAED",
}
PALETTE = ["#FDC8E0", "#FDC2D3", "#FECBD1", "#FCB2D4", "#F0C7DC"]


LAYOUT_BASE = dict(
    plot_bgcolor=COLORS["bg"],
    paper_bgcolor=COLORS["bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=13),
    margin=dict(t=48, b=16, l=8, r=8),
    hoverlabel=dict(bgcolor="#1E2130", font_color="#fff", bordercolor="#4169E1"),
)


def _fmp_get(endpoint, params=None):
    if not FMP_API_KEY:
        return None
    p = params or {}
    p["apikey"] = FMP_API_KEY
    try:
        r = requests.get(f"{FMP_BASE_URL}/{endpoint}", params=p, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"FMP error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# EXPENSE PIE
# ══════════════════════════════════════════════════════════════════════════════
def create_expense_pie_chart(expenses: dict) -> go.Figure:
    if not expenses:
        fig = go.Figure()
        fig.add_annotation(text="No expense data yet", showarrow=False,
                           font=dict(size=16, color=COLORS["text"]))
        fig.update_layout(**LAYOUT_BASE)
        return fig

    fig = go.Figure(go.Pie(
        labels=list(expenses.keys()),
        values=list(expenses.values()),
        hole=0.55,
        marker=dict(colors=PALETTE, line=dict(color="#0d1117", width=2)),
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
        textfont=dict(size=12),
        pull=[0.04 if i == 0 else 0 for i in range(len(expenses))],
    ))

    total = sum(expenses.values())
    fig.add_annotation(
        text=f"<b>₹{total:,.0f}</b><br><span style='font-size:11px'>Total</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=15, color=COLORS["text"]), xanchor="center",
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Expense Breakdown", font=dict(size=15, color=COLORS["text"])),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.5, font=dict(size=11, color=COLORS["text"])),
        showlegend=True,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INCOME VS EXPENSES BAR
# ══════════════════════════════════════════════════════════════════════════════
def create_income_expense_bar_chart(income: float, total_expenses: float, remaining: float) -> go.Figure:
    fig = go.Figure()
    bars = [
        ("Income",    income,         COLORS["success"]),
        ("Expenses",  total_expenses, COLORS["danger"]),
        ("Remaining", remaining,      COLORS["primary"] if remaining >= 0 else COLORS["warning"]),
    ]
    for label, val, color in bars:
        fig.add_trace(go.Bar(
            x=[label], y=[val], name=label,
            marker=dict(color=color, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=f"₹{val:,.0f}", textposition="outside",
            textfont=dict(color=COLORS["text"], size=12),
            hovertemplate=f"<b>{label}</b><br>₹%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Income vs Expenses", font=dict(size=15, color=COLORS["text"])),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False,
                   tickprefix="₹", tickformat=",", color=COLORS["text"]),
        xaxis=dict(color=COLORS["text"]),
        showlegend=False, bargap=0.35,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INVESTMENT GROWTH
# ══════════════════════════════════════════════════════════════════════════════
def create_investment_growth_chart(initial, monthly, years, rate) -> go.Figure:
    monthly_rate = rate / 12
    data = []
    val = initial
    contrib = initial
    data.append({"year": 0, "value": val, "contributions": contrib, "earnings": 0})
    for yr in range(1, years + 1):
        for _ in range(12):
            val = val * (1 + monthly_rate) + monthly
            contrib += monthly
        data.append({"year": yr, "value": val, "contributions": contrib, "earnings": val - contrib})

    df = pd.DataFrame(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["earnings"], fill="tozeroy", mode="lines",
        name="Returns", line=dict(color=COLORS["success"], width=2),
        fillcolor="rgba(0,200,150,0.15)",
        hovertemplate="Year %{x}<br>Returns: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["contributions"], fill="tozeroy", mode="lines",
        name="Contributions", line=dict(color=COLORS["primary"], width=2, dash="dot"),
        fillcolor="rgba(65,105,225,0.1)",
        hovertemplate="Year %{x}<br>Contributions: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["value"], mode="lines+markers",
        name="Total Value", line=dict(color=COLORS["warning"], width=3),
        marker=dict(size=5),
        hovertemplate="Year %{x}<br>Total: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Investment Growth @ {rate*100:.1f}% p.a.", font=dict(size=15, color=COLORS["text"])),
        xaxis=dict(title="Years", color=COLORS["text"], gridcolor=COLORS["grid"]),
        yaxis=dict(title="Value (₹)", tickprefix="₹", tickformat=",",
                   color=COLORS["text"], gridcolor=COLORS["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(color=COLORS["text"])),
        hovermode="x unified",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EXPENSE TREND
# ══════════════════════════════════════════════════════════════════════════════
def create_expense_trend_chart(expense_history: dict) -> go.Figure:
    if not expense_history:
        fig = go.Figure()
        fig.add_annotation(text="No expense history yet", showarrow=False,
                           font=dict(size=16, color=COLORS["text"]))
        fig.update_layout(**LAYOUT_BASE)
        return fig

    df = pd.DataFrame({
        "date":   pd.to_datetime(list(expense_history.keys())),
        "amount": list(expense_history.values()),
    }).sort_values("date")

    fig = go.Figure(go.Scatter(
        x=df["date"], y=df["amount"], mode="lines+markers",
        fill="tozeroy", line=dict(color=COLORS["primary"], width=2),
        fillcolor="rgba(65,105,225,0.12)",
        marker=dict(size=6, color=COLORS["primary"]),
        hovertemplate="%{x|%b %Y}<br>₹%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Expense Trend", font=dict(size=15, color=COLORS["text"])),
        xaxis=dict(color=COLORS["text"], gridcolor=COLORS["grid"]),
        yaxis=dict(tickprefix="₹", tickformat=",",
                   color=COLORS["text"], gridcolor=COLORS["grid"]),
        hovermode="x unified",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SAVINGS GOAL PROGRESS
# ══════════════════════════════════════════════════════════════════════════════
def create_savings_goal_progress_chart(goals: dict, current_savings: dict) -> go.Figure:
    if not goals or not current_savings:
        fig = go.Figure()
        fig.add_annotation(text="No savings goals yet", showarrow=False,
                           font=dict(size=16, color=COLORS["text"]))
        fig.update_layout(**LAYOUT_BASE)
        return fig

    names, targets, currents, pcts = [], [], [], []
    for name, target in goals.items():
        curr = current_savings.get(name, 0)
        pct  = (curr / target * 100) if target > 0 else 0
        names.append(name); targets.append(target)
        currents.append(curr); pcts.append(pct)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=targets, y=names, orientation="h", name="Target",
        marker=dict(color="rgba(255,255,255,0.08)"), showlegend=False,
        hovertemplate="Target: ₹%{x:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=currents, y=names, orientation="h", name="Progress",
        marker=dict(color=COLORS["success"]),
        text=[f"{p:.0f}%" for p in pcts], textposition="inside",
        hovertemplate="Current: ₹%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Goal Progress", font=dict(size=15, color=COLORS["text"])),
        barmode="overlay",
        xaxis=dict(tickprefix="₹", tickformat=",",
                   color=COLORS["text"], gridcolor=COLORS["grid"]),
        yaxis=dict(color=COLORS["text"]),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CANDLESTICK CHART  (FMP data)
# ══════════════════════════════════════════════════════════════════════════════
def create_candlestick_chart(ticker: str, period_days: int = 90) -> go.Figure:
    ticker   = ticker.upper().strip()
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=period_days + 30)).strftime("%Y-%m-%d")

    data = _fmp_get(f"historical-price-full/{ticker}",
                    {"from": start_date, "to": end_date})

    if not data or "historical" not in data or not data["historical"]:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for {ticker}. Check the ticker symbol.",
            showarrow=False, font=dict(size=14, color=COLORS["danger"]),
        )
        fig.update_layout(**LAYOUT_BASE, title=f"{ticker} — Not Found")
        return fig

    df = pd.DataFrame(data["historical"]).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df = df.tail(period_days).reset_index(drop=True)

    # indicators
    df["ema20"]    = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"]    = df["close"].ewm(span=50, adjust=False).mean()
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_std"]   = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name=ticker,
        increasing_line_color=COLORS["success"],
        decreasing_line_color=COLORS["danger"],
        increasing_fillcolor=COLORS["success"],
        decreasing_fillcolor=COLORS["danger"],
    ), row=1, col=1)

    # Bollinger band fill
    fig.add_trace(go.Scatter(
        x=pd.concat([df["date"], df["date"][::-1]]),
        y=pd.concat([df["bb_upper"], df["bb_lower"][::-1]]),
        fill="toself", fillcolor="rgba(65,105,225,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Bollinger Band", showlegend=True, hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["ema20"], name="EMA 20",
        line=dict(color=COLORS["warning"], width=1.5, dash="dot"),
        hovertemplate="%{x|%d %b}<br>EMA20: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["ema50"], name="EMA 50",
        line=dict(color=COLORS["secondary"], width=1.5, dash="dash"),
        hovertemplate="%{x|%d %b}<br>EMA50: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Volume
    colors_vol = [COLORS["success"] if c >= o else COLORS["danger"]
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"], name="Volume",
        marker_color=colors_vol, opacity=0.6,
        hovertemplate="%{x|%d %b}<br>Vol: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    # Latest price annotation
    if len(df) >= 2:
        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        change = latest["close"] - prev["close"]
        pct    = change / prev["close"] * 100
        sign   = "▲" if change >= 0 else "▼"
        color  = COLORS["success"] if change >= 0 else COLORS["danger"]
        fig.add_annotation(
            x=latest["date"], y=latest["high"],
            text=f"<b>{sign} {latest['close']:.2f}  ({pct:+.2f}%)</b>",
            showarrow=True, arrowhead=2, arrowcolor=color,
            font=dict(color=color, size=13),
            bgcolor="#1E2130", bordercolor=color, borderwidth=1,
            row=1, col=1,
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"<b>{ticker}</b> — Candlestick + Volume + Indicators",
            font=dict(size=16, color=COLORS["text"]),
        ),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=11, color=COLORS["text"])),
        yaxis=dict(color=COLORS["text"],  gridcolor=COLORS["grid"], side="right"),
        yaxis2=dict(color=COLORS["text"], gridcolor=COLORS["grid"],
                    title="Volume", side="right"),
        xaxis2=dict(color=COLORS["text"], gridcolor=COLORS["grid"]),
        hovermode="x unified",
        height=580,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ALLOCATION PIE
# ══════════════════════════════════════════════════════════════════════════════
def create_portfolio_allocation_chart(holdings_data: list) -> go.Figure:
    if not holdings_data:
        fig = go.Figure()
        fig.add_annotation(text="No portfolio data", showarrow=False,
                           font=dict(size=14, color=COLORS["text"]))
        fig.update_layout(**LAYOUT_BASE)
        return fig

    labels = [h["ticker"] for h in holdings_data]
    values = [h["current_value"] for h in holdings_data]
    total  = sum(values)

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=PALETTE, line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>₹{total:,.0f}</b><br><span style='font-size:11px'>Portfolio</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=COLORS["text"]), xanchor="center",
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Portfolio Allocation", font=dict(size=15, color=COLORS["text"])),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                    xanchor="center", x=0.5, font=dict(size=11, color=COLORS["text"])),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO P&L BAR
# ══════════════════════════════════════════════════════════════════════════════
def create_portfolio_pnl_chart(holdings_data: list) -> go.Figure:
    if not holdings_data:
        fig = go.Figure()
        fig.update_layout(**LAYOUT_BASE)
        return fig

    tickers = [h["ticker"] for h in holdings_data]
    pnl     = [h["gain_loss"] for h in holdings_data]
    pnl_pct = [h["gain_loss_pct"] for h in holdings_data]
    colors  = [COLORS["success"] if v >= 0 else COLORS["danger"] for v in pnl]

    fig = go.Figure(go.Bar(
        x=tickers, y=pnl,
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{p:+.1f}%" for p in pnl_pct],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=12),
        hovertemplate="<b>%{x}</b><br>P&L: ₹%{y:,.0f}<br>%{text}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Unrealised P&L by Holding",
                   font=dict(size=15, color=COLORS["text"])),
        yaxis=dict(tickprefix="₹", tickformat=",",
                   color=COLORS["text"], gridcolor=COLORS["grid"]),
        xaxis=dict(color=COLORS["text"]),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# NET WORTH GAUGE
# ══════════════════════════════════════════════════════════════════════════════
def create_net_worth_gauge(net_worth: float, target: float = None) -> go.Figure:
    tgt = target or max(abs(net_worth) * 2, 1_000_000)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=net_worth,
        number=dict(prefix="₹", valueformat=",.0f",
                    font=dict(color=COLORS["text"], size=28)),
        delta=dict(reference=0, valueformat=",.0f", prefix="₹"),
        gauge=dict(
            axis=dict(range=[0, tgt], tickformat=".2s", tickprefix="₹",
                      tickcolor=COLORS["text"], tickwidth=1),
            bar=dict(color=COLORS["primary"], thickness=0.25),
            bgcolor="rgba(255,255,255,0.05)",
            borderwidth=0,
            steps=[
                dict(range=[0, tgt * 0.33], color="rgba(255,71,87,0.15)"),
                dict(range=[tgt * 0.33, tgt * 0.66], color="rgba(255,165,2,0.15)"),
                dict(range=[tgt * 0.66, tgt], color="rgba(0,200,150,0.15)"),
            ],
            threshold=dict(line=dict(color=COLORS["success"], width=3),
                           value=tgt * 0.66),
        ),
        title=dict(text="Net Worth", font=dict(size=15, color=COLORS["text"])),
    ))
    fig.update_layout(**LAYOUT_BASE, height=280)
    return fig
