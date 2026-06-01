import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from database import (
    initialize_database,
    get_or_create_user,
    save_income,
    save_expenses,
    save_assets,
    save_liabilities,
    save_financial_goals,
    save_investment_portfolio,
    save_ai_insight,
    get_user_income,
    get_user_expenses,
    get_user_assets,
    get_user_liabilities,
    get_user_financial_goals,
    get_user_portfolio,
    get_user_insights,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Artha - AI-Powered Financial Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM STYLE ──────────────────────────────────────────────────────────────
def apply_custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .gradient-text {
        background: linear-gradient(to right, #4169E1, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    .chat-user {
        background: #4169E1; color: white;
        padding: 10px 16px; border-radius: 18px 18px 4px 18px;
        margin: 6px 0; max-width: 80%; margin-left: auto;
        font-size: 14px; line-height: 1.5;
    }
    .chat-bot {
        background: #f0f2f6; color: #1a1a1a;
        padding: 10px 16px; border-radius: 18px 18px 18px 4px;
        margin: 6px 0; max-width: 80%;
        font-size: 14px; line-height: 1.5;
        border-left: 3px solid #4169E1;
    }
    .chat-ts { font-size: 11px; color: #aaa; margin: 2px 0 10px; }
    .chat-ts.right { text-align: right; }

    .ai-card {
        background: #f0f4ff;
        border-left: 4px solid #4169E1;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #1a1a1a;
        font-size: 14px;
        line-height: 1.7;
    }
    .ai-card h4 { color: #4169E1; margin-top: 0; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; }

    .metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }
    </style>
    """, unsafe_allow_html=True)

# ── INIT ──────────────────────────────────────────────────────────────────────
initialize_database()
apply_custom_style()


from groq_ai import generate_financial_adivice, investement_advise, analyze_budget

from data_processing import (
    format_currency,
    calculate_budget_summary,
    calculate_investment_returns,
    calculate_loan_payment,
    generate_amortization_schedule,
    categorize_expenses,
)
from frontend import (
    create_expense_pie_chart,
    create_income_expense_bar_chart,
    create_investment_growth_chart,
    create_expense_trend_chart,
    create_savings_goal_progress_chart,
)
from moneyanalyser import (
    calculate_net_worth,
    calculate_emergency_fund_ratio,
    calculate_portfolio_metrics,
    calculate_debt_to_income_ratio,
    analyze_retirement_readiness,
    analyze_mortgage_affordability,
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "income": 0.0,
    "expenses": {},
    "assets": {},
    "liabilities": {},
    "financial_goals": [],
    "chat_history": [],
    "portfolio": [],
    "user_id": None,
    "username": None,
    "authenticated": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<h1><span class="gradient-text">Artha</span>: Your AI Financial Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size:1.1em; color:#555; font-style:italic;">Make smarter financial decisions with AI-powered insights</p>', unsafe_allow_html=True)

# ── AUTH ──────────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    st.sidebar.title("User Authentication")
    auth_option = st.sidebar.radio("Choose an option:", ["Login", "Register"])

    username = st.sidebar.text_input("Username")

    if auth_option == "Register":
        email = st.sidebar.text_input("Email (optional)")
    else:
        email = None

    btn_label = "Login" if auth_option == "Login" else "Register"
    if st.sidebar.button(btn_label):
        if username.strip():
            user_id = get_or_create_user(username.strip(), email)
            st.session_state.user_id = user_id
            st.session_state.username = username.strip()
            st.session_state.authenticated = True

            st.session_state.income = get_user_income(user_id)
            st.session_state.expenses = get_user_expenses(user_id)
            st.session_state.assets = get_user_assets(user_id)
            st.session_state.liabilities = get_user_liabilities(user_id)
            st.session_state.financial_goals = get_user_financial_goals(user_id)
            st.session_state.portfolio = get_user_portfolio(user_id)

            st.sidebar.success(f"Welcome {'back' if auth_option == 'Login' else ''}, {username}!")
            st.rerun()
        else:
            st.sidebar.error("Please enter a username")

    st.info("Please login or register to access all features.")
    st.markdown("""
    ### Welcome to Artha, your AI-powered financial assistant!
    Artha helps you:
    * Track your income and expenses
    * Monitor your net worth
    * Set and track financial goals
    * Get AI-powered financial advice

    Login or create an account to get started!
    """)
    st.stop()

# ── SIDEBAR NAV ───────────────────────────────────────────────────────────────
st.sidebar.title(f"👋 {st.session_state.username}")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a section:", ["Dashboard", "Budget Analyzer", "AI Financial Advisor"])

if st.sidebar.button("💾 Save Data"):
    with st.spinner("Saving your data..."):
        save_income(st.session_state.user_id, st.session_state.income)
        save_expenses(st.session_state.user_id, st.session_state.expenses)
        save_assets(st.session_state.user_id, st.session_state.assets)
        save_liabilities(st.session_state.user_id, st.session_state.liabilities)

        if st.session_state.financial_goals:
            save_financial_goals(st.session_state.user_id, st.session_state.financial_goals)

        if st.session_state.portfolio:
            save_investment_portfolio(st.session_state.user_id, st.session_state.portfolio)

    st.sidebar.success("✅ Data saved!")

if st.sidebar.button("🚪 Logout"):
    for key in defaults:
        st.session_state[key] = defaults[key]
    st.rerun()

# ── HELPER ────────────────────────────────────────────────────────────────────
def display_ai_advice(title, advice_text):
    st.markdown(f"### {title}")
    st.markdown(f'<div class="ai-card"><h4>🤖 AI-Generated Advice</h4>{advice_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)


# DASHBOARD

if page == "Dashboard":
    st.header("📊 Financial Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Financial Summary")

        net_worth, total_assets, total_liabilities = calculate_net_worth(
            st.session_state.assets, st.session_state.liabilities
        )
        st.metric("Net Worth", format_currency(net_worth))

        if st.session_state.expenses:
            total_expenses = sum(st.session_state.expenses.values())
            savings = st.session_state.income - total_expenses
            st.metric("Monthly Income", format_currency(st.session_state.income))
            st.metric("Monthly Expenses", format_currency(total_expenses))
            st.metric("Monthly Savings", format_currency(savings),
                      delta=f"{(savings/st.session_state.income*100):.1f}% savings rate" if st.session_state.income > 0 else None)
        else:
            total_expenses = 0

        if st.session_state.expenses and "Emergency Fund" in st.session_state.assets:
            monthly_expenses = sum(st.session_state.expenses.values())
            emergency_fund = st.session_state.assets.get("Emergency Fund", 0)
            months_covered = calculate_emergency_fund_ratio(emergency_fund, monthly_expenses)
            if months_covered is not None:
                st.metric("Emergency Fund Coverage", f"{months_covered:.1f} months",
                          delta="✅ Good" if months_covered >= 6 else "⚠️ Build this up")

        if st.session_state.income > 0:
            savings = st.session_state.income - total_expenses
            df = pd.DataFrame({
                "Metric": ["Net Worth", "Monthly Income", "Monthly Expenses", "Monthly Savings"],
                "Amount": [format_currency(net_worth), format_currency(st.session_state.income),
                           format_currency(total_expenses), format_currency(savings)]
            })
            st.table(df)

    with col2:
        if st.session_state.expenses:
            st.subheader("Expense Breakdown")
            fig = create_expense_pie_chart(st.session_state.expenses)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("➡️ Add your expenses in the **Budget Analyzer** to see a breakdown here.")

    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# BUDGET ANALYZER

elif page == "Budget Analyzer":
    st.header("💰 Budget Analyzer")

    st.subheader("Monthly Income")
    income = st.number_input(
        "Enter your monthly salary (Rs):",
        min_value=0.0,
        value=float(st.session_state.income),
        step=1000.0,
        format="%.2f",
    )
    st.session_state.income = income

    st.subheader("Monthly Expenses")
    expense_categories = [
        "Housing (Rent/Mortgage)",
        "Utilities (Electricity, Water, Gas)",
        "Groceries",
        "Transportation",
        "Health Care",
        "Entertainment",
        "Dining Out",
        "Shopping",
        "Education",
        "Insurance",
        "Savings",
        "Other",
    ]

    updated_expenses = {}
    left_cats = expense_categories[:6]
    right_cats = expense_categories[6:]
    col_l, col_r = st.columns(2)

    with col_l:
        for category in left_cats:
            val = st.number_input(
                f"{category} (Rs):",
                min_value=0.0,
                value=float(st.session_state.expenses.get(category, 0.0)),
                step=100.0,
                format="%.2f",
                key=f"exp_{category}",
            )
            if val > 0:
                updated_expenses[category] = val

    with col_r:
        for category in right_cats:
            val = st.number_input(
                f"{category} (Rs):",
                min_value=0.0,
                value=float(st.session_state.expenses.get(category, 0.0)),
                step=100.0,
                format="%.2f",
                key=f"exp_{category}",
            )
            if val > 0:
                updated_expenses[category] = val

    st.markdown("---")
    custom_category = st.text_input("➕ Add a custom expense category:")
    if custom_category:
        custom_value = st.number_input(
            f"{custom_category} (Rs):",
            min_value=0.0,
            value=float(st.session_state.expenses.get(custom_category, 0.0)),
            step=100.0,
            format="%.2f",
        )
        if custom_value > 0:
            updated_expenses[custom_category] = custom_value

    st.session_state.expenses = updated_expenses

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    budget_summary = calculate_budget_summary(income, updated_expenses)
    total_expenses = budget_summary["total_expenses"]
    remaining = budget_summary["remaining"]
    savings_rate = budget_summary["savings_rate"]

    st.markdown("---")
    st.subheader("Budget Summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Income", format_currency(income))
    with c2:
        st.metric("Expenses", format_currency(total_expenses))
    with c3:
        st.metric("Remaining", format_currency(remaining),
                  delta="surplus" if remaining >= 0 else "deficit")
    with c4:
        st.metric("Savings Rate", f"{savings_rate:.1f}%",
                  delta="✅ Good" if savings_rate >= 20 else "⚠️ Improve")

    if updated_expenses:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.subheader("Income vs Expenses")
            fig_bar = create_income_expense_bar_chart(income, total_expenses, remaining)
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_b2:
            st.subheader("Expense Breakdown")
            fig_pie = create_expense_pie_chart(updated_expenses)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Expense Details")
        df_exp = pd.DataFrame(
            [{"Category": k, "Amount": format_currency(v), "% of Income": f"{v/income*100:.1f}%" if income > 0 else "—"}
             for k, v in updated_expenses.items()]
        )
        st.dataframe(df_exp, use_container_width=True, hide_index=True)

    # ── AI ANALYSIS ──────────────────────────────────────────────────────────
    if updated_expenses and income > 0:
        st.markdown("---")
        st.subheader("🤖 AI Budget Analysis")
        with st.spinner("Analysing your budget with Groq AI..."):
            budget_analysis = analyze_budget(income, updated_expenses)

            if isinstance(budget_analysis, str):
                display_ai_advice("Budget Recommendations", budget_analysis)
            elif isinstance(budget_analysis, dict):
                if "analysis" in budget_analysis:
                    display_ai_advice("Budget Analysis", budget_analysis["analysis"])
                if "recommendations" in budget_analysis:
                    display_ai_advice("Recommendations", budget_analysis["recommendations"])

            if st.session_state.user_id:
                save_ai_insight(st.session_state.user_id, "budget_analysis", str(budget_analysis))


# AI FINANCIAL ADVISOR
elif page == "AI Financial Advisor":
    st.header("🤖 AI Financial Advisor")
    st.markdown("Ask me any financial question and I'll provide personalised advice based on your data.")

    st.subheader("💡 Suggested Topics")
    topics = [
        "How can I improve my savings rate?",
        "What's the best way to pay off my debt?",
        "How should I prioritize my financial goals?",
        "How much should I save for retirement?",
        "How can I build an emergency fund faster?",
        "Where should I invest my surplus income?",
    ]
    topic_cols = st.columns(3)
    for i, topic in enumerate(topics):
        with topic_cols[i % 3]:
            if st.button(topic, key=f"topic_{i}", use_container_width=True):
                financial_context = {
                    "income": st.session_state.income,
                    "expenses": st.session_state.expenses,
                    "assets": st.session_state.assets,
                    "liabilities": st.session_state.liabilities,
                    "goals": st.session_state.financial_goals,
                    "portfolio": st.session_state.portfolio,
                }
                with st.spinner("Thinking..."):
                    response = generate_financial_adivice(topic, financial_context)
                st.session_state.chat_history.append({"role": "user", "content": topic, "timestamp": datetime.now().strftime("%H:%M")})
                st.session_state.chat_history.append({"role": "assistant", "content": response, "timestamp": datetime.now().strftime("%H:%M")})
                st.rerun()

    st.markdown("---")

    user_query = st.text_input("💬 Your financial question:", placeholder="e.g., How can I reduce my monthly expenses?")
    if st.button("Ask Artha →", type="primary") and user_query.strip():
        financial_context = {
            "income": st.session_state.income,
            "expenses": st.session_state.expenses,
            "assets": st.session_state.assets,
            "liabilities": st.session_state.liabilities,
            "goals": st.session_state.financial_goals,
            "portfolio": st.session_state.portfolio,
        }
        with st.spinner("Thinking..."):
            response = generate_financial_adivice(user_query.strip(), financial_context)

        st.session_state.chat_history.append({"role": "user", "content": user_query.strip(), "timestamp": datetime.now().strftime("%H:%M")})
        st.session_state.chat_history.append({"role": "assistant", "content": response, "timestamp": datetime.now().strftime("%H:%M")})
        st.rerun()

    if st.session_state.chat_history:
        st.subheader("💬 Conversation")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-ts right">{msg["timestamp"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">{msg["content"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-ts">{msg["timestamp"]} · Artha</div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Ask a question above or pick a suggested topic to get started.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("*⚠️ Disclaimer: Artha provides general financial information only and is not a substitute for professional financial advice.*")
