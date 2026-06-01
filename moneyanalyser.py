import os
import numpy as np
import requests
from datetime import datetime, timedelta
#**********API Gettting******
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


def _fmp_get(endpoint: str, params: dict = None) -> dict | list | None:
    """Generic FMP API GET helper."""
    if not FMP_API_KEY:
        raise ValueError("FMP_API_KEY environment variable is not set.")
    p = params or {}
    p["apikey"] = FMP_API_KEY
    try:
        r = requests.get(f"{FMP_BASE_URL}/{endpoint}", params=p, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"FMP API error ({endpoint}): {e}")
        return None




def calculate_net_worth(assets: dict, liabilities: dict) -> tuple:
    """Calculate net worth from assets and liabilities."""
    assets_total = sum(assets.values()) if assets else 0
    liabilities_total = sum(liabilities.values()) if liabilities else 0
    net_worth = assets_total - liabilities_total
    return net_worth, assets_total, liabilities_total




def calculate_debt_to_income_ratio(monthly_debt_payments: float, monthly_income: float) -> float | None:
    """Debt-to-income ratio as a decimal."""
    if monthly_income <= 0:
        return None
    return monthly_debt_payments / monthly_income


def calculate_emergency_fund_ratio(emergency_fund: float, monthly_expenses: float) -> float | None:
    """Emergency fund coverage in months."""
    if monthly_expenses <= 0:
        return None
    return emergency_fund / monthly_expenses




def analyze_retirement_readiness(
    current_age, retirement_age, life_expectancy,
    current_savings, monthly_contribution, expected_return,
    desired_retirement_income
) -> dict:
    """Full retirement readiness analysis."""
    try:
        years_until_retirement = retirement_age - current_age
        retirement_years = life_expectancy - retirement_age

        fv_current = current_savings * (1 + expected_return) ** years_until_retirement

        monthly_return = expected_return / 12
        months = years_until_retirement * 12
        if monthly_return > 0:
            fv_contributions = monthly_contribution * (
                ((1 + monthly_return) ** months - 1) / monthly_return
            )
        else:
            fv_contributions = monthly_contribution * months

        projected_savings = fv_current + fv_contributions
        sustainable_annual_income = projected_savings * 0.04  # 4% rule
        income_gap = desired_retirement_income - sustainable_annual_income
        income_gap_pct = (income_gap / desired_retirement_income * 100
                          if desired_retirement_income > 0 else 0)

        return {
            "years_until_retirement": years_until_retirement,
            "retirement_years": retirement_years,
            "projected_savings": projected_savings,
            "sustainable_annual_income": sustainable_annual_income,
            "income_gap": income_gap,
            "income_gap_percentage": income_gap_pct,
            "on_track": sustainable_annual_income >= desired_retirement_income,
        }
    except Exception as e:
        return {"error": f"Retirement analysis error: {e}"}




def analyze_mortgage_affordability(
    income, debt, down_payment, interest_rate, term_years,
    property_tax_rate=0.01, insurance=100, pmi_rate=0.005
) -> dict:
    """Mortgage affordability using 28/36 rule."""
    try:
        monthly_income = income / 12
        front_end_max = monthly_income * 0.28
        back_end_max = monthly_income * 0.36 - debt
        max_housing_payment = min(front_end_max, back_end_max)

        monthly_rate = interest_rate / 12
        total_payments = term_years * 12
        price_ranges = []
        max_price = 0

        for multiple in range(1, 16):
            home_price = multiple * 50000
            loan_amount = home_price - down_payment
            if loan_amount <= 0:
                continue

            pmi_required = (down_payment / home_price) < 0.2
            monthly_pmi = (loan_amount * pmi_rate / 12) if pmi_required else 0

            if monthly_rate == 0:
                monthly_mortgage = loan_amount / total_payments
            else:
                monthly_mortgage = (
                    loan_amount
                    * (monthly_rate * (1 + monthly_rate) ** total_payments)
                    / ((1 + monthly_rate) ** total_payments - 1)
                )

            monthly_tax = (home_price * property_tax_rate) / 12
            monthly_total = monthly_mortgage + monthly_tax + insurance + monthly_pmi
            affordable = monthly_total <= max_housing_payment

            price_ranges.append({
                "home_price": home_price,
                "monthly_payment": monthly_total,
                "affordable": affordable,
                "details": {
                    "mortgage": monthly_mortgage,
                    "property_tax": monthly_tax,
                    "insurance": insurance,
                    "pmi": monthly_pmi,
                },
            })
            if affordable:
                max_price = home_price

        return {
            "max_housing_payment": max_housing_payment,
            "max_affordable_price": max_price,
            "price_ranges": price_ranges,
        }
    except Exception as e:
        return {"error": f"Mortgage affordability error: {e}"}



def get_stock_metrics(ticker: str) -> dict:
    """
    Get key metrics for a stock using FMP.
    Replaces the previous yfinance implementation.
    """
    ticker = ticker.upper().strip()

    # Quote endpoint
    quote_data = _fmp_get(f"quote/{ticker}")
    if not quote_data or not isinstance(quote_data, list):
        return {"error": f"Could not retrieve quote for {ticker}. Check the ticker symbol."}

    q = quote_data[0]
    current_price = q.get("price")
    price_52wk_high = q.get("yearHigh")
    price_52wk_low = q.get("yearLow")
    name = q.get("name", ticker)
    pe_ratio = q.get("pe")
    price_change_1d_pct = q.get("changesPercentage")  # FMP gives 1-day % change

    # Historical data for returns
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    hist_data = _fmp_get(
        f"historical-price-full/{ticker}",
        params={"from": start_date, "to": end_date, "serietype": "line"}
    )

    return_1mo = return_3mo = return_1yr = None
    if hist_data and "historical" in hist_data:
        prices = hist_data["historical"]  # newest first
        if len(prices) >= 21:
            return_1mo = ((current_price / prices[20]["close"]) - 1) * 100
        if len(prices) >= 63:
            return_3mo = ((current_price / prices[62]["close"]) - 1) * 100
        if len(prices) >= 252:
            return_1yr = ((current_price / prices[251]["close"]) - 1) * 100

    # Dividend yield via profile
    profile_data = _fmp_get(f"profile/{ticker}")
    dividend_yield = None
    if profile_data and isinstance(profile_data, list) and profile_data:
        dividend_yield = profile_data[0].get("lastDiv")

    return {
        "name": name,
        "current_price": current_price,
        "price_52wk_high": price_52wk_high,
        "price_52wk_low": price_52wk_low,
        "pe_ratio": pe_ratio,
        "dividend_yield": dividend_yield,
        "returns": {
            "1d_pct": price_change_1d_pct,
            "1mo": return_1mo,
            "3mo": return_3mo,
            "1yr": return_1yr,
        },
    }


# ── PORTFOLIO METRICS (FMP)

def calculate_portfolio_metrics(holdings: list) -> dict:
    """
    Calculate portfolio metrics using FMP for live prices.
    holdings = [{"ticker": "AAPL", "shares": 10, "cost_basis": 150.0}, ...]
    """
    if not holdings:
        return {
            "total_value": 0,
            "total_cost": 0,
            "total_gain_loss": 0,
            "total_gain_loss_pct": 0,
            "holdings_data": [],
        }

    holdings_data = []
    total_value = 0.0
    total_cost = 0.0

    for h in holdings:
        ticker = h["ticker"].upper()
        shares = h["shares"]
        cost_basis = h["cost_basis"]

        quote = _fmp_get(f"quote/{ticker}")
        if not quote or not isinstance(quote, list):
            continue

        latest_price = quote[0].get("price", 0)
        current_value = shares * latest_price
        total_cost_basis = shares * cost_basis
        gain_loss = current_value - total_cost_basis
        gain_loss_pct = (gain_loss / total_cost_basis * 100) if total_cost_basis > 0 else 0

        holdings_data.append({
            "ticker": ticker,
            "shares": shares,
            "cost_basis": cost_basis,
            "latest_price": latest_price,
            "current_value": current_value,
            "gain_loss": gain_loss,
            "gain_loss_pct": gain_loss_pct,
        })

        total_value += current_value
        total_cost += total_cost_basis

    total_gain_loss = total_value - total_cost
    total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0

    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_gain_loss": total_gain_loss,
        "total_gain_loss_pct": total_gain_loss_pct,
        "holdings_data": holdings_data,
    }
