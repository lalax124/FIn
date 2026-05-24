import os
import json
import google.generativeai as genai

# ── API KEY ───────────────────────────────────────────────────────────────────
# BUG FIX: original raised ValueError at import time if key missing —
# crashed the entire app before Streamlit could even render.
# Now falls back gracefully so users see the UI and a helpful error message.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

_model = None

def _get_model():
    """Lazy-load the Gemini model so missing API key doesn't crash at import."""
    global _model
    if _model is not None:
        return _model
    if not GOOGLE_API_KEY:
        raise ValueError(
            "Gemini API key not found. Add GOOGLE_API_KEY to your Streamlit secrets "
            "(.streamlit/secrets.toml) or environment variables."
        )
    genai.configure(api_key=GOOGLE_API_KEY)
    _model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        },
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ],
    )
    return _model

SYSTEM_PROMPT = """
You are a helpful financial advisor named ARTHA. You are an expert in personal finance,
budgeting, and investment — especially for the Indian market (SIP, PPF, ELSS, NPS, FD, RD).
Provide clear, concise, and actionable advice. Use ₹ / Rs for currency.
"""

def generate_financial_adivice(query, financial_content=None):
    """Generate financial advice based on the user's query and optional financial context."""
    try:
        model = _get_model()
        chat = model.start_chat(history=[])

        # BUG FIX: original called chat.send_message then read chat.last.text —
        # chat.last does not exist; send_message() returns the response directly.
        chat.send_message(SYSTEM_PROMPT)

        if financial_content:
            chat.send_message(f"User's financial context: {json.dumps(financial_content, default=str)}")

        response = chat.send_message(query)
        return response.text

    except ValueError as e:
        # API key not configured
        return f"⚠️ **Configuration Error:** {e}\n\nPlease add your `GOOGLE_API_KEY` to Streamlit secrets."
    except Exception as e:
        return f"⚠️ An error occurred while generating advice: {e}"


def analyze_budget(income, expenses):
    """Analyse the user's budget and return recommendations."""
    budget_data = {"salary_per_month": income, "expenses": expenses}
    prompt = f"""
Analyse the following budget and provide:
1. An overall analysis of the budget.
2. Specific recommendations for improvement.
3. Areas where the user can potentially save more.
4. A breakdown of the expenses as a percentage of income.

Budget Data:
{json.dumps(budget_data, indent=2)}

Return ONLY a JSON object with keys: overall_analysis, recommendations, improvements, expense_breakdown.
No markdown, no code fences — raw JSON only.
"""
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip accidental markdown fences if model ignores instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Return the raw text so the UI still shows something useful
            return raw

    except ValueError as e:
        return f"⚠️ **Configuration Error:** {e}"
    except Exception as e:
        return f"⚠️ An error occurred while analysing budget: {e}"


def investement_advise(risk_involved, investment_horizon, current_investments=None):
    """Provide investment advice based on risk tolerance and investment horizon."""
    investment_data = {
        "risk_involved": risk_involved,
        "investment_horizon": investment_horizon,
        "current_investments": current_investments,
    }
    prompt = f"""
Provide investment advice for an Indian investor based on:
{json.dumps(investment_data, indent=2)}

Consider the risk tolerance, investment horizon, and suggest specific instruments
(SIP, PPF, ELSS, NPS, FD, stocks, index funds etc.) with approximate expected returns.
"""
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        return response.text
    except ValueError as e:
        return f"⚠️ **Configuration Error:** {e}"
    except Exception as e:
        return f"⚠️ An error occurred while generating investment advice: {e}"


if __name__ == "__main__":
    print("Testing generate_financial_adivice:")
    advice = generate_financial_adivice(
        "How can I save more money each month?",
        {"income": 50000, "expenses": {"rent": 15000, "food": 10000}},
    )
    print(advice)

    print("\nTesting analyze_budget:")
    result = analyze_budget(50000, {"rent": 15000, "food": 10000, "utilities": 3000})
    print(result)

    print("\nTesting investement_advise:")
    inv = investement_advise("low", "long-term", {"stocks": 10000, "bonds": 5000})
    print(inv)
