import os
import json
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("The GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are Artha, an expert AI financial advisor specializing in personal finance for Indian users.
You give clear, concise, actionable advice on budgeting, saving, investing, and wealth management.
Always frame your advice in the Indian financial context (rupees, Indian markets, FD/RD, SIP, PPF, NPS, etc.) when relevant.
Be warm, practical, and direct. Avoid jargon unless you explain it."""

MODEL = "llama-3.3-70b-versatile"


def _chat(messages: list[dict], max_tokens: int = 1024) -> str:
    """Core chat wrapper for Groq API."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while contacting Groq AI: {e}"


def generate_financial_adivice(query: str, financial_content: dict = None) -> str:
    """Generates personalised financial advice based on user query and optional context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if financial_content:
        context_str = json.dumps(financial_content, indent=2, default=str)
        messages.append({
            "role": "user",
            "content": f"Here is my current financial situation:\n{context_str}"
        })
        messages.append({
            "role": "assistant",
            "content": "Thank you for sharing your financial details. I'll use this context to give you personalised advice."
        })

    messages.append({"role": "user", "content": query})
    return _chat(messages, max_tokens=1024)


def analyze_budget(income: float, expenses: dict) -> str:
    """Analyses the user's budget and provides structured recommendations."""
    budget_data = {"monthly_income_inr": income, "monthly_expenses_inr": expenses}

    prompt = f"""Analyse the following monthly budget and provide a comprehensive breakdown.

Budget Data:
{json.dumps(budget_data, indent=2)}

Please provide:
1. **Overall Assessment** – Is this budget healthy? What's the savings rate?
2. **Key Concerns** – Top 2-3 spending areas to watch
3. **Quick Wins** – Immediate actions to improve the budget
4. **50/30/20 Rule Check** – How does this compare to the ideal split (needs/wants/savings)?
5. **One-Line Summary** – A concise verdict

Keep it practical, specific, and actionable. Format with clear headings."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return _chat(messages, max_tokens=1024)


def investement_advise(risk_involved: str, investment_horizon: str, current_investments: dict = None) -> str:
    """Provides investment advice based on risk tolerance and horizon."""
    data = {
        "risk_tolerance": risk_involved,
        "investment_horizon": investment_horizon,
        "current_investments": current_investments or {}
    }

    prompt = f"""Provide personalised investment advice based on the following profile:
{json.dumps(data, indent=2)}

Cover:
1. Recommended asset allocation (% in equity, debt, gold, etc.)
2. Specific investment instruments suitable for this profile (SIP, PPF, NPS, FD, etc.)
3. Estimated expected returns
4. Key risks to be aware of
5. First concrete step to take

Be specific and actionable."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return _chat(messages, max_tokens=1024)
