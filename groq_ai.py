import os
import json
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are Artha, an expert AI financial advisor specializing in personal finance for Indian users.
You give clear, concise, actionable advice on budgeting, saving, investing, and wealth management.
Always frame your advice in the Indian financial context (rupees, Indian markets, FD/RD, SIP, PPF, NPS, etc.) when relevant.
Be warm, practical, and direct."""

# ── FIND FIRST WORKING CHAT MODEL ─────────────────────────────────────────────
def _get_working_model() -> str:
    skip = ["whisper", "tts", "embed", "vision", "guard", "distil", "playai"]
    try:
        all_models = [m.id for m in client.models.list().data]
        chat_models = [m for m in all_models if not any(s in m.lower() for s in skip)]
    except Exception:
        chat_models = []

    for model in chat_models:
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            return model
        except Exception:
            continue

    # absolute fallback — return first available even untested
    if chat_models:
        return chat_models[0]

    # last resort — show what models exist to help debug
    try:
        all_ids = [m.id for m in client.models.list().data]
        raise RuntimeError(f"No chat model worked. All models on your key: {all_ids}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Could not list models: {e}")


MODEL = _get_working_model()


def _chat(messages: list, max_tokens: int = 1024) -> str:
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
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if financial_content:
        context_str = json.dumps(financial_content, indent=2, default=str)
        messages.append({
            "role": "user",
            "content": f"Here is my current financial situation:\n{context_str}"
        })
        messages.append({
            "role": "assistant",
            "content": "Thank you. I will use this context to give you personalised advice."
        })
    messages.append({"role": "user", "content": query})
    return _chat(messages, max_tokens=1024)


def analyze_budget(income: float, expenses: dict) -> str:
    prompt = f"""Analyse this monthly budget and provide a comprehensive breakdown.

Monthly Income: Rs {income:,.0f}
Monthly Expenses:
{json.dumps(expenses, indent=2)}

Please provide:
1. Overall Assessment - savings rate and health of budget
2. Key Concerns - top 2-3 spending areas to watch
3. Quick Wins - immediate actions to improve budget
4. 50/30/20 Rule Check - how does this compare to ideal split
5. One-Line Summary - concise verdict

Keep it practical, specific and actionable."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    return _chat(messages, max_tokens=1024)


def investement_advise(risk_involved: str, investment_horizon: str,
                       current_investments: dict = None) -> str:
    data = {
        "risk_tolerance":       risk_involved,
        "investment_horizon":   investment_horizon,
        "current_investments":  current_investments or {},
    }
    prompt = f"""Provide personalised investment advice based on this profile:
{json.dumps(data, indent=2)}

Cover:
1. Recommended asset allocation (equity, debt, gold %)
2. Specific instruments suitable for this profile (SIP, PPF, NPS, FD, etc.)
3. Estimated expected returns
4. Key risks to be aware of
5. First concrete step to take

Be specific and actionable."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    return _chat(messages, max_tokens=1024)
