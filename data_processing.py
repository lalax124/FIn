
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def format_currency(amount):
    """Format a number as currency"""
    if amount >= 0:
        return f"Rs {amount:,.2f}"
    else:
        return f"-Rs {abs(amount):,.2f}"

def calculate_budget_summary(income, expenses):
    """
    Calculate budget summary statistics.
    
    Args:
        income (float): Total income
        expenses (dict): Dictionary of expenses by category
        
    Returns:
        dict: Budget summary statistics
    """
    if not expenses:
        return {
            "income": income,
            "total_expenses": 0,
            "remaining": income,
            "savings_rate": 100 if income > 0 else 0
        }
    
    total_expenses = sum(expenses.values())
    remaining = income - total_expenses
    savings_rate = (remaining / income) * 100 if income > 0 else 0
    
    return {
        "income": income,
        "total_expenses": total_expenses,
        "remaining": remaining,
        "savings_rate": savings_rate
    }

def fetch_stock_data(ticker, period="1y"):
    """
    Fetch stock data for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (e.g., "1d", "1mo", "1y")
        
    Returns:
        pandas.DataFrame: Stock data
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def calculate_investment_returns(initial_investment, monthly_contribution, years, rate_of_return):
    """
    Calculate investment returns over time.
    
    Args:
        initial_investment (float): Initial investment amount
        monthly_contribution (float): Monthly contribution amount
        years (int): Number of years for investment
        rate_of_return (float): Annual rate of return as a decimal (e.g., 0.07 for 7%)
        
    Returns:
        tuple: (final_amount, total_contributions, total_earnings)
    """
    # Convert annual rate to monthly
    monthly_rate = rate_of_return / 12
    
    # Total number of months
    months = years * 12
    
    # Calculate final amount using compound interest formula with monthly contributions
    final_amount = initial_investment * (1 + monthly_rate) ** months
    
    # Add in the effect of monthly contributions
    if monthly_rate > 0:
        final_amount += monthly_contribution * (((1 + monthly_rate) ** months - 1) / monthly_rate)
    else:
        final_amount += monthly_contribution * months
    
    # Calculate total contributions
    total_contribution = initial_investment + (monthly_contribution * months)
    
    # Calculate total earnings
    total_earnings = final_amount - total_contribution
    
    return final_amount, total_contribution, total_earnings

def calculate_loan_payment(principal, annual_rate, years):
    """
    Calculate monthly loan payment.
    
    Args:
        principal (float): Loan principal amount
        annual_rate (float): Annual interest rate as a decimal
        years (int): Loan term in years
        
    Returns:
        float: Monthly payment amount
    """
    # Convert annual rate to monthly
    monthly_rate = annual_rate / 12
    
    # Total number of payments
    n_payments = years * 12
    
    # Calculate monthly payment using loan formula
    if monthly_rate == 0:
        return principal / n_payments
    
    payment = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
    
    return payment

def generate_amortization_schedule(principal, annual_rate, years):
    """
    Generate loan amortization schedule.
    
    Args:
        principal (float): Loan principal amount
        annual_rate (float): Annual interest rate as a decimal
        years (int): Loan term in years
        
    Returns:
        pandas.DataFrame: Amortization schedule
    """
    # Calculate monthly payment
    monthly_payment = calculate_loan_payment(principal, annual_rate, years)
    
    # Convert annual rate to monthly
    monthly_rate = annual_rate / 12
    
    # Total number of payments
    n_payments = years * 12
    
    # Create amortization schedule
    schedule = []
    remaining_balance = principal
    
    for month in range(1, n_payments + 1):
        # Calculate interest payment
        interest_payment = remaining_balance * monthly_rate
        
        # Calculate principal payment
        principal_payment = monthly_payment - interest_payment
        
        # Update remaining balance
        remaining_balance -= principal_payment
        
        # Add to schedule
        schedule.append({
            'month': month,
            'payment': monthly_payment,
            'principal': principal_payment,
            'interest': interest_payment,
            'remaining_balance': max(0, remaining_balance)
        })
    
    return pd.DataFrame(schedule)

def categorize_expenses(transactions):
    """
    Categorize expenses from a list of transactions.
    
    Args:
        transactions (list): List of transaction dictionaries with 'amount' and 'description'
        
    Returns:
        dict: Expenses categorized by category
    """
    # Define common keywords for categories
    category_keywords = {
        'Housing': ['rent', 'mortgage', 'hoa', 'property tax'],
        'Utilities': ['electric', 'gas', 'water', 'internet', 'phone', 'utility'],
        'Groceries': ['grocery', 'groceries', 'supermarket', 'food'],
        'Transportation': ['gas', 'fuel', 'car', 'auto', 'transportation', 'uber', 'lyft', 'taxi'],
        'Dining': ['restaurant', 'cafe', 'coffee', 'dining', 'doordash', 'grubhub', 'takeout'],
        'Entertainment': ['movie', 'subscription', 'netflix', 'spotify', 'entertainment'],
        'Shopping': ['amazon', 'walmart', 'target', 'shopping', 'clothes', 'clothing'],
        'Health': ['doctor', 'medical', 'pharmacy', 'health', 'insurance', 'dental', 'vision'],
        'Education': ['school', 'tuition', 'book', 'course', 'education'],
        'Personal': ['haircut', 'gym', 'fitness', 'personal']
    }
    
    # Initialize categories
    categories = {category: 0 for category in category_keywords}
    categories['Other'] = 0
    
    # Categorize transactions
    for transaction in transactions:
        amount = transaction['amount']
        description = transaction['description'].lower()
        
        # Skip income (positive amounts)
        if amount <= 0:
            # Convert to positive expense amount
            expense_amount = abs(amount)
            
            # Determine category
            category_found = False
            for category, keywords in category_keywords.items():
                if any(keyword in description for keyword in keywords):
                    categories[category] += expense_amount
                    category_found = True
                    break
            
            # If no category matches, add to Other
            if not category_found:
                categories['Other'] += expense_amount
    
    # Remove categories with zero expenses
    return {k: v for k, v in categories.items() if v > 0}
