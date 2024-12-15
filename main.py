from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Jinja2 Templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
DATABASE = "financial_data.db"

def initialize_database():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                rent REAL NOT NULL,
                utilities REAL NOT NULL,
                groceries REAL NOT NULL,
                other_expenses REAL NOT NULL,
                earnings REAL NOT NULL,
                total_expenses REAL NOT NULL,
                savings REAL NOT NULL,
                post_investment REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()

initialize_database()

# Visualization for Investment Allocation
def visualize_investment_allocation(investment_amount, filename):
    """
    Creates and saves visualizations for investment allocation.
    """
    # Sample data for top companies
    data = {
        'Company': ['Apple', 'Amazon', 'Microsoft', 'Google', 'Tesla'],
        'Expected Return (%)': [12, 15, 10, 11, 20],
        'Risk (Volatility %)': [18, 22, 15, 17, 30]
    }
    df = pd.DataFrame(data)
    df['Weight'] = df['Expected Return (%)'] / df['Expected Return (%)'].sum()
    df['Investment ($)'] = df['Weight'] * investment_amount
    df['Post-Investment Amount ($)'] = df['Investment ($)'] * (1 + df['Expected Return (%)'] / 100)

    # Pie Chart
    pie_path = os.path.join("charts", f"{filename}_allocation_pie.png")
    plt.figure(figsize=(8, 6))
    plt.pie(df['Investment ($)'], labels=df['Company'], autopct='%1.1f%%', startangle=140)
    plt.title('Investment Allocation by Company')
    plt.savefig(pie_path)
    plt.close()

    # Line Chart: Cumulative Investment
    line_path = os.path.join("charts", f"{filename}_allocation_line.png")
    df['Cumulative Investment ($)'] = df['Investment ($)'].cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(df['Company'], df['Cumulative Investment ($)'], marker='o', color='green')
    plt.title('Cumulative Investment by Company')
    plt.xlabel('Company')
    plt.ylabel('Cumulative Investment ($)')
    plt.grid()
    plt.savefig(line_path)
    plt.close()

    return pie_path, line_path, df

# Visualization for User Data (Savings Growth, Expenses)
def visualize_user_data(name, rent, utilities, groceries, other_expenses, total_expenses, savings, filename):
    categories = ['Rent', 'Utilities', 'Groceries', 'Other Expenses']
    values = [rent, utilities, groceries, other_expenses]

    # Pie Chart: Monthly Expenses Breakdown
    pie_expenses_path = os.path.join("charts", f"{filename}_expenses_pie.png")
    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=140)
    plt.title('Monthly Expenses Breakdown')
    plt.savefig(pie_expenses_path)
    plt.close()

    # Bar Chart: Savings vs Expenses
    bar_savings_path = os.path.join("charts", f"{filename}_savings_bar.png")
    plt.figure(figsize=(8, 6))
    plt.bar(['Total Expenses', 'Savings'], [total_expenses, savings], color=['lightcoral', 'skyblue'])
    plt.title(f'{name} - Savings vs Total Expenses')
    plt.ylabel('Amount ($)')
    plt.savefig(bar_savings_path)
    plt.close()

    # Line Chart: Savings Growth Over Time
    months = list(range(1, 13))
    monthly_savings = [savings / 12 * i for i in months]
    line_savings_path = os.path.join("charts", f"{filename}_savings_growth.png")
    plt.figure(figsize=(10, 6))
    plt.plot(months, monthly_savings, marker='o', color='blue')
    plt.title(f'{name} - Savings Growth Over a Year')
    plt.xlabel('Month')
    plt.ylabel('Cumulative Savings ($)')
    plt.grid()
    plt.savefig(line_savings_path)
    plt.close()

    return pie_expenses_path, bar_savings_path, line_savings_path

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate", response_class=HTMLResponse)
async def calculate(
    request: Request,
    name: str = Form(...),
    rent: float = Form(0.0),
    utilities: float = Form(0.0),
    groceries: float = Form(0.0),
    other_expenses: float = Form(0.0),
    earnings: float = Form(...),
    ticker: str = Form("AAPL"),
):
    try:
        total_expenses = (rent + utilities + groceries + other_expenses) * 12
        savings = earnings - total_expenses
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"user_{timestamp.replace(':', '_').replace(' ', '_')}"

        # Save data to database
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_data (name, rent, utilities, groceries, other_expenses, earnings, total_expenses, savings, post_investment, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, rent, utilities, groceries, other_expenses, earnings, total_expenses, savings, 0, timestamp))
            conn.commit()

        # Generate visualizations if savings > 0
        if savings > 0:
            pie_path, bar_path, line_path, investment_df = visualize_investment_allocation(savings, filename)
            investment_table = investment_df[['Company', 'Investment ($)', 'Expected Return (%)', 'Risk (Volatility %)']].to_dict('records')
        else:
            pie_path, bar_path, line_path, investment_table = None, None, None, []

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "name": name,
                "total_expenses": total_expenses,
                "earnings": earnings,
                "savings": savings,
                "investment_table": investment_table,
            },
        )
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})

@app.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard(request: Request):
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, savings, post_investment, timestamp
                FROM user_data
                ORDER BY post_investment DESC
            """)
            leaderboard_data = cursor.fetchall()

        return templates.TemplateResponse(
            "leaderboard.html",
            {"request": request, "leaderboard_data": leaderboard_data, "enumerate": enumerate},
        )
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})
