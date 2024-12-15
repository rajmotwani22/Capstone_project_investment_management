from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from main import fetch_static_companies, visualize_investment_allocation_plotly  # Import helpers from main.py

# Fetch static company data
top_companies = fetch_static_companies()

# Create Dash app
dash_app = Dash(__name__)

# Layout of the Dash app
dash_app.layout = html.Div([
    html.H1("Investment Insights Dashboard", style={"textAlign": "center"}),

    # Dropdown for selecting investment strategy
    html.Div([
        html.Label("Select Investment Strategy:"),
        dcc.Dropdown(
            id="strategy-dropdown",
            options=[
                {"label": "Large Cap", "value": "large"},
                {"label": "Mid Cap", "value": "mid"},
                {"label": "Small Cap", "value": "small"}
            ],
            value="large",
            style={"width": "50%"}
        ),
        html.Label("Investment Amount ($):"),
        dcc.Slider(
            id="investment-slider",
            min=1000,
            max=50000,
            step=1000,
            value=10000,
            marks={i: f"${i}" for i in range(1000, 50001, 5000)}
        )
    ], style={"padding": "20px"}),

    # Div for interactive graphs
    html.Div([
        html.H3("Pie Chart: Investment Allocation", style={"textAlign": "center"}),
        dcc.Graph(id="pie-chart"),

        html.H3("Scatter Plot: Risk vs Reward", style={"textAlign": "center"}),
        dcc.Graph(id="scatter-plot"),

        html.H3("Line Chart: Cumulative Investment Growth", style={"textAlign": "center"}),
        dcc.Graph(id="line-chart")
    ])
])

# Callback to update the graphs dynamically based on user input
@dash_app.callback(
    [Output("pie-chart", "figure"),
     Output("scatter-plot", "figure"),
     Output("line-chart", "figure")],
    [Input("strategy-dropdown", "value"),
     Input("investment-slider", "value")]
)
def update_charts(strategy, investment_amount):
    """
    Update charts based on selected strategy and investment amount.
    """
    pie_chart, scatter_plot, line_chart = visualize_investment_allocation_plotly(
        investment_amount, strategy, top_companies
    )
    return pie_chart, scatter_plot, line_chart
