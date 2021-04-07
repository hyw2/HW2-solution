import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle
from datetime import date, datetime
from time import sleep
from momentum import *  # import everything from model

from helper_functions import *  # this statement imports all functions from your helper_functions file!

# Run your helper function to clear out any io files left over from old runs
# 1:
fig = go.Figure(
)

m = None

check_for_and_del_io_files()

# Make a Dash app!
app = dash.Dash(__name__)

# Define the layout.
app.layout = html.Div([
    # Section title
    html.H1("Section 1: Fetch & Display current IVV and Index data"),
    html.Br(),
    html.Div([
        html.Div(["Last number of days: ", dcc.Input(id='graph_n', value=5, type='number')]),
        html.Button('Update', id='refresh-graph', n_clicks=0),
        # Candlestick graph goes here:
        dcc.Graph(id='candlestick-graph', figure=fig)
    ]),
    html.H1("Section 2: Modify Default Algo Parameters"),
    # Another line break
    html.Br(),
    html.Div(
        [
            "Change parameters of current model: ",
            # Your text input object goes here:
            html.Div(["Alpha: ", dcc.Input(id='alpha', value=0.01, type='number')]),
            html.Div(["N: ", dcc.Input(id="n", value=5, type='number')]),
            html.Div(["Date: ", dcc.DatePickerSingle(
                id='date',
                min_date_allowed=date(2002, 8, 5),
                max_date_allowed=datetime.today(),
                initial_visible_month=date(2021, 4, 1),
                date=date(2021, 2, 5)
            )]),
            html.Div(id='model-output')
        ],
        # Style it so that the submit button appears beside the input.
        style={'display': 'inline-block'}
    ),
    html.Br(),
    # Submit button:
    html.Button('Retrain', id='submit-retrain', n_clicks=0),
    # Section title
    html.H1("Section 3: Backtesting Algo"),
    # Div to confirm what trade was made
    html.Div(id='output-backtest'),
    html.H1("Section 4: Model Trades"),
    html.Div(id='output-model-trades')
])


# Callback for IVV and index graphs
@app.callback(
    Output('candlestick-graph', 'figure'),
    [Input('refresh-graph', 'n_clicks')],
    State('graph_n', 'value')
)
def update_graph(n_clicks, value):
    with open("ticker_n.txt", "w") as f:
        f.write(str(value))

    while 'data.csv' not in listdir():
        sleep(0.1)
    df = pd.read_csv('data.csv')
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close']
            )
        ]
    )
    check_for_and_del_io_files()
    # Give the candlestick figure a title
    fig.update_layout(title=str(value) + " Candlestick Plot")
    return fig


# Callback for when model is retrained
@app.callback(
    Output('model-output', 'children'),
    Input('submit-retrain', 'n_clicks'),
    [State('alpha', 'value'), State('n', 'value'), State('date', 'date')],  # name of pair, trade amount,
)
def model_train(n_clicks, alpha, n, date_val):  # Still don't use n_clicks, but we need the dependency
    print("Model Train Called")
    msg = ''
    global m
    if n_clicks == 0:
        m = Model(n, True, alpha, date_val)
        msg += "loaded initial model"
    else:
        try:
            m = Model(n, True, alpha, date_val)
            msg += "Model Built"
        except:
            msg += "Error building model"
    return msg


# Run it!
if __name__ == '__main__':
    app.run_server(debug=True)
