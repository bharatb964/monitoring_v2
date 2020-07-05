import os
import pathlib
import numpy as np
import datetime as dt
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd 
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from scipy.stats import rayleigh
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go

sequence_cols=['cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4',
   's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
   's16', 's17', 's18', 's19', 's20', 's21']
seq_array_test_last=np.load('test_df.npy')
oo=np.load('test_out.npy')
dd=np.append(seq_array_test_last[:,-1],oo,axis=1)
test_df=pd.DataFrame(dd)
test_df.columns=sequence_cols+['Proba']

def plot_fig(dff,interval,col):
    df=dff.iloc[0:interval][[col,'Proba']]
    data = [go.Scatter(x = list(range(df.shape[0])),y = df[col].values,mode = 'lines+markers')]
    layout = go.Layout(xaxis=dict(title='Id',range=[0,max(20,df.shape[0]+5)]),yaxis=dict(title=col,range=[0,1]), )
    fig = go.Figure(data=data,layout=layout)
    return fig

def update_alert(dff,interval,threshold):
    if dff.iloc[interval]['Proba']>threshold:
        return "** ALERT **"

def log_alert(dff,interval,threshold):
    global l
    if dff.iloc[interval]['Proba']>threshold:
        l=str(list(range(0,interval)))
    return l

GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 3000)

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

app.layout = html.Div(
    [
        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4("DASH MONITORING APP", className="app__header__title"),
                        html.P(
                            "This app runs in real time and provide alerts if the threshhold reaches its limit.",
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
                
            ],
            className="app__header",
        ),
        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("PLOT OF S21 VS ID", className="graph__title")]
                        ),
                        dcc.Graph(
                            id="wind-speed",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),
                        dcc.Interval(
                            id="wind-speed-update",
                            interval=int(GRAPH_INTERVAL),
                            n_intervals=0,
                        ),
                    ],
                    className="two-thirds column wind__speed__container",
                ),
                html.Div(
                    [
                        # histogram
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "SELECT THE THRESHOLD",
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                html.Br(),
                                dcc.Input(id="bin-slider", type="number", placeholder="Threshold",style={'marginLeft': 25}),
                                html.Div([html.H6("----")]),
                                html.Div('ALERT WILL BE DISPLAYED HERE', style={'color': 'white', 'fontSize': 14}),
                                html.Div(' ', style={'color': 'white', 'fontSize': 24},id='alert'),
                                html.Br(),
                                html.Div('ALERT WILL BE LOGGED HERE', style={'color': 'white', 'fontSize': 14}),
                                html.Br(),
                                html.Div(' ', style={'color': 'white', 'fontSize': 14},id='alert_log'),
                            ],
                            className="graph__container first",
                        ),
                        # wind direction
                        
                    ],
                    className="one-third column histogram__direction",
                ),
            ],
            className="app__content",
        ),
    ],
    className="app__container",
)


@app.callback(
    Output("wind-speed", "figure"), [Input("wind-speed-update", "n_intervals")]
)
def gen_wind_speed(interval):
    fig = plot_fig(test_df,interval,'s21')
    return fig

@app.callback(
    Output("alert", "children"),
    [Input("wind-speed-update","n_intervals")],
    [State("bin-slider", "value")],
)
def show_num_bins(interval,threshold):
    alert=update_alert(test_df,interval,threshold)
    return alert

@app.callback(
    Output("alert_log", "children"),
    [Input("wind-speed-update","n_intervals")],
    [State("bin-slider", "value")],
)
def show_num_bins(interval,threshold):
    alert=log_alert(test_df,interval,threshold)
    return alert

if __name__ == "__main__":
    app.run_server(debug=False,host='127.0.0.5')
