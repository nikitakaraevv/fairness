import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_confidence(probs_dict):
    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=list(probs_dict.keys()), y=[probs_dict[key]['female_darker'] for key in probs_dict], name='female_darker',
                            line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=list(probs_dict.keys()), y=[probs_dict[key]['female_lighter'] for key in probs_dict], name = 'female_lighter',
                            line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=list(probs_dict.keys()), y=[probs_dict[key]['male_darker'] for key in probs_dict], name='male_darker',
                            line=dict(color='firebrick', width=4,
                                dash='dash') # dash options include 'dash', 'dot', and 'dashdot'
    ))
    fig.add_trace(go.Scatter(x=list(probs_dict.keys()), y=[probs_dict[key]['male_lighter'] for key in probs_dict], name='male_lighter',
                            line = dict(color='royalblue', width=4, dash='dash')))
    # Edit the layout
    fig.update_layout(
                    xaxis_title='Face size (in pixels)',
                    yaxis_title='Detector confidence'
                    )
    fig.show()
