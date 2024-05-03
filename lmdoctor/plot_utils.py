import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_projection_heatmap(all_projs, tokens, lastn_tokens_to_plot=None, saturate_at=3, figsize=(1000,600), aspect=None):
    """
    Projections by token/layer
    saturate_at ensures that large values don't dominate and can be adjusted. To get raw view, set to None.
    """
    plot_tokens = None
    if lastn_tokens_to_plot:
        plot_data = all_projs[:, -lastn_tokens_to_plot:].cpu().numpy()
        if tokens:
            plot_tokens = tokens[-lastn_tokens_to_plot:]
    else:
        plot_data = all_projs.cpu().numpy()
        if tokens:
            plot_tokens = tokens
    
    fig = px.imshow(plot_data, color_continuous_scale='RdYlGn', labels=dict(x="Token"), aspect=aspect)
    if saturate_at is not None:
        if saturate_at == -1:
            # set the max and min based on largest value in data
            absmax = np.max(np.abs(plot_data))
            fig.update_coloraxes(cmin=-absmax, cmax=absmax)
        else:
            fig.update_coloraxes(cmin=-saturate_at, cmax=saturate_at)
        
    if tokens:
        fig.update_xaxes(
            tickvals=list(range(len(plot_tokens))),
            ticktext=plot_tokens
        )
    
    fig.update_layout(
        width=figsize[0],
        height=figsize[1]
    )
    
    fig.show()


# def plot_scores_per_token(
#     readings, tokens, lastn_tokens_to_plot=None, detection_method=None, saturate_at=1, figsize=None, aspect=None, 
#     token_range=None, hold_plot=False):
#     """
#     Scores (e.g. lie detection scores) per token.
#     """
#     plot_tokens = None

#     if lastn_tokens_to_plot is not None and token_range is not None:
#         raise RuntimeError('Cannot set both lastn_tokens_to_plot and token_range simultaneously.')
    
#     if lastn_tokens_to_plot:
#         plot_data = readings[:, -lastn_tokens_to_plot:]
#         if tokens:
#             plot_tokens = tokens[-lastn_tokens_to_plot:]
#     elif token_range:
#         plot_data = readings[:, token_range[0]:token_range[1]]
#         if tokens:
#             plot_tokens = tokens[token_range[0]:token_range[1]]
#     else:
#         plot_data = readings
#         if tokens:
#             plot_tokens = tokens

#     fig = px.imshow(plot_data, color_continuous_scale='RdYlGn', labels=dict(x="Token"), aspect=aspect)

#     if detection_method == 'classifier':
#         fig.update_coloraxes(cmin=0, cmax=1)
#     else:
#         if saturate_at is not None:
#             if saturate_at == -1:
#                 # set the max and min based on largest value in data
#                 absmax = np.max(np.abs(plot_data))
#                 fig.update_coloraxes(cmin=-absmax, cmax=absmax)
#             else:
#                 fig.update_coloraxes(cmin=-saturate_at, cmax=saturate_at)


#     if tokens:
#         fig.update_xaxes(
#             tickvals=list(range(len(plot_tokens))),
#             ticktext=plot_tokens,
#             tickangle=-45,  # Tilts the labels at -45 degrees
#             tickfont=dict(size=20)  # Updates the font size to 12
#         )

#     fig.update_yaxes(
#         showticklabels=False
#     )

#     fig.update_layout(coloraxis_showscale=False)

#     if figsize:
#         fig.update_layout(
#             autosize=False,
#             width=figsize[0],  # or any other width
#             height=figsize[1]   # adjust the height accordingly
#         )

#     if hold_plot:
#         return fig
#     else:
#         fig.show()


def plot_scores_per_token(
    readings, tokens, detection_method=None, saturate_at=1,
    lastn_tokens_to_plot=None, token_ranges=None, auto_ranges_n=None,
    figsize=None, aspect=None, tickangle=-45, vertical_spacing=.1
):
    """
    Scores (e.g. lie detection scores) per token.
    """
    num_not_none = int(lastn_tokens_to_plot is not None) + int(token_ranges is not None) + int(auto_ranges_n is not None)

    if num_not_none > 1:
        raise RuntimeError('Choose one of lastn_tokens_to_plot, token_ranges, and auto_ranges.')
    elif num_not_none == 0:
        token_ranges = [[0, len(readings[0])]]
    elif lastn_tokens_to_plot:
        token_ranges = [[-lastn_tokens_to_plot, len(readings[0])]]
    elif auto_ranges_n:
        n = auto_ranges_n
        token_ranges = [(i, i+n) for i in range(0, len(readings[0]), n)]

    rows = len(token_ranges)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=False, vertical_spacing=vertical_spacing)
    tokens_plus_idx = [f"{token}({idx})" for idx, token in enumerate(tokens)]

    for i, tr in enumerate(token_ranges):
        plot_tokens = None
        plot_data = readings[:, tr[0]:tr[1]]
        if tokens:
            plot_tokens = tokens_plus_idx[tr[0]:tr[1]]

        fig.add_trace(
            go.Heatmap(
                z=plot_data, 
                x=plot_tokens,
                colorscale='RdYlGn',  # Using the RdYlGn color scale
                # coloraxis="coloraxis1"
            ),
            row=i+1, col=1
        )
    
    if detection_method == 'classifier':
        fig.update_coloraxes(cmin=0, cmax=1)
    else:
        if saturate_at is not None:
            if saturate_at == -1:
                # set the max and min based on largest value in data
                absmax = np.max(np.abs(plot_data))
                fig.update_coloraxes(cmin=-absmax, cmax=absmax)
            else:
                fig.update_coloraxes(cmin=-saturate_at, cmax=saturate_at)

    if tokens:
        fig.update_xaxes(
            tickangle=tickangle, 
        )

    fig.update_yaxes(
        showticklabels=False
    )

    # fig.update_layout(coloraxis_showscale=False)

    if figsize:
        fig.update_layout(
            autosize=False,
            width=figsize[0], 
            height=figsize[1]  
        )
    else:
        fig.update_layout(
            autosize=False,
            width=1000, 
            height=200+150*(rows-1)
        )

    fig.show()


def plot_projs_on_numberline(projs_1, projs_0):
    """
    Plot projections on numberline with a bit of jitter
    """
    df = pd.DataFrame({
        'Value': np.concatenate([projs_1, projs_0]),
        'label': ['1'] * len(projs_1) + ['0'] * len(projs_0)
    })
    df['y'] = np.random.uniform(-1, 1, df.shape[0])        
    fig = px.scatter(df, x='Value', y='y', color='label')
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, 
                     zeroline=True, zerolinecolor='black', zerolinewidth=3,
                     showticklabels=False)
    fig.update_layout(height=200, plot_bgcolor='white')
    
    fig.show()