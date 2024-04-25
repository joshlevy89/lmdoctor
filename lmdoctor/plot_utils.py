import plotly.express as px
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
    


def plot_scores_per_token(
    readings, tokens, lastn_tokens_to_plot=None, detection_method=None, saturate_at=1, figsize=None, aspect=None):
    """
    Scores (e.g. lie detection scores) per token.
    """
    plot_tokens = None
    if lastn_tokens_to_plot:
        plot_data = readings[:, -lastn_tokens_to_plot:]
        if tokens:
            plot_tokens = tokens[-lastn_tokens_to_plot:]
    else:
        plot_data = readings
        if tokens:
            plot_tokens = tokens

    fig = px.imshow(plot_data, color_continuous_scale='RdYlGn', labels=dict(x="Token"), aspect=aspect)

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
            tickvals=list(range(len(plot_tokens))),
            ticktext=plot_tokens,
            tickangle=-45,  # Tilts the labels at -45 degrees
            tickfont=dict(size=20)  # Updates the font size to 12
        )

    fig.update_yaxes(
        showticklabels=False
    )


    if figsize:
        fig.update_layout(
            autosize=False,
            width=figsize[0],  # or any other width
            height=figsize[1]   # adjust the height accordingly
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