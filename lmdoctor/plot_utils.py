import plotly.express as px

def plot_projection_heatmap(all_projs, tokens, lastn_tokens_to_plot=0, saturate_at=3):
    """
    Projections by token/layer
    saturate_at ensures that large values don't dominate and can be adjusted. To get raw view, set to None.
    """
    plot_data = all_projs[:, -lastn_tokens_to_plot:].cpu().numpy()
    plot_tokens = tokens[-lastn_tokens_to_plot:]
    
    fig = px.imshow(plot_data, color_continuous_scale='RdYlGn', labels=dict(x="Token"))
    if saturate_at:
        fig.update_coloraxes(cmin=-saturate_at, cmax=saturate_at)
    
    fig.update_xaxes(
        tickvals=list(range(len(plot_tokens))),
        ticktext=plot_tokens
    )
    
    fig.update_layout(
        width=1000,
        height=600
    )
    
    fig.show()

def plot_scores_per_token(readings, tokens, lastn_tokens_to_plot=0):
    """
    Scores (e.g. lie detection scores) per token.
    """
    plot_data = readings[:, -lastn_tokens_to_plot:]
    plot_tokens = tokens[-lastn_tokens_to_plot:]
    
    fig = px.imshow(plot_data, color_continuous_scale='RdYlGn', labels=dict(x="Token"))
    min_val = plot_data.min()
    max_val = plot_data.max()
    max_range = max(abs(min_val), abs(max_val))
    fig.update_coloraxes(cmin=-max_range, cmax=max_range)
    
    fig.update_xaxes(
        tickvals=list(range(len(plot_tokens))),
        ticktext=plot_tokens,
        tickangle=-45,  # Tilts the labels at -45 degrees
        tickfont=dict(size=20)  # Updates the font size to 12
    )
    
    fig.show()