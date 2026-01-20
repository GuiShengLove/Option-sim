"""
Plotly Custom Themes & Charts
=============================
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def _get_layout(title: str):
    """Common dark layout."""
    return go.Layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F0F6FC"),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(showgrid=True, gridcolor="#30363D"),
        yaxis=dict(showgrid=True, gridcolor="#30363D"),
        hovermode="x unified"
    )

def plot_simulation_paths(paths: np.ndarray, S0: float):
    """
    Plot Monte Carlo paths.
    """
    n_paths, n_steps = paths.shape
    fig = go.Figure()
    
    # Plot first 100 paths max to avoid lag
    limit = min(n_paths, 100)
    
    # Add paths
    for i in range(limit):
        fig.add_trace(go.Scatter(
            y=paths[i],
            mode='lines',
            line=dict(color='#3B82F6', width=1),
            opacity=0.15,
            showlegend=False,
            name=f"Path {i}"
        ))
        
    # Mean path
    mean_path = np.mean(paths, axis=0)
    fig.add_trace(go.Scatter(
        y=mean_path,
        mode='lines',
        line=dict(color='#F43F5E', width=2),
        name='Mean Path'
    ))
    
    # S0 Line
    fig.add_hline(y=S0, line_dash="dash", line_color="#F0F6FC", opacity=0.5)
    
    fig.update_layout(
        _get_layout("Monte Carlo Simulation Paths"),
        xaxis_title="Days",
        yaxis_title="Underlying Price"
    )
    
    return fig

def plot_pnl_distribution(res_df: pd.DataFrame):
    """
    Plot PnL histogram with probability density.
    """
    fig = px.histogram(
        res_df, 
        x="pnl", 
        nbins=20,
        marginal="box",
        color_discrete_sequence=['#8B5CF6'],
        title="PnL Distribution & Probability Density"
    )
    
    fig.update_layout(
        _get_layout("PnL Distribution"),
        xaxis_title="Profit / Loss (RMB)",
        yaxis_title="Frequency",
        bargap=0.1
    )
    
    # Add vertical line at 0
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#F0F6FC")
    
    return fig

def plot_vol_surface(df_chain: pd.DataFrame):
    """
    Plot 3D Volatility Surface (Strike vs Expiry vs Price/IV).
    MVP: Plot Price Surface since IV calc is placeholder.
    """
    # Create grid
    x = df_chain['strike']
    y = (df_chain['expiry_date'] - df_chain['trade_date']).dt.days
    z = df_chain['theoretical_price']
    
    fig = go.Figure(data=[go.Mesh3d(
        x=x,
        y=y,
        z=z,
        color='#8B5CF6',
        opacity=0.50
    )])
    
    fig.update_layout(
        _get_layout("Option Price Surface"),
        scene=dict(
            xaxis_title='Strike',
            yaxis_title='DTE',
            zaxis_title='Price',
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#30363D"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#30363D"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#30363D"),
        )
    )
    return fig
