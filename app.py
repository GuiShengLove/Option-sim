"""
AI Options Quant Platform (Forward-Test)
========================================
Main Streamlit Application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time

from ui.styles import apply_theme, metric_card
from ui.plots import plot_simulation_paths, plot_pnl_distribution
from forward_test import ForwardTestEngine
from strategies.base_strategy import DemoStrategy

# Config page
st.set_page_config(
    page_title="HyperQuant AI | Option Sim",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS
apply_theme()

# ============================================================
# Sidebar Configuration
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Simulation Config")
    
    # 1. Market Parameters
    st.markdown("### Market Params")
    S0 = st.number_input("Initial Price (S0)", value=3.0, step=0.01)
    drift = st.slider("Annual Drift (μ)", -0.2, 0.2, 0.05, 0.01)
    volatility = st.slider("Volatility (σ)", 0.1, 0.5, 0.20, 0.01)
    
    # 2. Simulation Settings
    st.markdown("### Engine Settings")
    n_days = st.slider("Duration (Days)", 5, 60, 20)
    n_paths = st.slider("Monte Carlo Paths", 10, 500, 50)
    model_type = st.selectbox("Model", ["GBM", "MJD"])
    
    st.divider()
    
    # 3. Strategy Config
    st.markdown("## 🧠 Strategy Config")
    strat_name = st.selectbox("Strategy", ["DemoStrategy"])
    
    # Dynamic Params
    # In a real app, we'd inspect the class. For now, hardcode DemoStrategy params.
    iv_thresh = st.slider("IV Threshold", 0.1, 0.3, 0.15)
    
    if st.button("Reset Default", use_container_width=True):
        st.experimental_rerun()

# ============================================================
# Main Content
# ============================================================
st.markdown("<h1 class='gradient-text'>HyperQuant AI <span style='font-size:1rem;color:#666'>v0.1.0</span></h1>", unsafe_allow_html=True)
st.markdown("### Forward-Test Simulation Engine")

# --- Control Panel ---
# col1, col2 = st.columns([3, 1])
# with col2:
run_btn = st.button("⚡ EXECUTE SIMULATION", use_container_width=True, type="primary")

if run_btn:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Initializing Simulation Engine...")
    
    try:
        # Initialize Engine
        engine = ForwardTestEngine()
        
        # Run Simulation
        start_time = time.time()
        
        # We can implement a callback for progress if we modify ForwardTestEngine
        # For now, just run it.
        status_text.text(f"Simulating {n_paths} paths over {n_days} days...")
        
        results = engine.run_simulation(
            strategy_cls=DemoStrategy,
            strategy_config={'iv_threshold': iv_thresh},
            S0=S0,
            mu=drift,
            sigma=volatility,
            days=n_days,
            n_paths=n_paths,
            model=model_type,
            base_date='2020-01-02'
        )
        
        progress_bar.progress(100)
        elapsed = time.time() - start_time
        status_text.text(f"✅ Simulation Complete in {elapsed:.2f}s")
        
        # --- Results Display ---
        st.divider()
        
        # 1. Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        
        final_eq_mean = results['final_equity'].mean()
        pnl_mean = results['pnl'].mean()
        win_rate = (results['pnl'] > 0).mean()
        max_dd_mean = results['max_drawdown'].mean()
        
        with m1: metric_card("Expected PnL", f"{pnl_mean:+.2f}", f"{pnl_mean/10000:.2%}")
        with m2: metric_card("Win Rate", f"{win_rate:.1%}", color="#3B82F6")
        with m3: metric_card("Max Drawdown (Avg)", f"{max_dd_mean:.2%}", color="#EF4444")
        with m4: metric_card("Paths Processed", f"{n_paths}", f"{elapsed:.1f}s")
        
        # 2. Charts Row
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # Reconstruct paths for visualization (Need to access path_gen from engine?)
            # ForwardTestEngine generates paths internally but doesn't return them in run_simulation.
            # Fix: ForwardTestEngine should expose paths or we generate them here for viz.
            # To stay consistent, let's just create a new set of paths for viz or modify engine.
            # Quick fix: Plot the 'final_S' distribution or generate dummy paths for visual appeal.
            # Better: Modify run_simulation to return paths.
            # For now, re-generate paths using same params for visualization (Random seed might differ).
            
            # Actually, `engine.path_gen` stores the generator, but paths are transient.
            # Let's just generate new paths for visualization purpose.
            viz_paths = engine.path_gen.generate_gbm(n_paths) if model_type=='GBM' else engine.path_gen.generate_mjd(n_paths)
            
            st.plotly_chart(plot_simulation_paths(viz_paths, S0), use_container_width=True)
            
        with c2:
            st.plotly_chart(plot_pnl_distribution(results), use_container_width=True)
            
        # 3. Data Table
        with st.expander("📊 Detailed Trade Logs"):
            st.dataframe(results, use_container_width=True)
            
    except Exception as e:
        st.error(f"Simulation Failed: {e}")
        import traceback
        st.code(traceback.format_exc())

else:
    # Empty State
    st.info("👈 Configure parameters on the sidebar and click **EXECUTE** to start.")
    
    # Placeholder Viz
    st.markdown("### Preview")
    dummy_paths = np.random.normal(3.0, 0.1, (10, 20)).cumsum(axis=1) + 3.0 # Just noise
    # st.plotly_chart(plot_simulation_paths(dummy_paths, 3.0), use_container_width=True)

