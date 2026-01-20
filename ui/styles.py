"""
UI Styles & CSS
===============
Defines the visual identity of the application.
"""
import streamlit as st

def apply_theme():
    """Inject custom CSS."""
    st.markdown("""
        <style>
        /* Main Background */
        .stApp {
            background-color: #0E1117;
            color: #F0F6FC;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #161B22;
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            color: #F0F6FC !important;
        }
        
        /* Gradient Text */
        .gradient-text {
            background: linear-gradient(90deg, #8B5CF6 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        
        /* Metrics Card */
        div[data-testid="metric-container"] {
            background-color: #161B22;
            border: 1px solid #30363D;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        /* Plotly Background */
        .js-plotly-plot .plotly .main-svg {
            background-color: transparent !important;
        }
        
        /* Buttons */
        div.stButton > button {
            background: linear-gradient(90deg, #8B5CF6 0%, #3B82F6 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.5);
        }
        
        /* Custom Card Class */
        .css-card {
            background-color: #161B22;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #30363D;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

def metric_card(label: str, value: str, delta: str = None, color: str = None):
    """
    Custom HTML metric card to bypass standard Streamlit look.
    """
    delta_html = ""
    if delta:
        color_hex = "#22C55E" if not delta.startswith("-") else "#EF4444"
        if color: color_hex = color
        delta_html = f'<span style="color: {color_hex}; font-size: 0.9rem; margin-left: 8px;">{delta}</span>'
        
    html = f"""
    <div style="background-color: #161B22; border: 1px solid #30363D; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
        <div style="color: #8B949E; font-size: 0.85rem; margin-bottom: 5px;">{label}</div>
        <div style="color: #F0F6FC; font-size: 1.5rem; font-weight: 600;">
            {value}
            {delta_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
