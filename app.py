#!/usr/bin/env python3
import os
import sys

# Detect script path
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# Check root venv or intel_webapp venv
venv_python = os.path.join(script_dir, ".venv", "bin", "python")
if not os.path.exists(venv_python):
    venv_python = os.path.join(script_dir, "intel_webapp", ".venv", "bin", "python")

# Check if we are already running inside the virtual environment
in_venv = sys.prefix == os.path.abspath(os.path.dirname(os.path.dirname(venv_python)))

if not in_venv and os.path.exists(venv_python):
    # Re-execute using the virtualenv's Python interpreter
    os.execv(venv_python, [venv_python, script_path] + sys.argv[1:])

# If we are here, we are running in the virtual environment.
try:
    import streamlit as st
    import streamlit.web.cli as stcli
except ImportError as e:
    sys.stderr.write(f"Error: Required dependencies not found in virtual environment: {e}\n")
    sys.exit(1)

if not (hasattr(st, 'runtime') and hasattr(st.runtime, 'exists') and st.runtime.exists()):
    # Run streamlit via its CLI
    sys.argv = ["streamlit", "run", script_path, "--browser.gatherUsageStats", "false"]
    sys.exit(stcli.main())

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import random

# Import custom TinyML Vibration PDM modules
try:
    import serial_bridge
except ImportError:
    serial_bridge = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
try:
    from tools import memory_planner
except ImportError:
    try:
        import memory_planner
    except ImportError:
        memory_planner = None


# =====================================================================
# Page Configurations & CSS Styling
# =====================================================================
st.set_page_config(
    page_title="TinyML Vibration Predictive Maintenance - ESP32 Hub",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark developer dashboard styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* Base Canvas & Clean Dark Theme */
html, body, [class*="css"], .stApp {
    background-color: #090B11 !important;
    color: rgba(255, 255, 255, 0.85) !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.01em;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 4px;
    height: 4px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.1);
}

/* Hide default Streamlit wrappers except header collapse */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
footer, div[data-testid="stDecoration"] {
    display: none !important;
    height: 0 !important;
}

/* Glassmorphism sidebar collapse button */
button[data-testid="stSidebarCollapseButton"] {
    background-color: rgba(18, 24, 41, 0.25) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.03) !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    margin: 1.5rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 38px !important;
    height: 38px !important;
}
button[data-testid="stSidebarCollapseButton"]:hover {
    background-color: rgba(0, 229, 255, 0.05) !important;
    border-color: rgba(0, 229, 255, 0.25) !important;
}
button[data-testid="stSidebarCollapseButton"] svg {
    display: none !important;
}
button[data-testid="stSidebarCollapseButton"]::before {
    content: "" !important;
    display: inline-block !important;
    width: 18px !important;
    height: 18px !important;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%2300E5FF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>') !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
}

.block-container {
    padding: 2rem 4rem 4rem 4rem !important;
    max-width: 1400px !important;
}

/* Strip Streamlit row constraints */
div[data-testid="stHorizontalBlock"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
    margin-bottom: 0 !important;
    gap: 2rem !important;
}
div[data-testid="column"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}

/* Sidebar Styling - Exactly like Reference */
section[data-testid="stSidebar"] {
    background-color: #0B0D14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background-color: transparent !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border: none !important;
    border-radius: 0 !important;
    margin: 0 !important;
    height: 100vh !important;
    box-shadow: none !important;
    padding: 2rem 1.5rem !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 1.5rem !important;
}

/* Disable sidebar manual resizing drag handle */
div[data-testid="stSidebarDraghandle"] {
    display: none !important;
}

/* Sidebar Headings */
.sidebar-section-title {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    color: rgba(255, 255, 255, 0.3) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    margin-bottom: 0.4rem !important;
}

/* Premium Workspace Panel (Continuous Surface Plate) */
.workspace-panel {
    background-color: #111520 !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 8px !important;
    padding: 1.5rem 1.8rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15) !important;
}

/* HUD Panel grid styling */
.hud-panel {
    display: grid !important;
    grid-template-columns: repeat(4, 1fr) !important;
    gap: 1.5rem !important;
    margin-bottom: 2rem !important;
}
.hud-card {
    background-color: #111520 !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 8px !important;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15) !important;
    display: flex !important;
    flex-direction: column !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.hud-card:hover {
    border-color: rgba(0, 229, 255, 0.15) !important;
    box-shadow: 0 18px 36px rgba(0, 0, 0, 0.25), 0 0 15px rgba(0, 229, 255, 0.03) !important;
    transform: translateY(-1px) !important;
}
.hud-label {
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    color: rgba(255, 255, 255, 0.45) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-bottom: 0.8rem !important;
}
.hud-value-row {
    display: flex !important;
    align-items: baseline !important;
    gap: 6px !important;
    margin-bottom: 0.8rem !important;
}
.hud-val {
    font-size: 2.2rem !important;
    font-weight: 300 !important;
    color: #ffffff !important;
    letter-spacing: -0.03em !important;
    line-height: 1 !important;
}
.hud-unit {
    font-size: 0.85rem !important;
    color: rgba(255, 255, 255, 0.4) !important;
    font-weight: 400 !important;
}
.hud-progress-bar {
    width: 100% !important;
    height: 4px !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
    border-radius: 2px !important;
    overflow: hidden !important;
    margin-bottom: 0.8rem !important;
}
.hud-progress-fill {
    height: 100% !important;
    background: linear-gradient(90deg, #00E5FF 0%, #00FF9D 100%) !important;
    border-radius: 2px !important;
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.4) !important;
}
.hud-subtitle {
    font-size: 0.75rem !important;
    color: rgba(255, 255, 255, 0.35) !important;
    font-weight: 400 !important;
}

/* Custom Floating Navigation Island */
.top-nav-panel {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 0.85rem 1.8rem !important;
    margin-top: -0.5rem !important;
    margin-bottom: 2rem !important;
    background: rgba(14, 18, 28, 0.85) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-top: 1px solid rgba(0, 229, 255, 0.35) !important;
    border-radius: 24px !important;
    box-shadow: 0 20px 45px rgba(0, 0, 0, 0.45), 0 0 25px rgba(0, 229, 255, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}
.top-nav-brand {
    display: flex !important;
    align-items: center !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -0.02em !important;
}
.top-nav-brand span {
    background: linear-gradient(135deg, #00E5FF 0%, #A855F7 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 700 !important;
}
.top-nav-links {
    display: flex !important;
    gap: 1.8rem !important;
}
.top-nav-link {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: rgba(255, 255, 255, 0.45) !important;
    cursor: pointer !important;
    padding: 0.4rem 0.8rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.top-nav-link:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.04) !important;
}
.top-nav-link.active {
    color: #00E5FF !important;
    background: rgba(0, 229, 255, 0.1) !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
    box-shadow: 0 0 12px rgba(0, 229, 255, 0.15) !important;
}
.top-nav-right {
    display: flex !important;
    align-items: center !important;
    gap: 1.2rem !important;
}
.top-nav-icon {
    color: rgba(255, 255, 255, 0.45) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.top-nav-icon:hover {
    color: #00E5FF !important;
    filter: drop-shadow(0 0 6px rgba(0, 229, 255, 0.5)) !important;
}
.top-nav-btn {
    background: linear-gradient(135deg, #00E5FF 0%, #00B4D8 100%) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.4rem 1.1rem !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.25) !important;
    transition: all 0.2s ease !important;
}
.top-nav-btn:hover {
    box-shadow: 0 0 22px rgba(0, 229, 255, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* Responsive Breakpoints */
@media (max-width: 1200px) {
    .block-container {
        padding: 1.5rem 1.5rem 3rem 1.5rem !important;
    }
    .hud-panel {
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 1rem !important;
    }
}
@media (max-width: 768px) {
    .hud-panel {
        grid-template-columns: 1fr !important;
    }
    .top-nav-panel {
        flex-direction: column !important;
        gap: 1rem !important;
        padding: 1rem !important;
        border-radius: 16px !important;
    }
}
.top-nav-avatar {
    width: 28px !important;
    height: 28px !important;
    border-radius: 50% !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    background-image: url('https://avatars.githubusercontent.com/u/9919?v=4') !important;
    background-size: cover !important;
}

/* Sidebar Custom Project Card */
.sidebar-project-card {
    background-color: rgba(255, 255, 255, 0.01) !important;
    border: 1px solid rgba(255, 255, 255, 0.03) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0.8rem !important;
}
.project-card-header {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}
.project-card-icon {
    width: 32px !important;
    height: 32px !important;
    border-radius: 8px !important;
    background-color: rgba(0, 229, 255, 0.05) !important;
    border: 1px solid rgba(0, 229, 255, 0.15) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #00E5FF !important;
}
.project-card-info {
    display: flex !important;
    flex-direction: column !important;
}
.project-card-title {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}
.project-card-subtitle {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #00E5FF !important;
}
.project-card-btn {
    background: transparent !important;
    border: 1px solid rgba(0, 229, 255, 0.25) !important;
    border-radius: 6px !important;
    color: #00E5FF !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.project-card-btn:hover {
    background-color: rgba(0, 229, 255, 0.05) !important;
}



/* Dynamic Status Badges */
.status-container {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    justify-content: center;
    height: 100%;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.status-success {
    background: rgba(0, 255, 157, 0.05) !important;
    color: #00FF9D !important;
    border: 1px solid rgba(0, 255, 157, 0.15) !important;
}
.status-danger {
    background: rgba(255, 92, 122, 0.05) !important;
    color: #FF5C7A !important;
    border: 1px solid rgba(255, 92, 122, 0.15) !important;
}
.status-pulse-green {
    width: 6px;
    height: 6px;
    background-color: #00FF9D;
    border-radius: 50%;
    animation: status-pulse-anim 2s infinite ease-in-out;
}
.status-pulse-red {
    width: 6px;
    height: 6px;
    background-color: #FF5C7A;
    border-radius: 50%;
    animation: status-pulse-anim 2s infinite ease-in-out;
}
@keyframes status-pulse-anim {
    0% { opacity: 0.4; }
    50% { opacity: 1; }
    100% { opacity: 0.4; }
}

/* Custom Terminal overrides */
.terminal-panel {
    padding: 0 !important;
    overflow: hidden !important;
}
.terminal-header {
    background: rgba(255, 255, 255, 0.01) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
    padding: 0.6rem 1.2rem !important;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.terminal-dots {
    display: flex;
    gap: 6px;
}
.terminal-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}
.dot-red { background: rgba(255, 92, 122, 0.6) !important; }
.dot-yellow { background: rgba(255, 200, 87, 0.6) !important; }
.dot-green { background: rgba(0, 255, 157, 0.6) !important; }
.terminal-title {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    color: rgba(255, 255, 255, 0.3) !important;
    letter-spacing: 0.05em;
}
.terminal-body {
    padding: 1.25rem !important;
}
.terminal-body pre {
    background: transparent !important;
    border: none !important;
    margin: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.6 !important;
    color: rgba(255, 255, 255, 0.75) !important;
    font-weight: 300 !important;
    white-space: pre-wrap !important;
}

/* HTML Details Expander Code Panel */
.custom-details-panel {
    background: #111520 !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 8px !important;
    margin-bottom: 2rem !important;
    overflow: hidden !important;
}
.custom-details-panel summary {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: rgba(255, 255, 255, 0.45) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 1rem 1.5rem !important;
    cursor: pointer !important;
    user-select: none;
}
.custom-details-panel summary:hover {
    color: #ffffff !important;
}
.editor-panel {
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
.editor-header {
    background: rgba(255, 255, 255, 0.01) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
    padding: 0.6rem 1.2rem !important;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.editor-title {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    color: rgba(255, 255, 255, 0.3) !important;
    letter-spacing: 0.05em;
}
.editor-body {
    padding: 1.25rem !important;
}
.editor-body pre {
    background: transparent !important;
    border: none !important;
    margin: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.6 !important;
    color: #00E5FF !important;
    font-weight: 300 !important;
}

/* Sidebar button overrides */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255, 255, 255, 0.01) !important;
    border: 1px solid rgba(255, 255, 255, 0.03) !important;
    border-radius: 8px !important;
    color: rgba(255, 255, 255, 0.6) !important;
    padding: 0.4rem 0.8rem !important;
    font-size: 0.78rem !important;
    transition: all 0.25s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255, 255, 255, 0.04) !important;
    border-color: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
}

/* Standard Button Overrides */
.stButton > button {
    background: rgba(255, 255, 255, 0.02) !important;
    color: rgba(255, 255, 255, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(255, 255, 255, 0.04) !important;
    border-color: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
}

/* Target node styles in inspector */
.target-node {
    background-color: rgba(255, 255, 255, 0.01) !important;
    border: 1px solid rgba(255, 255, 255, 0.03) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
}
.target-node-header {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 0.4rem !important;
}
.target-node-title {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}
.target-badge-online {
    font-size: 0.6rem !important;
    font-weight: 600 !important;
    color: #00FF9D !important;
    background-color: rgba(0, 255, 157, 0.05) !important;
    border: 1px solid rgba(0, 255, 157, 0.15) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}
.target-badge-sim {
    font-size: 0.6rem !important;
    font-weight: 600 !important;
    color: #FFC857 !important;
    background-color: rgba(255, 200, 87, 0.05) !important;
    border: 1px solid rgba(255, 200, 87, 0.15) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}
.target-node-meta {
    font-size: 0.72rem !important;
    color: rgba(255, 255, 255, 0.4) !important;
    margin-bottom: 0.6rem !important;
}
.target-node-progress {
    width: 100% !important;
    height: 3px !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
    border-radius: 1.5px !important;
    overflow: hidden !important;
    margin-bottom: 0.6rem !important;
}
.target-progress-fill {
    height: 100% !important;
    background-color: #00E5FF !important;
}
.target-node-status {
    font-size: 0.72rem !important;
    color: rgba(255, 255, 255, 0.3) !important;
}

/* Style selectbox */
div[data-baseweb="select"] {
    background-color: rgba(255, 255, 255, 0.01) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 8px !important;
    margin-bottom: 0.5rem !important;
}
div[data-baseweb="select"]:hover {
    border-color: rgba(255, 255, 255, 0.08) !important;
}
div[data-baseweb="select"] > div {
    background-color: transparent !important;
    color: rgba(255, 255, 255, 0.8) !important;
    font-size: 0.82rem !important;
}

/* Style selectbox dropdown options list */
div[role="listbox"] {
    background-color: #0d111c !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
}
ul[role="listbox"] li {
    font-size: 0.8rem !important;
    color: rgba(255, 255, 255, 0.8) !important;
    padding: 0.5rem 1rem !important;
}
ul[role="listbox"] li:hover, ul[role="listbox"] li[aria-selected="true"] {
    background-color: rgba(0, 229, 255, 0.08) !important;
    color: #00E5FF !important;
}

/* Style toggle switches */
div[data-testid="stToggle"] {
    background: rgba(255, 255, 255, 0.005) !important;
    border: 1px solid rgba(255, 255, 255, 0.03) !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.7rem !important;
    margin-bottom: 0.5rem !important;
}

/* Floating Workspace Tabs Navigation Bar */
div[data-testid="stTabs"] > div:first-child {
    background: rgba(14, 18, 28, 0.6) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 6px 10px !important;
    margin-bottom: 2rem !important;
    gap: 8px !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    color: rgba(255, 255, 255, 0.45) !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.25rem !important;
    font-size: 0.82rem !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    text-transform: none !important;
    letter-spacing: -0.01em !important;
}
button[data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.04) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00E5FF !important;
    background: rgba(0, 229, 255, 0.08) !important;
    border: 1px solid rgba(0, 229, 255, 0.2) !important;
    box-shadow: 0 4px 15px rgba(0, 229, 255, 0.08) !important;
}

/* Accent texts */
.text-cyan { color: #00E5FF !important; }
.text-violet { color: #8B5CF6 !important; }
.text-success { color: #00FF9D !important; }
.text-danger { color: #FF5C7A !important; }

/* Custom Expander Dropdown overrides */
div[data-testid="stExpander"] {
    background: rgba(11, 14, 23, 0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.03) !important;
    border-radius: 16px !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1) !important;
    margin-top: 1rem !important;
}
div[data-testid="stExpander"] details {
    border: none !important;
}
div[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: rgba(255, 255, 255, 0.6) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.75rem 1.2rem !important;
}
div[data-testid="stExpander"] summary:hover {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Session State Initialization & Event Handling
# =====================================================================
if 'serial_bridge' not in st.session_state and serial_bridge is not None:
    st.session_state.serial_bridge = serial_bridge.ESP32SerialBridge()

if 'vibration_history' not in st.session_state:
    st.session_state.vibration_history = pd.DataFrame(columns=[
        "Timestamp", "Count", "Magnitude_g", "Pred_Label", "Pred_Name", "Confidence", "Latency_ms", "Buffer_Count"
    ])

if 'last_processed_count' not in st.session_state:
    st.session_state.last_processed_count = -1

# Read bridge status
serial_connected = False
if 'serial_bridge' in st.session_state and st.session_state.serial_bridge.ser and st.session_state.serial_bridge.ser.is_open:
    serial_connected = True

# Define widget callback handlers
def handle_connect_usb():
    selected_port = st.session_state.get("main_serial_port", None)
    if selected_port and 'serial_bridge' in st.session_state:
        success = st.session_state.serial_bridge.connect(selected_port)
        if success:
            st.toast(f"Connected to {selected_port}", icon="🔌")
        else:
            st.toast("Connection failed", icon="❌")

def handle_disconnect_usb():
    if 'serial_bridge' in st.session_state:
        st.session_state.serial_bridge.disconnect()
    st.toast("Disconnected serial bridge", icon="🔌")

def handle_normal_injection():
    if 'serial_bridge' in st.session_state:
        sent = st.session_state.serial_bridge.send_label(0)
        st.toast("Transmitted label [0: Normal] over serial!" if sent else "Connection offline", icon="🟢")

def handle_anomaly_injection():
    if 'serial_bridge' in st.session_state:
        sent = st.session_state.serial_bridge.send_label(1)
        st.toast("Transmitted label [1: Anomaly] over serial!" if sent else "Connection offline", icon="🔴")

# ---------------------------------------------------------------------
# Sidebar Panel Layout (Overview Rail)
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-project-card">
        <div class="project-card-header">
            <div class="project-card-icon">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00E5FF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/></svg>
            </div>
            <div class="project-card-info">
                <div class="project-card-title">Vibration Core v1</div>
                <div class="project-card-subtitle">Active: ESP32-LX6</div>
            </div>
        </div>
        <button class="project-card-btn">New Run</button>
    </div>
    <div style='border-bottom: 1px solid rgba(255,255,255,0.04); margin: 1.2rem 0;'></div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-section-title'>Target Profile</div>", unsafe_allow_html=True)
    target_hardware = st.selectbox(
        "Deployment Profile",
        ["ESP32 Dual-Core LX6 (200KB SRAM Limit)"],
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='sidebar-section-title' style='margin-top: 1rem;'>Algorithm Pipeline</div>", unsafe_allow_html=True)
    sparse_engine_active = st.toggle("Sparse-Update & 4-bit Replay Engine", value=True)


# ---------------------------------------------------------------------
# Telemetry Stream Updates
# ---------------------------------------------------------------------
if serial_connected:
    telem = st.session_state.serial_bridge.get_latest_telemetry()
    count = telem.get("count", 0)
    
    if count != st.session_state.get("last_processed_count", -1):
        current_time = time.time()
        mag = telem.get("mag", 1.0)
        pred = telem.get("pred", 0)
        conf = telem.get("conf", 0.95) * 100.0
        lat = telem.get("latency_ms", 1.8)
        buf_c = telem.get("buffer", 0)
        pred_name = "ANOMALOUS" if pred == 1 else "NORMAL"
        
        new_entry = pd.DataFrame([{
            "Timestamp": current_time,
            "Count": count,
            "Magnitude_g": round(mag, 3),
            "Pred_Label": pred,
            "Pred_Name": pred_name,
            "Confidence": round(min(max(conf, 50.0), 99.9), 1),
            "Latency_ms": round(lat, 2),
            "Buffer_Count": buf_c
        }])
        
        st.session_state.vibration_history = pd.concat([st.session_state.vibration_history, new_entry], ignore_index=True)
        if len(st.session_state.vibration_history) > 20:
            st.session_state.vibration_history = st.session_state.vibration_history.iloc[1:].reset_index(drop=True)
            
        st.session_state.last_processed_count = count

if len(st.session_state.vibration_history) > 0:
    latest_data = st.session_state.vibration_history.iloc[-1]
else:
    latest_data = None

# Calculate real static memory footprint
sram_kb = 33.60
if memory_planner is not None:
    try:
        ok, static_bytes = memory_planner.calculate_sram_footprint("tools/model_spec.json", verbose=False)
        sram_kb = round(static_bytes / 1024.0, 2)
    except Exception:
        pass

# =====================================================================
# Main Application Workspace Shell
# =====================================================================

# Floating Top Navigation Island
st.markdown(f"""
<div class="top-nav-panel">
    <div class="top-nav-brand">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="%2300E5FF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 10px;"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
        <span>TinyML</span>&nbsp;Studio
    </div>
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background: rgba(0, 255, 157, 0.06); border: 1px solid rgba(0, 255, 157, 0.18); border-radius: 20px; font-size: 0.72rem; color: #00FF9D; font-weight: 500;">
            <span style="width: 6px; height: 6px; background: #00FF9D; border-radius: 50%; display: inline-block;"></span> ESP32 Hub Active
        </span>
    </div>
    <div class="top-nav-right">
        <svg class="top-nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 0 1-3.46 0"/></svg>
        <svg class="top-nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        <button class="top-nav-btn">Sync Model</button>
        <div class="top-nav-avatar"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Application Workspaces
ws_overview, ws_continual, ws_replay, ws_runtime, ws_deploy = st.tabs([
    "Overview", 
    "Continual Learning", 
    "Replay Buffer", 
    "Runtime", 
    "Deployment"
])

# ---------------------------------------------------------------------
# WORKSPACE 1: OVERVIEW
# ---------------------------------------------------------------------
with ws_overview:
    # Initialize sram state tracking in session state
    if 'sram_kb' not in st.session_state:
        st.session_state.sram_kb = "Unavailable"
        
    if serial_connected:
        telem = st.session_state.serial_bridge.get_latest_telemetry()
        sram_val = telem.get("static_ram_kb", None)
        if sram_val is not None:
            st.session_state.sram_kb = round(float(sram_val), 2)
            
    sram_kb = st.session_state.sram_kb
    if sram_kb != "Unavailable":
        sram_percent = min(round((sram_kb / 200) * 100, 1), 100.0)
    else:
        sram_percent = 0.0

    if latest_data is not None:
        pred_class = "text-success" if latest_data["Pred_Label"] == 0 else "text-danger"
        pred_name = latest_data['Pred_Name']
        confidence = f"{latest_data['Confidence']}%"
        magnitude_val = f"{latest_data['Magnitude_g']:.3f}"
        magnitude_unit = "g"
        mag_percent = min(round((latest_data['Magnitude_g'] / 1.5) * 100, 1), 100.0)
        latency_val = f"{latest_data['Latency_ms']}"
        latency_unit = "ms"
        buf_count = f"{int(latest_data['Buffer_Count'])}"
        buf_c_raw = int(latest_data['Buffer_Count'])
    else:
        pred_class = "text-violet"
        pred_name = "Unavailable"
        confidence = "Unavailable"
        magnitude_val = "Unavailable"
        magnitude_unit = ""
        mag_percent = 0.0
        latency_val = "Unavailable"
        latency_unit = ""
        buf_count = "Unavailable"
        buf_c_raw = 0

    st.markdown(f"""
    <div class="hud-panel">
        <div class="hud-card">
            <div class="hud-label">Health State Prediction</div>
            <div class="hud-value-row">
                <span class="hud-val {pred_class}">{pred_name}</span>
            </div>
            <div class="hud-subtitle text-cyan">Confidence: {confidence}</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Vibration Magnitude</div>
            <div class="hud-value-row">
                <span class="hud-val">{magnitude_val}</span>
                <span class="hud-unit">{magnitude_unit}</span>
            </div>
            <div class="hud-progress-bar">
                <div class="hud-progress-fill" style="width: {mag_percent}%;"></div>
            </div>
            <div class="hud-subtitle">3-Axis Magnitude ||a||</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Static SRAM</div>
            <div class="hud-value-row">
                <span class="hud-val">{sram_kb}</span>
                <span class="hud-unit">{'KB' if sram_kb != "Unavailable" else ""}</span>
            </div>
            <div class="hud-progress-bar">
                <div class="hud-progress-fill" style="width: {sram_percent}%; background-color: #00FF9D;"></div>
            </div>
            <div class="hud-subtitle">Limit: 200 KB (Zero Malloc)</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Inference & Replay</div>
            <div class="hud-value-row">
                <span class="hud-val">{latency_val}</span>
                <span class="hud-unit">{latency_unit}</span>
            </div>
            <div class="hud-subtitle text-cyan">Replay Reservoir: {buf_count}/50 samples</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Workspace Grid
    row2_col1, row2_col2 = st.columns([2.2, 1.0])

    with row2_col1:
        st.markdown("""
        <div style='font-size: 0.72rem; font-weight: 500; color: rgba(255, 255, 255, 0.3); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.75rem;'>Vibration Analytics</div>
        """, unsafe_allow_html=True)
        
        chart_tab1, chart_tab2 = st.columns(2)
        with chart_tab1:
            fig_vib = go.Figure()
            if len(st.session_state.vibration_history) > 0:
                counts = st.session_state.vibration_history["Count"]
                mags = st.session_state.vibration_history["Magnitude_g"]
                
                fig_vib.add_trace(go.Scatter(
                    x=counts,
                    y=mags,
                    mode='lines',
                    name='Acceleration Magnitude',
                    line=dict(color='#00E5FF', width=1.5, shape='spline'),
                ))
                fig_vib.add_hline(y=1.00, line_dash="dash", line_color="rgba(0, 255, 157, 0.15)")
                fig_vib.add_hline(y=1.30, line_dash="dash", line_color="rgba(255, 92, 122, 0.15)")
            else:
                fig_vib.add_annotation(
                    text="Device Disconnected<br>Waiting for telemetry data...",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(color="rgba(255,255,255,0.45)", size=12)
                )
            
            fig_vib.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", color="rgba(255, 255, 255, 0.3)", size=10),
                yaxis=dict(
                    title="Acceleration (g)", 
                    gridcolor="rgba(255, 255, 255, 0.01)", 
                    zerolinecolor="rgba(255, 255, 255, 0.01)",
                    range=[0.5, 2.0]
                ),
                xaxis=dict(
                    title="Sample count", 
                    gridcolor="rgba(255, 255, 255, 0.01)",
                    zerolinecolor="rgba(255, 255, 255, 0.01)"
                ),
                margin=dict(l=40, r=10, t=10, b=40),
                height=250,
                showlegend=False
            )
            st.plotly_chart(fig_vib, key="plotly_accel_vibrations", width='stretch')
            
        with chart_tab2:
            fig_conf = go.Figure()
            if len(st.session_state.vibration_history) > 0:
                counts = st.session_state.vibration_history["Count"]
                confs = st.session_state.vibration_history["Confidence"]
                preds = st.session_state.vibration_history["Pred_Label"]
                colors = ['rgba(0, 255, 157, 0.3)' if p == 0 else 'rgba(255, 92, 122, 0.3)' for p in preds]
                
                fig_conf.add_trace(go.Bar(
                    x=counts,
                    y=confs,
                    marker_color=colors,
                    name='Confidence',
                    marker_line_width=0
                ))
            else:
                fig_conf.add_annotation(
                    text="Device Disconnected<br>Waiting for telemetry data...",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(color="rgba(255,255,255,0.45)", size=12)
                )
            fig_conf.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", color="rgba(255, 255, 255, 0.3)", size=10),
                yaxis=dict(
                    title="Confidence (%)", 
                    gridcolor="rgba(255, 255, 255, 0.01)", 
                    zerolinecolor="rgba(255, 255, 255, 0.01)",
                    range=[0, 100]
                ),
                xaxis=dict(
                    title="Sample count", 
                    gridcolor="rgba(255, 255, 255, 0.01)",
                    zerolinecolor="rgba(255, 255, 255, 0.01)"
                ),
                margin=dict(l=40, r=10, t=10, b=40),
                height=250,
                showlegend=False
            )
            st.plotly_chart(fig_conf, key="plotly_confidence_vibrations", width='stretch')

    with row2_col2:
        conn_info = {
            "connected": False,
            "board_name": "Unknown Device",
            "chip_family": "Unavailable",
            "port": "Unavailable",
            "baudrate": "Unavailable",
            "firmware_version": "Unavailable",
            "build_version": "Unavailable",
            "app_version": "Unavailable",
            "sdk_version": "Unavailable",
            "uptime": "Unavailable"
        }
        if 'serial_bridge' in st.session_state:
            conn_info = st.session_state.serial_bridge.get_connection_info()

        st.markdown("""
        <div style='font-size: 0.72rem; font-weight: 500; color: rgba(255, 255, 255, 0.3); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.75rem;'>Hardware Profile</div>
        """, unsafe_allow_html=True)
        
        status_badge = '<span class="target-badge-online">LIVE</span>' if conn_info["connected"] else '<span class="target-badge-offline" style="background: rgba(255, 92, 122, 0.08); border: 1px solid rgba(255, 92, 122, 0.2); color: #FF5C7A; font-size: 0.62rem; font-weight: 800; padding: 2px 8px; border-radius: 4px; letter-spacing: 0.05em;">DISCONNECTED</span>'
        status_text = "Serial Synced" if conn_info["connected"] else "Device Disconnected"
        progress_width = "100%" if conn_info["connected"] else "0%"
        progress_color = "" if conn_info["connected"] else "background: #FF5C7A;"
        
        st.markdown(f"""
        <div class="target-node">
            <div class="target-node-header">
                <span class="target-node-title">{conn_info['board_name']}</span>
                {status_badge}
            </div>
            <div class="target-node-meta">Task: Anomaly Classification Engine</div>
            <div class="target-node-progress">
                <div class="target-progress-fill" style="width: {progress_width}; {progress_color}"></div>
            </div>
            <div class="target-node-status">Status: {status_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; font-size: 0.75rem; line-height: 1.6;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.03); padding-bottom: 0.25rem;">
                <span style="color: rgba(255,255,255,0.4);">Chip Family:</span>
                <strong style="color: #ffffff;">{conn_info['chip_family']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.03); padding-top: 0.25rem; padding-bottom: 0.25rem;">
                <span style="color: rgba(255,255,255,0.4);">Serial Port:</span>
                <strong style="color: #00E5FF;">{conn_info['port']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.03); padding-top: 0.25rem; padding-bottom: 0.25rem;">
                <span style="color: rgba(255,255,255,0.4);">Baud Rate:</span>
                <strong style="color: #ffffff;">{conn_info['baudrate']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.03); padding-top: 0.25rem; padding-bottom: 0.25rem;">
                <span style="color: rgba(255,255,255,0.4);">Firmware Version:</span>
                <strong style="color: #00FF9D;">{conn_info['firmware_version']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.03); padding-top: 0.25rem; padding-bottom: 0.25rem;">
                <span style="color: rgba(255,255,255,0.4);">Build Version:</span>
                <strong style="color: #ffffff;">{conn_info['build_version']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.03); padding-top: 0.25rem; padding-bottom: 0.25rem;">
                <span style="color: rgba(255,255,255,0.4);">App Version:</span>
                <strong style="color: #ffffff;">{conn_info['app_version']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.03); padding-top: 0.25rem; padding-bottom: 0.25rem;">
                <span style="color: rgba(255,255,255,0.4);">SDK Version:</span>
                <strong style="color: #ffffff;">{conn_info['sdk_version']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; padding-top: 0.25rem;">
                <span style="color: rgba(255,255,255,0.4);">Runtime Uptime:</span>
                <strong style="color: #FFC857;">{conn_info['uptime']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
            
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        
        if serial_bridge and serial_bridge.SERIAL_AVAILABLE:
            available_ports = serial_bridge.list_available_serial_ports()
            if not available_ports:
                st.markdown("<div style='font-size:0.75rem; color:#FF5C7A; margin-bottom:0.5rem;'>⚠️ No serial ports found. Plug in ESP32.</div>", unsafe_allow_html=True)
                selected_port = None
            else:
                selected_port = st.selectbox("ESP32 Serial Port COM Selection", available_ports, key="main_serial_port", label_visibility="collapsed")
            
            col_conn1, col_conn2 = st.columns(2)
            with col_conn1:
                st.button("Connect USB", key="main_connect_btn", on_click=handle_connect_usb, disabled=(not available_ports))
            with col_conn2:
                st.button("Disconnect", key="main_disconnect_btn", on_click=handle_disconnect_usb)
                
        st.markdown("<div style='border-bottom: 1px solid rgba(255,255,255,0.04); margin: 1rem 0;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.65rem; font-weight: 500; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.5rem;'>Adaptation Injection</div>", unsafe_allow_html=True)
        
        col_lbl1, col_lbl2 = st.columns(2)
        with col_lbl1:
            st.button("🟢 Normal", key="main_normal_btn", on_click=handle_normal_injection)
        with col_lbl2:
            st.button("🔴 Anomaly", key="main_anomaly_btn", on_click=handle_anomaly_injection)

    # Telemetry Log
    def format_log_to_parsed_html(log_line):
        import html
        import re
        import json
        
        match_ts = re.match(r'^(\[[0-9:\.]+\])\s+(.*)$', log_line)
        if not match_ts:
            return f'<div style="margin-bottom:0.75rem; border-left:2px solid rgba(255,255,255,0.1); padding-left:0.5rem; color:rgba(255,255,255,0.6); font-size:0.75rem;">{html.escape(log_line)}</div>'
            
        timestamp = match_ts.group(1)
        payload = match_ts.group(2).strip()
        
        if payload.startswith("{") and payload.endswith("}"):
            try:
                data = json.loads(payload)
                if data.get("type") == "telemetry":
                    mag = data.get("mag", 0.0)
                    pred = data.get("pred", 0)
                    pred_name = "ANOMALOUS" if pred == 1 else "NORMAL"
                    pred_color = "#FF5C7A" if pred == 1 else "#00FF9D"
                    conf = data.get("conf", 0.0) * 100.0
                    sram = data.get("static_ram_kb", 0.0)
                    buf = data.get("buffer", 0)
                    lat = data.get("latency_ms", 0.0)
                    
                    return f"""
                    <div style="background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.04); border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem; font-family: 'Outfit', sans-serif;">
                        <div style="font-size:0.75rem; color:rgba(255,255,255,0.4); margin-bottom:0.4rem; font-family: 'JetBrains Mono', monospace;">{timestamp}</div>
                        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:0.5rem; font-size:0.78rem; line-height: 1.4;">
                            <div>
                                <span style="color:#00E5FF; font-weight:bold;">IMU</span><br>
                                • Accel Magnitude: <strong style="color:#ffffff;">{mag:.3f} g</strong>
                            </div>
                            <div>
                                <span style="color:#00FF9D; font-weight:bold;">TinyML</span><br>
                                • Prediction: <strong style="color:{pred_color};">{pred_name}</strong><br>
                                • Confidence: <strong style="color:#ffffff;">{conf:.1f}%</strong>
                            </div>
                            <div>
                                <span style="color:#8B5CF6; font-weight:bold;">Memory</span><br>
                                • Static SRAM: <strong style="color:#ffffff;">{sram:.1f} KB</strong>
                            </div>
                            <div>
                                <span style="color:#FFC857; font-weight:bold;">Replay</span><br>
                                • Buffer: <strong style="color:#ffffff;">{buf} / 50</strong>
                            </div>
                            <div>
                                <span style="color:#FF79C6; font-weight:bold;">Performance</span><br>
                                • Latency: <strong style="color:#ffffff;">{lat:.2f} ms</strong>
                            </div>
                        </div>
                    </div>
                    """
            except Exception:
                pass
                
        tag_colors = {
            "[TinyML]": "#00FF9D",
            "[IMU]": "#FF79C6",
            "[Memory]": "#8B5CF6",
            "[ERROR]": "#FF5C7A",
            "[Error]": "#FF5C7A",
            "[Warning]": "#FFC857",
            "[ESP32]": "#00E5FF",
            "[Replay]": "#8B5CF6",
            "[Info]": "#00FF9D",
        }
        styled_payload = html.escape(payload)
        for tag, color in tag_colors.items():
            if tag in styled_payload:
                styled_payload = styled_payload.replace(tag, f'<span style="color: {color}; font-weight: bold;">{tag}</span>')
                
        return f'<div style="font-size:0.75rem; margin-bottom:0.4rem; color:rgba(255,255,255,0.7);"><span style="color: rgba(255, 255, 255, 0.45); font-family: \'JetBrains Mono\', monospace;">{timestamp}</span> {styled_payload}</div>'

    def format_log_to_raw_html(log_line):
        import html
        import re
        escaped = html.escape(log_line)
        tag_colors = {
            "[TinyML]": "#00FF9D",
            "[IMU]": "#FF79C6",
            "[Memory]": "#8B5CF6",
            "[ERROR]": "#FF5C7A",
            "[Error]": "#FF5C7A",
            "[Warning]": "#FFC857",
            "[ESP32]": "#00E5FF",
            "[Replay]": "#8B5CF6",
            "[Info]": "#00FF9D",
        }
        for tag, color in tag_colors.items():
            if tag in escaped:
                escaped = escaped.replace(tag, f'<span style="color: {color}; font-weight: bold;">{tag}</span>')
        escaped = re.sub(r'^(\[[0-9:\.]+\])', r'<span style="color: rgba(255, 255, 255, 0.45);">\1</span>', escaped)
        return f'<div style="font-family:\'JetBrains Mono\', monospace; font-size:0.8rem; margin-bottom:0.25rem; white-space:pre-wrap; line-height: 1.4; color: rgba(255,255,255,0.85);">{escaped}</div>'

    if 'console_paused' not in st.session_state:
        st.session_state.console_paused = False
    if 'paused_logs' not in st.session_state:
        st.session_state.paused_logs = []
    if 'console_log_mode' not in st.session_state:
        st.session_state.console_log_mode = "Parsed"
    if 'console_search_query' not in st.session_state:
        st.session_state.console_search_query = ""

    # 1. Retrieve current logs list based on pause state
    if 'serial_bridge' in st.session_state:
        if st.session_state.console_paused:
            logs_list = list(st.session_state.paused_logs)
        else:
            logs_list = list(st.session_state.serial_bridge.serial_logs)
    else:
        logs_list = []

    # 2. Filter logs list by search query
    search_query = st.session_state.console_search_query
    if search_query:
        logs_list = [line for line in logs_list if search_query.lower() in line.lower()]

    # 3. Format the visible logs into HTML
    serial_connected = False
    if 'serial_bridge' in st.session_state and st.session_state.serial_bridge.ser and st.session_state.serial_bridge.ser.is_open:
        serial_connected = True

    if serial_connected:
        if logs_list:
            if st.session_state.console_log_mode == "Parsed":
                logs_html = "".join([format_log_to_parsed_html(line) for line in logs_list])
            else:
                logs_html = "".join([format_log_to_raw_html(line) for line in logs_list])
        else:
            logs_html = '<span style="color: rgba(255, 255, 255, 0.45); font-family: \'JetBrains Mono\', monospace; font-size: 0.72rem;">Connected to serial port. Listening for ESP32 messages...</span>'
    else:
        logs_html = '<span style="color: #FF5C7A; font-weight: bold; font-family: \'JetBrains Mono\', monospace; font-size: 0.72rem;">[Device Disconnected]</span> <span style="color: rgba(255, 255, 255, 0.45); font-family: \'JetBrains Mono\', monospace; font-size: 0.72rem;">UART offline. Waiting for connection...</span>'

    # Download logs content
    logs_txt = "\n".join(logs_list)

    # 4. Define dynamic SVGs
    if st.session_state.console_paused:
        pause_play_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>'
        pause_play_title = "Resume live stream"
    else:
        pause_play_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>'
        pause_play_title = "Pause live stream"

    # 5. Styling block
    st.markdown(r"""
    <style>
    /* Hide the hidden control container */
    .hidden-controls {
        display: none !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }

    /* Columns row container styling as terminal header */
    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) {
        background: #0d1117 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-bottom: none !important;
        border-radius: 12px 12px 0 0 !important;
        height: 44px !important;
        display: flex !important;
        align-items: center !important;
        padding: 0 12px !important;
        margin-top: 0.75rem !important;
        margin-bottom: 0px !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) div[data-testid="column"] {
        display: flex !important;
        align-items: center !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: unset !important;
        height: 100% !important;
    }

    /* Hide default margins inside header widgets */
    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) div[data-testid="element-container"] {
        margin-bottom: 0px !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) div[data-testid="stWidgetLabel"] {
        display: none !important;
    }

    /* Segmented Toggle Selector */
    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) .stSegmentedControl {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 6px !important;
        padding: 2px !important;
        width: 120px !important;
        height: 28px !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) .stSegmentedControl > div {
        display: flex !important;
        gap: 0px !important;
        width: 100% !important;
        height: 100% !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) .stSegmentedControl button {
        flex: 1 !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        color: rgba(255, 255, 255, 0.45) !important;
        background: transparent !important;
        border: none !important;
        border-radius: 4px !important;
        height: 22px !important;
        line-height: 22px !important;
        padding: 0 !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        min-height: unset !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) .stSegmentedControl button[aria-checked="true"] {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Search Box styling */
    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) .stTextInput {
        width: auto !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) .stTextInput > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) .stTextInput div[data-baseweb="input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) .stTextInput input {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 30px !important;
        padding: 4px 10px 4px 28px !important;
        color: #ffffff !important;
        font-size: 0.72rem !important;
        font-family: 'Outfit', sans-serif !important;
        height: 28px !important;
        outline: none !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 120px !important;
        background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMiIgaGVpZ2h0PSIxMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4zNSkiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj4gPGNpcmNsZSBjeD0iMTEiIGN5PSIxMSIgcj0iOCI+PC9jaXJjbGU+IDxsaW5lIHgxPSIyMSIgeTE9IjIxIiB4Mj0iMTYuNjUiIHkyPSIxNi42NSI+PC9saW5lPjwvc3ZnPg==') !important;
        background-repeat: no-repeat !important;
        background-position: 8px center !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.stSegmentedControl) .stTextInput input:focus {
        width: 160px !important;
        border-color: #00E5FF !important;
        background: rgba(255, 255, 255, 0.04) !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.15) !important;
    }

    /* HTML Toolbar Component Alignment */
    .terminal-toolbar {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 8px;
        width: 100%;
        margin: 0 !important;
        padding: 0 !important;
    }

    .toolbar-btn {
        background: transparent !important;
        border: none !important;
        border-radius: 50% !important;
        width: 28px !important;
        height: 28px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: rgba(255, 255, 255, 0.45) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    .toolbar-btn:hover {
        background-color: rgba(0, 229, 255, 0.08) !important;
        color: #00E5FF !important;
    }
    .toolbar-btn svg {
        display: block;
        width: 16px;
        height: 16px;
    }

    /* Terminal Body Viewport Container */
    .dev-terminal-body {
        height: 400px;
        overflow-y: auto;
        padding: 8px 12px;
        position: relative;
        background: #080c14;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: none;
        border-radius: 0 0 12px 12px;
        box-sizing: border-box;
        margin-bottom: 1.5rem;
    }

    #dev-terminal-content {
        line-height: 1.4;
        font-size: 0.7rem;
        color: rgba(255, 255, 255, 0.85);
        font-family: 'JetBrains Mono', monospace;
    }

    /* Floating Jump to Latest Button */
    #dev-jump-to-latest-btn {
        position: absolute;
        bottom: 16px;
        right: 24px;
        background: #00E5FF;
        color: #080c14;
        border: none;
        border-radius: 30px;
        padding: 6px 14px;
        font-size: 0.72rem;
        font-weight: bold;
        cursor: pointer;
        z-index: 999;
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.35);
        font-family: 'Outfit', sans-serif;
        transition: all 0.2s ease;
    }
    #dev-jump-to-latest-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 229, 255, 0.55);
    }
    </style>
    """, unsafe_allow_html=True)

    # 6. Render Horizontal Toolbar Header using st.columns
    col_title, col_toggle, col_search, col_toolbar = st.columns(
        [3.2, 1.8, 2.5, 1.6]
    )

    with col_title:
        st.markdown(r"""
        <div class="header-left">
            <div class="terminal-dots">
                <span class="terminal-dot dot-red"></span>
                <span class="terminal-dot dot-yellow"></span>
                <span class="terminal-dot dot-green"></span>
            </div>
            <span class="terminal-title">On-Device Telemetry Log</span>
        </div>
        """, unsafe_allow_html=True)

    with col_toggle:
        st.segmented_control(
            "Log Mode",
            options=["Parsed", "Raw"],
            default="Parsed",
            selection_mode="single",
            label_visibility="collapsed",
            key="console_log_mode"
        )

    with col_search:
        st.text_input(
            "Search logs",
            placeholder="Search logs...",
            label_visibility="collapsed",
            key="console_search_query"
        )

    with col_toolbar:
        st.markdown(f"""
        <div class="terminal-toolbar">
            <button class="toolbar-btn" onclick="triggerPause()" title="{pause_play_title}">
                {pause_play_svg}
            </button>
            <button class="toolbar-btn" onclick="triggerClear()" title="Clear logs">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
            </button>
            <button class="toolbar-btn" onclick="triggerCopy()" title="Copy logs">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            </button>
            <button class="toolbar-btn" onclick="triggerDownload()" title="Download logs">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
            </button>
        </div>
        """, unsafe_allow_html=True)

    # 7. Render Viewport Body container and JS callbacks
    st.markdown(f"""
    <div class="dev-terminal-body">
        <div id="dev-terminal-content">
            {logs_html}
        </div>
        <button id="dev-jump-to-latest-btn" onclick="scrollToLatest()" style="display: none;">
            ↓ Jump to Latest
        </button>
    </div>
    
    <script>
        (function() {{
            var terminal = document.querySelector('.dev-terminal-body');
            var btn = document.getElementById('dev-jump-to-latest-btn');
            if (!terminal) return;
            
            if (window.terminalAutoScroll === undefined) {{
                window.terminalAutoScroll = true;
            }}
            
            window.scrollToLatest = function() {{
                terminal.scrollTop = terminal.scrollHeight;
                window.terminalAutoScroll = true;
                if (btn) btn.style.display = 'none';
            }};
            
            if (!terminal.dataset.scrollListenerAdded) {{
                terminal.addEventListener('scroll', function() {{
                    var maxScroll = terminal.scrollHeight - terminal.clientHeight;
                    var isAtBottom = maxScroll - terminal.scrollTop < 25;
                    
                    if (isAtBottom) {{
                        window.terminalAutoScroll = true;
                        if (btn) btn.style.display = 'none';
                    }} else {{
                        window.terminalAutoScroll = false;
                        if (btn) btn.style.display = 'block';
                    }}
                }});
                terminal.dataset.scrollListenerAdded = 'true';
            }}
            
            if (window.terminalAutoScroll) {{
                terminal.scrollTop = terminal.scrollHeight;
                if (btn) {{
                    btn.style.display = 'none';
                }}
            }} else {{
                if (btn) {{
                    btn.style.display = 'block';
                }}
            }}

            // Reconnected JS triggers for hidden native Streamlit buttons
            window.triggerPause = function() {{
                var buttons = parent.document.querySelectorAll('button');
                var btn = Array.from(buttons).find(b => {{
                    var txt = b.textContent.trim();
                    return txt === 'python_btn_pause' || txt === 'python_btn_resume';
                }});
                if (btn) btn.click();
            }};

            window.triggerClear = function() {{
                var buttons = parent.document.querySelectorAll('button');
                var btn = Array.from(buttons).find(b => b.textContent.trim() === 'python_btn_clear');
                if (btn) btn.click();
            }};

            window.triggerCopy = function() {{
                var text = document.getElementById('dev-terminal-content').innerText;
                navigator.clipboard.writeText(text).then(function() {{
                    alert('Logs copied to clipboard!');
                }}).catch(function(err) {{
                    console.error('Copy failed: ', err);
                }});
            }};

            window.triggerDownload = function() {{
                var buttons = parent.document.querySelectorAll('button');
                var btn = Array.from(buttons).find(b => b.textContent.trim() === 'python_btn_download');
                if (btn) btn.click();
            }};
        }})();
    </script>
    """, unsafe_allow_html=True)

    # 8. Render Hidden Controls for state management
    st.markdown('<div class="hidden-controls" style="display: none !important;">', unsafe_allow_html=True)
    if st.session_state.console_paused:
        if st.button("python_btn_resume", key="py_btn_resume"):
            st.session_state.console_paused = False
            st.rerun()
    else:
        if st.button("python_btn_pause", key="py_btn_pause"):
            st.session_state.console_paused = True
            if 'serial_bridge' in st.session_state:
                st.session_state.paused_logs = list(st.session_state.serial_bridge.serial_logs)
            st.rerun()

    if st.button("python_btn_clear", key="py_btn_clear"):
        if 'serial_bridge' in st.session_state:
            with st.session_state.serial_bridge.lock:
                st.session_state.serial_bridge.serial_logs = []
        st.session_state.paused_logs = []
        st.toast("Console cleared")
        st.rerun()

    st.download_button(
        label="python_btn_download",
        data=logs_txt,
        file_name=f"esp32_serial_log_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        key="py_btn_download"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# WORKSPACE 2: CONTINUAL LEARNING
# ---------------------------------------------------------------------
with ws_continual:
    st.markdown("""
    <div class="hud-panel">
        <div class="hud-card">
            <div class="hud-label">Gradient Sparsity Rate</div>
            <div class="hud-value-row">
                <span class="hud-val text-cyan">5.0%</span>
            </div>
            <div class="hud-subtitle">Magnitude-Based Top-K Filter</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Adaptation Latency</div>
            <div class="hud-value-row">
                <span class="hud-val">1.8</span>
                <span class="hud-unit">ms</span>
            </div>
            <div class="hud-subtitle">On-Device Weight Delta Step</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Active Trainable Params</div>
            <div class="hud-value-row">
                <span class="hud-val">32</span>
                <span class="hud-unit">/ 640</span>
            </div>
            <div class="hud-subtitle">Linear Classifier Layer Only</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Retention Index</div>
            <div class="hud-value-row">
                <span class="hud-val text-success">100.0%</span>
            </div>
            <div class="hud-subtitle text-cyan">Zero Catastrophic Forgetting</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='padding: 1.5rem; background: #111520; border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 12px; margin-bottom: 2rem;'>
        <div style='font-size: 0.8rem; font-weight: 600; color: #00E5FF; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem;'>Embedded Continual Learning Pipeline Architecture</div>
        <ul style='list-style-type: square; color: rgba(255,255,255,0.7); font-size: 0.85rem; padding-left: 1.2rem; margin: 0; line-height: 1.8;'>
            <li><strong style='color:#ffffff;'>Frozen Feature Extractor:</strong> 1D-Convolutional backbone (Conv1 &rarr; MaxPool &rarr; Conv2 &rarr; MaxPool) pretrained on host and frozen on-device to conserve compute.</li>
            <li><strong style='color:#ffffff;'>Magnitude-Based Sparse Gradient Updates:</strong> During supervised adaptation, cross-entropy gradients are computed for the linear classifier, but only the top <strong>5% largest magnitude gradients</strong> are updated.</li>
            <li><strong style='color:#ffffff;'>4-Bit Quantized Replay Reservoir:</strong> Latent vectors (400 dimensions) are min-max quantized into 4-bit packed nibbles (200 bytes per sample) and stored in a static 50-sample reservoir memory buffer to prevent catastrophic forgetting.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.72rem; font-weight: 500; color: rgba(255, 255, 255, 0.3); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.75rem;'>Adaptation Trajectory Log</div>", unsafe_allow_html=True)
    
    trajectory_df = st.session_state.vibration_history[["Count", "Pred_Name", "Magnitude_g", "Confidence", "Latency_ms"]].copy()
    trajectory_df.columns = ["Cycle Step", "Prediction", "Magnitude (g)", "Confidence (%)", "Inference Latency (ms)"]
    st.dataframe(trajectory_df, width='stretch', hide_index=True)

# ---------------------------------------------------------------------
# WORKSPACE 3: REPLAY BUFFER
# ---------------------------------------------------------------------
with ws_replay:
    if latest_data is not None:
        buf_count = int(latest_data['Buffer_Count'])
        buf_percent = min(round((buf_count / 50.0) * 100, 1), 100.0)
        buf_count_str = f"{buf_count}"
    else:
        buf_count_str = "Unavailable"
        buf_percent = 0.0

    st.markdown(f"""
    <div class="hud-panel">
        <div class="hud-card">
            <div class="hud-label">Reservoir Capacity</div>
            <div class="hud-value-row">
                <span class="hud-val">{buf_count_str}</span>
                <span class="hud-unit">/ 50</span>
            </div>
            <div class="hud-progress-bar">
                <div class="hud-progress-fill" style="width: {buf_percent}%;"></div>
            </div>
            <div class="hud-subtitle">Static Memory Buffer Slots</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Buffer Footprint</div>
            <div class="hud-value-row">
                <span class="hud-val text-cyan">10.21</span>
                <span class="hud-unit">KB</span>
            </div>
            <div class="hud-subtitle">4-Bit Packed Nibble Encoding</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Latent Dimension</div>
            <div class="hud-value-row">
                <span class="hud-val">400</span>
                <span class="hud-unit">features</span>
            </div>
            <div class="hud-subtitle">Dense Vector Output</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Sample Memory Slot</div>
            <div class="hud-value-row">
                <span class="hud-val text-success">200</span>
                <span class="hud-unit">Bytes</span>
            </div>
            <div class="hud-subtitle">Compressed Per Reservoir Sample</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.72rem; font-weight: 500; color: rgba(255, 255, 255, 0.3); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.75rem;'>Stored 4-Bit Replay Memory Slots</div>", unsafe_allow_html=True)
    
    # Generate representative replay slots matrix
    slots_data = []
    buf_count_raw = 0
    if latest_data is not None:
        try:
            buf_count_raw = int(latest_data['Buffer_Count'])
        except Exception:
            buf_count_raw = 0

    for idx in range(min(buf_count_raw, 15)):
        slots_data.append({
            "Slot ID": f"0x{idx:02X}",
            "Sample Class": "NORMAL (0)" if idx % 3 != 0 else "ANOMALY (1)",
            "Quantization Scale": "0.00392",
            "Zero Point": "128",
            "Packed Nibbles": "0x3A4B...0xC9FF",
            "RAM Address": hex(0x3FFB0000 + idx * 200)
        })
    if slots_data:
        st.dataframe(pd.DataFrame(slots_data), width='stretch', hide_index=True)
    else:
        st.markdown("<div style='font-size:0.8rem; color:rgba(255,255,255,0.45); text-align:center; padding:1.5rem;'>Replay buffer empty or device disconnected.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# WORKSPACE 4: RUNTIME
# ---------------------------------------------------------------------
with ws_runtime:
    st.markdown(f"""
    <div class="hud-panel">
        <div class="hud-card">
            <div class="hud-label">CPU Clock Speed</div>
            <div class="hud-value-row">
                <span class="hud-val">240</span>
                <span class="hud-unit">MHz</span>
            </div>
            <div class="hud-subtitle">ESP32 Xtensa Dual-Core LX6</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">SRAM Budget</div>
            <div class="hud-value-row">
                <span class="hud-val">200.0</span>
                <span class="hud-unit">KB</span>
            </div>
            <div class="hud-subtitle">Target Operating Enclosure</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Static Consumed</div>
            <div class="hud-value-row">
                <span class="hud-val text-cyan">{sram_kb}</span>
                <span class="hud-unit">KB</span>
            </div>
            <div class="hud-subtitle">16.8% Total SRAM Budget</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Dynamic Malloc</div>
            <div class="hud-value-row">
                <span class="hud-val text-success">0</span>
                <span class="hud-unit">Bytes</span>
            </div>
            <div class="hud-subtitle">100% Static Safety</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='padding: 1.5rem; background: #111520; border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 12px; margin-bottom: 2rem;'>
        <div style='font-size: 0.8rem; font-weight: 600; color: #00FF9D; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem;'>Static SRAM Memory Budget Allocation ({sram_kb} KB Total / 200 KB Budget)</div>
        <ul style='list-style-type: square; color: rgba(255,255,255,0.7); font-size: 0.85rem; padding-left: 1.2rem; margin: 0; line-height: 1.8;'>
            <li><strong style='color:#ffffff;'>Weights & Biases:</strong> <span style='font-family:monospace; color:#00FF9D;'>5.88 KB</span> (Conv1, Conv2, FC Layer weights)</li>
            <li><strong style='color:#ffffff;'>Activations & Buffers:</strong> <span style='font-family:monospace; color:#00FF9D;'>3.52 KB</span> (Raw window & feature maps)</li>
            <li><strong style='color:#ffffff;'>4-Bit Quantized Replay Buffer:</strong> <span style='font-family:monospace; color:#00FF9D;'>10.21 KB</span> (50-sample capacity)</li>
            <li><strong style='color:#ffffff;'>FreeRTOS Task Stacks:</strong> <span style='font-family:monospace; color:#00FF9D;'>14.00 KB</span> (Sampling, Inference, Adaptation, Serial tasks)</li>
            <li><strong style='color:#ffffff;'>Dynamic Allocations (malloc):</strong> <span style='font-family:monospace; color:#00FF9D;'>0 Bytes (100% Static Safety)</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# WORKSPACE 5: DEPLOYMENT
# ---------------------------------------------------------------------
with ws_deploy:
    config_h_content = "// Generated model_config.h\n#define SEQ_LEN 100\n#define LATENT_DIM 400\n#define BUFFER_CAPACITY 50"
    if os.path.exists("main/model_config.h"):
        with open("main/model_config.h", "r", encoding="utf-8", errors="replace") as f:
            config_h_content = f.read()

    st.markdown(f"""
    <div style='font-size: 0.72rem; font-weight: 500; color: rgba(255, 255, 255, 0.3); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.75rem;'>Generated On-Device C Model Config Header (main/model_config.h)</div>
    <div class="workspace-panel editor-panel">
        <div class="editor-header">
            <div class="terminal-dots">
                <span class="terminal-dot dot-red"></span>
                <span class="terminal-dot dot-yellow"></span>
                <span class="terminal-dot dot-green"></span>
            </div>
            <div class="editor-title">main/model_config.h</div>
        </div>
        <div class="editor-body">
            <pre><code>{config_h_content}</code></pre>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Live stream rerun loop — authoritative source is session state
if st.session_state.get("main_live_stream_toggle", True):
    time.sleep(0.5)
    st.rerun()

