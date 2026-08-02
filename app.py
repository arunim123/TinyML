import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import random
import os
import sys

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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"], .stApp {
    background-color: #080c14;
    color: #cbd5e1;
    font-family: 'Outfit', sans-serif;
}

section[data-testid="stSidebar"] {
    background-color: #0c1220 !important;
    border-right: 1px solid #1e293b !important;
}

.main-title {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    letter-spacing: -0.05rem;
}

.subtitle {
    color: #64748b;
    font-size: 1.0rem;
    margin-bottom: 1.5rem;
    text-transform: uppercase;
    letter-spacing: 0.15rem;
}

.hud-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 1.2rem;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}

.hud-card:hover {
    border-color: rgba(56, 189, 248, 0.5);
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.15);
    transform: translateY(-2px);
}

.hud-title {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05rem;
    margin-bottom: 0.4rem;
}

.hud-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f8fafc;
}

.hud-sub {
    font-size: 0.75rem;
    margin-top: 0.2rem;
}

.text-success { color: #10b981 !important; }
.text-warning { color: #f59e0b !important; }
.text-danger { color: #ef4444 !important; }
.text-cyan { color: #00f2fe !important; }

.banner-success {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid #10b981;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #34d399;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

.banner-warning {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #fbbf24;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

.terminal-header {
    background: #1e293b;
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.terminal-dots { display: flex; gap: 6px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.dot-red { background: #ef4444; }
.dot-yellow { background: #f59e0b; }
.dot-green { background: #10b981; }
.terminal-title { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Session State Initialization
# =====================================================================
if 'serial_bridge' not in st.session_state and serial_bridge is not None:
    st.session_state.serial_bridge = serial_bridge.ESP32SerialBridge()

if 'vibration_history' not in st.session_state:
    now = time.time()
    st.session_state.vibration_history = pd.DataFrame([
        {
            "Timestamp": now - (15 - i) * 0.2,
            "Count": i,
            "Magnitude_g": 1.0 + random.uniform(-0.03, 0.03),
            "Pred_Label": 0,
            "Pred_Name": "NORMAL",
            "Confidence": 98.5 + random.uniform(-1, 1),
            "Latency_ms": 1.8 + random.uniform(-0.1, 0.1),
            "Buffer_Count": min(i, 50)
        }
        for i in range(1, 16)
    ])

if 'sim_fault_active' not in st.session_state:
    st.session_state.sim_fault_active = False

if 'sim_step_count' not in st.session_state:
    st.session_state.sim_step_count = 15

# =====================================================================
# Sidebar Console & Serial Bridge Controls
# =====================================================================
with st.sidebar:
    st.markdown("<h2 style='margin-top: 0; color: #00f2fe;'>CONTROL CONSOLE</h2>", unsafe_allow_html=True)
    st.write("---")
    
    st.markdown("### 🔌 TARGET HARDWARE")
    target_hardware = st.selectbox(
        "Deployment Profile",
        ["ESP32 Dual-Core LX6 (200KB SRAM Limit)"]
    )
    
    st.write("")
    st.markdown("### ⚙️ CONTINUAL LEARNING PIPELINE")
    sparse_engine_active = st.toggle("Sparse-Update & 4-bit Replay Engine", value=True)
    
    st.write("---")
    st.markdown("### 🔌 USB SERIAL HARDWARE BRIDGE")
    
    serial_connected = False
    if serial_bridge and serial_bridge.SERIAL_AVAILABLE:
        available_ports = serial_bridge.list_available_serial_ports()
        if not available_ports:
            available_ports = ["Simulated Serial (No Hardware Detected)", "COM3", "COM4", "COM5", "/dev/ttyUSB0"]
        
        selected_port = st.selectbox("ESP32 Serial COM Port", available_ports)
        col_conn1, col_conn2 = st.columns(2)
        with col_conn1:
            if st.button("Connect USB"):
                if "Simulated" not in selected_port:
                    success = st.session_state.serial_bridge.connect(selected_port)
                    if success:
                        st.success(f"Connected to {selected_port}")
                    else:
                        st.error("Connection failed")
                else:
                    st.info("Simulated Serial Active")
        with col_conn2:
            if st.button("Disconnect"):
                if hasattr(st.session_state, 'serial_bridge'):
                    st.session_state.serial_bridge.disconnect()
                st.info("Disconnected")

        if hasattr(st.session_state, 'serial_bridge') and st.session_state.serial_bridge.ser and st.session_state.serial_bridge.ser.is_open:
            serial_connected = True
            st.markdown("<span style='color:#10b981; font-weight:bold;'>🟢 SERIAL BRIDGE CONNECTED</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#f59e0b;'>🟡 STREAMING SIMULATED TELEMETRY</span>", unsafe_allow_html=True)
            
        # Interactive Adaptation Controls
        st.write("---")
        st.markdown("### 🏷️ ON-DEVICE ADAPTATION LABELS")
        st.caption("Send real-time health label over USB serial to trigger on-device sparse update & 4-bit replay:")
        col_lbl1, col_lbl2 = st.columns(2)
        with col_lbl1:
            if st.button("🟢 Label: Normal"):
                sent = st.session_state.serial_bridge.send_label(0)
                st.toast("Transmitted target label [0: Normal] over serial!" if sent else "Simulated Label [0: Normal] queued", icon="🟢")
                st.session_state.sim_fault_active = False
        with col_lbl2:
            if st.button("🔴 Label: Anomaly"):
                sent = st.session_state.serial_bridge.send_label(1)
                st.toast("Transmitted target label [1: Anomaly] over serial!" if sent else "Simulated Label [1: Anomaly] queued", icon="🔴")
                st.session_state.sim_fault_active = True
    else:
        st.warning("pyserial module not available. Serial bridge operating in simulation mode.")

    st.write("---")
    st.markdown("### 📡 DRIFT SIMULATION")
    live_sim_on = st.toggle("Live Telemetry Stream", value=True)
    if not serial_connected:
        st.session_state.sim_fault_active = st.toggle("Induce Vibration Fault (Drift)", value=st.session_state.sim_fault_active)

    st.write("---")
    st.markdown(
        "<div style='font-size: 0.75rem; color: #475569; text-align: center;'>"
        "TinyML ESP32 Vibration PDM System v1.0<br>"
        "Continual Learning & Codegen Engine"
        "</div>",
        unsafe_allow_html=True
    )

# =====================================================================
# Update Telemetry Stream Data
# =====================================================================
if live_sim_on:
    st.session_state.sim_step_count += 1
    current_time = time.time()
    
    if serial_connected:
        telem = st.session_state.serial_bridge.get_latest_telemetry()
        mag = telem.get("mag", 1.0)
        pred = telem.get("pred", 0)
        conf = telem.get("conf", 0.95) * 100.0
        lat = telem.get("latency_ms", 1.8)
        buf_c = telem.get("buffer", 0)
    else:
        # Simulated vibration telemetry stream
        if not st.session_state.sim_fault_active:
            mag = 1.0 + np.random.normal(0, 0.04)
            pred = 0
            conf = 98.5 + np.random.normal(0, 0.8)
        else:
            mag = 1.45 + np.random.normal(0, 0.12)
            pred = 1
            conf = 94.2 + np.random.normal(0, 1.5)
        lat = 1.8 + np.random.normal(0, 0.05)
        buf_c = min(st.session_state.sim_step_count, 50)

    pred_name = "ANOMALOUS" if pred == 1 else "NORMAL"

    new_entry = pd.DataFrame([{
        "Timestamp": current_time,
        "Count": st.session_state.sim_step_count,
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

latest_data = st.session_state.vibration_history.iloc[-1]

# Calculate real static memory footprint
sram_kb = 33.60
if memory_planner is not None:
    try:
        ok, static_bytes = memory_planner.calculate_sram_footprint("tools/model_spec.json", verbose=False)

        sram_kb = round(static_bytes / 1024.0, 2)
    except Exception:
        pass

# =====================================================================
# Main Dashboard UI Layout
# =====================================================================
st.markdown("<h1 class='main-title'>TinyML Vibration Predictive Maintenance</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>ESP32 On-Device Continual Learning & Automatic Codegen Engine</div>", unsafe_allow_html=True)

# Status Banner
if latest_data["Pred_Label"] == 0:
    st.markdown(
        f"<div class='banner-success'>🟢 STATUS: SYSTEM HEALTHY — Machine vibration within baseline tolerances. "
        f"Inference Latency: {latest_data['Latency_ms']} ms | Memory: {sram_kb} KB Static SRAM</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"<div class='banner-warning'>🔴 STATUS: ANOMALY DETECTED — High vibration energy / harmonic breakdown! "
        f"Prediction Confidence: {latest_data['Confidence']}% | Trigger adaptation from sidebar.</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------
# Row 1: Live Hardware HUD / Telemetry Matrix
# ---------------------------------------------------------------------
hud_col1, hud_col2, hud_col3, hud_col4 = st.columns(4)

with hud_col1:
    pred_class = "text-success" if latest_data["Pred_Label"] == 0 else "text-danger"
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Health State Prediction</div>
        <div class="hud-value {pred_class}" style="font-size: 1.5rem;">{latest_data['Pred_Name']}</div>
        <div class="hud-sub text-cyan">Confidence: {latest_data['Confidence']}%</div>
    </div>
    """, unsafe_allow_html=True)

with hud_col2:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Vibration Magnitude</div>
        <div class="hud-value">{latest_data['Magnitude_g']:.3f} g</div>
        <div class="hud-sub text-cyan">3-Axis Magnitude ||a||</div>
    </div>
    """, unsafe_allow_html=True)

with hud_col3:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Static SRAM Footprint</div>
        <div class="hud-value">{sram_kb} KB</div>
        <div class="hud-sub text-success">Limit: 200 KB (Zero Malloc)</div>
    </div>
    """, unsafe_allow_html=True)

with hud_col4:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Inference & Replay</div>
        <div class="hud-value">{latest_data['Latency_ms']} ms</div>
        <div class="hud-sub text-cyan">Replay Buffer: {int(latest_data['Buffer_Count'])}/50 samples</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Row 2: Live Diagnostic Output Logs & Target Hardware Code
# ---------------------------------------------------------------------
row2_col1, row2_col2 = st.columns([3, 2])

with row2_col1:
    st.markdown("### 🖥️ ON-DEVICE TELEMETRY LOG")
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-dots">
            <span class="dot dot-red"></span>
            <span class="dot dot-yellow"></span>
            <span class="dot dot-green"></span>
        </div>
        <div class="terminal-title">esp32-serial-uart0 --baud 115200</div>
    </div>
    """, unsafe_allow_html=True)
    
    timestamp = time.strftime("%H:%M:%S")
    serial_status_text = "CONNECTED" if serial_connected else "SIMULATED STREAM"
    log_text = (
        f"[{timestamp}] [INIT] ESP32 Dual-Core LX6 initialized (200KB SRAM budget)\n"
        f"[{timestamp}] [IMU] Sampling 3-axis acceleration magnitude at 50Hz (20ms interval)\n"
        f"[{timestamp}] [TELEMETRY] #{int(latest_data['Count'])} | ||a||: {latest_data['Magnitude_g']:.3f}g | Pred: {latest_data['Pred_Name']} ({latest_data['Confidence']}%)\n"
        f"[{timestamp}] [MEMORY] Static RAM: {sram_kb} KB (Weights: 5.88KB, Act: 3.52KB, 4-bit Replay: 10.21KB, Stacks: 14KB)\n"
        f"[{timestamp}] [PERF] Latency: {latest_data['Latency_ms']} ms | Replay Buffer: {int(latest_data['Buffer_Count'])}/50\n"
        f"[{timestamp}] [STATUS] Serial Bridge Status: {serial_status_text}"
    )
    st.code(log_text, language="bash")

with row2_col2:
    st.markdown("### 🔩 EMITTED MODEL HEADERS")
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-dots">
            <span class="dot dot-red"></span>
            <span class="dot dot-yellow"></span>
            <span class="dot dot-green"></span>
        </div>
        <div class="terminal-title">main/model_config.h</div>
    </div>
    """, unsafe_allow_html=True)
    
    config_h_content = "// Generated model_config.h\n#define SEQ_LEN 100\n#define LATENT_DIM 400\n#define BUFFER_CAPACITY 50"
    if os.path.exists("main/model_config.h"):
        with open("main/model_config.h", "r", encoding="utf-8", errors="replace") as f:
            config_h_content = f.read()
            
    st.code(config_h_content, language="c")

# ---------------------------------------------------------------------
# Row 3: Plots & Visual Analytics
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 REAL-TIME VIBRATION & CONTINUAL LEARNING CHARTS")

row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    # Vibration Acceleration Magnitude Chart
    fig_vib = go.Figure()
    
    counts = st.session_state.vibration_history["Count"]
    mags = st.session_state.vibration_history["Magnitude_g"]
    
    fig_vib.add_trace(go.Scatter(
        x=counts,
        y=mags,
        mode='lines+markers',
        name='Acceleration Magnitude ||a|| (g)',
        line=dict(color='#00f2fe', width=3),
        marker=dict(size=6, color='#38bdf8')
    ))
    
    # Add horizontal baseline and anomaly threshold lines
    fig_vib.add_hline(y=1.00, line_dash="dash", line_color="#10b981", annotation_text="Normal Baseline (1.0g)")
    fig_vib.add_hline(y=1.30, line_dash="dash", line_color="#ef4444", annotation_text="Anomaly Threshold (1.3g)")

    fig_vib.update_layout(
        title={'text': "Live Acceleration Vector Magnitude ||a|| (g)", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title="Acceleration (g)", gridcolor="#1e293b", range=[0.5, 2.0]),
        xaxis=dict(title="Inference Sample Window Count", gridcolor="#1e293b"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=320,
        showlegend=False
    )
    
    st.plotly_chart(fig_vib, use_container_width=True)

with row3_col2:
    # Prediction Confidence & Adaptation History Chart
    fig_conf = go.Figure()
    
    confs = st.session_state.vibration_history["Confidence"]
    preds = st.session_state.vibration_history["Pred_Label"]
    
    colors = ['#10b981' if p == 0 else '#ef4444' for p in preds]
    
    fig_conf.add_trace(go.Bar(
        x=counts,
        y=confs,
        marker_color=colors,
        name='Prediction Confidence (%)'
    ))

    fig_conf.update_layout(
        title={'text': "Health State Prediction Confidence (%)", 'y':0.95, 'x':0.5, 'xanchor': 'center'},
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title="Confidence (%)", gridcolor="#1e293b", range=[0, 100]),
        xaxis=dict(title="Inference Sample Window Count", gridcolor="#1e293b"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=320,
        showlegend=False
    )
    
    st.plotly_chart(fig_conf, use_container_width=True)

# ---------------------------------------------------------------------
# Row 4: Technology Matrix & Memory Breakdown
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("### 💡 ESP32 CONTINUAL LEARNING ARCHITECTURE")

tab1, tab2 = st.tabs(["🧬 On-Device Continual Learning (Sparse + 4-Bit Replay)", "📊 Static Memory Breakdown"])

with tab1:
    st.markdown("""
    #### Embedded Continual Learning Pipeline
    - **Frozen Feature Extractor**: 1D-Convolutional backbone (Conv1 -> MaxPool -> Conv2 -> MaxPool) pretrained on host and frozen on-device.
    - **Magnitude-Based Sparse Gradient Updates**: During supervised adaptation, cross-entropy gradients are computed for the linear classifier, but only the top **5% largest magnitude gradients** are updated.
    - **4-Bit Quantized Replay Buffer**: Latent vectors ($400$ dimensions) are min-max quantized into 4-bit packed nibbles ($200$ bytes per sample) and stored in a static $50$-sample reservoir memory buffer to prevent catastrophic forgetting.
    """)

with tab2:
    st.markdown(f"""
    #### Static SRAM Memory Budget Allocation ({sram_kb} KB Total / 200 KB Budget)
    - **Weights & Biases**: **5.88 KB** (Conv1, Conv2, FC Layer weights)
    - **Activations & Buffers**: **3.52 KB** (Raw window & feature maps)
    - **4-Bit Quantized Replay Buffer**: **10.21 KB** (50-sample capacity)
    - **FreeRTOS Task Stacks**: **14.00 KB** (Sampling, Inference, Adaptation, Serial tasks)
    - **Dynamic Allocations (`malloc`)**: **0 Bytes (100% Static Safety)**
    """)

# Live stream rerun loop
if live_sim_on:
    time.sleep(0.5)
    st.rerun()
