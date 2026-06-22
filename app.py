import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import random

# =====================================================================
# Page Configurations & CSS Styling
# =====================================================================
st.set_page_config(
    page_title="EdgeOrchestrate AI - Hackathon Hub",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium developer dashboard styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

/* Base style settings */
html, body, [class*="css"], .stApp {
    background-color: #080c14;
    color: #cbd5e1;
    font-family: 'Outfit', sans-serif;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0c1220 !important;
    border-right: 1px solid #1e293b !important;
}

/* Custom header colors with beautiful gradients */
.main-title {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    letter-spacing: -0.05rem;
}

.subtitle {
    color: #64748b;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    text-transform: uppercase;
    letter-spacing: 0.15rem;
}

/* Metric / HUD Card styles */
.hud-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 12px;
    padding: 1.2rem;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}

.hud-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.1);
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

/* Color codes for HUD sub-text */
.text-success { color: #10b981 !important; }
.text-warning { color: #f59e0b !important; }
.text-danger { color: #ef4444 !important; }
.text-cyan { color: #00f2fe !important; }

/* Status banners */
.banner-success {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 8px;
    padding: 1rem;
    color: #34d399;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

.banner-warning {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px;
    padding: 1rem;
    color: #fbbf24;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

.banner-danger {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 8px;
    padding: 1rem;
    color: #f87171;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

/* Custom terminal font and size */
.stCodeBlock, code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Form layout overrides */
div[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Compiler Engine Backend Logic
# =====================================================================
class EdgeCompilerEngine:
    """
    Evaluates compilation metrics and generates mock compiler logs
    and target C++/OpenVINO IR code for edge devices.
    """
    
    @staticmethod
    def compile(target_device: str, sparse_update_on: bool, sensors: dict) -> dict:
        result = {}
        
        # Pull sensor readings for context-aware compiling logs
        vwc = sensors.get('vwc', 35.0)
        temp = sensors.get('temp', 22.5)
        perm = sensors.get('perm', 4.5)
        volt = sensors.get('volt', 3.3)
        
        timestamp = time.strftime("%H:%M:%S")
        
        if target_device == "Ultra-Low-Power MCU (STM32/ESP32)":
            result["sram_limit_kb"] = 256
            if not sparse_update_on:
                result["status"] = "CRASHED"
                result["memory_footprint_kb"] = 1950
                result["precision"] = "FP32 (Standard)"
                result["sparsity"] = "0% (Dense)"
                result["speedup"] = "N/A (Crashed)"
                result["active_layers"] = "0 / 30 Layers"
                result["energy_mw"] = 0.0
                
                result["logs"] = (
                    f"[{timestamp}] [INIT] Analyzing Target Silicon: STM32 / ESP32 MCU (256KB SRAM Target Limit)\n"
                    f"[{timestamp}] [INFO] Peripheral telemetry network detected: VWC={vwc:.1f}%, Temp={temp:.1f}°C, Permittivity={perm:.2f}, Bus Voltage={volt:.2f}V\n"
                    f"[{timestamp}] [STAGE 1] Resolving standard floating-point backpropagation gradient buffers...\n"
                    f"[{timestamp}] [STAGE 1] Allocation request: Weight matrix (650KB) + Activation map buffers (800KB) + Gradient history (500KB)\n"
                    f"[{timestamp}] [ERROR] Total requested heap/stack footprint: 1,950 KB\n"
                    f"[{timestamp}] [CRITICAL] Out of Memory (OOM) / System Stack Overflow!\n"
                    f"[{timestamp}] [CRITICAL] Requested footprint (1,950 KB) exceeds strict hardware limit (256 KB) by 1,694 KB.\n"
                    f"[{timestamp}] [FATAL] Stack pointer address collided with sensor peripheral bus interrupt vector.\n"
                    f"[{timestamp}] [FATAL] Device compilation aborted. System crashed."
                )
                
                result["emitted_code"] = (
                    "// ERROR: Compilation aborted due to Out of Memory (OOM)\n"
                    "// Memory requested: 1,950 KB | Limit: 256 KB\n"
                    "// Enable 'Sparse-Update Engine' to bypass memory wall."
                )
            else:
                result["status"] = "SUCCESS"
                result["memory_footprint_kb"] = 218
                result["precision"] = "4-bit Quantized Latent Replay"
                result["sparsity"] = "90% Sparsity"
                result["speedup"] = "1.0x (Baseline)"
                result["active_layers"] = "3 / 30 Layers"
                result["energy_mw"] = 14.5
                
                result["logs"] = (
                    f"[{timestamp}] [INIT] Analyzing Target Silicon: STM32 / ESP32 MCU (256KB SRAM Target Limit)\n"
                    f"[{timestamp}] [INFO] Peripheral telemetry network detected: VWC={vwc:.1f}%, Temp={temp:.1f}°C, Permittivity={perm:.2f}, Bus Voltage={volt:.2f}V\n"
                    f"[{timestamp}] [STAGE 1] Activating Dynamic Sparse-Update Neuromorphic Architecture...\n"
                    f"[{timestamp}] [STAGE 1] Applying Magnitude-Based Dynamic Sparsity: Bypassing 90% of low-salience activation maps.\n"
                    f"[{timestamp}] [STAGE 2] Applying 4-bit Quantized Latent Replay: Compressing history buffers into strict 32KB scratch block.\n"
                    f"[{timestamp}] [STAGE 3] Target-specific code emission utilizing ARM CMSIS-NN / ESP-NN intrinsics.\n"
                    f"[{timestamp}] [STAGE 3] Optimizing active training layers: 3 trainable projection heads, 27 frozen backbone layers.\n"
                    f"[{timestamp}] [SUCCESS] Active memory footprint footprint reduced to 218 KB (within 256KB SRAM boundary).\n"
                    f"[{timestamp}] [SUCCESS] Emit target: bare-metal C++ header and assembly optimization macros.\n"
                    f"[{timestamp}] [SUCCESS] Compilation completed successfully."
                )
                
                result["emitted_code"] = (
                    "// Auto-Generated ARM CMSIS-NN Optimized Sparsity Kernel\n"
                    "#include \"arm_math.h\"\n"
                    "#include \"arm_nnfunctions.h\"\n\n"
                    "#define BUFFER_SIZE_KB 32\n"
                    "#define SPARSITY_RATIO 0.90f // 90% salience bypass\n"
                    "#define ACTIVE_LAYERS 3\n\n"
                    "// 4-bit latent replay reconstruction helper\n"
                    "inline q7_t dequantize_4bit(uint8_t packed_val, float scale, float zero_point) {\n"
                    "    return (q7_t)((((float)(packed_val & 0x0F)) - zero_point) * scale);\n"
                    "}\n\n"
                    "void train_sparse_step(q7_t* activations, q7_t* weights, q31_t* bias) {\n"
                    "    // Bypassing 90% low-salience activation maps\n"
                    "    for (int i = 0; i < ACTIVATION_LEN; i++) {\n"
                    "        if (abs(activations[i]) < SALIENCE_THRESHOLD) {\n"
                    "            continue; // Skip gradient step for this node\n"
                    "        }\n"
                    "        // Update weights using 4-bit compressed latent historical buffer\n"
                    "        q7_t history = dequantize_4bit(history_buffer[i], 0.12f, -8.0f);\n"
                    "        weights[i] += (history * LR_COEFF) >> 8;\n"
                    "    }\n"
                    "}"
                )
                
        elif target_device == "Mid-Range Edge Core (ARM Cortex-A/Pi)":
            result["sram_limit_kb"] = 4096  # Virtual limit for comparison
            result["status"] = "WARNING"
            result["memory_footprint_kb"] = 2048
            result["precision"] = "FP16 / INT8 Mixed"
            result["sparsity"] = "0% (Dense)"
            result["speedup"] = "2.8x (CPU Threaded)"
            result["active_layers"] = "30 / 30 Layers"
            result["energy_mw"] = 480.0
            
            result["logs"] = (
                f"[{timestamp}] [INIT] Analyzing Target Silicon: ARM Cortex-A / Raspberry Pi Architecture\n"
                f"[{timestamp}] [INFO] Peripheral telemetry network detected: VWC={vwc:.1f}%, Temp={temp:.1f}°C, Permittivity={perm:.2f}, Bus Voltage={volt:.2f}V\n"
                f"[{timestamp}] [STAGE 1] Allocating standard TensorFlow Lite Micro runtime environment...\n"
                f"[{timestamp}] [STAGE 2] Checking memory requirements: 2,048 KB allocated in RAM tensor arena.\n"
                f"[{timestamp}] [WARN] Device operating under full dense parameters. Core temperature expected to rise.\n"
                f"[{timestamp}] [WARN] Active battery voltage: {volt:.2f}V. Continuous dense computation may result in thermal throttling.\n"
                f"[{timestamp}] [WARN] Recommend enabling Sparse-Update Engine to optimize duty-cycle and reduce energy draw by 90%.\n"
                f"[{timestamp}] [STAGE 3] Generating flatbuffer schema serialization (.tflite).\n"
                f"[{timestamp}] [SUCCESS] Compiled with warning. Emitted TFLite executor binary."
            )
            
            result["emitted_code"] = (
                "// TensorFlow Lite Micro Execution Graph configuration\n"
                "#include \"tensorflow/lite/micro/micro_interpreter.h\"\n"
                "#include \"tensorflow/lite/schema/schema_generated.h\"\n\n"
                "// Standard Dense FP16 / INT8 Mixed Precision Config\n"
                "const int tensor_arena_size = 2048 * 1024; // 2 MB RAM allocation\n"
                "alignas(16) uint8_t tensor_arena[tensor_arena_size];\n\n"
                "void execute_inference_step() {\n"
                "    TfLiteStatus invoke_status = interpreter->Invoke();\n"
                "    if (invoke_status != kTfLiteOk) {\n"
                "        error_reporter->Report(\"TFLite Execution Failed\\n\");\n"
                "    }\n"
                "}"
            )
            
        else:  # Next-Gen Intel Ultra Compute (Intel Core Ultra NPU / GPU)
            result["sram_limit_kb"] = 32768  # 32MB NPU cache limit
            result["status"] = "SUCCESS"
            result["memory_footprint_kb"] = 8192
            result["precision"] = "FP32 Precision (Unpruned)"
            result["sparsity"] = "0% (Full Dense)"
            result["speedup"] = "14.2x (OpenVINO IR NPU)"
            result["active_layers"] = "30 / 30 Layers"
            result["energy_mw"] = 15000.0  # 15W NPU/GPU block
            
            result["logs"] = (
                f"[{timestamp}] [INIT] Analyzing Target Silicon: Intel Core Ultra NPU / GPU Block Detected\n"
                f"[{timestamp}] [INFO] Peripheral telemetry network detected: VWC={vwc:.1f}%, Temp={temp:.1f}°C, Permittivity={perm:.2f}, Bus Voltage={volt:.2f}V\n"
                f"[{timestamp}] [STAGE 1] Next-Gen hardware detected. Unpruning all 30 layers to maximize telemetry regression accuracy.\n"
                f"[{timestamp}] [STAGE 2] Bypassing 4-bit loss penalty constraints; routing parameters to full float32 pathways.\n"
                f"[{timestamp}] [STAGE 3] Target-specific translation: Compiling graph into OpenVINO Intermediate Representation (IR).\n"
                f"[{timestamp}] [STAGE 4] Routing execution graph directly to hardware Intel NPU Compute Blocks.\n"
                f"[{timestamp}] [SUCCESS] Compiled OpenVINO XML/BIN Intermediate Representation.\n"
                f"[{timestamp}] [SUCCESS] NPU Direct Acceleration enabled. Performance multiplier: 14.2x compute speedup.\n"
                f"[{timestamp}] [SUCCESS] Compilation completed successfully."
            )
            
            result["emitted_code"] = (
                "<!-- OpenVINO Intermediate Representation (IR) Graph Segment -->\n"
                "<net name=\"EdgeOrchestrate_Intel_Ultra\" version=\"11\">\n"
                "    <layers>\n"
                "        <layer id=\"0\" name=\"telemetry_input\" type=\"Parameter\" version=\"opset10\">\n"
                "            <data element_type=\"f32\" shape=\"1, 4\"/>\n"
                "            <output>\n"
                "                <port id=\"0\" precision=\"FP32\" names=\"telemetry_input\"/>\n"
                "            </output>\n"
                "        </layer>\n"
                "        <layer id=\"1\" name=\"dense_unpruned_proj\" type=\"FullyConnected\" version=\"opset1\">\n"
                "            <data output_size=\"256\" precision=\"FP32\"/>\n"
                "            <input>\n"
                "                <port id=\"0\" precision=\"FP32\"/>\n"
                "            </input>\n"
                "            <output>\n"
                "                <port id=\"1\" precision=\"FP32\"/>\n"
                "            </output>\n"
                "        </layer>\n"
                "    </layers>\n"
                "    <edges>\n"
                "        <edge from-layer=\"0\" from-port=\"0\" to-layer=\"1\" to-port=\"0\"/>\n"
                "    </edges>\n"
                "</net>"
            )
            
        return result

# =====================================================================
# Session State Initialization (Telemetry Tracking & History)
# =====================================================================
if 'vwc' not in st.session_state:
    st.session_state.vwc = 35.0
if 'temp' not in st.session_state:
    st.session_state.temp = 22.5
if 'perm' not in st.session_state:
    st.session_state.perm = 4.5
if 'volt' not in st.session_state:
    st.session_state.volt = 3.3
if 'history' not in st.session_state:
    # Initialize mock history for timeline charts
    st.session_state.history = pd.DataFrame({
        "Iteration": list(range(1, 11)),
        "VWC": [35.0 + random.uniform(-1, 1) for _ in range(10)],
        "Temp": [22.5 + random.uniform(-0.5, 0.5) for _ in range(10)],
        "Permittivity": [4.5 + random.uniform(-0.1, 0.1) for _ in range(10)],
        "Voltage": [3.3 + random.uniform(-0.05, 0.05) for _ in range(10)]
    })
if 'iter_count' not in st.session_state:
    st.session_state.iter_count = 10

# =====================================================================
# Sidebar Console Controls
# =====================================================================
with st.sidebar:
    st.markdown("<h2 style='margin-top: 0; color: #00f2fe;'>CONTROL CONSOLE</h2>", unsafe_allow_html=True)
    st.write("---")
    
    st.markdown("### 🔌 TARGET SILICON")
    target_hardware = st.selectbox(
        "Deployment Profile",
        [
            "Ultra-Low-Power MCU (STM32/ESP32)",
            "Mid-Range Edge Core (ARM Cortex-A/Pi)",
            "Next-Gen Intel Ultra Compute (Core Ultra NPU)"
        ]
    )
    
    st.write("")
    
    # Custom colored toggle switch using st.checkbox styled
    st.markdown("### ⚙️ OPTIMIZATION PIPELINE")
    sparse_engine_active = st.toggle("Sparse-Update Engine (DSD-NA)", value=True, help="Enable Magnitude-Based Sparsity (90% bypass) & 4-bit Quantized Latent Replay.")
    
    st.write("---")
    
    st.markdown("### 📡 TELEMETRY SIMULATION")
    live_sim_on = st.toggle("Live Telemetry Stream", value=True)
    
    # Handle Live simulation vs Manual slider adjustment
    if live_sim_on:
        # Generate new drifted values
        st.session_state.vwc = np.clip(st.session_state.vwc + np.random.normal(0, 0.4), 0.0, 100.0)
        st.session_state.temp = np.clip(st.session_state.temp + np.random.normal(0, 0.15), -20.0, 60.0)
        st.session_state.perm = np.clip(st.session_state.perm + np.random.normal(0, 0.05), 1.0, 80.0)
        st.session_state.volt = np.clip(st.session_state.volt + np.random.normal(0, 0.01), 1.8, 5.0)
        
        # Increment iterations and append to data history
        st.session_state.iter_count += 1
        new_row = pd.DataFrame([{
            "Iteration": st.session_state.iter_count,
            "VWC": st.session_state.vwc,
            "Temp": st.session_state.temp,
            "Permittivity": st.session_state.perm,
            "Voltage": st.session_state.volt
        }])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
        # Keep sliding history window at 15 items
        if len(st.session_state.history) > 15:
            st.session_state.history = st.session_state.history.iloc[1:].reset_index(drop=True)
            
        # Display read-only metrics
        st.slider("Volumetric Water Content (VWC %)", 0.0, 100.0, float(st.session_state.vwc), disabled=True)
        st.slider("Soil Temperature (°C)", -20.0, 60.0, float(st.session_state.temp), disabled=True)
        st.slider("Dielectric Permittivity", 1.0, 80.0, float(st.session_state.perm), disabled=True)
        st.slider("Bus Voltage (V)", 1.8, 5.0, float(st.session_state.volt), disabled=True)
    else:
        # Manual slide adjustments
        st.session_state.vwc = st.slider("Volumetric Water Content (VWC %)", 0.0, 100.0, float(st.session_state.vwc))
        st.session_state.temp = st.slider("Soil Temperature (°C)", -20.0, 60.0, float(st.session_state.temp))
        st.session_state.perm = st.slider("Dielectric Permittivity", 1.0, 80.0, float(st.session_state.perm))
        st.session_state.volt = st.slider("Bus Voltage (V)", 1.8, 5.0, float(st.session_state.volt))

    st.write("---")
    # Small metadata footer
    st.markdown(
        "<div style='font-size: 0.75rem; color: #475569; text-align: center;'>"
        "Intel Next-Gen AI Workforce Hackathon<br>"
        "Orchestration Framework Pitch v2.0"
        "</div>",
        unsafe_allow_html=True
    )

# Gather sensor dict for compiling
current_sensors = {
    'vwc': st.session_state.vwc,
    'temp': st.session_state.temp,
    'perm': st.session_state.perm,
    'volt': st.session_state.volt
}

# Run the compilation engine
compiler_data = EdgeCompilerEngine.compile(target_hardware, sparse_engine_active, current_sensors)


# =====================================================================
# Main Dashboard UI Layout
# =====================================================================
# Header Section
st.markdown("<h1 class='main-title'>EdgeOrchestrate AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Cross-Platform Neuromorphic Middleware Engine & Compiler</div>", unsafe_allow_html=True)

# Compiler Banner Status
if compiler_data["status"] == "SUCCESS":
    st.markdown(
        f"<div class='banner-success'>🟢 STATUS: COMPILE SUCCESSFUL. Model graph validated and routed to "
        f"{target_hardware.split(' ')[0]} target. Active profile: {compiler_data['precision']}</div>",
        unsafe_allow_html=True
    )
elif compiler_data["status"] == "WARNING":
    st.markdown(
        f"<div class='banner-warning'>🟡 STATUS: COMPILE SUCCESS WITH WARNINGS. Model deployed on "
        f"{target_hardware.split(' ')[0]} in dense format. Optimization bypass warnings active. Check logs.</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"<div class='banner-danger'>🔴 STATUS: CRITICAL ERROR. Out of Memory (OOM) Stack Overflow. Target device halted. "
        f"Activate 'Sparse-Update Engine' on sidebar to compression-route.</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------
# Row 1: Live Hardware HUD / Telemetry Matrix
# ---------------------------------------------------------------------
hud_col1, hud_col2, hud_col3, hud_col4 = st.columns(4)

with hud_col1:
    mem_color = "text-success" if compiler_data["status"] == "SUCCESS" else ("text-warning" if compiler_data["status"] == "WARNING" else "text-danger")
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Memory Footprint</div>
        <div class="hud-value">{compiler_data['memory_footprint_kb']} KB</div>
        <div class="hud-sub {mem_color}">Limit: {compiler_data['sram_limit_kb']} KB Target SRAM</div>
    </div>
    """, unsafe_allow_html=True)

with hud_col2:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Precision Pipeline</div>
        <div class="hud-value" style="font-size: 1.3rem; line-height: 2.1rem;">{compiler_data['precision']}</div>
        <div class="hud-sub text-cyan">Quantized Latent Replay</div>
    </div>
    """, unsafe_allow_html=True)

with hud_col3:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Active Layers</div>
        <div class="hud-value">{compiler_data['active_layers']}</div>
        <div class="hud-sub text-cyan">Pruning Strategy: {compiler_data['sparsity']}</div>
    </div>
    """, unsafe_allow_html=True)

with hud_col4:
    speedup_class = "text-cyan" if "Intel" in target_hardware else "text-warning"
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Hardware Acceleration</div>
        <div class="hud-value">{compiler_data['speedup']}</div>
        <div class="hud-sub {speedup_class}">Power Draw: {compiler_data['energy_mw']:.1f} mW</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Row 2: Live Diagnostic Output Logs & Target Hardware Code
# ---------------------------------------------------------------------
row2_col1, row2_col2 = st.columns([3, 2])

with row2_col1:
    st.markdown("### 🖥️ COMPILE INTERFACE TERMINAL LOG")
    
    # Styled Terminal window wrapper
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-dots">
            <span class="dot dot-red"></span>
            <span class="dot dot-yellow"></span>
            <span class="dot dot-green"></span>
        </div>
        <div class="terminal-title">edgeorchestrate-compiler --verbose</div>
    </div>
    """, unsafe_allow_html=True)
    st.code(compiler_data["logs"], language="bash")

with row2_col2:
    st.markdown("### 🔩 EMITTED HARDWARE TARGET REPRESENTATION")
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-dots">
            <span class="dot dot-red"></span>
            <span class="dot dot-yellow"></span>
            <span class="dot dot-green"></span>
        </div>
        <div class="terminal-title">emitted_binary_ir.h / net.xml</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Set syntax highlighting based on code block content
    code_lang = "xml" if "Intel" in target_hardware else "cpp"
    st.code(compiler_data["emitted_code"], language=code_lang)

# ---------------------------------------------------------------------
# Row 3: Plots & Visual Analytics
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 INTERACTIVE DIAGNOSTICS & HARDWARE BENCHMARKS")

row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    # Memory Demand Bar Chart
    mem_target_limit = compiler_data["sram_limit_kb"]
    mem_allocated = compiler_data["memory_footprint_kb"]
    
    fig_mem = go.Figure()
    
    # Add target hardware limit bar
    fig_mem.add_trace(go.Bar(
        x=["Hardware SRAM Limit", "Model Footprint"],
        y=[mem_target_limit, mem_allocated],
        marker_color=["rgba(56, 189, 248, 0.6)", "rgba(244, 63, 94, 0.85)" if compiler_data["status"] == "CRASHED" else "rgba(16, 185, 129, 0.85)"],
        text=[f"{mem_target_limit} KB", f"{mem_allocated} KB"],
        textposition='auto',
        width=[0.5, 0.5]
    ))
    
    # Custom layout
    fig_mem.update_layout(
        title={
            'text': "Memory Footprint Demands vs. Silicon Limits",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title="Memory Size (KB)",
            gridcolor="#1e293b",
            zerolinecolor="#1e293b",
        ),
        xaxis=dict(
            gridcolor="#1e293b"
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=320,
        showlegend=False
    )
    
    # Add strict horizontal red line representing the 256KB Microcontroller SRAM limit
    if target_hardware == "Ultra-Low-Power MCU (STM32/ESP32)":
        fig_mem.add_hline(
            y=256,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Strict 256KB SRAM Barrier Wall",
            annotation_position="top right",
            annotation_font=dict(color="#f87171", size=10)
        )
        
    st.plotly_chart(fig_mem, use_container_width=True)

with row3_col2:
    # Execution Timeline Simulation
    fig_time = go.Figure()
    
    iterations = st.session_state.history["Iteration"]
    
    # Simulating iteration compute time based on state
    if "Intel" in target_hardware:
        # High speedup, Intel openVINO speedup
        sparse_time = [random.uniform(0.7, 0.9) for _ in range(len(iterations))]
        dense_time = [random.uniform(11.5, 13.0) for _ in range(len(iterations))]
    elif "Pi" in target_hardware:
        sparse_time = [random.uniform(12.0, 15.0) for _ in range(len(iterations))]
        dense_time = [random.uniform(40.0, 45.0) for _ in range(len(iterations))]
    else:  # MCU
        if not sparse_engine_active:
            # Crashed, no data
            sparse_time = []
            dense_time = []
        else:
            sparse_time = [random.uniform(32.0, 38.0) for _ in range(len(iterations))]
            dense_time = [random.uniform(280.0, 310.0) for _ in range(len(iterations))] # Standard float update time estimation
            
    if compiler_data["status"] == "CRASHED":
        # Display simple crash notice placeholder on chart
        fig_time.add_annotation(
            text="CORE CRASHED (OOM)<br>No execution data generated.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#f87171", size=18, family="Outfit")
        )
    else:
        # Add Standard compute times
        fig_time.add_trace(go.Scatter(
            x=iterations,
            y=dense_time,
            mode='lines+markers',
            name='Standard Dense Training Iteration',
            line=dict(color='#f43f5e', width=2, dash='dot'),
            marker=dict(size=6)
        ))
        # Add EdgeOrchestrate optimized times
        fig_time.add_trace(go.Scatter(
            x=iterations,
            y=sparse_time,
            mode='lines+markers',
            name='EdgeOrchestrate Neuromorphic Sparse',
            line=dict(color='#00f2fe', width=3),
            marker=dict(size=7)
        ))
        
    fig_time.update_layout(
        title={
            'text': "Execution Latency Over Iterations (ms)",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title="Computation Latency (ms)",
            gridcolor="#1e293b",
            zerolinecolor="#1e293b",
        ),
        xaxis=dict(
            title="Iteration Step",
            gridcolor="#1e293b"
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=320,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9)
        )
    )
    
    st.plotly_chart(fig_time, use_container_width=True)

# ---------------------------------------------------------------------
# Row 4: Technology Explainer Tabs
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("### 💡 CORE TECHNOLOGICAL MATRIX")

tab1, tab2 = st.tabs(["🧬 Neuromorphic Sparse Updates (DSD-NA)", "⚡ Intel NPU Acceleration & OpenVINO"])

with tab1:
    st.markdown("""
    #### Dynamic Sparse-Update Neuromorphic Architecture (DSD-NA)
    The **DSD-NA** middleware pipeline makes on-device training feasible under strict hardware limitations (e.g., STM32 MCU with 256KB SRAM) where traditional AI engines crash with OOM errors:
    
    * **Magnitude-Based Dynamic Sparsity**: Instead of executing complete backward weight matrix iterations, the runtime analyzes activation salience on-the-fly, bypassing up to 90% of gradients that don't pass the threshold.
    * **4-Bit Quantized Latent Replay**: Reconstructs previous state activations from a 4-bit compressed memory ring-buffer during weight adjustments, cutting activation memory storage requirements from several megabytes down to a mere 32KB.
    * **Hardware Native Translation**: Outputs highly optimized C++ code invoking specialized assembly instructions (ARM CMSIS-NN / ESP-NN) to utilize DSP instructions directly on-chip.
    """)

with tab2:
    st.markdown("""
    #### Auto-Scaling and OpenVINO Compilation
    When deployment targets with advanced silicon capabilities (like the **Intel Core Ultra NPU or GPU**) are detected by the middleware, it scales accuracy and throughput automatically:
    
    * **Automatic Layer Unpruning**: Bypasses the 90% sparsity target and 4-bit loss quantization boundaries, enabling dense, high-fidelity FP32 training paths to leverage maximum accuracy.
    * **OpenVINO Compiler Pipeline**: Cross-compiles the system configuration directly into OpenVINO XML/BIN Intermediate Representation (IR) graphs.
    * **Intel NPU Optimization**: Harnesses dedicated neural compute blocks on Core Ultra architectures, achieving up to a **14.2x compute speedup** while minimizing CPU thermal load and system power draw.
    """)

# Live stream page rerun block
if live_sim_on:
    time.sleep(0.5)
    st.rerun()
