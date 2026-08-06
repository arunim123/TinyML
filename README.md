# TinyML Vibration Predictive Maintenance on ESP32

An end-to-end TinyML predictive maintenance framework for vibration-based health monitoring on ESP32 microcontrollers. Combines an automated host Python codegen compiler, a static-allocation FreeRTOS C continual-learning runtime, and a real-time Streamlit USB serial telemetry dashboard.

---

## Hardware Pinout & Wiring Diagram

Connect your **MPU6050** (or ADXL345) IMU breakout board to the ESP32 dev board:

```text
+-----------------------+              +-----------------------+
|   ESP32 Dev Board     |              |   MPU6050 / ADXL345   |
|                       |              |      IMU Sensor       |
|                 3.3V  +--------------+ VCC                   |
|                  GND  +--------------+ GND                   |
|        GPIO 21 (SDA)  +--------------+ SDA                   |
|        GPIO 22 (SCL)  +--------------+ SCL                   |
+-----------------------+              +-----------------------+
```

---

## System Architecture

```text
+-----------------------+     +-----------------------+     +-----------------------+
|  Host Model Spec &    | --> | Host Python Codegen   | --> | Auto-Generated C      |
|  Synthetic Training   |     | & Memory Planner      |     | model_config.h        |
| (tools/model_spec.json|     | (tools/codegen.py)    |     | weights.h             |
+-----------------------+     +-----------------------+     +-----------+-----------+
                                                                        |
                                                                        v
+-----------------------+     +-----------------------+     +-----------+-----------+
| Streamlit Dashboard   | <-- | USB Serial Bridge     | <-- | ESP-IDF C Runtime Engine  |
| & Adaptation Control  |     | UART Telemetry        |     | Static FreeRTOS Tasks     |
| (app.py)              |     | (serial_bridge.py)    |     | (main/main.c & tiny_cl.c) |
+-----------------------+     +-----------------------+     +-----------------------+
```

---

## End-to-End Workflow & Execution Guide

### 1. Host Python Codegen Pipeline
Train the feature extractor, check the static SRAM memory budget, and generate target C headers:
```bash
python tools/train_vibration_model.py
python tools/memory_planner.py
python tools/codegen.py
```

### 2. Desktop C Simulation Validation
Validate generated model headers against host C simulation harness before flashing hardware:
```bash
gcc -I. -Imain tools/desktop_sim.c -o tools/desktop_sim.exe -lm
./tools/desktop_sim.exe
```

### 3. ESP32 Sensor Self-Test (Wiring Verification)
Verify physical IMU I2C communication over UART serial:
1. Open `main/CMakeLists.txt` and set SRCS to `"self_test_main.c" "imu_driver.c"`.
2. Build and flash:
```bash
idf.py build
idf.py -p COMx flash monitor
```

### 4. ESP32 Full Firmware Build & Flash
Build and flash the complete continual-learning runtime:
1. Ensure `main/CMakeLists.txt` registers `"main.c" "imu_driver.c" "tiny_cl.c"`.
2. Build and flash:
```bash
idf.py build
idf.py -p COMx flash monitor
```

### 5. Launch Real-Time USB Serial Dashboard
Launch the Streamlit dashboard, select your ESP32 COM port in the sidebar, and connect:
```bash
streamlit run app.py
```

---

## Static Memory & Performance Metrics

- **Target Platform**: ESP32 Dual-Core LX6 (200 KB Usable SRAM Budget)
- **Total Static SRAM Allocation**: **33.60 KB**
  - Model Weights & Biases: **5.88 KB**
  - Activations & Buffers: **3.52 KB**
  - 4-bit Quantized Replay Memory: **10.21 KB** (50 sample capacity)
  - FreeRTOS Task Stacks: **14.00 KB** (Sampling, Inference, Adaptation, Serial tasks)
- **Heap Memory Allocations (`malloc`)**: **0 Bytes (100% Static)**
- **Inference Latency**: **~1.8 ms** per 100-sample window