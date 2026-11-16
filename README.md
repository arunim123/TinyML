# TinyML Adaptive Edge Framework

On-device continual learning for ESP32-class microcontrollers, applied to **vibration-based predictive maintenance**. A tiny CNN feature extractor stays frozen; a small classifier head keeps adapting on the chip while the machine's vibration behaviour drifts, using **sparse updates** and a **4-bit quantized latent replay buffer** so it does not forget what it already knew.

The whole runtime is static-allocation C (zero heap allocations), runs as FreeRTOS tasks on the ESP32, and streams live telemetry to a Streamlit dashboard over USB serial.

> **Project history.** This repository started as a small C demo of continual learning on synthetic signals (kept in [`legacy/`](legacy/continual_learning_demo.c)), then grew into the ESP32 firmware, the host tooling and the dashboard. This README describes the merged state of all those lines of work.

---

## What problem it solves

A model trained once on a healthy machine degrades when conditions change: the sensor ages, the mount loosens, the temperature or load shifts (**concept drift**). Retraining in the cloud needs data transfer and connectivity. Fine-tuning naively on the device causes **catastrophic forgetting**, where the model learns the new condition and loses the old one.

This project keeps adaptation on the device and cheap enough for a microcontroller:

| Idea | How it is done here |
|---|---|
| Cheap training | Feature extractor is **frozen**; only the linear head is trained, so no conv gradients or activations are stored for training. |
| Fewer updates | **Sparse head updates**: only the largest-magnitude gradients (about 5 percent) are applied. Four selection rules are implemented and switchable. |
| No forgetting | **Latent replay**: old latent vectors (not raw signals) are stored, **4-bit quantized**, in a small reservoir and replayed during adaptation. |
| Fits the chip | All memory is statically allocated and accounted for; the reported footprint includes gradient, selection and dequantization scratch. |

---

## System architecture

```text
+-----------------------+     +-----------------------+     +-----------------------+
|  Host model spec &    | --> | Host Python codegen   | --> | Auto-generated C      |
|  training             |     | & memory planner      |     | model_config.h        |
| (tools/model_spec.json|     | (tools/codegen.py)    |     | weights.h             |
|  train_vibration_...) |     | (memory_planner.py)   |     |                       |
+-----------------------+     +-----------------------+     +-----------+-----------+
                                                                        |
                                                                        v
+-----------------------+     +-----------------------+     +-----------+-----------+
| Streamlit dashboard   | <-- | USB serial bridge     | <-- | ESP-IDF C runtime     |
| & adaptation control  |     | JSON telemetry        |     | static FreeRTOS tasks |
| (app.py)              |     | (serial_bridge.py)    |     | (main/main.c,         |
|                       |     |                       |     |  main/tiny_cl.c)      |
+-----------------------+     +-----------------------+     +-----------------------+
```

### On-device pipeline

```text
IMU (MPU6050 / ADXL345, I2C, 50 Hz)
   -> window of 100 samples
   -> per-window standardisation (zero mean, unit variance)
   -> frozen Conv1D(1->8, k5) + pool -> Conv1D(8->16, k5) + pool
   -> latent vector (400 values)
   -> linear classifier head (2 classes: Normal / Anomaly)   <-- the only part that learns
        ^                                   |
        |  sparse update (top ~5% gradients)|
        +--- 4-bit quantized latent replay reservoir (50 samples)
```

### FreeRTOS tasks (all statically created)

| Task | Priority | Role |
|---|---|---|
| `imu_sampling` | 5 | Reads the IMU at 50 Hz into the window buffer |
| `inference` | 4 | Runs feature extraction and prediction at 10 Hz, publishes latency and confidence |
| `adaptation` | 3 | On a label command: stores the latent in the reservoir, applies the sparse head update, then a replay step |
| `serial_io` | 2 | Emits JSON telemetry and parses label commands |

---

## Key implementation details

- **Per-window standardisation** (`tiny_cl_standardise_window`): each window is normalised before the conv layers (population std, epsilon 1e-8). Host-trained weights must be trained with the same normalisation to transfer.
- **Four head-update rules** (`TinyCLUpdateRule`, chosen at `tiny_cl_init`):
  - `TINY_CL_UPDATE_DENSE`: update every weight.
  - `TINY_CL_UPDATE_SORT`: top-k by a full comparison sort (baseline sparse rule, the firmware default).
  - `TINY_CL_UPDATE_INTROSELECT`: top-k via quickselect partition, expected O(n).
  - `TINY_CL_UPDATE_THRESHOLD`: running-threshold heuristic, single pass, no selection step.
- **Quantized latent replay**: min-max 4-bit quantization of latent vectors (values 0 to 15 plus min and max per vector), dequantized when replayed. Replay can be toggled with `tiny_cl_set_replay_enabled`.
- **Honest memory accounting**: `tiny_cl_get_stats` reports head, reservoir, activation, gradient and dequantization storage separately, plus flash (frozen weights) as a separate figure. An earlier revision omitted the gradient, selection and dequantization scratch and understated the true static footprint by about 2.5x.
- **Static allocation**: no `malloc` in the runtime.

### Static memory (desktop simulation of the merged core)

| Component | Bytes |
|---|---|
| Head parameters + adaptation state | 3,228 |
| Replay reservoir (50 samples, 4-bit) | 10,472 |
| Extractor activation scratch | 10,000 |
| Gradient accumulator + selection scratch | 9,608 |
| Dequantization staging | 1,600 |
| **Total engine SRAM** | **34,908 (about 34.1 KB)** |
| Frozen weights (flash, `.rodata`, not SRAM) | 6,024 |

FreeRTOS task stacks (about 14 KB, from the earlier README) are in addition to the engine figure. Target budget in `tools/model_spec.json` is 200 KB SRAM.

---

## Repository layout

```text
main/                    ESP-IDF firmware component
  main.c                   FreeRTOS tasks, telemetry, label commands
  tiny_cl.c / tiny_cl.h    Continual-learning engine (extractor, head, sparse update, replay, stats)
  imu_driver.c / .h        MPU6050 / ADXL345 I2C driver
  self_test_main.c         IMU wiring self-test entry point
  model_config.h           Generated model constants
  weights.h                Frozen extractor + head weights (see caveat below)
tools/                   Host tooling
  model_spec.json          Architecture and adaptation parameters
  train_vibration_model.py Host training of the feature extractor
  memory_planner.py        Static SRAM budget check
  codegen.py               Generates model_config.h and weights.h from model_spec + weights.json
  desktop_sim.c            Host C simulation harness (no hardware needed)
app.py                   Streamlit dashboard (live telemetry, console, label buttons)
serial_bridge.py         USB serial bridge between the ESP32 and the dashboard
docs/vibration_demo_script.md   Step-by-step drift and adaptation demo
legacy/continual_learning_demo.c  Original single-file C demo on synthetic PPG-like signals
```

---

## Getting started

### 1. Desktop simulation (no hardware)

```bash
gcc -O2 -std=c11 -Wall tools/desktop_sim.c -lm -o tools/desktop_sim
./tools/desktop_sim
```

Prints the memory breakdown, runs 100 streaming inference and adaptation cycles on synthetic windows and confirms there are no heap allocations. The accuracy it prints is on the simulator's own generated data and is a sanity check, not a benchmark.

### 2. Hardware wiring

```text
ESP32                    MPU6050 / ADXL345
3.3V          ------->   VCC
GND           ------->   GND
GPIO 21 (SDA) ------->   SDA
GPIO 22 (SCL) ------->   SCL
```

### 3. IMU self-test

Set `SRCS` in `main/CMakeLists.txt` to `"self_test_main.c" "imu_driver.c"`, then:

```bash
idf.py build
idf.py -p COMx flash monitor
```

### 4. Full firmware

Register `"main.c" "imu_driver.c" "tiny_cl.c"` in `main/CMakeLists.txt`, then:

```bash
idf.py build
idf.py -p COMx flash monitor
```

### 5. Dashboard

```bash
streamlit run app.py
```

Select the ESP32 serial port in the sidebar and connect. Follow [`docs/vibration_demo_script.md`](docs/vibration_demo_script.md) to induce drift and trigger on-device adaptation.

### Serial protocol

The firmware prints one JSON line per update at 115200 baud, for example:

```json
{"type":"telemetry","count":120,"pred":1,"conf":0.973,"mag":1.21,"sparsity":0.05,"buffer":50,"static_ram_kb":33.6,"latency_ms":1.8}
```

Commands accepted on the same port: `{"cmd":"label","val":0|1}` or `LABEL:0` / `LABEL:1`.

---

## Research findings

The accompanying study (13-page paper) benchmarked the four update rules over 10 random seeds under a recalibrated, realistic task (about 90 percent nominal accuracy, about 51 percent under drift). Summary of its conclusions:

- On-device adaptation recovered roughly 20 percentage points under drift but stayed below the offline-trained ceiling.
- Naive replay hurt retention in that setup.
- Dense updates beat every sparse rule: conditional writes block vectorization and selection overhead cancels the theoretical savings.

The paper and the benchmark scripts are not part of this repository, so these numbers are reported from that study rather than reproduced here.

---

## Known caveats

- **Weights and the host training script.** The checked-in `main/weights.h` pairs with the per-window-standardised core and includes a trained classifier head. `tools/train_vibration_model.py` and `tools/weights.json` predate that standardisation step; regenerating `weights.h` with `tools/codegen.py` will produce weights that do not match the current firmware until the training script applies the same standardisation.
- **Telemetry RAM field.** `main/main.c` still reports a hardcoded `static_ram_kb` of 33.6 and `app.py` displays that figure; the corrected engine total is about 34.1 KB plus task stacks. These should read from `tiny_cl_get_stats`.
- **Firmware not compiled in CI.** The ESP-IDF build was not run when branches were merged; the desktop simulation and Python syntax were checked.
- **Data.** The desktop simulator and the legacy demo use synthetic signals. The firmware itself has been run on a real ESP32 with a real IMU sensor.

---

## Legacy continual-learning demo

`legacy/continual_learning_demo.c` is the original dependency-free C experiment: a frozen Conv1D extractor and trainable linear classifier on synthetic PPG-like waveforms, comparing standard training against sparse training with 4-bit latent replay.

```bash
gcc -O2 -std=c11 legacy/continual_learning_demo.c -lm -o tinyml_c && ./tinyml_c
```
