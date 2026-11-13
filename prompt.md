# TinyML Vibration Predictive Maintenance — Project Plan + Agent Prompt

Scope locked in: **ESP32 only**, **one application** (vibration-based predictive
maintenance) built fully end-to-end, **full auto-codegen** pipeline, **real**
Streamlit dashboard bridged to the device over **USB serial**. Sensor: MPU6050
or ADXL345 IMU.

---

## Part 1 — Step-by-step procedure (for you, human-readable)

### Phase 0 — Environment & repo prep
1. Confirm your ESP-IDF install matches what `iteration_1`'s
   `sdkconfig.defaults.esp32` / `dependencies.lock` expect.
2. From the repo, cut a new branch off `iteration_1`, e.g. `vibration-pdm`.
3. Gather hardware: ESP32 dev board, MPU6050 (or ADXL345) breakout, jumper
   wires, breadboard, USB cable, and — if you want a realistic "wear"
   demonstration — something that vibrates at two distinguishable states
   (e.g. a small desk fan/motor you can unbalance, or just controlled taps/shakes).

### Phase 1 — Sensor bring-up (you do the physical wiring)
4. Agent writes an I2C driver + a minimal "sensor self-test" firmware that
   just prints raw accelerometer readings over serial.
5. **You** wire the IMU to the ESP32 (exact pins specified by the agent in
   Phase 1 output) and flash the self-test firmware yourself.
6. **You** confirm in the serial monitor that readings look sane (values
   change when you move the board, settle when still).

### Phase 2 — Model spec & codegen pipeline (pure software, host-side)
7. Agent defines a JSON model-spec schema (window length, conv channel
   counts, latent dim, number of classes, replay buffer capacity, sparsity
   ratio, adaptation mode, learning rate).
8. Agent writes a small Python training script that pretrains the frozen
   convolutional feature extractor (on synthetic or your own captured
   vibration data) and exports its weights.
9. Agent writes a memory planner that sizes every static buffer from the
   spec and checks the total against your ESP32's usable SRAM budget.
10. Agent writes the code generator that turns (spec + trained weights)
    into `model_config.h` + `weights.h` — the actual "compiler" step from
    the abstract, now real instead of mocked.
11. Agent validates generated code against a desktop C simulation (an
    evolution of the existing `main.c`) before it ever touches hardware.

### Phase 3 — On-device runtime integration
12. Agent refactors the existing continual-learning C logic (frozen feature
    extractor, sparse last-layer updates, 4-bit quantized latent replay)
    into a reusable static library driven by the generated headers.
13. Agent wires that library to real IMU sampling instead of the old ADC
    soil-moisture code path.
14. Agent implements the on-device serial protocol (telemetry out,
    label/command in).
15. **You** build and flash this firmware yourself, then confirm inference +
    adaptation logs look correct over serial.

### Phase 4 — Real dashboard
16. Agent replaces the mocked `EdgeCompilerEngine` in `app.py` with: a real
    build trigger that reports actual binary size, and a live serial bridge
    (pyserial) that streams real telemetry into the existing HUD/chart
    layout.
17. Agent adds a labeling control so you can mark the current window
    "Normal" / "Anomalous" from the dashboard, which drives the on-device
    adaptation step.
18. **You** run the dashboard against the real, flashed board and confirm
    live data shows up.

### Phase 5 — Validation & demo script
19. Agent defines a repeatable "induce drift, then adapt" demo procedure
    (e.g., change the vibration source partway through a session) and
    captures before/after accuracy, latency, and memory numbers.
20. Use these real numbers to replace the illustrative ones in your review
    slides.

### Phase 6 — Documentation
21. Agent writes README, wiring diagram (text/ASCII), architecture doc, and
    a "how to run this end to end" guide.

---

## Part 2 — Master prompt for Claude Code / Antigravity

Copy everything in the fenced block below into your agent session.

```
PROJECT: TinyML Vibration-Based Predictive Maintenance on ESP32

CONTEXT
I have an existing repo (arunim123/TinyML) with three branches:
- main / intel_webapp: a Streamlit demo dashboard (app.py) that currently
  MOCKS a compiler backend (hardcoded logs/output), plus a standalone
  desktop C simulation (main.c) implementing a continual-learning algorithm:
  a frozen 1D-conv feature extractor + a trainable linear classifier,
  magnitude-based sparse gradient updates (touch only the top ~5-10% of
  gradients by magnitude), and a 4-bit quantized latent replay buffer
  (reservoir sampling) used to resist catastrophic forgetting.
- iteration_1: a real ESP-IDF project that ports this same algorithm onto
  real ESP32 hardware — static memory only (no malloc), FreeRTOS tasks,
  and (currently) reads a soil-moisture-style ADC sensor as its input.

GOAL
Build ONE complete, working, end-to-end system: vibration-based predictive
maintenance on a real ESP32 board with an IMU (MPU6050 or ADXL345 — confirm
which registers/address to target; default to MPU6050 over I2C unless told
otherwise, but isolate the sensor behind a small imu_driver.h/.c interface so
swapping is a one-file change). Reuse and refactor the existing algorithm —
do not throw it away and start over. Do NOT build STM32 support, and do NOT
build the keyword-spotting or gesture-recognition demos — those are
explicitly out of scope right now.

SCOPE — build all of the following:

1. Model spec + codegen pipeline (host-side, Python)
   - A JSON schema describing: window length, conv1/conv2 channel counts,
     kernel size, pooling, latent dim, number of classes, replay buffer
     capacity, base sparsity ratio, adaptation mode (last-layer sparse
     update, and — as a stretch goal only if time allows — prototype-based
     learning), learning rate, quantization bit-width.
   - A training script that pretrains the frozen feature extractor on
     synthetic vibration signals (two or more distinguishable "health"
     states) if I don't have a real labeled dataset yet, and exports its
     weights as numpy arrays.
   - A memory planner that computes the static SRAM footprint (weights +
     activations + replay buffer + stack headroom) from the spec and flags
     it clearly if it would exceed a configurable SRAM budget (default
     assume ~200 KB usable on classic ESP32 after WiFi/BT stack overhead —
     make this a named constant I can adjust).
   - A code generator that emits `model_config.h` and `weights.h` from
     (spec + trained weights). This is the real "compiler" step — treat it
     as the centerpiece deliverable, not a mock.
   - A regression test that runs the generated config through a desktop C
     simulation (extend the existing main.c-style harness) and confirms
     the numbers are sane BEFORE anything touches hardware.

2. On-device firmware (ESP-IDF, C, targets classic ESP32 unless I say
   otherwise)
   - Refactor the existing algorithm (frozen feature extractor, sparse
     classifier updater, 4-bit quantized reservoir replay buffer) out of
     iteration_1/main/main.c into a small reusable static library that is
     parameterized entirely by the generated model_config.h / weights.h —
     no hardcoded dimensions in the library itself.
   - Write an I2C IMU driver that samples acceleration magnitude at a fixed
     rate into the window buffer the feature extractor expects.
   - FreeRTOS task structure: sampling task -> inference task -> adaptation
     task -> serial I/O task, all static allocation, no heap.
   - A serial protocol over USB (pick a simple, documented framing — e.g.
     newline-delimited JSON is fine, doesn't need to be binary) that:
     (a) streams telemetry out: raw window summary stats, latest
         prediction + confidence, memory footprint, per-inference latency,
         adaptation events (sparsity ratio used, replay buffer occupancy)
     (b) accepts commands in: a label correction for the current/most
         recent window ("normal" / "anomalous"), a request to force an
         adaptation step, a request for a full telemetry snapshot.
   - A minimal standalone "sensor self-test" build (Phase 1) that just
     prints raw IMU readings, separate from the full pipeline, so I can
     verify wiring before the full firmware exists.

3. Real dashboard (Streamlit, app.py)
   - Replace the mocked EdgeCompilerEngine with a real build step: invoke
     `idf.py build` (or accept a pre-built .bin path if the sandbox can't
     run the ESP-IDF toolchain) and report the ACTUAL resulting binary /
     static-allocation size from the generated model_config.h, not a
     hardcoded number.
   - Build a serial bridge module using pyserial that opens a configurable
     COM/tty port, parses the telemetry protocol from firmware item 2, and
     feeds the existing HUD cards / charts with live data instead of the
     random-walk simulation currently there.
   - Add a labeling control (buttons: "Normal" / "Anomalous") that sends a
     label-correction command back over serial, so I can drive supervised
     continual-learning adaptation live from the dashboard.
   - Keep the existing dark HUD/terminal-log visual style; just make the
     data underneath it real.

4. Validation & docs
   - A written "induce drift, then watch it adapt" demo script: how to
     physically change the vibration source mid-session and what I should
     see change in the dashboard (prediction flipping, then re-stabilizing
     after a few adaptation cycles).
   - README covering: repo layout, how to run the codegen step, how to
     build the firmware, and how to run the dashboard against a real board.
   - A short architecture note describing how the pieces fit together
     (spec -> codegen -> firmware / spec -> dashboard's memory display).

HARD CONSTRAINTS
- No malloc / heap allocation anywhere in the on-device C code — static
  arenas only, sized by the memory planner.
- Everything on-device must fit the SRAM budget the memory planner checks
  against; if a proposed architecture doesn't fit, shrink the spec and say so
  rather than silently exceeding the budget.
- Don't touch STM32 tooling, the other two demo apps, or unrelated parts of
  the repo.
- Work in small, reviewable increments that roughly follow phases 1-6 below.
  Commit (or at least clearly checkpoint) after each phase.

HARDWARE BOUNDARY — READ CAREFULLY
I will physically wire the IMU and flash every build myself. You must NOT:
- attempt to flash a device,
- attempt to open a real serial port to a physical board,
- assume any wiring has been done,
- invent sensor readings and present them as if they came from real
  hardware.
Instead, whenever a step requires physical hardware interaction (wiring,
flashing, plugging in a USB cable, choosing a COM port), STOP, clearly list
the exact steps and commands I need to run myself (pin numbers, `idf.py`
commands, expected serial output), and wait for me to confirm it worked
before moving on to code that depends on that hardware state.

SUGGESTED PHASE ORDER (match Part 1 of the plan I already have)
Phase 0: repo/branch setup.
Phase 1: I2C driver + sensor self-test firmware only -> hand off to me to
  wire + flash + confirm readings.
Phase 2: model spec, training script, memory planner, code generator,
  desktop validation harness — all pure software, no hardware needed.
Phase 3: refactor algorithm into reusable library, wire to real IMU driver,
  serial protocol -> hand off to me to build + flash + confirm.
Phase 4: real dashboard + serial bridge + labeling control -> hand off to me
  to run against the real board.
Phase 5: drift/adaptation demo script + captured real numbers.
Phase 6: README + architecture doc.

Start with Phase 0 and Phase 1. Ask me anything you need about my exact
IMU part, ESP-IDF version, or SRAM budget before writing code, rather than
guessing silently.
```

---

### Notes on how to use this

- Paste the fenced block as-is into Claude Code or Antigravity as your
  opening message — it's self-contained.
- If the agent asks which exact IMU you have (MPU6050 vs ADXL345), have the
  part number or breakout board label ready — I2C address and register map
  differ between them.
- The hardware-boundary section is load-bearing: it's what stops the agent
  from trying to flash a board it can't see or fabricating "live" sensor
  data. Keep it in even if you trim other parts of the prompt.