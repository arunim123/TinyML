# TinyML Continual Learning Demo

This project is a small, self-contained C implementation of a TinyML-style continual learning experiment.
It simulates a lightweight classifier that is first trained on a baseline dataset and then adapted to a shifted
"evolved" dataset while trying to avoid catastrophic forgetting.

The code is designed to run without external machine learning dependencies. Everything is implemented directly in
C, using only the standard library and `libm` for math functions.

## What the project does

The program generates synthetic 1D sensor-like signals that resemble a simple heart-rate / PPG style waveform.
It creates two versions of the data:

- Baseline data: the original distribution.
- Evolved data: a slightly shifted distribution that simulates concept drift.

It then runs two training pipelines:

1. Standard training
   - Trains the classifier normally with full updates.
   - Adapts the same model to the evolved data without replay.
   - Measures how much performance is retained on the original baseline data.

2. Sparse training + quantized latent replay
   - Trains a second model with sparse gradient updates.
   - Stores old latent activations in a replay buffer using 4-bit-style quantization.
   - Replays quantized old representations while learning the evolved data.
   - Measures baseline retention and evolved-data performance.

At the end, the program prints a metrics report and a success message if the sparse + replay setup maintains
strong accuracy on both datasets.

## Why this exists

This is a learning and demonstration project, not a production ML system.
It is meant to show how several TinyML-inspired ideas fit together:

- Training on a constrained model.
- Handling data drift.
- Reducing update cost with sparse parameter updates.
- Compressing replay memory with low-bit storage.
- Comparing old-task retention against new-task learning.

## Key ideas in the implementation

### Synthetic data generation

The program does not load a real dataset.
Instead, it generates noisy sine-wave sequences with two labels:

- `0`: one waveform pattern.
- `1`: a slightly different waveform pattern.

The evolved dataset adds extra distribution shift so the model has to adapt to changed conditions.

### Model structure

The model is a tiny neural network made of:

- A frozen 1D convolutional feature extractor.
- A trainable linear classifier on top of the extracted latent vector.

The feature extractor is intentionally frozen to mimic a low-power embedded setting where only a small part of the
model is adapted on-device.

### Sparse updates

When sparse mode is enabled, the classifier update only applies to the gradients with the largest magnitudes.
This simulates a low-update regime where only a small percentage of parameters are changed each step.

### Quantized latent replay

Instead of replaying raw input samples, the program stores latent vectors produced by the feature extractor.
These latents are quantized into 4-bit-style values in the range `0..15`, along with min/max values needed to
restore them later.

When replaying, the stored latent values are dequantized back into floats before being passed through the classifier.
This keeps the replay buffer small while preserving the old information needed for continual learning.

## Files

- `main.c` - Complete program source.
- `tinyml_c` - Native binary produced after compilation.

## Build requirements

You only need a C compiler and the math library.

Recommended:

- GCC or Clang
- A Unix-like environment such as Linux or macOS

## Build instructions

From the project directory:

```bash
gcc -O2 -std=c11 main.c -lm -o tinyml_c
```

The `-lm` flag is required because the code uses math functions such as `sinf`, `sqrtf`, `logf`, and `cosf`.

## Run instructions

After building, run:

```bash
./tinyml_c
```

The program will:

1. Generate synthetic baseline and evolved datasets.
2. Train and evaluate the standard pipeline.
3. Train and evaluate the sparse + replay pipeline.
4. Print a final comparison report.

## Example output

The exact numbers will vary a little by random seed and data difficulty, but the output will look roughly like this:

```text
Generating Synthetic Data...

==================================================
Test Pipeline 1: STANDARD TRAINING (100% updates)
==================================================
--- Training on Baseline Data (Standard) ---
Epoch 1/10 | Loss: ... | Acc: ...%
...
Accuracy on Baseline Test Split: ...%

==================================================
Test Pipeline 2: SPARSE TRAINING (~5% updates) + QLR (100 sample capacity)
==================================================
--- Training on Baseline Data (w/ sparsity & buffer fill) ---
...

==================================================
Final Output Metrics Report
==================================================
...
```

## Important implementation notes

- The project does not use PyTorch, NumPy, or any other external ML library.
- The replay buffer stores quantized latent vectors, not the original raw signals.
- Only the classifier is updated during training; the feature extractor remains frozen.
- The dataset is synthetic, so the reported metrics are experiment results, not real-world model performance.
- The program is intentionally simple and meant for demonstration, testing, and learning.

## How the 4-bit replay works

The replay buffer uses a simple min-max quantization scheme:

1. Find the minimum and maximum value in a latent vector.
2. Scale each latent value into an integer from `0` to `15`.
3. Store those quantized values together with the min/max range.
4. When replaying, convert the values back into floating-point numbers.

This is why the project is described as using "4-bit" replay.
It is a compressed representation of old latent data, not a full integer-only model.

## Limitations

This code is a simulation, not a benchmark-grade continual learning system.

- It uses synthetic data.
- It does not train a full end-to-end deep network.
- The sparse update rule is a simplified magnitude-based heuristic.
- The quantization is approximate and meant to demonstrate the idea of compressed replay.

## Suggested next steps

If you want to extend the project, useful improvements would be:

- Add a `Makefile`.
- Add command-line options for seed, epochs, and buffer size.
- Save metrics to a CSV file.
- Add a real sensor dataset.
- Compare more replay strategies.

## License

No license has been added yet. Add one if you plan to share or publish the project.