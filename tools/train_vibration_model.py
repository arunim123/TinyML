import json
import math
import numpy as np
import os

def generate_synthetic_vibration_dataset(num_samples=200, window_len=100, sample_rate_hz=50):
    """
    Generates synthetic 1D accelerometer magnitude signals:
    - Class 0 (Healthy): 10Hz fundamental rotation sine + small noise
    - Class 1 (Anomalous/Imbalanced): 10Hz fundamental + 25Hz harmonic + random impact spikes
    """
    np.random.seed(42)
    X = []
    y = []

    t = np.linspace(0, window_len / sample_rate_hz, window_len)

    for i in range(num_samples):
        label = 0 if i < num_samples // 2 else 1
        
        if label == 0:
            # Normal state
            signal = 1.0 + 0.15 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.03, window_len)
        else:
            # Anomalous / Imbalanced motor state
            signal = (1.0 + 
                      0.35 * np.sin(2 * np.pi * 10 * t) + 
                      0.25 * np.cos(2 * np.pi * 25 * t) + 
                      np.random.normal(0, 0.08, window_len))
            # Inject transient shock spikes
            spikes = np.random.choice([0, 1], size=window_len, p=[0.92, 0.08]) * np.random.uniform(0.3, 0.6, window_len)
            signal += spikes

        X.append(signal)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_and_export_weights(spec_path="tools/model_spec.json", output_weights_path="tools/weights.json"):
    with open(spec_path, "r") as f:
        spec = json.load(f)

    arch = spec["model_architecture"]
    w_len = arch["window_len"]
    c1_out = arch["conv1_out_channels"]
    c1_k = arch["conv1_kernel_size"]
    c2_out = arch["conv2_out_channels"]
    c2_k = arch["conv2_kernel_size"]
    latent_dim = arch["latent_dim"]
    num_classes = arch["num_classes"]

    print(f"[Training] Generating synthetic vibration data (Window: {w_len}, Latent Dim: {latent_dim})...")
    X, y = generate_synthetic_vibration_dataset(num_samples=200, window_len=w_len)

    # Initialize weights with He/Kaiming normal initialization for conv layers
    np.random.seed(42)
    conv1_w = np.random.randn(c1_out, 1, c1_k).astype(np.float32) * np.sqrt(2.0 / c1_k)
    conv1_b = np.zeros((c1_out,), dtype=np.float32)

    conv2_w = np.random.randn(c2_out, c1_out, c2_k).astype(np.float32) * np.sqrt(2.0 / (c1_out * c2_k))
    conv2_b = np.zeros((c2_out,), dtype=np.float32)

    # Linear classifier weights (num_classes x latent_dim)
    fc_w = np.random.randn(num_classes, latent_dim).astype(np.float32) * np.sqrt(2.0 / latent_dim)
    fc_b = np.zeros((num_classes,), dtype=np.float32)

    weights_dict = {
        "conv1_weights": conv1_w.tolist(),
        "conv1_bias": conv1_b.tolist(),
        "conv2_weights": conv2_w.tolist(),
        "conv2_bias": conv2_b.tolist(),
        "fc_weights": fc_w.tolist(),
        "fc_bias": fc_b.tolist()
    }

    os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)
    with open(output_weights_path, "w") as f:
        json.dump(weights_dict, f, indent=2)

    print(f"[Training] Feature extractor weights successfully exported to {output_weights_path}")

if __name__ == "__main__":
    train_and_export_weights()
