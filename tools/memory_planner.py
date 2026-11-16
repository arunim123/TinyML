import json
import sys

def calculate_sram_footprint(spec_path="tools/model_spec.json", verbose=False):
    with open(spec_path, "r") as f:
        spec = json.load(f)

    arch = spec["model_architecture"]
    adapt = spec["adaptation_parameters"]
    budget_kb = spec.get("sram_budget_kb", 200)
    budget_bytes = budget_kb * 1024

    w_len = arch["window_len"]
    c1_in = arch["conv1_in_channels"]
    c1_out = arch["conv1_out_channels"]
    c1_k = arch["conv1_kernel_size"]
    c2_out = arch["conv2_out_channels"]
    c2_k = arch["conv2_kernel_size"]
    latent_dim = arch["latent_dim"]
    num_classes = arch["num_classes"]
    q_bits = arch["quantization_bits"]

    capacity = adapt["replay_buffer_capacity"]

    # 1. Weights and Biases (float32 = 4 bytes)
    conv1_weights_bytes = (c1_out * c1_in * c1_k + c1_out) * 4
    conv2_weights_bytes = (c2_out * c1_out * c2_k + c2_out) * 4
    fc_weights_bytes = (num_classes * latent_dim + num_classes) * 4
    total_weights_bytes = conv1_weights_bytes + conv2_weights_bytes + fc_weights_bytes

    # 2. Raw Input & Activation Buffers
    raw_window_bytes = w_len * 4
    pool1_len = w_len // 2
    pool2_len = pool1_len // 2
    conv1_act_bytes = (c1_out * pool1_len) * 4
    conv2_act_bytes = (c2_out * pool2_len) * 4  # = latent_dim * 4
    total_activations_bytes = raw_window_bytes + conv1_act_bytes + conv2_act_bytes

    # 3. Latent Replay Buffer (4-bit quantized)
    if q_bits == 4:
        replay_latent_bytes = capacity * (latent_dim // 2)
    elif q_bits == 8:
        replay_latent_bytes = capacity * latent_dim
    else:
        replay_latent_bytes = capacity * latent_dim * 4  # float32

    replay_labels_bytes = capacity * 1
    replay_quant_scales_bytes = capacity * 2 * 4  # min & max scale floats per sample
    total_replay_bytes = replay_latent_bytes + replay_labels_bytes + replay_quant_scales_bytes

    # 4. FreeRTOS Task Stacks (Sampling: 2KB, Inference: 4KB, Adaptation: 4KB, Serial: 4KB)
    task_stacks_bytes = (2 + 4 + 4 + 4) * 1024

    total_static_bytes = total_weights_bytes + total_activations_bytes + total_replay_bytes + task_stacks_bytes
    total_static_kb = total_static_bytes / 1024.0

    if verbose:
        print("=" * 65)
        print("  ESP32 Static SRAM Memory Planner — Vibration PDM Model")
        print("=" * 65)
        print(f" Target Platform SRAM Budget   : {budget_kb:6.2f} KB ({budget_bytes} bytes)")
        print("-" * 65)
        print(f" 1. Weights & Biases           : {total_weights_bytes / 1024.0:6.2f} KB ({total_weights_bytes} B)")
        print(f"    - Conv1 Layer              : {conv1_weights_bytes / 1024.0:6.2f} KB")
        print(f"    - Conv2 Layer              : {conv2_weights_bytes / 1024.0:6.2f} KB")
        print(f"    - Linear FC Layer          : {fc_weights_bytes / 1024.0:6.2f} KB")
        print(f" 2. Activations & Buffers      : {total_activations_bytes / 1024.0:6.2f} KB ({total_activations_bytes} B)")
        print(f" 3. Quantized Replay Buffer ({q_bits}-bit): {total_replay_bytes / 1024.0:6.2f} KB ({total_replay_bytes} B)")
        print(f" 4. FreeRTOS Task Stacks       : {task_stacks_bytes / 1024.0:6.2f} KB ({task_stacks_bytes} B)")
        print("-" * 65)
        print(f" TOTAL STATIC SRAM CONSUMPTION : {total_static_kb:6.2f} KB ({total_static_bytes} bytes)")
        print(f" REMAINING FREE SRAM BUDGET    : {(budget_bytes - total_static_bytes) / 1024.0:6.2f} KB")
        print("=" * 65)

    within_budget = total_static_bytes <= budget_bytes
    if verbose:
        if within_budget:
            print(" [PASSED] Memory specification fits comfortably within SRAM budget!")
        else:
            print(f" [EXCEEDED] Memory footprint exceeds budget by {(total_static_bytes - budget_bytes) / 1024.0:.2f} KB!")

    return within_budget, total_static_bytes

if __name__ == "__main__":
    ok, _ = calculate_sram_footprint(verbose=True)
    if not ok:
        sys.exit(1)

