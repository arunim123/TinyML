import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -----------------
# 1. Synthetic Data Engine
# -----------------
def generate_synthetic_data(num_samples=1000, seq_len=100, is_evolved=False):
    """
    Generate 1D Heart Rate (PPG) signals.
    Baseline: Normal HR patterns.
    Evolved: Lower resting HR, shifted recovery curves.
    """
    # Fix seed for reproducibility
    # np.random.seed(42)  # Optional: removing to allow variability if called multiple times
    
    X = np.zeros((num_samples, 1, seq_len), dtype=np.float32)
    y = np.zeros((num_samples,), dtype=np.int64)
    
    t = np.linspace(0, 10, seq_len)
    for i in range(num_samples):
        # We create a binary classification setup for synthetic patterns
        # 0: "Resting", 1: "Active"
        label = np.random.randint(0, 2)
        
        freq = 1.0 if label == 0 else 2.0
        amp = 1.0 if label == 0 else 1.5
        
        if is_evolved:
            # Concept drift: Evolved heart rate implies lower baseline frequency and amplitude
            freq *= 0.8
            amp *= 0.8
            offset = -0.5  # shifted recovery
        else:
            offset = 0.0
            
        noise = np.random.normal(0, 0.1, seq_len)
        signal = amp * np.sin(2 * np.pi * freq * t) + offset + noise
        X[i, 0, :] = signal
        y[i] = label
        
    return torch.tensor(X), torch.tensor(y)

# -----------------
# 2. Model Architecture
# -----------------
class DynamicSparseModel(nn.Module):
    def __init__(self, seq_len=100, use_sparsity=False):
        super(DynamicSparseModel, self).__init__()
        
        # Frozen Feature Extractor (2-layer 1D-CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Calculate resulting latent dimension after pooling
        self.latent_dim = (seq_len // 4) * 16
        
        # Adaptive Classifier (will undergo on-device training)
        self.classifier = nn.Linear(self.latent_dim, 2)
        
        # -----------------
        # 3. Dynamic Sparsity Engine Configuration
        # -----------------
        self.use_sparsity = use_sparsity
        self.base_sparsity = 0.05  # Target is 5% updates
        self.update_ratio_S = self.base_sparsity
        
        if self.use_sparsity:
            # Register backward hook on the Adaptive Classifier's weights
            self.classifier.weight.register_hook(self._sparse_update_hook)
            if self.classifier.bias is not None:
                self.classifier.bias.register_hook(self._sparse_update_hook)

    def set_loss_magnitude(self, loss_val):
        """Update update_ratio_S adjusting the percentage of updated weights based on loss magnitude."""
        # Simple heuristic: If loss is higher, allow slightly more updates to help convergence
        # If loss is small, push towards 5% sparsity. Let's bound it [0.01, 0.10].
        dynamic_ratio = self.base_sparsity + (loss_val * 0.02)
        self.update_ratio_S = min(max(dynamic_ratio, 0.01), 0.10)

    def _sparse_update_hook(self, grad):
        """Calculate magnitude-based salience threshold to zero out gradients."""
        num_elements = grad.numel()
        k = int(num_elements * self.update_ratio_S)
        
        if k == 0:
            return torch.zeros_like(grad)
        if k >= num_elements:
            return grad
            
        # Find threshold value for the top-K% magnitudes
        abs_grad = torch.abs(grad)
        # Flatten and get the k-th largest value; kthvalue finds the k-th smallest, 
        # so we ask for (N - k + 1)-th smallest element
        threshold = torch.kthvalue(abs_grad.flatten(), num_elements - k + 1).values
        
        # Mask where gradient magnitude is >= threshold
        mask = abs_grad >= threshold
        return grad * mask.float()

    def forward(self, x):
        # x shape: (B, 1, seq_len)
        features = self.feature_extractor(x)
        # Output of penultimate layer (latent vectors)
        latent = features.view(features.size(0), -1)
        out = self.classifier(latent)
        return out, latent


# -----------------
# 4. Quantized Latent Replay (QLR) Simulation
# -----------------
class QuantizedLatentReplayBuffer:
    def __init__(self, capacity=100):
        self.capacity = capacity
        # Storing tuples of (quantized_tensor, min_val, max_val)
        self.buffer_latents = []
        self.buffer_labels = []
        self.samples_seen = 0
        
    def quantize_4bit(self, latents):
        """Simulate 4-bit integer quantization (0-15 range)."""
        # latents shape: (B, D)
        min_val, _ = torch.min(latents, dim=1, keepdim=True)
        max_val, _ = torch.max(latents, dim=1, keepdim=True)
        
        range_val = max_val - min_val
        # Prevent division by zero
        range_val[range_val == 0] = 1e-5
        
        # Scale continuous values to [0, 15] integer range
        quantized = torch.round(15.0 * (latents - min_val) / range_val)
        quantized = torch.clamp(quantized, 0, 15).to(torch.int8)
        
        return quantized, min_val, max_val

    def dequantize_4bit(self, quantized, min_val, max_val):
        """Dequantize 4-bit representation back to float representation."""
        quantized_float = quantized.to(torch.float32)
        range_val = max_val - min_val
        dequantized = (quantized_float / 15.0) * range_val + min_val
        return dequantized

    def add(self, latents, labels):
        """Add to the buffer using Reservoir Sampling."""
        quantized, min_val, max_val = self.quantize_4bit(latents)
        
        batch_size = latents.size(0)
        for i in range(batch_size):
            self.samples_seen += 1
            if len(self.buffer_latents) < self.capacity:
                self.buffer_latents.append((quantized[i], min_val[i], max_val[i]))
                self.buffer_labels.append(labels[i])
            else:
                j = np.random.randint(0, self.samples_seen)
                if j < self.capacity:
                    self.buffer_latents[j] = (quantized[i], min_val[i], max_val[i])
                    self.buffer_labels[j] = labels[i]
                    
    def sample(self, batch_size):
        """Sample and de-quantize from the buffer."""
        if len(self.buffer_latents) == 0:
            return None, None
            
        indices = np.random.choice(
            len(self.buffer_latents), 
            size=min(batch_size, len(self.buffer_latents)), 
            replace=False
        )
        
        sampled_latents = []
        sampled_labels = []
        
        for idx in indices:
            q, min_v, max_v = self.buffer_latents[idx]
            q = q.unsqueeze(0)
            min_v = min_v.unsqueeze(0)
            max_v = max_v.unsqueeze(0)
            
            deq = self.dequantize_4bit(q, min_v, max_v)
            sampled_latents.append(deq)
            sampled_labels.append(self.buffer_labels[idx])
            
        replay_latents = torch.cat(sampled_latents, dim=0)
        replay_labels = torch.stack(sampled_labels, dim=0)
        
        return replay_latents, replay_labels
        

# -----------------
# 5. Validation Suite
# -----------------
def train_model(model, dataloader, epochs=5, lr=0.01, buffer=None, use_replay=False, name=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    print(f"--- Training on {name} ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            out, latents = model(X)
            loss = criterion(out, y)
            
            # Mix with Quantized Latent Replay
            if use_replay and buffer is not None:
                replay_latents, replay_labels = buffer.sample(batch_size=X.size(0))
                if replay_latents is not None:
                    # Pass the replays straight into the classifier
                    replay_out = model.classifier(replay_latents)
                    loss_replay = criterion(replay_out, replay_labels)
                    # We equally weight the current data loss and replay loss
                    loss = 0.5 * loss + 0.5 * loss_replay
                    
            # Dynamically adapt sparsity threshold
            if hasattr(model, 'set_loss_magnitude'):
                model.set_loss_magnitude(loss.item())
                
            loss.backward()
            optimizer.step()
            
            # Populate reservoir when initially training on baseline
            if buffer is not None and not use_replay:
                buffer.add(latents.detach(), y)
            
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {100.*correct/total:.2f}%")

def evaluate_model(model, dataloader, name=""):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            out, _ = model(X)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    acc = 100. * correct / total
    print(f"Accuracy on {name}: {acc:.2f}%")
    return acc

def check_sparsity(model):
    """Verify that magnitude-based sparsity is working on the gradients."""
    grad = model.classifier.weight.grad
    if grad is not None:
        zeros_count = (grad == 0).sum().item()
        total_count = grad.numel()
        print(f"[Check] Sparsity in Classifier Weights: {zeros_count}/{total_count} gradients are exactly zero ({(zeros_count/total_count)*100:.2f}%).")
    else:
        print("[Check] No gradients found.")

# -----------------
# 6. Output Metrics Driver
# -----------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Generating Synthetic Data...")
    X_base_train, y_base_train = generate_synthetic_data(num_samples=1000, seq_len=100, is_evolved=False)
    X_base_test, y_base_test = generate_synthetic_data(num_samples=400, seq_len=100, is_evolved=False)
    
    X_evol_train, y_evol_train = generate_synthetic_data(num_samples=1000, seq_len=100, is_evolved=True)
    X_evol_test, y_evol_test = generate_synthetic_data(num_samples=400, seq_len=100, is_evolved=True)
    
    from torch.utils.data import TensorDataset, DataLoader
    train_base_loader = DataLoader(TensorDataset(X_base_train, y_base_train), batch_size=32, shuffle=True)
    test_base_loader = DataLoader(TensorDataset(X_base_test, y_base_test), batch_size=32, shuffle=False)
    
    train_evol_loader = DataLoader(TensorDataset(X_evol_train, y_evol_train), batch_size=32, shuffle=True)
    test_evol_loader = DataLoader(TensorDataset(X_evol_test, y_evol_test), batch_size=32, shuffle=False)
    
    # --------
    # Pipeline 1: Standard Training
    # --------
    print("\n" + "="*50)
    print("Test Pipeline 1: STANDARD TRAINING (100% updates)")
    print("="*50)
    model_std = DynamicSparseModel(use_sparsity=False)
    
    train_model(model_std, train_base_loader, epochs=10, name="Baseline Data (Standard)")
    acc_base_std = evaluate_model(model_std, test_base_loader, name="Baseline Test Split")
    
    # Continual Learning naive (no replay)
    train_model(model_std, train_evol_loader, epochs=10, name="Evolved Data (Naive CL without Replay)")
    acc_evol_std = evaluate_model(model_std, test_evol_loader, name="Evolved Test Split")
    acc_base_cl_std = evaluate_model(model_std, test_base_loader, name="Baseline Test Split (After Naive CL)")


    # --------
    # Pipeline 2: Sparse Training + QLR
    # --------
    print("\n" + "="*50)
    print("Test Pipeline 2: SPARSE TRAINING (~5% updates) + QLR (100 sample capacity)")
    print("="*50)
    model_sparse = DynamicSparseModel(use_sparsity=True)
    buffer = QuantizedLatentReplayBuffer(capacity=100)
    
    train_model(model_sparse, train_base_loader, epochs=10, buffer=buffer, use_replay=False, name="Baseline Data (w/ sparsity & buffer fill)")
    check_sparsity(model_sparse)
    acc_base_sparse = evaluate_model(model_sparse, test_base_loader, name="Baseline Test Split")
    
    # Continual Learning Sparse + Replay
    train_model(model_sparse, train_evol_loader, epochs=10, buffer=buffer, use_replay=True, name="Evolved Data (w/ Sparse updates & QLR)")
    check_sparsity(model_sparse)
    acc_evol_sparse = evaluate_model(model_sparse, test_evol_loader, name="Evolved Test Split")
    acc_base_cl_sparse = evaluate_model(model_sparse, test_base_loader, name="Baseline Test Split (After Sparse CL + QLR)")
    
    
    # --------
    # Final Reporting
    # --------
    print("\n" + "="*50)
    print("Final Output Metrics Report")
    print("="*50)
    print("a) Standard Training vs. Sparse Training (Initial Learning):")
    print(f"   Baseline Acc: Standard = {acc_base_std:.2f}% | Sparse = {acc_base_sparse:.2f}%")
    print("\nb) Accuracy maintenance on old data (Catastrophic Forgetting test):")
    print(f"   Navie Standard Continual Learning Maintained Acc : {acc_base_cl_std:.2f}%")
    print(f"   Sparse Update + 4-bit QLR Maintained Acc         : {acc_base_cl_sparse:.2f}%")
    print(f"\nc) New Evolved Data Classification:")
    print(f"   Evolved Acc: Standard = {acc_evol_std:.2f}% | Sparse + QLR = {acc_evol_sparse:.2f}%")
    
    if acc_evol_sparse > 85.0 and acc_base_cl_sparse > 85.0:
        print("\nSUCCESS FLAG: Model achieved >85% accuracy on both datasets utilizing the 5% updates engine and QLR!")
    else:
        print("\nWARNING: Accuracy threshold not fully met. Hyperparameters might need tuning depending on initialized weight states.")

if __name__ == "__main__":
    main()
