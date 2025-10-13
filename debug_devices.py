"""
Run this script to diagnose device mismatch issues.
It will show you exactly where tensors are located.
"""

import torch
from src.data_module.importer import DataImporter
from src.utils.cnn_builder import CNNBuilder
from src.utils.network_utils import (
    LayerConfig,
    LayerType,
    OutChannels,
    KernelSize,
    ActivationFunction,
    NetworkConfig,
)
from torch.nn import CrossEntropyLoss
from src.classification_module.train import Trainer

print("=" * 60)
print("DEVICE DIAGNOSTICS")
print("=" * 60)

# Check CUDA availability
print(f"\n1. CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")

# Test DataImporter
print("\n2. Testing DataImporter...")
data_importer = DataImporter(max_per_class=50)
print(f"   Dataset data device: {data_importer.data.device}")
print(f"   Dataset labels device: {data_importer.labels.device}")

# Test DataLoader
print("\n3. Testing DataLoader...")
train_loader, test_loader = data_importer.get_as_cnn(batch_size=64, test_split=0.2)
sample_batch = next(iter(train_loader))
X_sample, y_sample = sample_batch
print(f"   Batch X device: {X_sample.device}")
print(f"   Batch y device: {y_sample.device}")

# Test model creation
print("\n4. Testing Model Creation...")
config = NetworkConfig(
    layers=[
        LayerConfig(
            layer_type=LayerType.CONV,
            out_channels=OutChannels.CH_16,
            kernel_size=KernelSize.KS_3,
            activation=ActivationFunction.RELU,
        ),
    ]
)
cnn_builder = CNNBuilder(rl_config=config)
model = cnn_builder.build()

print("   Model built")
first_param = next(model.parameters())
print(f"   Model device (before .to()): {first_param.device}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
first_param = next(model.parameters())
print(f"   Model device (after .to({device})): {first_param.device}")

# Test all model parameters
print("\n5. Checking ALL model parameters...")
all_on_same_device = True
expected_device = first_param.device
for i, (name, param) in enumerate(model.named_parameters()):
    if param.device != expected_device:
        print(f"   ❌ MISMATCH: {name} is on {param.device}, expected {expected_device}")
        all_on_same_device = False
    if i < 3:  # Show first 3
        print(f"   ✓ {name}: {param.device}")
if all_on_same_device:
    print(f"   ✓ All parameters on {expected_device}")

# Test Trainer initialization
print("\n6. Testing Trainer initialization...")
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
trainer = Trainer(
    dataloaders=(train_loader, test_loader),
    model=model,
    loss_function=CrossEntropyLoss().to(device),
    optimizer=optimizer,
)
print(f"   Trainer device: {trainer.device}")
first_param = next(trainer.model.parameters())
print(f"   Trainer's model device: {first_param.device}")

# Test actual training step
print("\n7. Testing one training step...")
try:
    # Get a batch
    X, y = next(iter(trainer.train_loader))
    print(f"   Before .to() - X device: {X.device}, y device: {y.device}")

    # Manually do what trainer.train() does
    X = X.to(trainer.device, non_blocking=True)
    y = y.to(trainer.device, non_blocking=True)
    print(f"   After .to() - X device: {X.device}, y device: {y.device}")

    # Check model parameters again
    first_param = next(trainer.model.parameters())
    print(f"   Model parameters device: {first_param.device}")

    # Try forward pass
    trainer.model.train()
    predictions = trainer.model(X)
    print("   ✓ Forward pass successful!")
    print(f"   Predictions device: {predictions.device}")

    # Try loss calculation
    loss = trainer.loss_function(predictions, y)
    print("   ✓ Loss calculation successful!")
    print(f"   Loss device: {loss.device}")

except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)
