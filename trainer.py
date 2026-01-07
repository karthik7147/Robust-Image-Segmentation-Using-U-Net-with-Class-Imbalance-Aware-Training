import torch
from torch.utils.data import DataLoader

from assets.dataset import SegmentationDataset
from assets.model import UNet
from assets.losses import get_losses


# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATA
# =========================
dataset = SegmentationDataset(
    "data/train/images",
    "data/train/masks"
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

# =========================
# MODEL
# =========================
model = UNet().to(device)

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

model.apply(init_weights)

# =========================
# OPTIMIZER
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =========================
# LOSSES
# =========================
bce_loss, combined_loss = get_losses(device)

# =========================
# -------- STAGE 1 --------
# =========================
print("Stage 1: BCE-only warmup")

for epoch in range(5):
    model.train()
    epoch_loss = 0

    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)

        optimizer.zero_grad()
        loss = bce_loss(model(img), mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Stage 1 | Epoch {epoch+1}/5] Loss: {epoch_loss/len(loader):.4f}")

# =========================
# -------- STAGE 2 --------
# =========================
print("Stage 2: BCE + Dice training")

for epoch in range(50):
    model.train()
    epoch_loss = 0

    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)

        optimizer.zero_grad()
        loss = combined_loss(model(img), mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Stage 2 | Epoch {epoch+1}/50] Loss: {epoch_loss/len(loader):.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(
    model.state_dict(),
    "Model_Weights/hypothesis_final_full_saved_model.pth"
)

print("Training complete. Model saved.")
