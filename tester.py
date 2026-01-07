import os
import torch
from torch.utils.data import DataLoader

from assets.dataset import SegmentationDataset
from assets.model import UNet


# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# DATASET (TEST ONLY)
# =========================
test_dataset = SegmentationDataset(
    image_dir="data/test/images",
    mask_dir="data/test/masks"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


# =========================
# MODEL LOADING
# =========================
model = UNet().to(device)

WEIGHTS_PATH = "Model_Weights/hypothesis_final_full_saved_model.pth"

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"Model weights not found at {WEIGHTS_PATH}. "
        "Please train the model first."
    )

model.load_state_dict(
    torch.load(WEIGHTS_PATH, map_location=device)
)

model.eval()
print("✅ Model loaded successfully")


# =========================
# INFERENCE
# =========================
with torch.no_grad():
    for idx, (img, mask) in enumerate(test_loader):
        img = img.to(device)

        logits = model(img)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.25).float()

        # Optional: print progress
        print(f"Inference done for test image {idx + 1}/{len(test_loader)}")
