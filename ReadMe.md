# Task-2: Image Segmentation

## Requirements
- Python 3.8 or higher
- PyTorch
- OpenCV
- NumPy

## Project Structure
data/
├── train/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/

Python_Scripts/
├── trainer.py
├── tester.py
└── assets/

Model_Weights/
├── hypothesis_final_full_saved_model.pth

## How to Train
```bash
python Python_Scripts/trainer.py
