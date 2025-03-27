# BME_5710_Project_Team_5

## MRI Super-Resolution Project

This repository contains a PyTorch implementation for MRI image super-resolution using deep learning techniques.

## Project Structure

```
.
├── checkpoints/            # Model checkpoints
├── configs/                # Configuration files
├── data/                   # Data directory
│   ├── train/              # Training data
│   │   ├── high-res/       # High-resolution training images
│   │   └── low-res/        # Low-resolution training images
│   └── val/                # Validation data
│       ├── high-res/       # High-resolution validation images
│       └── low-res/        # Low-resolution validation images
├── logs/                   # Log files
├── results/                # Results and visualizations
├── src/                    # Source code
│   ├── models/             # Model definitions
│   │   ├── __init__.py     # Model imports
│   │   ├── losses.py       # Loss functions
│   │   ├── resnet.py       # Residual block implementations
│   │   └── unet.py         # U-Net based architecture
│   ├── utils/              # Utility functions
│   │   ├── __init__.py     # Utility imports
│   │   ├── data.py         # Data loading utilities
│   │   ├── metrics.py      # Evaluation metrics
│   │   └── visualization.py # Visualization functions
│   ├── evaluate.py         # Evaluation script
│   ├── inference.py        # Inference script
│   └── train.py            # Training script
└── requirements.txt        # Project dependencies
```

## Setup Instructions

### 1. Set up a virtual environment with Python

Open a terminal in VSCode (Terminal > New Terminal) and run:

```bash
# Navigate to the repository directory
cd /path/to/your/repo

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Ubuntu/Linux:
source venv/bin/activate

# On Windows (This may be wrong, not on Windows machine):
# venv\Scripts\activate
```

You should now see `(venv)` at the beginning of your terminal prompt, indicating that the virtual environment is active.

### 2. Install the required dependencies

Install all dependencies from the requirements.txt file:

```bash
# Make sure you're in the activated virtual environment
pip install -r requirements.txt
```

### 3. Configure VSCode to use the virtual environment

1. In VSCode, press `Ctrl+Shift+P` to open the command palette
2. Type "Python: Select Interpreter" and select it
3. Choose the interpreter located in your virtual environment (e.g., `./venv/bin/python`)

## Data Format

The code expects paired MRI images in TIFF format (.tif) organized in the data directory structure. Each low-resolution image should have a corresponding high-resolution image with the same filename.

## Dependencies

The project requires the following main dependencies:
- PyTorch and torchvision
- NumPy
- Pillow
- Matplotlib
- tensorboard
- tqdm
- PyYAML
- scikit-image (optional, for improved image processing)

See `requirements.txt` for the complete list of dependencies and version requirements.

## Note

Some function implementations in the codebase are marked with `pass` and need to be completed before running the scripts.