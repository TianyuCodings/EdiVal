#!/bin/bash

# Unified setup script for vLLM and Grounding DINO
# This script addresses common dependency conflicts between the two libraries

sudo apt install libgl1
set -e

# Create fresh environment
conda create -n edival python=3.10 -y
source activate edival

# Set CUDA environment variables
# This is for Grounding DINO to use the correct CUDA version. 
# Please have a look at https://github.com/IDEA-Research/GroundingDINO README for more details if this is not working for you.
CUDA_PATH=/usr/local/cuda-12.1
if [ -n "$CUDA_PATH" ]; then
    export CUDA_HOME="$CUDA_PATH"
    echo "Found CUDA at: $CUDA_HOME"
fi
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc 
source ~/.bashrc
source activate edival


pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# Install vLLM-specific dependencies that might conflict
pip install transformers==4.45.2  # Pin to avoid conflicts
pip install tokenizers==0.19.1
pip install accelerate==0.34.2
pip install pydantic==2.8.2  # Specific version to avoid conflicts
pip install tabulate

# Install other vLLM dependencies
pip install fastapi
pip install uvicorn
pip install openai
pip install datasets
pip install ray>=2.9


echo "=== Installing Grounding DINO Dependencies ==="
# Install dependencies that are compatible with vLLM
pip install opencv-python==4.8.1.78
pip install pillow==10.0.1
pip install matplotlib
pip install scipy
pip install scikit-image
pip install gdown

# Install supervision with specific version to avoid conflicts
pip install supervision==0.22.0

# Install other Grounding DINO requirements
pip install addict
pip install yapf
pip install timm==0.9.16  # Pin timm version for stability
pip install pycocotools
pip install pandas
pip install tabulate

echo "=== Setting Up Grounding DINO ==="
cd "$(dirname "$0")/.."

# Clone or navigate to GroundingDINO
if [ ! -d "GroundingDINO" ]; then
    echo "Cloning GroundingDINO repository..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git
fi

# # Modify requirements.txt to avoid conflicts
# echo "=== Modifying GroundingDINO requirements for compatibility ==="
# cat > requirements_modified.txt << 'EOF'
# torch>=2.0.0
# torchvision>=0.15.0
# transformers>=4.21.0
# addict
# yapf
# timm>=0.6.7
# numpy
# opencv-python
# supervision>=0.22.0
# pycocotools
# EOF

# Install Grounding DINO with modified requirements
echo "Installing Grounding DINO with compatibility fixes..."
python -m pip install -e ./GroundingDINO --no-build-isolation --config-settings editable_mode=compat
pip install vllm==0.8.4

echo "=== Downloading Pre-trained Weights ==="
cd GroundingDINO
mkdir -p weights
cd weights
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi

cd ../../
pip install diffusers
pip install opencv-python
pip install tabulate
pip install tensorboard
pip install transformers==4.57.6