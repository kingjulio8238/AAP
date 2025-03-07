# Install basic requirements
pip install torch torchvision
pip install opencv-python
pip install setuptools

# Install SEEM
pip install git+https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git@package

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install Semantic-SAM
pip install git+https://github.com/UX-Decoder/Semantic-SAM.git@package

# # Install DCNv2
# pip install dcnv2

# Install detectron2-xyz (common error fix)
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'

#Demo dependencies (pure_SoM)
# Install MPI implementation first
sudo apt-get install -y libopenmpi-dev
pip install mpi4py

# Set CUDA paths explicitly based on your system
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Create symbolic links if needed
if [ ! -d "/usr/local/cuda" ]; then
  sudo mkdir -p /usr/local/cuda/bin
  sudo ln -sf $CUDA_HOME /usr/local/cuda
  sudo ln -sf $CUDA_HOME/bin/nvcc /usr/local/cuda/bin/nvcc
fi

# Modify make.sh to use CUDA 12.4 without version detection
cat > ops/make.sh << 'EOF'
#!/usr/bin/env bash

# Hardcoded for CUDA 12.4
echo "Using CUDA 12.4"
export TORCH_CUDA_ARCH_LIST="5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"

# Install dependencies first
python -m pip install --upgrade pip
python -m pip install ninja
python -m pip install -U setuptools wheel
python -m pip install -U torch torchvision
python -m pip install git+https://github.com/facebookresearch/detectron2.git

# Build the extension
python setup.py build install
EOF

# Make the script executable
chmod +x ops/make.sh

# Patch the source files to fix PyTorch compatibility issues
cat > patch_ms_deform_attn.py << 'EOF'
#!/usr/bin/env python3
import os

# Fix ms_deform_attn.h
with open('ops/src/ms_deform_attn.h', 'r') as f:
    content = f.read()

# Replace deprecated type() calls with device().is_cuda()
content = content.replace('value.type().is_cuda()', 'value.device().is_cuda()')

with open('ops/src/ms_deform_attn.h', 'w') as f:
    f.write(content)

# Fix ms_deform_attn_cuda.cu
with open('ops/src/cuda/ms_deform_attn_cuda.cu', 'r') as f:
    content = f.read()

# Replace AT_DISPATCH_FLOATING_TYPES with newer version
content = content.replace('AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda"', 
                         'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda"')
content = content.replace('AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda"', 
                         'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda"')

with open('ops/src/cuda/ms_deform_attn_cuda.cu', 'w') as f:
    f.write(content)
EOF

# Make the patch script executable and run it
chmod +x patch_ms_deform_attn.py
python patch_ms_deform_attn.py

# Compile MultiScaleDeformableAttention CUDA operators
# cd ops
# bash make.sh
# cd ..

