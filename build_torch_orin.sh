export USE_NCCL=0
export USE_DISTRIBUTED=1
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0

# Orin is based on Ampere Achitecture
export TORCH_CUDA_ARCH_LIST="8.7"

export PYTORCH_BUILD_VERSION=2.5.0
export PYTORCH_BUILD_NUMBER=1

python3 setup.py bdist_wheel
