# Video Preprocessing with CUDA, PyCUDA, and PyNvCodec

This repository demonstrates a GPU-based pipeline for video preprocessing using CUDA kernels for cropping, resizing, padding, and normalization. It also includes a Python pipeline that uses NVIDIA’s PyNvCodec for video decoding and PyCUDA for launching GPU kernels.

## Prerequisites

Before running the project, make sure you have the following installed and configured:

1. **CUDA Toolkit**  
   Ensure the CUDA Toolkit is installed and your environment variables (e.g., `CUDA_HOME`) are set correctly.

2. **PyCUDA**  
   PyCUDA provides a Python interface to CUDA. Install it via pip:
   ```bash
   pip install pycuda
