import tensorflow as tf
import torch
import subprocess
import os

from aksetup_helper import get_pybind_include


def get_tensorflow_version():
    print("TensorFlow version:", tf.__version__)
    print("CUDA version used by TensorFlow:", tf.sysconfig.get_build_info().get('cuda_version', 'Unknown'))
    print("cuDNN version used by TensorFlow:", tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown'))


def get_pytorch_version():
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
    else:
        print("CUDA is not available")


get_tensorflow_version()
get_pybind_include()
