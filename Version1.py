import subprocess
import os


def get_tensorflow_version():
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    print("CUDA version used by TensorFlow:", tf.sysconfig.get_build_info().get('cuda_version', 'Unknown'))
    print("cuDNN version used by TensorFlow:", tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown'))


def get_pytorch_version():
    import torch
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
    else:
        print("CUDA is not available")


def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        print("CUDA version:\n", output)
    except FileNotFoundError:
        print("CUDA is not installed or nvcc is not in the PATH")


def get_cudnn_version():
    cudnn_header_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include\cudnn_version.h'
    if os.path.isfile(cudnn_header_path):
        try:
            output = subprocess.check_output(
                ['findstr', '/C:#define CUDNN_MAJOR', '/C:#define CUDNN_MINOR', '/C:#define CUDNN_PATCHLEVEL',
                 cudnn_header_path]
            ).decode('utf-8')
            lines = output.strip().split('\n')
            cudnn_major = lines[0].split()[-1]
            cudnn_minor = lines[1].split()[-1]
            cudnn_patch = lines[2].split()[-1]
            print(f"cuDNN version: {cudnn_major}.{cudnn_minor}.{cudnn_patch}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while fetching cuDNN version: {e}")
    else:
        print("cuDNN version file not found or cuDNN is not installed")


def main():
    print("Checking TensorFlow version and dependencies...\n")
    get_tensorflow_version()

    print("\nChecking PyTorch version and dependencies...\n")
    get_pytorch_version()

    print("\nChecking CUDA version...\n")
    get_cuda_version()

    print("\nChecking cuDNN version...\n")
    get_cudnn_version()


if __name__ == "__main__":
    main()
