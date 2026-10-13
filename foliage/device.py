"""Hardware detection: CUDA for PyTorch matchers, OpenCL/threads for OpenCV."""
import os

import cv2


def setup_opencv(threads: int | None = None) -> dict:
    """Use every CPU core and OpenCL (GPU-backed UMat ops) when available."""
    cv2.setNumThreads(threads or os.cpu_count() or 1)
    cv2.ocl.setUseOpenCL(cv2.ocl.haveOpenCL())
    cuda = 0
    try:
        cuda = cv2.cuda.getCudaEnabledDeviceCount()
    except (AttributeError, cv2.error):
        pass
    return {"threads": cv2.getNumThreads(), "opencl": cv2.ocl.useOpenCL(), "opencv_cuda_devices": cuda}


def torch_device(prefer: str = "auto"):
    """Pick cuda > mps > cpu. Returns None if PyTorch is not installed."""
    try:
        import torch
    except ImportError:
        return None
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe() -> str:
    info = setup_opencv()
    dev = torch_device()
    torch_s = "not installed" if dev is None else str(dev)
    return (f"OpenCV threads={info['threads']} opencl={info['opencl']} "
            f"cuda_devices={info['opencv_cuda_devices']} | torch device={torch_s}")
