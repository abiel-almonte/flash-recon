"""
C++/CUDA runtime providing a GPU-resident pipeline for camera capture, preprocessing, and PyTorch integration.
"""
from __future__ import annotations
import torch
import typing
__all__: list[str] = ['Camera', 'FrameGenerator', 'GraphCachedModule']
class Camera:
    """
    Wrapper around a V4L2 camera device
    """
    def __init__(self, device: str) -> None:
        """
        Open a camera at the given device path (e.g., '/dev/video0').
        """
    def close(self) -> None:
        """
        Close the opened camera
        """
    def print_formats(self) -> None:
        """
        Print all supported camera formats.
        """
    def print_selected_format(self) -> None:
        """
        Print the currently selected camera format.
        """
    def set_format(self, index: int) -> None:
        """
        Set the capture format.
        """
    def stream(self) -> FrameGenerator:
        """
        Return a FrameGenerator that yields frames from this Camera.
        """
class FrameGenerator:
    """
    Iterator that yields preprocessed frames from a Camera as torch.Tensors
    """
    def __iter__(self) -> FrameGenerator:
        """
        Return the iterator object itself.
        """
    def __next__(self) -> torch.Tensor:
        """
        Advance to the next frame.
        
        Raises StopIteration when no frames remain.
        """
class GraphCachedModule:
    """
    Modfier that compiles and captures CUDA graph for the given PyTorch module
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Launch CUDA graph
        """
    def __init__(self, module: typing.Any) -> None:
        """
        PyTorch module to compile and capture
        """
    def capture(self, example_tensor: torch.Tensor) -> None:
        """
        Capture CUDA graph with example input
        """
    def is_captured(self) -> bool:
        """
        Return True if CUDA graph has been captured
        """
