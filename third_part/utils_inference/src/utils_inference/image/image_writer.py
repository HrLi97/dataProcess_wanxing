import os
import numpy as np
import torch
from torchvision import transforms


class TensorSaveImage:
    @staticmethod
    def save_torch_tensor_jpg_nhwc(tensor: torch.Tensor, save_path):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        TensorSaveImage.save_torch_tensor_jpg_nchw(
            tensor.contiguous().permute(0, 3, 1, 2), save_path=save_path
        )

    @staticmethod
    def save_torch_tensor_jpg_nchw(tensor: torch.Tensor, save_path):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        if torch.max(tensor).item() > 2:
            tensor = tensor / 255.0

        tensor = tensor.clamp(0, 1.0)
        to_pil_image = transforms.ToPILImage()
        image = to_pil_image(tensor[0].cpu())

        base_dir = os.path.dirname(save_path)
        if len(base_dir) > 0:
            os.makedirs(base_dir, exist_ok=True)
        image.save(save_path)

    @staticmethod
    def save_numpy_tensor_jpg_nchw(tensor: np.ndarray, save_path):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        TensorSaveImage.save_numpy_tensor_jpg_nhwc(
            tensor.transpose(0, 2, 3, 1), save_path=save_path
        )

    @staticmethod
    def save_numpy_tensor_jpg_nhwc(tensor: np.ndarray, save_path):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        if np.max(tensor) <= 1:
            tensor = tensor * 255

        to_pil_image = transforms.ToPILImage()
        image = to_pil_image(tensor[0])

        base_dir = os.path.dirname(save_path)
        if len(base_dir) > 0:
            os.makedirs(base_dir, exist_ok=True)
        image.save(save_path)
