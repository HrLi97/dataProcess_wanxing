import torch
import numpy as np
from .video_data import VideoData

class ViewObject:
    def show_self(self, expand_obj=False, expand_none=False) -> str:
        strs = ""
        for attr, value in vars(self).items():
            if value is None:
                if expand_none:
                    strs += f"\n{attr}: None"
            elif isinstance(value, (torch.Tensor, VideoData)):
                strs += f"\n{attr}: {type(value)}, {value.shape}, {value.dtype}, {value.device}"
            elif isinstance(value, np.ndarray):
                strs += f"\n{attr}: {type(value)}, {value.shape}, {value.dtype}"
            elif isinstance(
                value, (list, tuple, dict, range, set, frozenset, bytes, bytearray)
            ):
                strs += f"\n{attr}: {type(value)}, len={len(value)}"
            elif isinstance(value, (int, float, str, complex, bool)):
                strs += f"\n{attr}: {value}"
            else:
                if expand_obj:
                    strs += f"\n{attr}: {type(expand_obj)}"
        return strs[1:] if len(strs) > 0 else ""
