import torch
from torch.nn.functional import pad


class AudioData:
    def __init__(self, tensor: torch.Tensor, sample_rate=-1, tag=None):
        """
        tensor: data to manager
        format: data's format, NCHW or NHWC
        indexs: a tag to record where it come from
        """
        self.data_ = tensor
        self.sample_rate_ = sample_rate
        self.tag = tag

    def to_device(self, device, inplace=True):
        if inplace:
            self.data_ = self.data_.to(device)
            return self
        else:
            return AudioData(
                tensor=self.data_.to(device),
                sample_rate=self.sample_rate_,
                tag=self.tag,
            )

    def to_dtype(self, dtype: torch.dtype, inplace=True):
        if inplace:
            self.data_ = self.data_.to(dtype)
            return self
        else:
            return AudioData(
                tensor=self.data_.to(dtype),
                sample_rate=self.sample_rate_,
                tag=self.tag,
            )

    def padding_to(self, length, mode="constant", value=0):
        """
        padding length to give length with mode

        mode: 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'

        value: fill value for 'constant' padding. Default: 0
        """
        if length <= self.length:
            return AudioData(
                self.data[..., :length],
                sample_rate=self.sample_rate,
                tag=self.tag,
            )

        return AudioData(
            pad(self.data, pad=[0, length - self.length], mode=mode, value=value),
            sample_rate=self.sample_rate,
            tag=self.tag,
        )

    def release(self):
        self.data_ = None
        self.sample_rate = -1
        self.indexs = None

    @property
    def valid(self):
        return self.data_ is not None

    @property
    def data(self) -> torch.Tensor:
        return self.data_

    @data.setter
    def data(self, value: torch.Tensor):
        if hasattr(self, "embedding"):
            delattr(self, "embedding")
        self.data_ = value

    @property
    def device(self):
        return self.data_.device

    @property
    def device(self):
        return self.data_.device

    @property
    def dtype(self):
        return self.data_.dtype

    @property
    def shape(self):
        return self.data_.shape

    @property
    def length(self):
        return self.shape[1]

    @property
    def channel(self):
        return self.shape[0]

    @property
    def sample_rate(self):
        return self.sample_rate_
