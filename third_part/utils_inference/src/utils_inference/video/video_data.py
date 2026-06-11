import torch
from typing import List, Callable, Any, Optional
import math
from torchvision.transforms.v2 import Pad
from torchvision.transforms.functional import gaussian_blur


class VideoData:
    def __init__(
        self,
        tensor: torch.Tensor,
        format="NCHW",
        indexs=None,
        scale=None,
    ):
        """
        tensor: data to manager
        format: data's format, NCHW or NHWC
        indexs: a tag to record where it come from
        scale: 1/sclae to norm to float32
        """
        self.data_ = tensor
        self.format_ = format.upper()
        self.indexs = indexs
        self.scale = scale

    def clone(self) -> "VideoData":
        return VideoData(
            tensor=self.data_.clone(),
            format=self.format,
            indexs=self.indexs,
            scale=self.scale,
        )

    def shallow_copy(self) -> "VideoData":
        return VideoData(
            tensor=self.data_,
            format=self.format,
            indexs=self.indexs,
            scale=self.scale,
        )

    def narraw(self, dim: int, start, length: int) -> "VideoData":
        return VideoData(
            tensor=self.data_.narrow(dim=dim, start=start, length=length),
            format=self.format,
            indexs=self.indexs,
            scale=self.scale,
        )

    def __getitem__(self, key):
        return VideoData(
            tensor=self.data_[key],
            format=self.format,
            indexs=self.indexs,
            scale=self.scale,
        )

    def values_like(self, value: int | float = None) -> "VideoData":
        new = VideoData(
            tensor=torch.empty_like(self.data_),
            format=self.format,
            indexs=self.indexs,
            scale=self.scale,
        )
        if value is not None:
            new.data_[:] = value
        return new

    def set_scale(self, scale):
        self.scale = scale

    def is_norm(self) -> bool:
        if (
            self.data_.dtype == torch.uint8
            or self.data_.dtype == torch.uint16
            or self.data_.dtype == torch.uint32
            or self.data_.dtype == torch.uint64
        ):
            return False
        else:
            return True

    def try_norm(self, dtype=torch.float32, allow_deduce=True) -> bool:
        if self.is_norm():
            self.data_ = self.data_.to(dtype)
            return True

        scale = self.scale
        if scale is None and allow_deduce:
            if self.data_.dtype == torch.uint8:
                scale = 2**8 - 1
            elif self.data_.dtype == torch.uint16:
                scale = 2**16 - 1
            elif self.data_.dtype == torch.uint32:
                scale = 2**32 - 1
            elif self.data_.dtype == torch.uint64:
                scale = 2**64 - 1
        if scale is None:
            return False
        else:
            self.scale = scale
            self.data_ = self.data_.to(torch.float32)
            self.data_.div_(scale)
            self.data_ = self.data_.to(dtype)
            return True

    def try_denorm(self, img_type=None, scale=None) -> bool:
        if not self.is_norm():
            return True

        if scale is None:
            scale = self.scale

        if scale is None:
            return False

        self.data_ = self.data_.to(torch.float32)
        self.data_.mul_(scale)
        if img_type is None:
            if scale < 2**8:
                img_type = torch.uint8
            elif scale < 2**16:
                img_type = torch.uint16
            elif scale < 2**32:
                img_type = torch.uint32
            elif scale < 2**64:
                img_type = torch.uint64
        self.data_ = self.data_.to(img_type)
        return True

    def to(
        self,
        device: torch.device = None,
        dtype: torch.device = None,
        format: str = None,
    ) -> "VideoData":
        if device is not None or dtype is not None:
            self.data_ = self.data_.to(dtype=dtype, device=device)

        if format is not None:
            self.to_format(format=format)
        return self

    def to_dtype(self, dtype: torch.dtype) -> "VideoData":
        return self.to(dtype=dtype)

    def to_device(self, device) -> "VideoData":
        return self.to(device=device)
        return self

    def to_format(self, format="NCHW") -> "VideoData":
        if format.upper() == "NCHW":
            self.to_nchw()
        else:
            self.to_nhwc()
        return self

    def to_nchw(self) -> "VideoData":
        if self.format == "NCHW":
            pass
        else:
            self.data_ = self.data_.movedim(-1, 1)
            self.format_ = "NCHW"
        return self

    def to_nhwc(self) -> "VideoData":
        if self.format == "NHWC":
            pass
        else:
            self.data_ = self.data_.movedim(1, -1)
            self.format_ = "NHWC"
        return self

    @torch.no_grad()
    def align_to(
        self,
        align_w: Optional[int] = None,
        align_h: Optional[int] = None,
        fill: int = 0,
        padding_mode: str = "edge",
    ) -> "VideoData":
        """
        align width and height to

        :param align_w: None means no change
        :param align_h: None means no change
        :param padding_mode: ['constant', 'edge', 'reflect', 'symmetric']
        """
        width_pad, height_pad = 0, 0
        if align_w is not None and align_w > 0:
            width_new = math.ceil(self.width / align_w) * align_w
            width_pad = width_new - self.width

        if align_h is not None and align_h > 0:
            height_new = math.ceil(self.height / align_h) * align_h
            height_pad = height_new - self.height

        if width_pad <= 0 and height_pad <= 0:
            return self

        pad = Pad(
            padding=[0, 0, width_pad, height_pad], fill=fill, padding_mode=padding_mode
        )
        if self.format == "NCHW":
            self.data_ = pad(self.data_)  # type:torch.Tensor
        else:
            self.data_ = pad(self.data.movedim(-1, 1))
            self.data_ = self.data_.movedim(1, -1)

        return self

    @torch.no_grad()
    def to_gray(self, mid_type=torch.float32) -> "VideoData":
        raw_dtype = self.dtype
        gray_conv_kernel = torch.tensor(
            [[[[0.299]], [[0.587]], [[0.114]]]],
            dtype=mid_type,
            device=self.device,
            requires_grad=False,
        )
        if self.format == "NHWC":
            self.data_ = self.data_.movedim(-1, 1)

        self.data_ = torch.nn.functional.conv2d(
            self.data_.to(dtype=mid_type), gray_conv_kernel, stride=1, padding=0
        ).to(raw_dtype)

        if self.format == "NHWC":
            self.data_ = self.data_.movedim(1, -1)
        return self

    @torch.no_grad()
    def merge_channel(self) -> "VideoData":
        if self.channel <= 1:
            return self

        if self.format == "NHWC":
            self.data_ = self.data_.max(dim=-1, keepdim=True)
        else:
            self.data_ = self.data_.max(dim=1, keepdim=True)

    @torch.no_grad()
    def clam(self, min: int, max: int) -> "VideoData":
        self.data_.clamp_(min=min, max=max)
        return self

    @torch.no_grad()
    def repeat_num_to(
        self, num_frame: int, recycle: bool = False, check_num: bool = True
    ) -> "VideoData":
        """
        repeat num_frame to give length

        :param num_frame: final num_frame
        :type num_frame: int
        :param recycle: padding last frame with cycle, like [0,1 2, ...,8,9 to 0,1,2, ...,8,9,8,7,6,...,2,1,0,1,2,...],False means repeat last frame
        :type recycle: bool
        :param check_num: if num_frame>ori_num_frame, will raise exception
        :type check_num: bool
        :return: itself
        :rtype: VideoData
        """
        if check_num:
            assert (
                self.num_frame <= num_frame
            ), f"bad length of origin num_frame={self.num_frame} > {num_frame}"

        padding = num_frame - self.num_frame
        if padding < 0:
            self.data_ = self.data_[:num_frame]
            return self
        elif padding == 0:
            return self

        if self.num_frame < 2 or not recycle:
            self.data_ = torch.cat(
                [
                    self.data_,
                    self.data_[-1:, ...].repeat(padding, 1, 1, 1),
                ],
                dim=0,
            )
        else:
            frames = [self.data_]
            idx = self.num_frame
            step = -1
            for _ in range(padding):
                idx += step
                frames.append(self.data_.narrow(dim=0, start=idx, length=1))
                if idx == 0:
                    step = 1
                elif idx == (self.num_frame - 1):
                    step = -1

            self.data_ = torch.cat(
                frames,
                dim=0,
            )
        return self

    @torch.no_grad()
    def crop(self, bbox: List[int] = [0, 0, -1, -1]) -> "VideoData":
        """
        shallow crop video data

        :param bbox: (x,y,w,h)
        :type bbox: List[int]
        """
        if bbox[2] < 0:
            bbox[2] = self.width
        if bbox[3] < 0:
            bbox[3] = self.height

        if self.format == "NHWC":
            return self[:, bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :]
        else:
            return self[:, :, bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]

    @torch.no_grad()
    def splice(self, left_top: tuple[int], video_data: "VideoData") -> "VideoData":
        if video_data.format != self.format or video_data.device != self.device:
            raise ValueError(
                f"fail to splice video data, self=({self}), video_data={video_data}"
            )

        dtype = self.dtype
        if self.format == "NCHW":
            self.data_[
                :,
                :,
                left_top[1] : left_top[1] + video_data.height,
                left_top[0] : left_top[0] + video_data.width,
            ] = video_data.data_
        else:
            self.data_[
                :,
                left_top[1] : left_top[1] + video_data.height,
                left_top[0] : left_top[0] + video_data.width,
                :,
            ] = video_data.data_
        self.data_ = self.data_.to(dtype)
        return True

    def set_value(
        self, bbox: tuple[int], value, start: int = 0, length: int = -1
    ) -> "VideoData":
        """
        set value for value in bbox

        :param bbox: (x,y,w,h)
        :type bbox: tuple[int]
        :param value: value to set
        :rtype: VideoData
        """
        dtype = self.dtype
        length = length if length > 0 else self.num_frame

        if self.format == "NCHW":
            self.data_[
                start : start + length,
                :,
                bbox[1] : bbox[1] + bbox[3],
                bbox[0] : bbox[0] + bbox[2],
            ] = value
        else:
            self.data_[
                start : start + length,
                bbox[1] : bbox[1] + bbox[3],
                bbox[0] : bbox[0] + bbox[2],
                :,
            ] = value
        self.data_ = self.data_.to(dtype)
        return True

    @torch.no_grad()
    def gussian_dialate(
        self,
        dialate_iter: int = 20,
        ksize: tuple[int] = (3, 3),
        guss_dialate_sigma=0.8,
        remain_distance: int = 2,
    ) -> "VideoData":
        """
        only make sense while video is mask
        """
        assert self.is_norm()

        dilate_conv_kernel = torch.zeros(
            (self.data_.size(1), 1, *ksize),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        dilate_conv_kernel[:, :, ksize[0] // 2, :] = 1     # fmt: skip
        dilate_conv_kernel[:, :, :, ksize[1] // 2,] = 1    # fmt: skip

        dtype = self.dtype
        mask = self.data_
        if self.format == "NHWC":
            mask = mask.movedim(-1, 1)

        mask = torch.where(mask > 0, 1.0, 0.0)
        for _ in range(dialate_iter):
            mask = torch.nn.functional.conv2d(
                mask,
                dilate_conv_kernel,
                stride=1,
                padding=(ksize[0] // 2, ksize[1] // 2),
                groups=mask.size(1),
            )
            mask = torch.where(mask > 0, 1.0, 0.0)

        for _ in range(dialate_iter + remain_distance):
            mask = gaussian_blur(
                mask,
                kernel_size=ksize,
                sigma=guss_dialate_sigma,
            )

        self.data_ = self.data_ + (1 - self.data_) * mask
        if self.format == "NHWC":
            self.data_ = self.data_.movedim(1, -1)

        self.data_ = self.data_.to(dtype)
        return self

    def dialate(
        self,
        dialate_iter: int = 20,
        ksize: tuple[int] = (3, 3),
    ) -> "VideoData":
        if dialate_iter < 1:
            return self

        assert self.is_norm()
        dilate_conv_kernel = torch.zeros(
            (self.data_.size(1), 1, *ksize),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        dilate_conv_kernel[:, :, ksize[0] // 2, :] = 1     # fmt: skip
        dilate_conv_kernel[:, :, :, ksize[1] // 2,] = 1    # fmt: skip

        mask = self.data_
        if self.format == "NHWC":
            mask = mask.movedim(-1, 1)

        mask = torch.where(mask > 0, 1.0, 0.0)
        for _ in range(dialate_iter):
            mask = torch.nn.functional.conv2d(
                mask,
                dilate_conv_kernel,
                stride=1,
                padding=(ksize[0] // 2, ksize[1] // 2),
                groups=mask.size(1),
            )
            mask = torch.where(mask > 0, 1.0, 0.0)

        if self.format == "NHWC":
            mask = mask.movedim(1, -1)
        self.data_ = mask.to(dtype=self.data_.dtype)
        return self

    def __copy__(self):
        return self.shallow_copy()

    def __deepcopy__(self, memo):
        return self.clone()

    def __str__(self):
        return f"VideoData(id={id(self)},tid={id(self.data)}) with shape={self.shape}, dtype={self.dtype}, device={self.device}, scale={self.scale}"

    @property
    def data(self) -> torch.Tensor:
        return self.data_

    @data.setter
    def data(self, value):
        self.data_ = value

    @property
    def format(self):
        return self.format_.upper()

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
    def shape_nchw(self):
        return [self.num_frame, self.channel, self.height, self.width]

    @property
    def shape_nhwc(self):
        return [self.num_frame, self.height, self.width, self.channel]

    @property
    def num_frame(self):
        return self.shape[self.format.index("N")]

    @property
    def channel(self):
        return self.shape[self.format.index("C")]

    @property
    def height(self):
        return self.shape[self.format.index("H")]

    @property
    def width(self):
        return self.shape[self.format.index("W")]
