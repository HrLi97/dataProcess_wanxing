from typing import Optional, List
from .video_data import VideoData
from abc import ABC, abstractmethod


class VideoInfo(ABC):
    """
    interface class to read video info, use directly can meet exception
    """

    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.num_frame = 0
        self.channel = 0
        self.width = 0
        self.height = 0
        self.fps = 0
        self.bit_rate = 0

        self.pix_fmt = None
        self.color_space = None
        self.color_transfer = None
        self.color_primaries = None
        self.duration = None

        self.sample_aspect_ratio = None
        self.display_aspect_ratio = None

        self.addition_tag = None

    @property
    def meta_data(self) -> dict:
        return {
            "file_path": self.file_path,
            "num_frame": self.num_frame,
            "channel": self.channel,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "bit_rate": self.bit_rate,
            "pix_fmt": self.pix_fmt,
            "color_space": self.color_space,
            "color_transfer": self.color_transfer,
            "color_primaries": self.color_primaries,
            "duration": self.duration,
            "is_high_bit": self.is_high_bit,
            "rgb_fmt": self.rgb_fmt,
            "sample_aspect_ratio": self.sample_aspect_ratio,
            "display_aspect_ratio": self.display_aspect_ratio,
        }

    @abstractmethod
    def init(self, file_path: str = None) -> bool:
        self.file_path = file_path
        self.num_frame = 0
        self.channel = 0
        self.width = 0
        self.height = 0
        self.fps = 0
        self.bit_rate = 0

        self.pix_fmt = None
        self.color_space = None
        self.color_transfer = None
        self.color_primaries = None
        self.duration = None

        self.addition_tag = None
        return False

    @abstractmethod
    def load_data_by_index(
        self, indexs: List[int] = [], device="cpu", format="NHWC"
    ) -> VideoData:
        return None

    def load_data(self, offset, max_num_frame, sample_rate=1):
        if max_num_frame < 0:
            max_num_frame = self.num_frame

        indexs = range(
            offset,
            min(offset + sample_rate * max_num_frame, self.num_frame),
            sample_rate,
        )[:max_num_frame]

        return self.load_data_by_index(list(indexs))

    def read_device(self, device_str: str) -> tuple:
        device = "cpu"
        index = 0
        if ":" in device_str:
            data = device_str.split(":")
            device, index = data[0].strip(), int(data[1].strip())
        else:
            device = device_str
            index = 0
        return device.lower(), index

    def close(self):
        pass

    @property
    def shape_nhwc(self):
        return [self.num_frame, self.height, self.width, self.channel]

    @property
    def shape_nchw(self):
        return [self.num_frame, self.channel, self.height, self.width]

    @property
    def valid(self) -> bool:
        return self.num_frame > 0

    @property
    def is_high_bit(self) -> bool:
        high_pix_fmt = [
            "yuv420p10le",
            "yuv422p10le",
            "yuv444p10le",
            "yuv420p10be",
            "yuv422p10be",
            "yuv444p10be",
            "yuv420p12le",
            "yuv422p12le",
            "yuv444p12le",
            "yuv420p12be",
            "yuv422p12be",
            "yuv444p12be",
        ]

        return self.pix_fmt in high_pix_fmt

    @property
    def rgb_fmt(self) -> str:
        if not self.is_high_bit:
            return "rgb24"
        else:
            return "rgb48le"


class VideoInfoNone(VideoInfo):
    """
    an empty VideoInfo, try to load frame will raise Exception
    """

    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.num_frame = 0
        self.channel = 0
        self.width = 0
        self.height = 0
        self.fps = 0
        self.bit_rate = 0

    def init(self, file_path: str = None) -> bool:
        self.file_path = file_path
        self.num_frame = 0
        self.channel = 0
        self.width = 0
        self.height = 0
        self.fps = 0
        self.bit_rate = 0
        return False

    def load_data_by_index(
        self, indexs: List[int] = [], device="cpu", format="NHWC"
    ) -> VideoData:
        raise Exception("you call and empty VideoInfo")

    def load_data(self, offset, max_num_frame, sample_rate=1):
        raise Exception("you call and empty VideoInfo")

    def read_device(self, device_str: str) -> tuple:
        raise Exception("you call and empty VideoInfo")
