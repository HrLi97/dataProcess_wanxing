import os
from typing import List
from .audio_data import AudioData
import torch
from abc import ABC, abstractmethod


class AudioInfo(ABC):
    """
    interface class to read audio info, use directly can meet exception
    """

    def __init__(self, file_path=None, target_sample_rate=None):
        self.file_path = file_path

        self.duration = 0.0  # s
        self.channel = 0

        # real after sample
        self.length_per_channel = 0
        self.sample_rate = -1

        # file origin
        self.ori_sample_rate = -1
        self.ori_length_per_channel = 0

        os.environ["DECORD_EOF_RETRY_MAX"] = "4464640"  # 436*10240
        self.init(file_path=file_path, target_sample_rate=target_sample_rate)

    @abstractmethod
    def init(self, file_path=None, target_sample_rate=None) -> bool:
        self.duration = 0.0  # s
        self.channel = 0

        # real after sample
        self.length_per_channel = 0
        self.sample_rate = -1

        # file origin
        self.ori_sample_rate = -1
        self.ori_length_per_channel = 0
        return False

    @abstractmethod
    def load_data_by_index(
        self, indexs: List[int] = [], device="cpu", sample_rate=1.0
    ) -> AudioData:
        return None

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

    # @test_time(enable=GlobalValues.ENABLE_PER)
    @torch.no_grad()
    def load_data(self, offset=0, max_num_frame=-1, sample_rate=1, device="cpu"):
        if max_num_frame < 0:
            max_num_frame = self.length

        indexs = range(
            offset,
            min(int(offset + sample_rate * max_num_frame), self.length),
            sample_rate,
        )[:max_num_frame]

        return self.load_data_by_index(
            list(indexs), device=device, sample_rate=sample_rate
        )

    @property
    def length(self):
        return len(self)

    @property
    def ori_length(self):
        return self.ori_length_per_channel

    @property
    def valid(self):
        return self.length > 0

    def close(self):
        pass

    def __len__(self):
        return self.length_per_channel

    def __str__(self):
        return f"file_path: {self.file_path}, duration: {self.duration}s, channel: {self.channel}, length_per_channel: {self.length_per_channel}, sample_rate: {self.sample_rate}, ori_sample_rate: {self.ori_sample_rate}, ori_length_per_channel: {self.ori_length_per_channel}"


class AudioInfoNone(AudioInfo):
    """
    an empty AudioInfo, try to load frame will raise Exception
    """

    def __init__(self, file_path=None, target_sample_rate=None):
        self.file_path = file_path

        self.duration = 0.0  # s
        self.channel = 0

        # real after sample
        self.length_per_channel = 0
        self.sample_rate = -1

        # file origin
        self.ori_sample_rate = -1
        self.ori_length_per_channel = 0

    def init(self, file_path=None, target_sample_rate=None) -> bool:
        self.duration = 0.0  # s
        self.channel = 0

        # real after sample
        self.length_per_channel = 0
        self.sample_rate = -1

        # file origin
        self.ori_sample_rate = -1
        self.ori_length_per_channel = 0
        return False

    def load_data_by_index(
        self, indexs: List[int] = [], device="cpu", sample_rate=1.0
    ) -> AudioData:
        raise Exception("you call and empty AudioInfo")

    def load_data(self, offset=0, max_num_frame=-1, sample_rate=1, device="cpu"):
        raise Exception("you call and empty AudioInfo")
    
    def read_device(self, device_str: str) -> tuple:
        raise Exception("you call and empty AudioInfo")
