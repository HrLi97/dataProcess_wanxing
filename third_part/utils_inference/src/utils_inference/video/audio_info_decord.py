import os
import decord
from typing import List
from .audio_data import AudioData
import torch
from .audio_info import AudioInfo


class AudioInfoDecord(AudioInfo):
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

    def init(self, file_path=None, target_sample_rate=None) -> bool:
        self.file_path = file_path
        if file_path is None or not os.path.exists(file_path):
            self.duration = 0.0  # s
            self.channel = 0

            # real after sample
            self.length_per_channel = 0
            self.sample_rate = -1

            # file origin
            self.ori_sample_rate = -1
            self.ori_length_per_channel = 0
            return False
        else:
            reader = decord.AudioReader(self.file_path, sample_rate=-1)

            # file origin
            self.ori_length_per_channel = reader._num_samples_per_channel
            self.ori_sample_rate = reader.sample_rate
            del reader

            if target_sample_rate is not None:
                reader = decord.AudioReader(
                    self.file_path, sample_rate=target_sample_rate
                )

            self.duration = reader._duration
            self.channel = reader._num_channels

            # real after sample
            self.length_per_channel = reader._num_samples_per_channel
            self.sample_rate = reader.sample_rate
            del reader
            return True

    # @test_time(enable=GlobalValues.ENABLE_PER)
    def load_data_by_index(
        self, indexs: List[int] = [], device=decord.cpu(0), sample_rate=1.0
    ) -> AudioData:
        device_str, device_id = self.read_device(device)
        if device_str == "cpu":
            device = decord.cpu(device_id)
        else:
            raise Exception(
                f"AudioInfoDecord device only support cpu, {device_str} is given"
            )

        indexs = [index for index in indexs if (index >= 0 and index < self.length)]
        if len(indexs) < 1:
            return None

        from torch.utils import dlpack

        reader = decord.AudioReader(
            self.file_path, ctx=device, sample_rate=self.sample_rate
        )
        ori = reader.get_batch(indexs)
        return AudioData(
            dlpack.from_dlpack(ori.to_dlpack()),
            sample_rate=self.sample_rate / sample_rate,
            tag=indexs,
        )
