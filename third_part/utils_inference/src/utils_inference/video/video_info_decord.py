import os
import decord
from typing import List
from .video_data import VideoData
from ..bench.time_bench import test_time
from ..common.global_values import GlobalValues
from .video_info_ffmpeg import VideoInfoFfmpeg


class VideoInfoDecord(VideoInfoFfmpeg):
    """
    read video by decord, only support 8bit video
    """

    def __init__(self, file_path: str = None):
        super().__init__(file_path)

        os.environ["DECORD_EOF_RETRY_MAX"] = "4464640"  # 436*10240
        self.init(file_path=file_path)

    @test_time(enable=GlobalValues.ENABLE_PER)
    def load_data_by_index(
        self, indexs: List[int] = [], device="cpu", format="NHWC"
    ) -> VideoData:
        if self.is_high_bit:
            print(
                f"warning: high-bit video is not supported by VideoInfoDecord because of the limit of decord, you many get error image."
            )

        device_str, device_id = self.read_device(device)
        if device_str == "cpu":
            device = decord.cpu(device_id)
        elif device_str == "cuda" or device_str == "gpu":
            device = decord.gpu(device_id)
        else:
            raise Exception(
                f"VideoInfoDecord device only support cuda and cpu, {device_str} is given"
            )

        indexs = [index for index in indexs if (index >= 0 and index < self.num_frame)]
        if len(indexs) < 1:
            return None

        from torch.utils import dlpack

        reader = decord.VideoReader(self.file_path, ctx=device)
        ori = reader.get_batch(indexs)
        return VideoData(
            dlpack.from_dlpack(ori.to_dlpack()), format="NHWC", indexs=indexs, scale=255
        ).to_format(format)
