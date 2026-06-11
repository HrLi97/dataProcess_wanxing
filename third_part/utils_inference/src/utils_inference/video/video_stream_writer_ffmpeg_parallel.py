import subprocess
from .video_stream_writer_ffmpeg import VideoStreamWriterFfmpeg
import torch.distributed as dist
import torch


# ffprobe -v quiet -select_streams v:0 -show_streams -print_format json test.mp4
class VideoStreamWriterFfmpegParallel(VideoStreamWriterFfmpeg):
    def __init__(
        self,
        via_device: torch.device,
        save_path: str = None,
        writer: subprocess.Popen = None,
        refer_file: str = None,
        loglevel="warning",
        write_async: bool = True,
        write_rank=0,
        group=None,
        **kwargs
    ):
        self.__communicate__ = dist.is_initialized()
        self.__src_rank__ = write_rank
        self.__is_work_rank__ = (not dist.is_initialized()) or (
            dist.get_rank() == self.__src_rank__
        )

        self.__group__ = group
        self.__via_device__ = via_device

        super().__init__(
            save_path=save_path,
            writer=writer,
            refer_file=refer_file,
            loglevel=loglevel,
            write_async=write_async,
            enable=self.__is_work_rank__,
            **kwargs,
        )

    def exchange_data(self, objects=None):
        if self.__communicate__:
            if self.__is_work_rank__:
                data = [objects]
            else:
                data = [None]
            dist.broadcast_object_list(
                object_list=data,
                src=self.__src_rank__,
                group=self.__group__,
                device=self.__via_device__,
            )
            return data[0]
        else:
            return objects

    def Write(self, data: bytes, return_info: bool = False) -> bool | tuple[bool, str]:
        if self.__is_work_rank__:
            status, info = super().Write(data, return_info=True)
        else:
            status, info = None, None

        status, info = self.exchange_data(objects=(status, info))

        if not return_info:
            return status
        else:
            return status, info
