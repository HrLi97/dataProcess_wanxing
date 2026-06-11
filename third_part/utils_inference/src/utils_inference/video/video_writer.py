import os
import numpy as np
import torch
import os
import shutil
import numpy as np
from .video_data import VideoData
from .video_stream_writer import VideoStreamWriter
from .video_stream_writer_ffmpeg import VideoStreamWriterFfmpeg
from .video_info_ffmpeg import VideoInfoFfmpeg
from .video_info import VideoInfo
import subprocess
import ffmpeg


class TensorSaveVideo:
    @staticmethod
    @torch.no_grad()
    def videodata_to_video(
        data: VideoData, path: str, fps, write_to=True, append=False
    ):
        """
        This function will automatically convert the image to RGB888 format and store it. It can accept floating-point number input.
        """
        if data.format == "NCHW":
            TensorSaveVideo.torch_nchw_to_video(
                tensor=data.data, path=path, fps=fps, write_to=write_to, append=append
            )
        else:
            TensorSaveVideo.torch_nhwc_to_video(
                tensor=data.data, path=path, fps=fps, write_to=write_to, append=append
            )

    @staticmethod
    @torch.no_grad()
    def videodata_to_stream(
        data: VideoData,
        writer: VideoStreamWriter,
        write_to=True,
    ):
        """
        This function is not responsible for data format conversion. Please ensure that the data matches the expected format of the writer.
        """
        if data.format == "NCHW":
            return TensorSaveVideo.torch_nchw_to_video_stream(
                tensor=data.data,
                writer=writer,
                write_to=write_to,
            )
        else:
            return TensorSaveVideo.torch_nhwc_to_video_stream(
                tensor=data.data,
                writer=writer,
                write_to=write_to,
            )

    @staticmethod
    @torch.no_grad()
    def torch_nchw_to_video(
        tensor: torch.Tensor, path: str, fps, write_to=True, append=False
    ):
        """
        This function will automatically convert the image to RGB888 format and store it. It can accept floating-point number input.
        """
        return TensorSaveVideo.torch_nhwc_to_video(
            tensor=tensor.movedim(1, 3),
            path=path,
            fps=fps,
            write_to=write_to,
            append=append,
        )

    @staticmethod
    @torch.no_grad()
    def torch_nhwc_to_video(
        tensor: torch.Tensor, path: str, fps, write_to=True, append=False
    ):
        """
        This function will automatically convert the image to RGB888 format and store it. It can accept floating-point number input.
        """
        if not write_to or tensor is None:
            return

        if tensor.dtype not in [torch.uint8, torch.uint16, torch.uint32]:
            tensor = (tensor * 255).to(torch.uint8)

        _, h, w, c = tensor.shape

        write_path = path
        if append and os.path.exists(path):
            write_path = "_cache_.mp4"
            raw_video = VideoInfoFfmpeg(path)
            assert (
                raw_video.height == h
                and raw_video.width == w
                and raw_video.channel == c
            ), f"append mode video shape should be same as raw-shape, width={raw_video.width}, height={raw_video.height}, channel={raw_video.channel}"

            ffmpeg_writer = VideoStreamWriterFfmpeg(
                save_path=write_path, refer_file=path
            )
            MAX_CHUNK = 300
            start = 0
            while start < raw_video.num_frame:
                chunk_size = min(MAX_CHUNK, raw_video.num_frame - start)
                if chunk_size < 1:
                    break
                TensorSaveVideo.videodata_to_stream(
                    raw_video.load_data_by_index(
                        indexs=list(range(start, start + chunk_size)),
                        __debug_test_time__=False,
                        device="cpu",
                        format="NHWC",
                    ),
                    writer=ffmpeg_writer,
                )
                start += chunk_size
            raw_video.close()

        else:
            ffmpeg_writer = VideoStreamWriterFfmpeg(
                save_path=write_path,
                writer=VideoStreamWriterFfmpeg.create_writer(
                    save_path=write_path,
                    width=w,
                    height=h,
                    fps=fps,
                    input_fmt="gray" if c == 1 else "rgb24",
                ),
            )

        TensorSaveVideo.torch_nhwc_to_video_stream(tensor=tensor, writer=ffmpeg_writer)
        ffmpeg_writer.Close()

        if write_path != path:
            shutil.move(write_path, path)
        return None

    @staticmethod
    @torch.no_grad()
    def torch_nchw_to_video_stream(
        tensor: torch.Tensor,
        writer: VideoStreamWriter,
        write_to=True,
    ):
        """
        This function is not responsible for data format conversion. Please ensure that the data matches the expected format of the writer.
        """
        return TensorSaveVideo.torch_nhwc_to_video_stream(
            tensor=tensor.movedim(1, 3),
            writer=writer,
            write_to=write_to,
        )

    @staticmethod
    def torch_nhwc_to_video_stream(
        tensor: torch.Tensor,
        writer: VideoStreamWriter,
        write_to=True,
    ):
        """
        This function is not responsible for data format conversion. Please ensure that the data matches the expected format of the writer.
        """
        if not write_to:
            return

        tensor_numpy_nhwc = tensor.cpu().numpy()
        for idx in range(tensor_numpy_nhwc.shape[0]):
            status, info = writer.Write(
                np.ascontiguousarray(tensor_numpy_nhwc[idx]).tobytes(), return_info=True
            )
            assert status, f"fail to write video, {info}"
        return None

    @staticmethod
    def resize_mp4(input_path: str, resize_to: tuple, output_path: str = None) -> int:
        return TensorSaveVideo.resize_video(
            input_path=input_path, resize_to=resize_to, output_path=output_path
        )

    @staticmethod
    def repeate_mp4(
        input_path: str,
        repeate_to: int,
        output_path: str = None,
        check_frame: bool = True,
    ):
        return TensorSaveVideo.repeate_video(
            input_path=input_path,
            repeate_to=repeate_to,
            output_path=output_path,
            check_frame=check_frame,
        )

    @staticmethod
    def repeate_video(
        input_path: str,
        repeate_to: int,
        output_path: str = None,
        check_frame: bool = True,
    ):
        raw_video = VideoInfoFfmpeg(input_path)
        ffmpeg_writer = VideoStreamWriterFfmpeg(
            save_path=output_path, refer_file=input_path
        )

        assert repeate_to >= raw_video.num_frame or (
            not check_frame
        ), f"bad op to repeate video with frame-{raw_video.num_frame} to {repeate_to}"

        assert raw_video.num_frame > 0, "raw video is empty, can not repeate"

        MAX_CHUNK = 300
        start = 0
        while start < raw_video.num_frame:
            chunk_size = min(MAX_CHUNK, raw_video.num_frame - start)
            if chunk_size < 1:
                break
            TensorSaveVideo.videodata_to_stream(
                raw_video.load_data_by_index(
                    indexs=list(range(start, start + chunk_size)),
                    __debug_test_time__=False,
                    device="cpu",
                    format="NHWC",
                ),
                writer=ffmpeg_writer,
            )
            start += chunk_size

        last_frame = raw_video.load_data_by_index(
            indexs=[raw_video.num_frame - 1],
            __debug_test_time__=False,
            device="cpu",
            format="NHWC",
        )
        for _ in range(repeate_to - raw_video.num_frame):
            TensorSaveVideo.videodata_to_stream(
                last_frame,
                writer=ffmpeg_writer,
            )
        raw_video.close()
        ffmpeg_writer.Close()

    @staticmethod
    def resize_video(input_path: str, resize_to: tuple, output_path: str = None) -> int:
        raw_video = VideoInfoFfmpeg(input_path)
        ffmpeg_writer = VideoStreamWriterFfmpeg(
            save_path=None,
            writer=VideoStreamWriterFfmpeg.create_writer(
                save_path=output_path,
                width=raw_video.width,
                height=raw_video.height,
                fps=raw_video.fps,
                resize_to=resize_to,
            ),
        )
        MAX_CHUNK = 300
        start = 0
        while start < raw_video.num_frame:
            chunk_size = min(MAX_CHUNK, raw_video.num_frame - start)
            if chunk_size < 1:
                break
            TensorSaveVideo.videodata_to_stream(
                raw_video.load_data_by_index(
                    indexs=list(range(start, start + chunk_size)),
                    __debug_test_time__=False,
                    device="cpu",
                    format="NHWC",
                ),
                writer=ffmpeg_writer,
            )
            start += chunk_size
        raw_video.close()
        ffmpeg_writer.Close()

    @staticmethod
    def add_audio_to_video(video_path: str, audio_path: str, output_path: str):
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-filter_complex",
            "[1:a]apad",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-loglevel",
            "quiet",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    @staticmethod
    def copy_audio_to_video(audio_source_video: str, video: str, save_path: str):
        """
        merge `audio of audio_source_video` and `video`, save as `output_video`
        """

        audio_input = ffmpeg.input(audio_source_video)
        video_input = ffmpeg.input(video)

        audio = audio_input.audio
        video = video_input.video

        (
            ffmpeg.output(
                audio, video, save_path, vcodec="copy", acodec="copy", shortest=None
            )
            .overwrite_output()
            .run()
        )

    @staticmethod
    def copy_to_video_rgb24(
        raw_video: VideoInfo, write_path: str, fps: int = 0, value_scale: float = 1.0
    ):
        fps = raw_video.fps if fps <= 0 else fps
        fps = 25 if fps <= 0 else fps
        ffmpeg_writer = VideoStreamWriterFfmpeg(
            save_path=write_path,
            writer=VideoStreamWriterFfmpeg.create_writer(
                save_path=write_path,
                width=raw_video.width,
                height=raw_video.height,
                fps=fps,
                input_fmt="gray" if raw_video.channel == 1 else "rgb24",
            ),
        )

        MAX_CHUNK = 100
        start = 0
        while start < raw_video.num_frame:
            chunk_size = min(MAX_CHUNK, raw_video.num_frame - start)
            if chunk_size < 1:
                break
            data = raw_video.load_data_by_index(
                indexs=list(range(start, start + chunk_size)),
                __debug_test_time__=False,
                device="cpu",
                format="NHWC",
            )  # type:VideoData
            data.try_norm()
            data.data.mul_(value_scale)
            data.scale = 255
            data.try_denorm()
            TensorSaveVideo.videodata_to_stream(
                data,
                writer=ffmpeg_writer,
            )
            start += chunk_size
        raw_video.close()
        ffmpeg_writer.Close()
