import time
from collections import deque
import torch.distributed as dist
import sys


def speed_str(speed: float, threshold=1.0) -> str:
    """

    :param threshold: not less than threshold will use s/iter
    """
    if speed >= threshold:
        return f"{speed:.2f} s/iter"
    else:
        if speed < 0.001:
            return f">1000 iter/s"
        return f"{1.0/speed:.2f} iter/s"


class ProgressBar:
    """
    generate sub progress-bar
    """

    def __init__(
        self,
        total: float = 1,
        desc: str = "Sub-Task: ",
        rich_progress=None,
        avg_time_step: float = 10,
        enable: bool = None,
    ):
        """
        :param total: totoal progress
        :type total: float
        :param desc: tip-text to show in head of progress-bar
        :type desc: str
        :param rich_progress: use rich progress bar if give, or use tqdm instead
        :type rich_progress: rich.progress.Progress
        :param avg_time_step: use last `avg_time_step` step to get avg speed and show in tail
        :type avg_time_step: float
        :param enable: make no affect to code if enable is False. None means `enable=((not dist.is_initialized()) or (dist.get_rank() == 0))`
        :type enable: bool
        """
        self.__enable__ = (
            enable
            if enable is not None
            else ((not dist.is_initialized()) or (dist.get_rank() == 0))
        )
        self.__time_deque__ = deque()  # deque[(int,time)]
        self.__avg_time_step__ = avg_time_step

        self.__rich_progress__ = rich_progress
        self.__num_step__ = total
        self.__progress_bar__ = None
        self.__tip__ = desc

        self.__start_time__ = time.time()
        self.__sum_step__ = 0

    def __enque_current_time__(self):
        if not self.__enable__:
            return

        if self.__avg_time_step__ < 1:
            return
        self.__time_deque__.append((self.__sum_step__, time.time()))

        while len(self.__time_deque__) > 2:
            if (
                self.__time_deque__[-1][0] - self.__time_deque__[1][0]
                >= self.__avg_time_step__
            ):
                self.__time_deque__.popleft()
            else:
                break

    def __get_speed__(self) -> float:
        if not self.__enable__:
            return

        if self.__sum_step__ < 1:
            return 0

        self.__enque_current_time__()
        return (self.__time_deque__[-1][1] - self.__time_deque__[0][1]) / (
            self.__time_deque__[-1][0] - self.__time_deque__[0][0]
        )

    def __get_total_speed__(self) -> float:
        if not self.__enable__:
            return

        if self.__sum_step__ < 1 or len(self.__time_deque__) < 2:
            return 0

        return (self.__time_deque__[-1][1] - self.__start_time__) / self.__sum_step__

    def __enter__(self) -> "ProgressBar":
        if not self.__enable__:
            print(f"hide progress-bar")
            return self

        if self.__num_step__ < 1:
            return self

        if self.__rich_progress__ is None:
            from tqdm import tqdm

            self.__progress_bar__ = tqdm(
                total=self.__num_step__,
                desc=self.__tip__,
            )
        else:
            from rich.progress import Progress

            assert isinstance(
                self.__rich_progress__, Progress
            ), "only rich.progress.Progress is supported"

            self.__progress_bar__ = self.__rich_progress__.add_task(
                f"[cyan]{self.__tip__}", total=self.__num_step__, speed=""
            )
        self.__start_time__ = time.time()
        self.__sum_step__ = 0
        self.__enque_current_time__()
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        if not self.__enable__:
            return self

        if self.__progress_bar__ is None:
            return

        if self.__rich_progress__ is None:
            self.__progress_bar__.close()
        else:
            from rich.progress import Progress

            if self.__get_total_speed__() > 0:
                print(
                    f"process end: step={self.__sum_step__:.2f}, e2e avg speed={speed_str(speed=self.__get_total_speed__())}, latest avg speed={speed_str(speed=self.__get_speed__())}"
                )

            self.__rich_progress__: Progress
            self.__rich_progress__.remove_task(self.__progress_bar__)
        self.__start_time__ = None
        self.__sum_step__ = 0
        self.__time_deque__.clear()

    def update(self, n: float = 1, completed: float = None, show_speed=True):
        if not self.__enable__:
            return self

        if completed is not None:
            n = completed - self.__sum_step__

        if self.__progress_bar__ is None:
            return
        self.__sum_step__ += n

        self.__enque_current_time__()

        if self.__rich_progress__ is not None:
            # update progress bar
            self.__rich_progress__.update(
                self.__progress_bar__,
                advance=n if completed is None else None,
                completed=completed,
                speed=(
                    speed_str(speed=self.__get_speed__(), threshold=0.1)
                    if show_speed
                    else ""
                ),
            )
        else:
            new_n = completed or (n + self.__progress_bar__.n)

            # get max .000, and format 2.0 to 2
            if abs(round(new_n) - new_n) < 10e-6:
                new_n = round(new_n)

            new_n = int(new_n * 1000) / 1000
            if (new_n % 1) < 10e-6:
                new_n = int(new_n)

            self.__progress_bar__.n = new_n
            self.__progress_bar__.refresh()
