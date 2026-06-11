from .logger import FTSyncLogger


class FTSyncLoggerParallel(FTSyncLogger):
    """
    In a parallel environment, automatically distinguish the printed contents of different nodes, and the master node can automatically roll-over and store log files.

    use example:

    with FTSyncLoggerParallel("test.log", False, 0):    # global_rank need set to real rank
        print("hello")
    """

    def __init__(
        self,
        filename: str = None,
        only_master_print: bool = False,
        global_rank: int = -1,
        console=None,
        file_log_ranks: list[int] = [0],
    ):
        self.rank = global_rank

        if filename is not None and self.rank in file_log_ranks:
            paths = filename.split(".")
            if len(paths) == 1:
                paths.append(f"rank{self.rank}")
            else:
                paths.insert(len(paths) - 1, f"rank{self.rank}")
            filename = ".".join(paths)
        else:
            filename = None

        super().__init__(
            filename=filename,
            prompt=f"[node-{self.rank}]",
            console=console,
            skip_print=only_master_print and self.rank != 0,
        )
