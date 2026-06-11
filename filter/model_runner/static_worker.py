"""占位 Worker，当对应模型被禁用或加载失败时使用。"""


class StaticScoreWorker:
    def __init__(self, result):
        self.result = result

    def score(self, *args, **kwargs):
        return dict(self.result)

    def close(self):
        pass
