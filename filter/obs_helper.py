import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

try:
    from obs import ObsClient, GetObjectHeader, PutObjectHeader
    _HAS_OBS_SDK = True
except ImportError:
    _HAS_OBS_SDK = False


def add_obs_args(parser):
    """向 argparse.ArgumentParser 添加 OBS 相关参数。"""
    grp = parser.add_argument_group("OBS (华为云对象存储)")
    grp.add_argument("--obs", action="store_true", help="启用 OBS 模式：从 OBS 下载视频处理后上传结果")
    grp.add_argument("--obs_ak", default=os.environ.get("OBS_AK", ""), help="Access Key (或环境变量 OBS_AK)")
    grp.add_argument("--obs_sk", default=os.environ.get("OBS_SK", ""), help="Secret Key (或环境变量 OBS_SK)")
    grp.add_argument("--obs_endpoint", default=os.environ.get("OBS_ENDPOINT", ""), help="Endpoint URL")
    grp.add_argument("--obs_bucket", default=os.environ.get("OBS_BUCKET", ""), help="Bucket 名称 (下载+上传共用)")
    grp.add_argument("--obs_download_bucket", default="", help="下载 Bucket (为空则用 --obs_bucket)")
    grp.add_argument("--obs_upload_bucket", default="", help="上传 Bucket (为空则用 --obs_bucket)")
    grp.add_argument("--obs_upload_prefix", default="", help="上传路径前缀，如 'result/0622/video_filter'")
    grp.add_argument("--obs_tmp_dir", default="", help="本地临时目录 (默认系统 tmpdir)")
    grp.add_argument("--obs_download_workers", type=int, default=4, help="并发下载线程数")


class ObsIO:
    """OBS 下载/上传/清理管理器。当 --obs 未启用时所有操作为 no-op。"""

    def __init__(self, args):
        self.enabled = getattr(args, "obs", False)
        if not self.enabled:
            return

        if not _HAS_OBS_SDK:
            raise ImportError("OBS SDK 未安装，请运行: pip install esdk-obs-python")

        self.ak = args.obs_ak
        self.sk = args.obs_sk
        self.endpoint = args.obs_endpoint
        self.download_bucket = args.obs_download_bucket or args.obs_bucket
        self.upload_bucket = args.obs_upload_bucket or args.obs_bucket
        self.upload_prefix = args.obs_upload_prefix.strip("/")

        # 本地临时目录
        self.tmp_root = Path(args.obs_tmp_dir) if args.obs_tmp_dir else Path(tempfile.gettempdir()) / "obs_filter"
        self.tmp_root.mkdir(parents=True, exist_ok=True)

        if not all([self.ak, self.sk, self.endpoint, self.download_bucket]):
            raise ValueError("OBS 模式需要: --obs_ak, --obs_sk, --obs_endpoint, --obs_bucket")

    def _client(self):
        return ObsClient(access_key_id=self.ak, secret_access_key=self.sk, server=self.endpoint)

    # ── 下载 ──────────────────────────────────────────────────────────────

    def download(self, obs_key, local_path=None, retries=3):
        """从 OBS 下载文件到本地。返回本地路径。
        
        Args:
            obs_key: OBS 对象 key（如 'data/video/001.mp4'）
            local_path: 指定本地路径；为 None 则自动生成临时路径
            retries: 重试次数
        """
        if not self.enabled:
            return obs_key  # 非 OBS 模式，key 即为本地路径

        if local_path is None:
            # 保留原始文件名，加 pid 避免冲突
            basename = Path(obs_key).name
            local_dir = self.tmp_root / f"dl_{os.getpid()}"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = str(local_dir / basename)

        for attempt in range(retries):
            client = self._client()
            try:
                resp = client.getObject(
                    self.download_bucket, obs_key, local_path,
                    headers=GetObjectHeader()
                )
                if resp.status < 300:
                    return local_path
                print(f"  [OBS] 下载失败 ({attempt+1}/{retries}): {obs_key} status={resp.status}")
            except Exception:
                print(f"  [OBS] 下载异常 ({attempt+1}/{retries}): {obs_key}\n{traceback.format_exc()}")
            finally:
                client.close()
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))

        raise IOError(f"OBS 下载失败 (重试 {retries} 次): {obs_key}")

    # ── 上传 ──────────────────────────────────────────────────────────────

    def upload(self, local_path, obs_key=None, retries=3, metadata=None):
        """将本地文件上传到 OBS。返回上传后的 obs_key。
        
        Args:
            local_path: 本地文件路径
            obs_key: OBS 对象 key；为 None 则根据 upload_prefix + 文件名自动生成
            retries: 重试次数
        """
        if not self.enabled:
            return local_path  # 非 OBS 模式直接返回本地路径

        if obs_key is None:
            basename = Path(local_path).name
            obs_key = f"{self.upload_prefix}/{basename}" if self.upload_prefix else basename

        for attempt in range(retries):
            client = self._client()
            try:
                resp = client.putFile(
                    self.upload_bucket, obs_key, local_path,
                    metadata, PutObjectHeader()
                )
                if resp.status < 300:
                    return obs_key
                print(f"  [OBS] 上传失败 ({attempt+1}/{retries}): {obs_key} status={resp.status}")
            except Exception:
                print(f"  [OBS] 上传异常 ({attempt+1}/{retries}): {obs_key}\n{traceback.format_exc()}")
            finally:
                client.close()
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))

        raise IOError(f"OBS 上传失败 (重试 {retries} 次): {local_path} → {obs_key}")

    # ── 上传目录下所有文件 ────────────────────────────────────────────────

    def upload_dir(self, local_dir, obs_prefix=None):
        """递归上传本地目录到 OBS。返回 {local_path: obs_key} 映射。"""
        if not self.enabled:
            return {}
        local_dir = Path(local_dir)
        if obs_prefix is None:
            obs_prefix = self.upload_prefix
        result = {}
        for f in local_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(local_dir)
                obs_key = f"{obs_prefix}/{rel}" if obs_prefix else str(rel)
                self.upload(str(f), obs_key)
                result[str(f)] = obs_key
        return result

    # ── 清理 ──────────────────────────────────────────────────────────────

    def cleanup(self, *paths):
        """删除本地临时文件/目录。"""
        for p in paths:
            if not p:
                continue
            p = Path(p)
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception:
                pass

    def cleanup_tmp(self):
        """清理当前进程的所有临时文件。"""
        tmp_dir = self.tmp_root / f"dl_{os.getpid()}"
        self.cleanup(tmp_dir)

    # ── 上下文管理器 ──────────────────────────────────────────────────────

    class temp_download:
        """上下文管理器：自动下载 + 处理完后清理本地文件。
        
        Usage:
            with obs_io.temp_download(obs_key) as local_path:
                result = process(local_path)
        """
        def __init__(self, obs_io, obs_key):
            self.obs_io = obs_io
            self.obs_key = obs_key
            self.local_path = None

        def __enter__(self):
            self.local_path = self.obs_io.download(self.obs_key)
            return self.local_path

        def __exit__(self, *exc):
            if self.local_path:
                self.obs_io.cleanup(self.local_path)
            return False

    def make_obs_key(self, sub_dir, filename):
        """生成上传 obs_key: {upload_prefix}/{sub_dir}/{filename}"""
        parts = [self.upload_prefix, sub_dir, filename]
        return "/".join(p for p in parts if p)

    # ── 判断路径是否为 OBS key ────────────────────────────────────────────

    @staticmethod
    def is_obs_path(path):
        """简单判断：不以 / 开头的视为 OBS key。"""
        return path and not path.startswith("/")
