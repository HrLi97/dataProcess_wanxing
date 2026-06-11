# utils_inference
some tools to help deploy LM

# 0. Install
To install `utils_inference` as a pip package.
```bash
pip install -e .
```

---

```python
import utils_inference
```

# 1. Functions

## 1.1 time bench

| types | support |
| ----- | :-----: |
| CPU   |    ✅    |
| CUDA  |    ✅    |

```python
from utils_inference.bench.time_bench import test_time
from utils_inference.bench.cuda_bench import test_time_cuda

@test_time_cuda()
def myfunc1():
    # .......

@test_time()
def myfunc2():
    # .......
```

## 1.2 Image

### 1.2.1 Image Encode

* Input:
  
    | types         | support |
    | ------------- | :-----: |
    | numpy.ndarray |    ✅    |
    | torch.Tensor  |    ✅    |
    | RGB888        |    ✅    |
    | NCHW          |    ✅    |
    | NHWC          |    ✅    |

* Output:
  
    | types  | support |
    | ------ | :-----: |
    | RGB888 |    ✅    |
    | Other  |    ❌    |

```python
from utils_inference.image.image_writer import TensorSaveImage

TensorSaveImage.save_torch_tensor_jpg_nhwc(tensor, "data/test.jpg")
```

## 1.3 Log

* Input:
  
    | types      | support |
    | ---------- | :-----: |
    | Sigle Node |    ✅    |
    | Parallel   |    ✅    |

* Output:
  
    | types    | support |
    | -------- | :-----: |
    | Terminal |    ✅    |
    | File     |    ✅    |

Usage:

```python
from utils_inference.log.logger_parallel import FTSyncLoggerParallel
from utils_inference.log.logger import FTSyncLogger

# signal
with FTSyncLogger("test.log"):
    print("hello")

# parallel
with FTSyncLoggerParallel("test.log", False, 0):    # global_rank need set to real rank
    print("hello")

```

## 1.4 Video

* Input:
  
    | Tools      | support |
    | ---------- | :-----: |
    | coco       |    ✅    |
    | ffmpeg     |    ✅    |
    | decord     |    ✅    |
    | extensible |    ✅    |

* Output:
  
    | Tools     | support |
    | --------- | :-----: |
    | stream    |    ✅    |
    | no stream |    ✅    |

```python
from utils_inference.video.video_info_decord import VideoInfoDecord
from utils_inference.video.video_info_ffmpeg import VideoInfoFfmpeg
from utils_inference.video.video_info_coco import VideoInfoCoco

# use decord
video1 = VideoInfoDecord("test_video1.mp4")
width = video1.width
height = video1.height
during = video1.duration
data1 = video1.load_data_by_index(indexs=[0,1,3,5,6],device="cuda",format="NCHW")

# use ffmpeg
video2 = VideoInfoFfmpeg("test_video2.mp4")
width = video2.width
height = video2.height
during = video2.duration
data2 = video1.load_data_by_index(indexs=[0,1,2,3,4,5,6],device="cpu",format="NHWC")

# read coco
video3 = VideoInfoCoco("video3.json")
width = video3.width
height = video3.height
during = video3.duration
data3 = video3.load_data_by_index(indexs=[0,1,3,4,7,8,9],device="cpu",format="NHWC")
```

