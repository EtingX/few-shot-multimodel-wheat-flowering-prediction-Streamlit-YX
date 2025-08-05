# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.34"

from ultralytics_YX.data.explorer.explorer import Explorer
from ultralytics_YX.models import RTDETR, SAM, YOLO, YOLOWorld, YOLOv10
from ultralytics_YX.models.fastsam import FastSAM
from ultralytics_YX.models.nas import NAS
from ultralytics_YX.utils import ASSETS, SETTINGS as settings
from ultralytics_YX.utils.checks import check_yolo as checks
from ultralytics_YX.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
    "YOLOv10"
)
