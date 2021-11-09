from .dota import DotaDatasetv1
from .registry import DATASETS


@DATASETS.register_module
class HRSC2016Dataset(DotaDatasetv1):
    CLASSES = ('ship',)