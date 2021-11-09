from .dota import DotaDatasetv1
from .registry import DATASETS


@DATASETS.register_module
class UCASAODDataset(DotaDatasetv1):
    CLASSES = ('car', 'airplane',)