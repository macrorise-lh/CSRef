# Configuration for HuggingFace dataset loading
from torchvision.transforms import transforms

from csref.config import LazyCall
from csref.datasets.dataset_speech_hf import HuggingFaceSpeechRefCOCODataSet

from .train import train

dataset = LazyCall(HuggingFaceSpeechRefCOCODataSet)(
    dataset="refcoco_speech",
    # dataset="refcoco+_speech", 
    # dataset="refcocog_speech",
    # dataset="srefface",
    # dataset="srefface+",
    # dataset="sreffaceg",

    max_durations=None,

    # HuggingFace specific parameters
    cache_dir="./hf_cache",  # Directory to cache downloaded datasets
    
    # Audio processing parameters
    use_trim=True,
    target_sample_rate=16000,

    # Original input image shape
    input_shape=[416, 416],
    flip_lr=False,

    # Basic transforms
    transforms=LazyCall(transforms.Compose)(
        transforms=[
            LazyCall(transforms.ToTensor)(),
            LazyCall(transforms.Normalize)(
                mean=train.data.mean,
                std=train.data.std,
            )
        ]
    ),

    # Candidate transforms
    candidate_transforms={
        # "RandAugment": RandAugment(2, 9),
        # "ElasticTransform": A.ElasticTransform(p=0.5),
        # "GridDistortion": A.GridDistortion(p=0.5),
        # "RandomErasing": transforms.RandomErasing(
        #     p = 0.3,
        #     scale = (0.02, 0.2),
        #     ratio=(0.05, 8),
        #     value="random",
        # )
    },

    # Dataset splits
    split="train",
)
