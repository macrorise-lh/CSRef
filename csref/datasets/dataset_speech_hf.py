"""
Updated dataset module for loading CSRef datasets from Hugging Face Hub.

This module provides compatibility with the existing training code while loading
data from Hugging Face datasets instead of local files.
"""

import random
from typing import Dict, List, Optional, Union

import cv2
import librosa
import numpy as np
import torch
import torch.utils.data as Data
from datasets import load_dataset
from PIL import Image

from csref.utils.distributed import is_main_process
from .utils import label2yolobox


class HuggingFaceSpeechRefCOCODataSet(Data.Dataset):
    """
    Dataset class for speech-based referring expression comprehension using Hugging Face datasets.
    
    This class loads datasets from Hugging Face Hub instead of local files,
    maintaining compatibility with existing training pipelines.
    """
    
    # Mapping from dataset names to HuggingFace repo names
    HF_REPO_MAPPING = {
        'refcoco_speech': 'lihong-huang/refcoco-speech',
        'refcoco+_speech': 'lihong-huang/refcoco-plus-speech', 
        'refcocog_speech': 'lihong-huang/refcocog-speech',
        'srefface': 'lihong-huang/srefface',
        'srefface+': 'lihong-huang/srefface-plus',
        'sreffaceg': 'lihong-huang/sreffaceg'
    }
    
    def __init__(self,
                 dataset: str = "refcoco_speech",
                 split: str = "train",
                 input_shape: tuple = (416, 416),
                 flip_lr: bool = True,
                 transforms=None,
                 candidate_transforms: dict = None,
                 max_durations: Optional[float] = None,
                 use_trim: bool = True,
                 target_sample_rate: int = 16000,
                 cache_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize the HuggingFace-based dataset.
        
        Args:
            dataset: Name of the dataset to load
            split: Dataset split ('train', 'val', 'test', 'testA', 'testB' or combinations)
            input_shape: Target shape for input images
            flip_lr: Whether to apply random left-right flipping
            transforms: Image transformations to apply
            candidate_transforms: Dictionary of candidate transformations for augmentation
            max_durations: Maximum duration in seconds for audio clips
            use_trim: Whether to trim silence from audio clips
            target_sample_rate: Target sample rate for audio resampling
            cache_dir: Directory to cache downloaded datasets
        """
        super(HuggingFaceSpeechRefCOCODataSet, self).__init__()
        
        # Validate dataset name
        assert dataset in self.HF_REPO_MAPPING, f"Dataset {dataset} not supported"
        
        self.dataset_name = dataset
        self.split = split
        self.input_shape = input_shape
        self.target_sample_rate = target_sample_rate
        self.use_trim = use_trim
        self.max_durations = max_durations
        
        # Load dataset from Hugging Face Hub
        repo_name = self.HF_REPO_MAPPING[dataset]
        
        if is_main_process():
            print(f"Loading dataset {dataset} from {repo_name}...")
        
        try:
            self.hf_dataset = load_dataset(repo_name, cache_dir=cache_dir)
        except Exception as e:
            if is_main_process():
                print(f"Error loading dataset from {repo_name}: {e}")
                print("Falling back to local dataset loading...")
            # Fallback to original dataset class
            from .dataset_speech import SpeechRefCOCODataSet
            raise ImportError("HuggingFace dataset not available, please use original dataset class")
        
        # Handle multiple splits (e.g., 'train+val')
        splits = split.split('+')
        
        # Collect data from specified splits
        self.data = []
        for split_name in splits:
            if split_name in self.hf_dataset:
                split_data = self.hf_dataset[split_name]
                self.data.extend(split_data)
                if is_main_process():
                    print(f"Loaded {len(split_data)} samples from split '{split_name}'")
            else:
                if is_main_process():
                    print(f"Warning: Split '{split_name}' not found in dataset")
        
        if len(self.data) == 0:
            raise ValueError(f"No data found for splits: {splits}")
        
        # Set up transformations
        self.flip_lr = flip_lr if split == 'train' else False
        self.transforms = transforms
        self.candidate_transforms = candidate_transforms if split == 'train' else {}
        
        # Store data size
        self.data_size = len(self.data)
        
        if is_main_process():
            print(f"Dataset size: {self.data_size}")
    
    def load_and_process_audio(self, audios_data: list, sent_ids: list) -> np.ndarray:
        """
        Load and process audio data from HuggingFace format.
        Randomly selects one audio file similar to original training logic.
        
        Args:
            audios_data: List of audio data dictionaries from HuggingFace dataset
            sent_ids: List of sentence IDs for reference
            
        Returns:
            Processed audio waveform as numpy array
        """
        # Randomly select one audio file (maintaining original training behavior)
        if isinstance(audios_data, list) and len(audios_data) > 0:
            selected_audio = audios_data[np.random.choice(len(audios_data))]
        else:
            # Fallback for single audio
            selected_audio = audios_data
        
        # Get audio array and sampling rate
        audio_array = np.array(selected_audio["array"], dtype=np.float32)
        orig_sr = selected_audio["sampling_rate"]
        
        # Resample if needed
        if orig_sr != self.target_sample_rate:
            audio_array = librosa.resample(
                y=audio_array, 
                orig_sr=orig_sr, 
                target_sr=self.target_sample_rate
            )
        
        # Trim silence if enabled
        if self.use_trim:
            audio_array, _ = librosa.effects.trim(audio_array)
        
        # Truncate to maximum duration if specified
        if self.max_durations is not None:
            max_frames = int(self.max_durations * self.target_sample_rate)
            if len(audio_array) > max_frames:
                audio_array = audio_array[:max_frames]
        
        return audio_array
    
    def preprocess_info(self, img: np.ndarray, box: np.ndarray, 
                       iid: int, aid: int, lr_flip: bool = False) -> tuple:
        """
        Preprocess image and bounding box information.
        
        Args:
            img: Input image as numpy array
            box: Bounding box coordinates in [x, y, width, height] format
            iid: Image ID
            aid: Annotation ID  
            lr_flip: Whether left-right flip was applied
            
        Returns:
            Tuple of processed image, adjusted bounding box, and image info
        """
        # Get original image dimensions
        h, w, _ = img.shape
        imgsize = self.input_shape[0]
        
        # Calculate aspect ratio and new dimensions
        new_ar = w / h
        if new_ar < 1:  # Portrait
            nh = imgsize
            nw = int(nh * new_ar)
        else:  # Landscape
            nw = imgsize  
            nh = int(nw / new_ar)
        
        # Calculate padding
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2
        
        # Resize image
        img = cv2.resize(img, (nw, nh))
        
        # Create padded image
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        sized[dy:dy + nh, dx:dx + nw, :] = img
        
        # Store image info
        info_img = (h, w, nh, nw, dx, dy, iid, aid)
        
        # Convert bounding box to YOLO format
        sized_box = label2yolobox(box, info_img, imgsize, lrflip=lr_flip)
        
        return sized, sized_box, info_img
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of processed audio, image, bounding boxes, and image information
        """
        # Get data sample
        sample = self.data[idx]
        
        # Load and process audio (randomly select from multiple audios)
        audio_iter = self.load_and_process_audio(sample["audios"], sample["sent_ids"])  # TODO 
        
        # Get image
        image_iter = sample["image"]
        if isinstance(image_iter, Image.Image):
            image_iter = np.array(image_iter)
        
        # Convert RGB to BGR for compatibility with OpenCV operations
        if image_iter.shape[-1] == 3:
            image_iter = cv2.cvtColor(image_iter, cv2.COLOR_RGB2BGR)
        
        # Get bounding box based on dataset type
        if self.dataset_name in ['refcoco_speech', 'refcoco+_speech', 'refcocog_speech']:
            gt_box_iter = np.array([sample["bbox"]], dtype=np.float32)
        else:  # sRefFACE variants
            gt_box_iter = np.array([sample["fbox"]], dtype=np.float32)
        
        # Get image and sentence IDs
        iid = sample["image_id"]
        sent_id = sample["sent_ids"][0] if sample["sent_ids"] else 0
        
        # Convert back to RGB for transformations
        image_iter = cv2.cvtColor(image_iter, cv2.COLOR_BGR2RGB)
        
        # Apply random transformations
        ops = None
        if len(list(self.candidate_transforms.keys())) > 0:
            ops = random.choices(list(self.candidate_transforms.keys()), k=1)[0]
        
        if ops is not None and ops != 'RandomErasing':
            image_iter = self.candidate_transforms[ops](image=image_iter)['image']
        
        # Apply random flip
        flip_box = False
        if self.flip_lr and random.random() < 0.5:
            image_iter = image_iter[:, ::-1]  # Flip horizontally
            flip_box = True
        
        # Preprocess image and bounding box
        image_iter, box_iter, info_iter = self.preprocess_info(
            image_iter, gt_box_iter.copy(), iid, sent_id, flip_box
        )
        
        return (
            audio_iter,
            self.transforms(image_iter),
            torch.from_numpy(box_iter).float(),
            torch.from_numpy(gt_box_iter).float(), 
            np.array(info_iter)
        )
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.data_size
    
    def shuffle_list(self, list_data: list) -> None:
        """Shuffle a list in place."""
        random.shuffle(list_data)


# # For backward compatibility, provide an alias
# SpeechRefCOCODataSet = HuggingFaceSpeechRefCOCODataSet
