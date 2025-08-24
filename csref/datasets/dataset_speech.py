# coding=utf-8
# Copyright 2025 The CSRef Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dataset module for speech-based referring expression comprehension in the CSRef project.

This module implements a dataset class for handling speech data with reference annotations,
supporting various datasets including RefCOCO_speech variants and SRefFACE variants.
The dataset provides functionality to load and preprocess audio-visual data for training
and evaluating models that understand referring expressions in speech.

Key features:
- Loading and preprocessing speech audio with resampling and trimming
- Loading and preprocessing images with bounding box annotations
- Support for multiple datasets with different annotation formats
- Data augmentation transformations for training
- Efficient data loading with PyTorch Dataset interface

Classes:
    SpeechRefCOCODataSet: Main dataset class for speech-based referring expression comprehension
"""

import json
import os
import random

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.utils.data as Data

from csref.utils.distributed import is_main_process
from .utils import label2yolobox


class SpeechRefCOCODataSet(Data.Dataset):
    """
    Dataset class for speech-based referring expression comprehension tasks.
    
    This class handles loading and preprocessing of speech audio and corresponding
    image annotations for various referring expression datasets. It supports
    multiple datasets with different annotation formats and provides functionality
    for data augmentation and preprocessing.
    
    Attributes:
        split (str): Dataset split ('train', 'val', 'test', or combinations)
        dataset (str): Name of the dataset being used
        audio_root (str): Root directory for audio files
        target_sample_rate (int): Target sample rate for audio resampling
        speakers (list): List of available speakers for audio selection
        use_trim (bool): Whether to trim silence from audio
        max_durations (float): Maximum duration for audio clips
        speech_refs_anno (list): List of speech reference annotations
        image_path (str): Path to image directory
        input_shape (tuple): Input shape for images
        flip_lr (bool): Whether to apply random left-right flipping
        candidate_transforms (dict): Available data transformations
        transforms: Applied transformations
        data_size (int): Size of the dataset
    """
    def __init__(self,
                 ann_path,
                 image_path,
                 audio_root,
                 # mask_path,
                 input_shape,
                 speakers,
                 flip_lr,
                 transforms,
                 candidate_transforms,
                 max_durations=None,
                 split="train",
                 dataset="refcoco_speech",
                 use_trim=True,
                 target_sample_rate=16000,
                 # only_people=False
                 ):
        """
        Initialize the SpeechRefCOCODataSet.
        
        Args:
            ann_path (dict): Dictionary mapping dataset names to annotation file paths
            image_path (dict): Dictionary mapping dataset names to image directory paths
            audio_root (str): Root directory for audio files
            input_shape (tuple): Target shape for input images (height, width)
            speakers (list): List of available speaker identifiers
            flip_lr (bool): Whether to apply random left-right flipping during training
            transforms: Image transformations to apply
            candidate_transforms (dict): Dictionary of candidate transformations for augmentation
            max_durations (float, optional): Maximum duration in seconds for audio clips
            split (str, optional): Dataset split ('train', 'val', 'test', or combinations)
            dataset (str, optional): Name of the dataset to use
            use_trim (bool, optional): Whether to trim silence from audio clips
            target_sample_rate (int, optional): Target sample rate for audio resampling
        """
        super(SpeechRefCOCODataSet, self).__init__()
        self.split = split
        
        # Validate dataset name
        assert dataset in ['refcoco_speech', 'refcoco+_speech', 'refcocog_speech', 'srefface', 'srefface+', 'sreffaceg']
        self.dataset = dataset

        # Store audio-related parameters
        self.audio_root = audio_root
        self.target_sample_rate = target_sample_rate
        self.speakers = speakers
        self.use_trim = use_trim
        self.max_durations = max_durations

        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        # Load annotation data from JSON file
        stat_refs_list = json.load(open(ann_path[dataset], 'r'))
        total_refs_list = []

        # Handle multiple splits (e.g., 'train+val')
        splits = split.split('+')

        # Collect annotations for specified splits
        speech_refs_anno = []
        self.speech_refs_anno = []

        for split_ in splits:
            speech_refs_anno += stat_refs_list[split_]

        self.speech_refs_anno = speech_refs_anno

        # Extract all references (currently unused but preserved)
        refs = []

        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)

        for split in total_refs_list:
            for ann in total_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)

        # Set image path based on dataset
        self.image_path = image_path[dataset]
        
        # Store input shape for image preprocessing
        self.input_shape = input_shape

        # Enable flipping only for training split
        self.flip_lr = flip_lr if split == 'train' else False

        # Define run data size
        self.data_size = len(self.speech_refs_anno)

        if is_main_process():
            print(' ========== Dataset size:', self.data_size)
        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        if is_main_process():
            print('Finished!')
            print('')

        # Set up transformations based on split
        if split == 'train':
            self.candidate_transforms = candidate_transforms
        else:
            self.candidate_transforms = {}

        self.transforms = transforms


    def get_audio_by_sent_id(self, sent_id):
        """
        Load and preprocess audio file by sentence ID.
        
        This method loads an audio file corresponding to the given sentence ID,
        randomly selects a speaker, resamples the audio to the target sample rate,
        and optionally trims silence from the beginning and end.
        
        Args:
            sent_id (str): Sentence ID identifying the audio file to load
            
        Returns:
            numpy.ndarray: Preprocessed audio waveform as a float32 array
        """
        # Randomly select a speaker from the available speakers
        speaker = self.speakers[np.random.choice(len(self.speakers))]
        # Construct the full path to the audio file
        path = os.path.join(self.audio_root, speaker, f"{sent_id}.wav")
        # Read audio file and get original sample rate
        wav, origin_sample_rate = sf.read(path, dtype="float32")
        # Resample audio to target sample rate (default: 16000Hz)
        resampled_wav = librosa.resample(y=wav, orig_sr=origin_sample_rate, target_sr=self.target_sample_rate)
        # Remove silence from the beginning and end of the audio if enabled
        if self.use_trim:
            resampled_wav, _ = librosa.effects.trim(resampled_wav)
        return resampled_wav

    def load_audio(self, idx):
        """
        Load audio data for a given dataset index.
        
        This method randomly selects one of the sentence IDs associated with the
        given index, loads the corresponding audio, and optionally truncates it
        to the maximum duration if specified.
        
        Args:
            idx (int): Index of the item in the dataset
            
        Returns:
            numpy.ndarray: Audio waveform as a float32 array, possibly truncated
        """
        # Get all sentence IDs associated with this dataset item
        sent_ids = self.speech_refs_anno[idx]["sent_ids"]
        # Randomly select one sentence ID from the available options
        sent_id = sent_ids[np.random.choice(len(sent_ids))]

        # Load and preprocess the audio file
        audio = self.get_audio_by_sent_id(sent_id)

        # Truncate audio to maximum duration if specified
        if self.max_durations is not None:
            # Calculate maximum number of frames to keep
            n_kept_frames = self.max_durations * self.target_sample_rate
            if len(audio) > n_kept_frames:
                # Truncate audio from the beginning
                audio = audio[0: n_kept_frames]

        return audio

    # def preprocess_info(self, img, mask, box, iid, aid, lr_flip=False):
    def preprocess_info(self, img, box, iid, aid, lr_flip=False):
        """
        Preprocess image and bounding box information.
        
        This method resizes the image while maintaining aspect ratio, pads it to
        the target input size, and adjusts the bounding box coordinates accordingly.
        The processed image is centered in a square canvas with gray padding.
        
        Args:
            img (numpy.ndarray): Input image as a BGR numpy array
            box (numpy.ndarray): Bounding box coordinates in [x, y, width, height] format
            iid (int): Image ID
            aid (int): Annotation ID
            lr_flip (bool, optional): Whether left-right flip was applied. Defaults to False.
            
        Returns:
            tuple: Processed image, adjusted bounding box, and image info tuple
                - sized (numpy.ndarray): Resized and padded image
                - sized_box (numpy.ndarray): Adjusted bounding box in YOLO format
                - info_img (tuple): Image information (h, w, nh, nw, dx, dy, iid, aid)
        """
        # Get original image dimensions
        h, w, _ = img.shape
        # Get target image size (assuming square input)
        imgsize = self.input_shape[0]
        # Calculate aspect ratio
        new_ar = w / h
        
        # Determine new dimensions while maintaining aspect ratio
        if new_ar < 1:  # Portrait orientation
            nh = imgsize
            nw = nh * new_ar
        else:  # Landscape orientation
            nw = imgsize
            nh = nw / new_ar
        nw, nh = int(nw), int(nh)

        # Calculate padding to center the image
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

        # Resize image while maintaining aspect ratio
        img = cv2.resize(img, (nw, nh))
        # Create a square canvas with gray padding (127 = mid-gray)
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        # Place the resized image in the center of the canvas
        sized[dy:dy + nh, dx:dx + nw, :] = img
        # Store image information for bounding box transformation
        info_img = (h, w, nh, nw, dx, dy, iid, aid)

        # Convert bounding box to YOLO format with padding adjustments
        sized_box = label2yolobox(box, info_img, self.input_shape[0], lrflip=lr_flip)
        # return sized, sized_mask, sized_box, info_img
        return sized, sized_box, info_img

    def load_img_feats(self, idx):
        """
        Load image and bounding box features for a given dataset index.
        
        This method loads the image file corresponding to the dataset index and
        extracts the appropriate bounding box based on the dataset type. Different
        datasets use different field names for bounding boxes (bbox for RefCOCO_speech
        variants, fbox for SRFFace variants).

        Args:
            idx (int): Index of the item in the dataset
            
        Returns:
            tuple: Image, bounding box, and image ID
                - image (numpy.ndarray): Loaded image as BGR numpy array
                - box (numpy.ndarray): Bounding box coordinates
                - iid (int): Image ID
        """
        # Construct the image file path using COO format
        img_path = os.path.join(self.image_path, 'COCO_train2014_%012d.jpg' % self.speech_refs_anno[idx]['iid'])
        # Load the image using OpenCV
        image = cv2.imread(img_path)

        # mask = None

        # box = np.array([self.speech_refs_anno[idx]['bbox']])

        # Handle different dataset types with different bounding box field names
        if self.dataset in ['refcoco_speech', 'refcoco+_speech', 'refcocog_speech']:
            # RefCOCO_speech variants use 'bbox' field
            box = np.array([self.speech_refs_anno[idx]['bbox']])
        elif self.dataset in ['srefface', 'srefface+', 'sreffaceg']:
            # SRFFace variants use 'fbox' field
            box = np.array([self.speech_refs_anno[idx]['fbox']])
        return image, box, self.speech_refs_anno[idx]['iid']

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        This method loads and preprocesses the audio and image data for the given index,
        applies data augmentation transformations if in training mode, and returns the
        processed data as tensors suitable for model input.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            tuple: Processed audio, image, bounding boxes, and image information
                - audio_iter (numpy.ndarray): Processed audio waveform
                - transformed_image: Transformed image tensor
                - box_iter (torch.Tensor): Processed bounding box in YOLO format
                - gt_box_iter (torch.Tensor): Original bounding box
                - info_iter (numpy.ndarray): Image information tuple
        """
        # Load audio data for the given index
        audio_iter = self.load_audio(idx)

        # Get sentence IDs and randomly select one
        sent_ids = self.speech_refs_anno[idx]["sent_ids"]
        sent_id = sent_ids[np.random.choice(len(sent_ids))]

        # Load image features (image and bounding box)
        # image_iter, mask_iter, gt_box_iter, mask_id, iid = self.load_img_feats(idx)
        # image_iter, gt_box_iter, mask_id, iid = self.load_img_feats(idx)
        image_iter, gt_box_iter, iid = self.load_img_feats(idx)

        # Convert image from BGR to RGB format
        image_iter = cv2.cvtColor(image_iter, cv2.COLOR_BGR2RGB)
        ops = None

        # Randomly select a data augmentation transformation if available
        if len(list(self.candidate_transforms.keys())) > 0:
            ops = random.choices(list(self.candidate_transforms.keys()), k=1)[0]

        # Apply the selected transformation (except RandomErasing which is applied later)
        if ops is not None and ops != 'RandomErasing':
            image_iter = self.candidate_transforms[ops](image=image_iter)['image']

        # Apply random left-right flip if enabled and in training mode
        flip_box = False
        if self.flip_lr and random.random() < 0.5:
            image_iter = image_iter[::-1]  # Flip image horizontally
            flip_box = True

        # Preprocess image and bounding box with resizing and padding
        image_iter, box_iter, info_iter = self.preprocess_info(image_iter, gt_box_iter.copy(),
                                                                          iid, sent_id, flip_box)

        # Return processed data as tensors
        return \
            audio_iter, \
            self.transforms(image_iter), \
            torch.from_numpy(box_iter).float(), \
            torch.from_numpy(gt_box_iter).float(), \
            np.array(info_iter)

    def __len__(self):
        """
        Return the size of the dataset.
        
        This method returns the number of items in the dataset, which is stored
        in the data_size attribute set during initialization.
        
        Returns:
            int: Number of items in the dataset
        """
        return self.data_size

    def shuffle_list(self, list):
        """
        Shuffle a list in place.
        
        This method takes a list and shuffles its elements randomly using the
        random.shuffle function. The shuffling is done in place, meaning the
        original list is modified.
        
        Args:
            list (list): The list to be shuffled
            
        Returns:
            None: The list is shuffled in place
        """
        random.shuffle(list)
