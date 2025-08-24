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
LibriSpeech Dataset for Contrastive Semantic Alignment (CSA) Training.

LibriSpeech is a large-scale corpus of approximately 1000 hours of read English speech
derived from audiobooks in the LibriVox project. It is widely used for speech recognition
research and has become a standard benchmark in the field. The dataset contains speech
from multiple speakers with diverse accents and reading styles, making it ideal for
training robust speech-text alignment models.

In the CSA training process, this dataset serves as the primary source of paired
audio-text samples that are used to train the model to learn meaningful representations
by contrasting positive (matching) audio-text pairs with negative (non-matching) pairs.
The dataset implementation includes features for audio preprocessing, data loading
optimization, and flexible split handling to support various training configurations.

Key features of this implementation:
- Efficient loading of audio files and transcripts
- Audio preprocessing including resampling, silence trimming, and duration limiting
- Support for train, validation, and test splits
- Optimized for multi-process data loading in distributed training environments
- Flexible configuration options to adapt to different training requirements

For more information about LibriSpeech, see: https://www.openslr.org/12
"""

import os

import librosa
from torch.utils.data import Dataset
from csref.utils.distributed import is_main_process

import soundfile as sf


class LibriSpeechDataset(Dataset):
    """
    PyTorch Dataset implementation for LibriSpeech audio data and transcripts.
    
    This class provides a comprehensive interface for the LibriSpeech corpus, designed
    specifically for training Contrastive Semantic Alignment (CSA) models. The dataset
    contains approximately 1000 hours of English speech from audiobooks, read by
    multiple speakers with diverse accents and styles.
    
    The implementation handles the complex directory structure of LibriSpeech and
    provides efficient access to audio-transcript pairs with various preprocessing
    options. It's optimized for both single-process and distributed training scenarios.
    
    Key features:
    1. Flexible split support (train, validation, test) with predefined configurations
    2. Audio preprocessing pipeline including resampling, silence trimming, and duration limiting
    3. Efficient file loading and transcript pairing with minimal memory overhead
    4. Support for distributed training with process-safe data loading
    5. Configurable preprocessing parameters to adapt to different model requirements
    
    Directory structure:
    LibriSpeech/
        |--train-clean-100/    # 100 hours of clean speech for training
        |--train-clean-360/    # 360 hours of clean speech for training
        |--train-other-500/    # 500 hours of more varied speech for training
        |--dev-clean/          # Clean speech for validation
        |--dev-other/          # More varied speech for validation
        |--test-clean/         # Clean speech for testing
        |--test-other/         # More varied speech for testing
    
    Each split contains directories organized by speaker ID and chapter ID:
        split/speaker_id/chapter_id/
            |--*.flac          # Audio files
            |--*.txt           # Transcript files (audio_id transcript_text)
    
    Usage:
        # Create a dataset instance for training
        dataset = LibriSpeechDataset(
            root_dir='/path/to/LibriSpeech',
            train_split='train',
            max_durations=10.0,  # Limit audio to 10 seconds
            use_trim=True,        # Remove silence
            target_sample_rate=16000
        )
        
        # Access a sample
        waveform, transcript = dataset[0]
        
        # Use with DataLoader for batch processing
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    def __init__(self, root_dir, train_split='train', max_durations=None,
                 use_trim=True,
                 target_sample_rate=16000):
        """
        Initialize the LibriSpeech dataset with specified configuration parameters.
        
        This constructor sets up the dataset by configuring the data splits, preprocessing
        options, and loading the file paths and transcripts. The initialization process
        involves determining which subsets of LibriSpeech to include based on the split type,
        and then scanning the directory structure to build lists of audio files and their
        corresponding transcripts.
        
        Args:
            root_dir (str): Root directory of the LibriSpeech dataset. This should be the
                path that contains the various split directories (train-clean-100, dev-clean, etc.).
            train_split (str): Dataset split to use. Valid options are:
                - 'train': Includes train-clean-100, train-clean-360, and train-other-500
                - 'val': Includes dev-clean, dev-other, test-clean, and test-other
                - 'test': Includes only test-clean and test-other
                Defaults to 'train'.
            max_durations (float, optional): Maximum duration in seconds for audio clips.
                If specified, audio clips longer than this value will be truncated to
                the specified duration. This is useful for ensuring uniform sequence lengths
                during training. If None, no duration limiting is applied. Defaults to None.
            use_trim (bool): Whether to apply silence trimming to audio clips using
                librosa.effects.trim(). This removes leading and trailing silence from
                the audio. 
            target_sample_rate (int): Target sample rate for audio resampling. All audio
                files will be resampled to this rate regardless of their original sample rate.
                Common values include 16000 (16kHz) for speech processing. Defaults to 16000.
                
        Raises:
            ValueError: If train_split is not one of the valid options ('train', 'val', 'test').
            
        Note:
            The initialization process scans the entire directory structure for the specified
            splits, which may take some time for large datasets. The file paths and transcripts
            are stored in memory for efficient access during training.
        """
        # Store configuration parameters for later use during audio loading
        self.root_dir = root_dir  # Base directory of the LibriSpeech dataset
        self.target_sample_rate = target_sample_rate  # Target sample rate for all audio (e.g., 16000 Hz)
        self.use_trim = use_trim  # Whether to remove silence from audio clips
        self.max_durations = max_durations  # Maximum duration in seconds for audio clips (None for no limit)

        # Define dataset splits based on the specified split type
        # LibriSpeech organizes data into different subsets based on recording quality and purpose
        if train_split == 'train':
            # Training split includes all training subsets (~960 hours total)
            # - train-clean-100: 100 hours of clean speech
            # - train-clean-360: 360 hours of clean speech
            # - train-other-500: 500 hours of more varied speech with different conditions
            self.splits = [
                'train-clean-100',
                'train-clean-360',
                'train-other-500'
            ]
        elif train_split == 'val':
            # Validation split includes development and test subsets
            self.splits = [
                'dev-clean',
                'dev-other',
                # 'test-clean',
                # 'test-other'
            ]
        elif train_split == 'test':
            # Test split includes only the official test subsets
            self.splits = [
                'test-clean',
                'test-other'
            ]
        else:
            raise ValueError(f"Invalid split: {train_split}. Must be 'train', 'val', or 'test'.")

        # Initialize lists to store file paths and transcripts
        # These lists maintain a one-to-one correspondence: speech_files[i] matches transcripts[i]
        self.speech_files = []    # List of full paths to audio files (.flac)
        self.transcripts = []     # List of text transcripts corresponding to each audio file

        # Load data from the specified splits
        self._load_data()

        # Print dataset information if this is the main process
        if is_main_process():
            print(f'====== Dataset {train_split} loaded! ======')
            print('Max durations:', max_durations, '\n',
                  'Trimmed:', use_trim, '\n',
                  'Target sample rate:', target_sample_rate, '\n',
                  'num of samples:', len(self.speech_files)
                  )
            print(f'====== Dataset {train_split} loaded! ======')

    def _load_data(self):
        """
        Load audio files and transcripts from the LibriSpeech dataset directory structure.
        
        This method performs a comprehensive traversal of the LibriSpeech directory structure
        to build lists of audio file paths and their corresponding transcripts. The loading
        process is optimized by first collecting all transcripts for a chapter into a dictionary,
        then matching audio files to their transcripts based on file names.
        
        The directory structure is organized hierarchically as:
        root_dir/split/speaker_id/chapter_id/
            - *.txt files contain transcripts (one file per chapter with multiple lines)
            - *.flac files contain audio data (one file per utterance)
        
        Each transcript file contains multiple lines in the format:
            "audio_id transcript_text"
        where audio_id corresponds to the name of the audio file (without extension).
        
        The method follows a two-pass approach for each chapter:
        1. First pass: Read all transcript files and build a dictionary mapping audio IDs
           to their transcript texts
        2. Second pass: Process all audio files and pair them with their corresponding
           transcripts using the dictionary
        
        This approach minimizes disk I/O by reading each transcript file only once,
        regardless of the number of audio files in the chapter. 
              
        Note:
            This method is called during initialization and may take significant time
            for large datasets due to the extensive directory traversal and file reading.
        """
        # Iterate through each dataset split (train, val, or test subsets)
        for split in self.splits:
            # Construct the full path to the current split directory
            split_path = os.path.join(self.root_dir, split)

            # Navigate through the hierarchical directory structure: split -> speaker -> chapter
            for speaker in os.listdir(split_path):
                speaker_path = os.path.join(split_path, speaker)
                # Skip non-directory files that might be present in the speaker directory
                if not os.path.isdir(speaker_path):
                    continue
                    
                # Process each chapter directory within the speaker directory
                for chapter in os.listdir(speaker_path):
                    chapter_path = os.path.join(speaker_path, chapter)
                    # Skip non-directory files that might be present in the chapter directory
                    if not os.path.isdir(chapter_path):
                        continue

                    # Create a dictionary to store transcripts for this chapter
                    # Key: audio file ID (without extension), Value: transcript text
                    transcripts_dict = {}

                    # First pass: Load all transcript files for this chapter
                    # This approach minimizes disk I/O by reading each transcript file only once
                    for file in os.listdir(chapter_path):
                        if file.endswith('.txt'):
                            file_path = os.path.join(chapter_path, file)

                            # Read the transcript file and parse each line
                            with open(file_path, 'r') as f:
                                for line in f:
                                    # Each line format: "audio_id transcript_text"
                                    # Split on first space only to preserve transcript text with spaces
                                    key, value = line.strip().split(' ', 1)
                                    transcripts_dict[key] = value
                    
                    # Second pass: Load all audio files and pair them with their transcripts
                    for file in os.listdir(chapter_path):
                        if file.endswith('.flac'):
                            file_path = os.path.join(chapter_path, file)
                            # Store the full path to the audio file for later loading
                            self.speech_files.append(file_path)

                            # Extract the audio ID from the filename (remove .flac extension)
                            audio_id = file.split('.')[0]
                            # Retrieve the corresponding transcript from our dictionary
                            transcript = transcripts_dict[audio_id]
                            self.transcripts.append(transcript)

    def load_audio(self, idx):
        """
        Load and preprocess an audio file by index with full preprocessing pipeline.
        
        This method retrieves an audio file from the dataset using the provided index
        and applies the complete preprocessing pipeline as configured during initialization.
        The preprocessing includes loading the audio, resampling to the target sample rate,
        trimming silence (if enabled), and limiting duration (if specified).
        
        The method serves as a high-level interface that orchestrates the audio loading
        and preprocessing steps, delegating the actual file loading to get_audio_by_path()
        and applying additional duration-based processing if needed.
        
        Args:
            idx (int): Index of the audio file to load. This index corresponds to the
                position in self.speech_files list, which is populated during initialization.
                Must be in the range [0, len(self)-1].
                
        Returns:
            numpy.ndarray: Preprocessed audio waveform as a 1D array of float32 values.
                The waveform has been resampled to target_sample_rate, trimmed of silence
                (if use_trim=True), and truncated to max_durations (if specified).
                The array shape is (num_samples,) where num_samples depends on the
                audio duration and preprocessing settings.
            
        Note:
            This method is called internally by __getitem__() when accessing dataset
            samples. It can also be called directly for custom audio loading scenarios.
            
            The actual audio loading and basic preprocessing is delegated to
            get_audio_by_path(), which handles the file I/O and resampling/trimming
            operations. This method adds duration limiting on top of those operations.
        """
        # Retrieve the full path to the audio file from our pre-loaded list
        speech_file = self.speech_files[idx]

        # Load and apply basic preprocessing (resampling and silence trimming)
        audio = self.get_audio_by_path(speech_file)

        # Apply duration limiting if specified during initialization
        if self.max_durations is not None:
            # Calculate the maximum number of samples to keep based on duration and sample rate
            n_kept_frames = int(self.max_durations * self.target_sample_rate)
            
            # Truncate the audio if it exceeds the maximum duration
            if len(audio) > n_kept_frames:
                audio = audio[0: n_kept_frames]

        return audio

    def get_audio_by_path(self, path):
        """
        Load and preprocess an audio file from a specified file path.
        
        This method handles the low-level audio loading and preprocessing operations
        for a single audio file. It reads the audio data using soundfile, resamples
        it to the target sample rate if necessary, and applies silence trimming if
        enabled. This method serves as the core audio processing function used by
        load_audio() for dataset samples.
        
        The preprocessing pipeline includes:
        1. Audio loading with soundfile.read() to get raw waveform and sample rate
        2. Resampling to the target sample rate using librosa.resample()
        3. Optional silence trimming using librosa.effects.trim()
        
        Args:
            path (str): Full path to the audio file to be loaded. 
                
        Returns:
            numpy.ndarray: Preprocessed audio waveform as a 1D array of float32 values.
                The waveform has been resampled to target_sample_rate and trimmed of
                silence (if use_trim=True). The array shape is (num_samples,) where
                num_samples depends on the audio duration and preprocessing settings.
        """
        # Load the audio file using soundfile with explicit float32 dtype
        wav, origin_sample_rate = sf.read(path, dtype="float32")

        # Resample the audio to the target sample rate if necessary
        resampled_wav = librosa.resample(y=wav, orig_sr=origin_sample_rate, target_sr=self.target_sample_rate)

        # Apply silence trimming if specified during initialization
        # This removes leading and trailing silence, focusing on the actual speech content
        if self.use_trim:
            # librosa.effects.trim returns (trimmed_audio, (start_index, end_index))
            # We only need the trimmed audio, so we discard the index information
            resampled_wav, _ = librosa.effects.trim(resampled_wav)
            
        return resampled_wav

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of audio files in the dataset. This equals the length of
                self.speech_files and self.transcripts lists, which are guaranteed
                to have the same length.
                
        Note:
            This value is determined during initialization when _load_data() scans
            the directory structure and builds the file lists. It remains constant
            for the lifetime of the dataset instance.
        """
        return len(self.speech_files)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset by index with full preprocessing applied.
        
        This method implements the standard Python indexing interface for the dataset,
        allowing access to individual audio-transcript pairs. It loads the audio file
        specified by the index, applies the complete preprocessing pipeline (resampling,
        silence trimming, and duration limiting), and returns both the processed waveform
        and its corresponding text transcript.
        
        This method is the primary access point used by PyTorch's DataLoader when
        creating batches for training. It ensures that all returned samples have been
        consistently preprocessed according to the dataset configuration.
        
        Args:
            idx (int): Index of the sample to retrieve. 
                
        Returns:
            tuple: A tuple containing two elements:
                - waveform (numpy.ndarray): Preprocessed audio waveform as a 1D array
                  of float32 values. The waveform has been resampled to target_sample_rate,
                  trimmed of silence (if use_trim=True), and truncated to max_durations
                  (if specified). The array shape is (num_samples,).
                - transcript (str): Text transcript corresponding to the audio file,
                  containing the spoken text that matches the audio content.
            
        Note:
            This method delegates the actual audio loading and preprocessing to
            the load_audio() method, which in turn uses get_audio_by_path() for
            the low-level audio processing. This layered approach allows for
            modular design and code reuse.
            
            The transcript is retrieved directly from the pre-loaded self.transcripts
            list, which was populated during initialization. This avoids repeated
            file I/O operations for transcript data.
        """
        # Load and preprocess the audio waveform
        waveform = self.load_audio(idx)
        # Get the corresponding transcript
        transcript = self.transcripts[idx]

        return waveform, transcript
