# 🎤 Contrastive Semantic Alignment for Speech Referring Expression Comprehension (CSRef)

![CSRef Logo](https://img.shields.io/badge/CSRef-v1.0-blue) ![Python](https://img.shields.io/badge/python-3.9.23-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red)

This repository contains the implementation of the approach described in the paper "CSRef: Contrastive Semantic Alignment for Speech Referring Expression Comprehension". 🚀


## 📋 Project Overview

### What is CSRef?

CSRef is a deep learning framework designed to comprehend referring expressions in speech and localize the corresponding objects in images. The framework employs a two-stage training approach:

1. **CSA Stage**: A pretraining stage that learns to align speech and text semantics through contrastive learning. It leverages the structured semantic space of text to guide the representation learning of raw speech.
2. **SREC Stage**: The main training stage that leverages the speech encoder from the CSA stage to perform referring expression comprehension by aligning speech with visual features.

### Key Features and Capabilities

- **Two-stage training approach**: First learns speech-text alignment, then applies it to speech-visual tasks
- **Multi-modal fusion**: Integrates speech and visual modalities effectively
- **Flexible architecture**: Supports various speech encoders and visual backbones

### Potential Applications and Use Cases

- **Human-computer interaction**: Enabling natural language control of computer vision systems
- **Robotic vision**: Allowing robots to understand and locate objects based on verbal descriptions


## 🛠️ Installation Instructions

### Prerequisites

- **Python**: 3.9.23 (tested with this version)
- **CUDA**: 12.6 or higher (for GPU support)
- **PyTorch**: 2.8 or higher
- **Operating System**: Linux (tested on Ubuntu 22.04)

### Step-by-Step Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/macrorise-lh/CSRef.git
   cd CSRef
   ```

2. **Create a conda virtual environment**

   ```bash
   conda create -n csref python=3.9
   conda activate csref
   ```

3. **Install PyTorch**

   ```bash
   # For CUDA 12.6
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```

4. **Install dependencies from requirements.txt**

   ```bash
   pip install -r requirements.txt
   ```


## 💾 Data Preparation

Before training, you need to download and prepare the required datasets:

### Speech Referring Expressions Annotations

We provide two methods to obtain the speech referring expressions annotations:

#### Method 1: Automatic Download from [Hugging Face](https://huggingface.co/collections/lihong-huang/speech-referring-expression-comprehension-srec-68a97ed74ea0b45b56dcc4f9)

The simplest way is to use the Hugging Face dataset integration. When you run training with the `_hf` configuration files, the datasets will be automatically downloaded:

```bash
# Example: This will automatically download RefCOCO speech dataset from Hugging Face
CUDA_VISIBLE_DEVICES=0 PORT=23451 bash tools/train_speech.sh configs/csref_refcoco_speech_hf.py 1
```

Available datasets with automatic download:
- `configs/csref_refcoco_speech_hf.py` - RefCOCO_speech dataset
- `configs/csref_refcoco+_speech_hf.py` - RefCOCO+_speech dataset  
- `configs/csref_refcocog_speech_hf.py` - RefCOCOg_speech dataset
- `configs/csref_srefface_hf.py` - SRRefFace dataset
- `configs/csref_srefface+_hf.py` - SRRefFace+ dataset
- `configs/csref_sreffaceg_hf.py` - SRRefFaceG dataset

#### Method 2: Manual Download from [ModelScope](https://modelscope.cn/datasets/lihongh/CSRef_data)

Alternatively, you can manually download the complete dataset and pre-trained model weights:

```bash
# Download from ModelScope
# Follow the link: https://modelscope.cn/datasets/lihongh/CSRef_data
# Extract files to the appropriate directories in the data folder following the Project Structure
```

**Advantages of Manual Download:**
- Complete offline access to all datasets
- Faster training startup (no download time)

**Data Organization:** 📁 After manual download, organize the files according to the directory structure shown in the [Project Structure](#-project-structure) section.

### 📦 Additional Required Datasets and Weights

1. **Download [LibriSpeech ASR dataset](https://www.openslr.org/12/) for CSA pre-training**

   ```bash
   # Create directory
   mkdir -p data/audios
   
   # Download and extract LibriSpeech
   cd data/audios
   # train sets - 960 hours
   wget https://www.openslr.org/resources/12/train-other-500.tar.gz
   wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
   wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
   # dev sets
   wget https://www.openslr.org/resources/12/dev-other.tar.gz
   wget https://www.openslr.org/resources/12/dev-clean.tar.gz
   
   tar -xvzf train-other-500.tar.gz
   tar -xvzf train-clean-360.tar.gz
   tar -xvzf train-clean-100.tar.gz
   tar -xvzf dev-other.tar.gz
   tar -xvzf dev-clean.tar.gz
   cd ../../
   ```

2. **Download [COCO images](https://cocodataset.org/#download)**

   ```bash
   # Create directory
   mkdir -p data/images
   
   # Download and extract COCO train2014 images
   cd data/images
   wget http://images.cocodataset.org/zips/train2014.zip
   unzip train2014.zip
   rm train2014.zip
   cd ../../
   ```

3. **Download pre-trained encoders**

   ```bash
   # Create directory
   mkdir -p data/weights

   # Download BERT and Wav2Vec2 models
   cd data/weights
   git lfs install
   git clone https://huggingface.co/facebook/wav2vec2-base
   git clone https://huggingface.co/google-bert/bert-base-uncased
   cd ../../

   # Download CSA pretrained Speech Encoder
   wget https://modelscope.cn/datasets/lihongh/CSRef_data/resolve/master/data/weights/CSA_speech_encoder.pth

   # Download pretrained visual backbone CSPDarkNet 
   # following https://github.com/luogen1996/SimREC/blob/main/DATA_PRE_README.md#pretrained-weights
   # or https://modelscope.cn/datasets/lihongh/CSRef_data/resolve/master/data/weights/cspdarknet_coco.pth
   ```


## 🚀 Usage Examples

### 🏋️ Training

#### 🎯 CSA Stage Training

The CSA stage learns semantic alignment between speech and text modalities:

```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=0 PORT=23450 bash tools/train_CSA.sh configs/csref_CSA_librispeech.py 1

# Multi-GPU training (4 GPUs)
CUDA_VISIBLE_DEVICES=1,2,3,4 PORT=23450 bash tools/train_CSA.sh configs/csref_CSA_librispeech.py 4
```

Key parameters:

- `CUDA_VISIBLE_DEVICES`: Specifies which GPUs to use
- `PORT`: Port number for distributed training
- `configs/csref_CSA_librispeech.py`: Configuration file for CSA stage
- `4`: Number of GPUs to use

#### 🔍 SREC Stage Training

The SREC stage uses the trained speech encoder to perform referring expression comprehension:

```bash
# Single GPU training on RefCOCO+
CUDA_VISIBLE_DEVICES=0 PORT=23451 bash tools/train_speech.sh configs/csref_refcoco_speech.py 1
```

You can also train on other datasets by using different configuration files:

- `configs/csref_refcoco_speech.py` / `configs/csref_refcoco_speech_hf.py`: For RefCOCO_speech dataset
- `configs/csref_refcoco+_speech.py` / `configs/csref_refcoco+_speech_hf.py`: For RefCOCO+_speech dataset
- `configs/csref_refcocog_speech.py` / `configs/csref_refcocog_speech_hf.py`: For RefCOCOg_speech dataset
- `configs/csref_srefface.py` / `configs/csref_srefface_hf.py`: For SRRefFace dataset
- `configs/csref_srefface+.py` / `configs/csref_srefface+_hf.py`: For SRRefFace+ dataset
- `configs/csref_sreffaceg.py` / `configs/csref_sreffaceg_hf.py`: For SRRefFaceG dataset

**Note:** Use configuration files with `_hf` suffix for automatic Hugging Face dataset download, or without `_hf` suffix if you have manually downloaded and organized the data.

### 📊 Evaluation

```bash
# Evaluate SREC model
# Using automatically downloaded Hugging Face datasets
CUDA_VISIBLE_DEVICES=0 PORT=23451 bash tools/eval_speech.sh configs/csref_refcoco_speech_hf.py 1 data/weights/csref_refcoco_speech.pth

# Using manually downloaded datasets  
CUDA_VISIBLE_DEVICES=0 PORT=23451 bash tools/eval_speech.sh configs/csref_refcoco_speech.py 1 data/weights/csref_refcoco_speech.pth
```

**Note:** Make sure to use the corresponding configuration file (`_hf` or non-`_hf`) that matches your data preparation method.


## 📂 Project Structure

The CSRef project is organized as follows:

```
CSRef/
├── configs/                  # Configuration files
│   ├── csref_*.py            # Main configuration files for different datasets
│   └── common/               # Common configuration modules
│       ├── dataset_*.py      # Dataset configurations
│       ├── optim.py          # Optimizer configurations
│       ├── train.py          # Training configurations
│       └── models/           # Model configurations
├── csref/                    # Core library code
│   ├── config/               # Configuration management
│   ├── datasets/             # Dataset handling
│   ├── layers/               # Neural network layers
│   ├── models/               # Model definitions
│   │   ├── backbones/        # Visual backbones
│   │   ├── heads/            # Detection heads
│   │   ├── losses/           # Loss functions
│   │   ├── speech_encoders/  # Speech encoders
│   │   ├── text_encoder/     # Text encoders
│   │   └── utils/            # Model utilities
│   ├── scheduler/            # Learning rate schedulers
│   └── utils/                # Utility functions
├── tools/                    # Training and evaluation scripts
│   ├── train_*.py            # Training scripts
│   ├── train_*.sh            # Training shell scripts
│   ├── eval_*.py             # Evaluation scripts
│   └── eval_*.sh             # Evaluation shell scripts
├── data/                     # Data directory (to be created by user)
│   ├── audios/               # Audio files
│   │   ├── LibriSpeech/
│   │   ├── refcoco_speech/
│   │   ├── refcoco+_speech/
│   │   └── refcocog_speech/
│   ├── images/               # Image files
│   │   └── train2014/        # COCO train2014 images
│   ├── anns/                 # Annotation files
│   │   ├── general_object/   # General object annotations (RefCOCO/RefCOCO+/RefCOCOg)
│   │   └── face_centric/     # Face-centric annotations (SRRefFace series)
│   ├── hf_cache/             # Hugging Face dataset cache (auto-created)
│   └── weights/              # Pre-trained model weights
│       ├── wav2vec2-base/    # Wav2Vec2 base model
│       ├── bert-base-uncased/  # BERT base uncased model
│       ├── CSA_speech_encoder.pth  # Pre-trained CSA speech encoder
│       └── csref_*.pth       # Trained CSRef model weights (if downloaded manually)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore                # Git ignore rules
```


## License Information

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement

Thanks a lot for the nicely organized code from the following repos: 

- [SimREC](https://github.com/luogen1996/SimREC)
