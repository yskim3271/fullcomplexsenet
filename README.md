# FullComplex-SENet: Full Complex Spectrum Speech Enhancement Network

A PyTorch implementation of **FullComplex-SENet**, a speech enhancement model that operates on full complex-valued spectrogram representations. The model leverages complex-valued neural network operations and time-frequency dual-path processing for effective noise suppression.

## Features

- **Full Complex-valued Operations**: Native complex convolutions, normalization, and activations following Cauchy's integral rules
- **Time-Frequency Dual-Path Processing**: Separate processing paths for temporal and frequency dimensions using LKFCA blocks
- **Dense Dilated Convolutions**: Multi-scale feature extraction with dilated separable convolutions
- **MetricGAN Discriminator**: PESQ-guided adversarial training for perceptually optimized enhancement
- **Hydra Configuration**: Flexible experiment management with YAML-based configs
- **Multiple Dataset Support**: VoiceBank+DEMAND via HuggingFace or local files
- **Comprehensive Evaluation**: PESQ, STOI, CSIG, CBAK, COVL, and SegSNR metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/FullComplex-senet.git
cd FullComplex-senet

# Install dependencies
pip install torch torchaudio librosa hydra-core omegaconf
pip install pesq pystoi scipy numpy
pip install datasets  # for HuggingFace dataset support
pip install tensorboard psutil
```

### Training

```bash
# Train with default configuration
python train.py

# Train with HuggingFace VoiceBank-DEMAND dataset
python train.py use_huggingface=true

# Train with custom model configuration
python train.py model=fullcomplex
```

### Inference

```bash
# Run evaluation on test set
python test.py
```

## Model Architecture

### ComplexSENet

The core model consists of three main components:

```
Input Complex Spectrogram [B, F, T, 2]
            ↓
    ComplexDenseEncoder
    (Complex Conv2d + Dense Dilated Blocks)
            ↓
    Time-Frequency Blocks (×4)
    (LKFCA with Multi-scale Gated Convolutions)
            ↓
    ComplexMaskDecoder
    (Complex Mask Estimation with Bounded Activation)
            ↓
Output Enhanced Spectrogram [B, F, T, 2]
```

### Key Components

| Component | Description |
|-----------|-------------|
| `ComplexConv2d` | Complex-valued 2D convolution following Cauchy multiplication |
| `ComplexIN2d` | Complex instance normalization with shared affine parameters |
| `ComplexLeakyModReLU` | Magnitude-gated activation preserving phase information |
| `ComplexDSDDB` | Dense Separable Dilated Dense Block for multi-scale features |
| `TS_BLOCK` | Time-Frequency dual-path processing block |
| `LKFCA_Block` | Large Kernel Feature-wise Channel Attention block |
| `GCGFN` | Gated Convolution Gated Feed-forward Network |

### Model Variants

| Model | Config File | Parameters |
|-------|-------------|------------|
| ComplexSENet | `conf/model/fullcomplex.yaml` | `dense_channel=64, beta=2.0, num_tsblock=4` |

## Training Details

### Loss Functions

The model is trained with a combination of losses:

- **Complex Loss**: MSE between clean and estimated complex spectrograms
- **Consistency Loss**: Ensures STFT-iSTFT consistency
- **Metric Loss**: MetricGAN adversarial loss guided by PESQ scores

```python
loss_gen = loss_metric * 0.05 + loss_complex * 2 + loss_consistency * 0.1
```

### Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 5e-4 |
| Betas | (0.8, 0.99) |
| LR Decay | 0.99 (per epoch) |
| Batch Size | 2 |
| Epochs | 200 |

### STFT Configuration

| Parameter | Value |
|-----------|-------|
| n_fft | 400 |
| hop_size | 100 |
| win_size | 400 |
| compress_factor | 0.3 |

## Datasets

### VoiceBank+DEMAND

The model is trained and evaluated on the VoiceBank+DEMAND dataset:

- **Training**: 11,572 utterances from 28 speakers
- **Test**: 824 utterances from 2 speakers
- **Sampling Rate**: 16 kHz

#### Option 1: HuggingFace (Recommended)

```bash
python train.py use_huggingface=true
```

The dataset will be automatically downloaded from [JacobLinCool/VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k).

#### Option 2: Local Files

1. Prepare file lists in `VoiceBank+DEMAND/` directory:
   - `training_list.txt`
   - `test_list.txt`

2. File list format (pipe-separated):
```
utterance_id|/path/to/noisy.wav|/path/to/clean.wav
```

3. Run training:
```bash
python train.py use_huggingface=false
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management.

### Main Config (`conf/config.yaml`)

```yaml
# Dataset
segment: 32000
n_fft: 400
hop_size: 100
win_size: 400
compress_factor: 0.3

# Training
batch_size: 2
epochs: 200
lr: 5e-4
lr_decay: 0.99
optim: adamW

# Logging
log_interval: 500
validation_interval: 1000
evaluation_interval: 50
best_models_num: 5
```

### Model Config (`conf/model/fullcomplex.yaml`)

```yaml
model_lib: complexsenet
model_class: ComplexSENet

param:
  dense_channel: 64
  beta: 2.0
  num_tsblock: 4
```

### Override Examples

```bash
# Change batch size and learning rate
python train.py batch_size=4 lr=1e-4

# Use different model configuration
python train.py model=fullcomplex model.param.dense_channel=48

# Change output directory
python train.py hydra.run.dir=./experiments/exp1
```

## Project Structure

```
FullComplex-senet/
├── conf/
│   ├── config.yaml          # Main configuration
│   └── model/
│       └── fullcomplex.yaml # Model configuration
├── models/
│   ├── __init__.py
│   ├── complexsenet.py      # Main model architecture
│   └── discriminator.py     # MetricGAN discriminator
├── VoiceBank+DEMAND/        # Dataset file lists
│   ├── training_list.txt
│   └── test_list.txt
├── outputs/                 # Training outputs (auto-generated)
├── train.py                 # Training entry point
├── test.py                  # Testing script
├── evaluate.py              # Evaluation functions
├── enhance.py               # Enhancement/inference utilities
├── solver.py                # Training loop and logic
├── data.py                  # Dataset and data processing
├── utils.py                 # Utility functions
├── compute_metrics.py       # Metric computation (PESQ, STOI, etc.)
├── pcs400.py                # PCS400 processing
├── parse_list.py            # File list parsing utilities
└── run.sh                   # Training shell script
```

## Evaluation Metrics

The model is evaluated using standard speech enhancement metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| PESQ | Perceptual Evaluation of Speech Quality | -0.5 ~ 4.5 |
| STOI | Short-Time Objective Intelligibility | 0 ~ 1 |
| CSIG | Signal distortion (MOS-like) | 1 ~ 5 |
| CBAK | Background intrusiveness (MOS-like) | 1 ~ 5 |
| COVL | Overall quality (MOS-like) | 1 ~ 5 |
| SegSNR | Segmental Signal-to-Noise Ratio | dB |

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- torchaudio
- librosa
- hydra-core
- omegaconf
- pesq
- pystoi
- scipy
- numpy
- datasets (for HuggingFace)
- tensorboard
- psutil

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{fullcomplex-senet,
  author = {Yunsik Kim},
  title = {FullComplex-SENet: Full Complex Spectrum Speech Enhancement Network},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-username/FullComplex-senet}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- VoiceBank+DEMAND dataset: [Valentini et al.](https://datashare.ed.ac.uk/handle/10283/2791)
- Metric computation code adapted from [CMGAN](https://github.com/ruizhecao96/CMGAN)
- MetricGAN discriminator inspired by [MetricGAN+](https://github.com/speechbrain/speechbrain)

## Contact

- **Author**: Yunsik Kim
- **Affiliation**: POSTECH
- **Email**: [ys.kim@postech.ac.kr]
