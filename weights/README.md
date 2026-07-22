---
license: cc-by-nc-4.0
tags:
  - computer-vision
  - 6d-pose-estimation
  - object-detection
  - robotics
  - foundationpose
library_name: foundationpose
---

# FoundationPose Model Weights

Pre-trained weights for [FoundationPose](https://github.com/NVlabs/FoundationPose) 6D object pose estimation model.

## Model Details

- **Refiner weights:** `2023-10-28-18-33-37/model_best.pth`
- **Scorer weights:** `2024-01-11-20-02-45/model_best.pth`
- **Source:** [Official FoundationPose release](https://github.com/NVlabs/FoundationPose)
- **Paper:** [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects (CVPR 2024)](https://arxiv.org/abs/2312.08344)

## Model Architecture

FoundationPose is a unified foundation model for 6D object pose estimation and tracking, supporting both:
- **Model-based setup**: Using CAD models
- **Model-free setup**: Using reference images (16-20 views)

## Files

```
.
├── 2023-10-28-18-33-37/
│   ├── config.yml
│   └── model_best.pth (refiner model)
└── 2024-01-11-20-02-45/
    ├── config.yml
    └── model_best.pth (scorer model)
```

## Usage

### Download Weights

```python
from huggingface_hub import snapshot_download

# Download all weights
weights_path = snapshot_download(
    repo_id="gpue/foundationpose-weights",
    local_dir="./weights"
)
```

### Use with FoundationPose Space

This model repository is designed to work with the [gpue/foundationpose](https://huggingface.co/spaces/gpue/foundationpose) Space.

Set environment variables:
```bash
FOUNDATIONPOSE_MODEL_REPO=gpue/foundationpose-weights
USE_HF_WEIGHTS=true
USE_REAL_MODEL=true
```

### Local Usage

```python
import torch
from pathlib import Path

# Load refiner
refiner_weights = torch.load("weights/2023-10-28-18-33-37/model_best.pth")

# Load scorer
scorer_weights = torch.load("weights/2024-01-11-20-02-45/model_best.pth")
```

## Performance

- **Accuracy**: State-of-the-art on BOP benchmark (as of 2024/03)
- **Speed**: Real-time capable with GPU acceleration
- **Generalization**: Works on novel objects without fine-tuning

## Citation

If you use these weights, please cite:

```bibtex
@inproceedings{wen2023foundationpose,
  title={FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects},
  author={Wen, Bowen and Yang, Wei and Kautz, Jan and Birchfield, Stan},
  booktitle={CVPR},
  year={2024}
}
```

## License

These weights are from the official FoundationPose release and are subject to NVIDIA's [Source Code License](https://github.com/NVlabs/FoundationPose/blob/main/LICENSE.txt).

**Key restrictions:**
- Non-commercial use only
- No redistribution of derivative works
- Academic and research purposes

## Related Resources

- **Paper**: https://arxiv.org/abs/2312.08344
- **Code**: https://github.com/NVlabs/FoundationPose
- **Project Page**: https://nvlabs.github.io/FoundationPose/
- **Inference Space**: https://huggingface.co/spaces/gpue/foundationpose

## Model Card

**Developed by:** NVIDIA Research (Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield)

**Model type:** Transformer-based 6D pose estimator

**Training data:** Large-scale synthetic dataset

**Intended use:** 6D object pose estimation and tracking for robotics and AR/VR applications

**Out-of-scope:** Commercial deployment (due to license restrictions)
