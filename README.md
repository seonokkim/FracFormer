# FracFormer: Semi-supervised Learning for Vertebrae and Fracture Classification on 3D Radiographs with Transformers

This repository is the official implementation of FracFormer: Semi-supervised Learning for Vertebrae and Fracture Classification on 3D Radiographs with Transformers. Our framework applies transformer-based models to detect vertebrae fractures, incorporating a Vision Transformer for spine detection and a Swin Transformer for fracture identification. This approach leverages the strengths of transformers in medical imaging to advance vertebral and fracture classification accuracy on 3D radiographic data. The full thesis is available [here](https://dcollection.korea.ac.kr/srch/srchDetail/000000270509).

## Overview

### FracFormer Architecture
The architecture consists of:
1. **Vertebrae Network**: Predicts vertebra visibility labels (C1–C7).
2. **Fracture Network**: Predicts fracture probabilities using pseudo-labels from the Vertebrae Network.

<p align="center">
  <img src="figures/figure_5_Overview of FracFormer.png" width="800">
</p>

### Components
#### 1. Vertebrae Network
- Uses Vision Transformers (ViT) for vertebra visibility prediction.
<p align="center">
  <img src="figures/figure_6_ Vertebrae Network with ViT .png" width="700">
</p>

#### 2. Fracture Network
- Employs Swin Transformers for detecting fractures.
<p align="center">
  <img src="figures/figure_9_Fracture Network with Swin Transformer.png" width="700">
</p>

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/seonokkim/FracFormer.git
   cd FracFormer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up directories:
   ```bash
   mkdir dataset models figures utils
   ```

---

## Dataset Preparation

1. Download the dataset from the [RSNA Cervical Spine Fracture Detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/data).
   Place the dataset into the `dataset` directory with the following structure:
   ```
   dataset/
   ├── train_images/
   ├── test_images/
   ├── train.csv
   ├── test.csv
   ```

2. Preprocess the dataset:
   ```bash
   python dataset/dataset.py
   ```

---

## Training

### 1. Train Vertebrae Network
The first stage predicts vertebra visibility using Vision Transformers:
```bash
python models/fracformer.py
```
This step:
- Trains the **VertebraeNet**.
- Generates pseudo-labels for vertebra visibility (C1–C7).

### 2. Train Fracture Network
The second stage detects fractures using Swin Transformers:
```bash
python models/fracformer.py
```
This step:
- Trains the **FractureNet** using the Vertebrae Network predictions.

---

## Testing

1. Ensure trained models are saved in the `models/checkpoints` directory:
   ```
   models/checkpoints/
   ├── vertebraenet_fold0.tph
   ├── vertebraenet_fold1.tph
   ├── fracturenet_fold0.tph
   ├── fracturenet_fold1.tph
   ```

2. Run the inference script:
   ```bash
   python test.py
   ```

3. Results:
   - Generates predictions for fractures.
   - Outputs classification reports and metrics like AUC-ROC.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This work builds upon the RSNA Cervical Spine Fracture Detection dataset and leverages cutting-edge Transformer architectures (ViT and Swin Transformers).

---

