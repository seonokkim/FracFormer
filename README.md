# FracFormer: Semi-supervised Learning for Vertebrae and Fracture Classification on 3D Radiographs with Transformers
This repository is the official implementation of FracFormer: Semi-supervised Learning for Vertebrae and Fracture Classification on 3D Radiographs with Transformers. Our framework applies transformer-based models to detect vertebrae fractures, incorporating a Vision Transformer for spine detection and a Swin Transformer for fracture identification. This approach leverages the strengths of transformers in medical imaging to advance vertebral and fracture classification accuracy on 3D radiographic data.


<img src="https://github.com/user-attachments/assets/5a074c7a-4cea-460d-bfdb-06523ac54dc7" width="50%" />


![image](https://github.com/user-attachments/assets/b37d4a4f-c832-495e-ad3f-d19de4bb827a)

## Testing the Model

To evaluate the performance of the FracFormer model on test data, you can use the `test.py` script. Follow the steps below:

### 1. Ensure Required Files and Directories
- **Test Data**: Ensure the `./dataset/test.csv` file exists. This file should contain metadata about the test cases.
- **Test Images**: Ensure the test DICOM images are located in the directory specified by `Config.TEST_IMAGES_PATH`.
- **Checkpoints**: Ensure the model checkpoints are available in the directory specified by `Config.CHECKPOINTS_PATH`. Each model checkpoint file should match the names in `Config.MODEL_NAMES`.

### 2. Run the `test.py` Script
Use the following command to run the test script:

```bash
python test.py
```

## Work in Progress 🚧

This repository is currently under active development, with regular updates expected.
