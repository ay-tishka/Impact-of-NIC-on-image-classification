# Impact of NIC on image classification

This project investigates how AI-based image compression—particularly neural image compression (NIC)—affects image classification performance. It evaluates the trade-off between compression efficiency and the retention of discriminative features across both general and task-specific datasets.

## Project Overview

Traditional compression methods (e.g., JPEG) often sacrifice subtle features important for classification. This project compares conventional methods with state-of-the-art NIC models, focusing on their impact on downstream tasks like medical image diagnosis and license plate recognition.

**Input image text detection**            |  **Compressed Image text detection**
:-------------------------:|:-------------------------:
![](https://github.com/ay-tishka/Impact-of-NIC-on-image-classification/blob/main/experiments/license%20analysis/uncomp_better_1.png)   |  ![](https://github.com/ay-tishka/Impact-of-NIC-on-image-classification/blob/main/experiments/license%20analysis/comp_worse_1.png)


## Key Contributions
- Compared Cheng2020-anchor and Cheng2020-attn NIC models against JPEG using PSNR, SSIM, VIF, and BPP.
- Analyzed performance degradation in classification on compressed medical images (ISIC2018 dataset).
- Explored surprising improvements in OCR accuracy for license plate recognition after compression.

## Datasets Used
- Kodak – Used for evaluating compression performance as a baseline.
- ISIC2018 – Skin lesion classification with high-resolution medical images.
- US License Plates – For OCR evaluation via PaddleOCR.
- Aro

# User guide

## CompressAI For Colab users:
```bash
pip install compressai
```
It will require the notebook to be restarted for once.

## CompressAI Local Installation Guide

```bash
python3 -m venv compressaienv
source compressaienv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio
python3 -m pip install matplotlib pandas
python3 -m pip install numpy scipy
python3 -m pip install ipykernel
python3 -m pip install jupyter
python3 -m ipykernel install --user --name=compressaienv
python3 -m pip install pybind11
module load compilers/gcc-8.3.0
```

Clone [CompressAI](https://github.com/InterDigitalInc/CompressAI) 
```
git clone https://github.com/InterDigitalInc/CompressAI compressai
cd compressai
pip install wheel
python3 setup.py bdist_wheel --dist-dir dist/
pip install dist/compressai-*.whl
```


## Experiments Summary
**0. Testing Cheng2020-Anchor and Cheng2020-Attn AI-based compression model for Kodak dataset**
- Task: Investigate models' performance
- Metric: PSNR, SSIM, VIF, BPP, MSE, MAE
- Dataset: https://www.kaggle.com/datasets/sherylmehta/kodak-dataset
- Notable finding: Rate-distortion trade-off by increasing the quality of compression model
- Running: `experiments/AICompression.ipynb`
  
**1.Skin Lesion Classification Summary**

- **Model**: DenseNet201 (pretrained + custom)
- **Task**: 9-class skin lesion classification
- **Data**: [ISIC Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic/data)  
  → Place `Train` and `Test` in `experiments/image_compression/ISIC-skin-cancer`
- **Weights**: [`skin_disease_model.h5`](https://www.kaggle.com/code/muhammadsamarshehzad/skin-cancer-classification-densenet201-99-acc/output) → Save in `experiments/`
- **Metrics**: Accuracy, F1, Cohen’s Kappa
- **Finding**: Up to 20% accuracy drop on compressed images
- **Note**: Run `AICompression.ipynb` first to generate degraded test images before using `DenseNet121_Aug_Clf (2).ipynb`


<!---
- Classifier: DenseNet201 (pretrained and custom-trained)
- Task: Skin lesion classification (9-class)
- Datasets: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic/data

  Make sure that Test and Train directories lie in `experiments/image_compression/ISIC-skin-cancer`
- Pretrained model for test in `experiments/AICompression.ipynb`: https://www.kaggle.com/code/muhammadsamarshehzad/skin-cancer-classification-densenet201-99-acc/output

  Make sure downloaded `skin_disease_model.h5` file lies in `experiments` directory.
- Metric: Accuracy, F1 score, Cohen’s Kappa
- Notable finding: Up to 20% drop in accuracy on compressed images vs original
- Remark: In order to run notebook `experiments/image_classification/DenseNet121_Aug_Clf (2).ipynb`, first run through Experiment 2 in `experiments/AICompression.ipynb` in order to generate degraded test images. Those images will lie in specific folder `decompressed` inside each label folder of test dataset.
-->



**2. OCR on Compressed Images**

- **Tool**: PaddleOCR (PP-OCRv3)
- **Setup**: US license plate dataset, Cheng2020-anchor & attn, Q1/Q3/Q6
- **Finding**: Q3 compression improved OCR in **~12.3%** of cases
- **Insight**: Anchor models outperformed; compression smoothed noise, boosted clarity


## Authors
QuocViet Pham, Thanakrit Lerdmatayakul, Arofenitra Rarivonjy, Tikhon Mavrin


## Related works




