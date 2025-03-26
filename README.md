# Impact of NIC on image classification

This project investigates how AI-based image compression—particularly neural image compression (NIC)—affects image classification performance. It evaluates the trade-off between compression efficiency and the retention of discriminative features across both general and task-specific datasets.

## 📌 Overview

Traditional compression methods (e.g., JPEG) often sacrifice subtle features important for classification. This project compares conventional methods with state-of-the-art NIC models, focusing on their impact on downstream tasks like medical image diagnosis and license plate recognition.

## 🧠 Key Contributions
- Compared Cheng2020-anchor and Cheng2020-attn NIC models against JPEG using PSNR, SSIM, VIF, and BPP.
- Analyzed performance degradation in classification on compressed medical images (ISIC2018 dataset).
- Explored surprising improvements in OCR accuracy for license plate recognition after compression.

## 📊 Datasets
- Kodak – Used for evaluating compression performance as a baseline.
- ISIC2018 – Skin lesion classification with high-resolution medical images.
- US License Plates – For OCR evaluation via PaddleOCR.

## 🧪 Experiments Summary
**0. Testing Cheng2020-Anchor and Cheng2020-Attn AI-based compression model for Kodak dataset
- Task: Investigate models' performance
- Metric: PSNR, SSIM, VIF, BPP, MSE, MAE
- Dataset: https://www.kaggle.com/datasets/sherylmehta/kodak-dataset
- Notable finding: Rate-distortion trade-off by increasing the quality of compression model
- Running: `experiments/AICompression.ipynb`
  
**1. Classification**
- Classifier: DenseNet201 (pretrained and custom-trained)
- Task: Skin lesion classification (9-class)
- Datasets: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic/data

  Make sure that Test and Train directories lie in `experiments/image_compression/ISIC-skin-cancer`
- Pretrained model for test in `experiments/AICompression.ipynb`: https://www.kaggle.com/code/muhammadsamarshehzad/skin-cancer-classification-densenet201-99-acc/output

  Make sure downloaded `skin_disease_model.h5` file lies in `experiments` directory.
- Metric: Accuracy, F1 score, Cohen’s Kappa
- Notable finding: Up to 20% drop in accuracy on compressed images vs original
- Remark: In order to run notebook `experiments/image_classification/DenseNet121_Aug_Clf (2).ipynb`, first run through Experiment 2 in `experiments/AICompression.ipynb` in order to generate degraded test images. Those images will lie in specific folder `decompressed` inside each label folder of test dataset.

**2. OCR on Compressed Images**
- OCR Tool: PaddleOCR (PP-OCRv3 pipeline)
- Finding: Compression (Q3) improved OCR in ~12.3% of cases
- Hypothesis: Compression smooths noise and enhances character clarity
