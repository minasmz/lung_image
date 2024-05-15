## Question

Dataset: The COVID-19 Radiography Database, which is a database of chest X-ray images and corresponding lung masks for 3616 COVID-19 positive, 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection), and 1345 Viral Pneumonia cases. The dataset can be downloaded from: [COVID-19 Radiography Database](https://drive.google.com/drive/folders/1RLBO1o7ngJrG57QCQLe-7Eg4UFQL__Od?usp=sharing)

We place the downloaded dataset in a folder names *"images"*

***Task:*** Design and train a deep learning classifier that takes an x-ray image as the input and assigns the input image to one of the four classes: COVID-19 positive, Normal, Lung Opacity, and Viral Pneumonia.

## Classification Result

200 samples are used for training, 40 for validation, and 200 for testing. The training stops when F1-score does not increase for 8 epoche (patience=8)

- **Test Accuracy:** 0.5350
- **Test F1 Score (weighted):** 0.3729
- **Test AUC (one-vs-rest):** 0.6667

***Task:*** Design and train an image segmentation network to segment the left and right lungs from the x-ray images.

## Segmentation Result

200 samples are used for training, 40 for validation, and 200 for testing. The training stops when mean_iou does not increase for 5 epoche (patience=5)

- **Test Mean IoU:** 0.1362
