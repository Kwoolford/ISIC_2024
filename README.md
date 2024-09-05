# Skin Cancer Classification with PyTorch

The goal of this project was to deepen my understanding of image classification techniques, model training, and evaluation within PyTorch. The focus was on building a deep learning model capable of distinguishing between malignant and non-malignant skin lesions using both image data and tabular metadata.

## Project Overview

- **Objective**: To develop an effective deep learning model for classifying skin lesions as malignant or non-malignant based on images and associated metadata.
- **Dataset**: The dataset comes from the ISIC 2024 Challenge, containing images of skin lesions along with metadata such as age, sex, anatomical site, and lesion size.
- **Tools & Frameworks**: PyTorch for model development, training, and evaluation; Pandas and Scikit-Learn for data preprocessing; and various Python libraries for data handling and visualization.

## Model Architecture

The model architecture consists of two primary components:

1. **Image Feature Extractor**:  
   A pretrained ResNet-50 model extracts features from skin lesion images. The final fully connected layer of ResNet-50 is removed to utilize the extracted feature vectors directly.
   
2. **Tabular Data Processor**:  
   A feedforward neural network processes the additional metadata (e.g., age, sex, anatomical site). This data is one-hot encoded and concatenated with numeric features for model input.

3. **Combined Model**:  
   The outputs from the Image Feature Extractor and the Tabular Data Processor are concatenated and passed through a series of fully connected layers for final classification.

## Key Features

- **Data Augmentation**: Applied random transformations such as horizontal and vertical flips, rotations, and color jittering to augment the image data.
- **Oversampling**: Addressed class imbalance by oversampling the minority class, ensuring that the model learns effectively from both classes.
- **Training**: The model was trained using Binary Cross-Entropy Loss with the Adam optimizer, leveraging both GPU and CPU environments for efficient computation.
- **Evaluation**: The model was evaluated using accuracy, precision, recall, and F1-score metrics, with a detailed confusion matrix analysis to identify strengths and areas for improvement.

## Results - Test Dataset

- **Image Count**: 100,265 total images  
  - Negative Images: 100,167  
  - Positive Images: 98  

- **Performance Metrics**:  
  - **Accuracy**: 99.95%  
  - **Precision**: 66.67%  
  - **Recall**: 95.92%  
  - **F1 Score**: 78.66%  

### Confusion Matrix

- **True Negatives (TN)**: 100,120  
- **False Positives (FP)**: 47  
- **False Negatives (FN)**: 4  
- **True Positives (TP)**: 94  

## Reflection on Results

I was very pleased with the model's performance, achieving an accuracy of 99.95%. When confronted with a malignant lesion, the model correctly identified it 95.92% of the time. While there is still room for improvement, especially in reducing the ~4% false negative rate, the model's ability to "find a needle in a haystack" 96% of the time is a commendable achievement for my first competition.

## Conclusion

This project provided valuable insights into image classification tasks using PyTorch, particularly in handling multimodal data (images + tabular data). Future steps involve experimenting with different architectures, hyperparameters, and data augmentation techniques to further improve the model's ability to accurately classify malignant lesions.
