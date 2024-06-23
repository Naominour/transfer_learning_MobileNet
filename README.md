# Transfer Learning with MobileNetV1

<div align="center">
  <img src="images\image.webp" alt="Alpaca Image" style="width:400px;">

  ![TensorFlow](https://img.shields.io/badge/Skill-TensorFlow-yellow)
  ![Transfer Learning](https://img.shields.io/badge/Skill-Transfer%20Learning-green)
  ![Fine-tuning](https://img.shields.io/badge/Skill-Fine%20tuning-orange)
  ![Image Data Augmentation](https://img.shields.io/badge/Skill-Image%20Data%20Augmentation-brightgreen)
  ![Python Programming](https://img.shields.io/badge/Skill-Python%20Programming-blue)

</div>

This project uses transfer learning with MobileNetV1 to create an Alpaca/Not Alpaca image classifier. The project demonstrates the application of transfer learning using a pre-trained Convolutional Neural Network (CNN) to classify images with high accuracy.

```bash
├── Transfer_learning_with_MobileNet_v1.ipynb     # Containing the code for training and evaluating the Alpaca/Not Alpaca classifier.
├── test_utils.py     # Python script with utility functions for testing the model.
├── dataset           # Directory containing the Alpaca/Not Alpaca images.
└── images
```

## Frameworks and Libraries
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange.svg?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.3.3-red.svg?style=flat&logo=keras)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-blue.svg?style=flat&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6.2-green.svg?style=flat&logo=matplotlib)

## Project Architecture
1. **Data Preparation**:
   - Loading and preprocessing the dataset, including `image augmentation`.

2. **Model Construction**:
   - Using `MobileNetV1` as the base model and adding custom layers on top.

3. **Training**:
   - Compiling and training the model with the prepared dataset.

4. **Evaluation**:
   - Assessing the performance of the model on the validation set.

5. **Testing**:
   - Using utility functions to test the model with new images.

## Key Features
- **Transfer Learning**: Utilises MobileNetV1, a pre-trained model, to leverage existing features learned from a large dataset.
- **Image Augmentation**: Applies random transformations to images to improve model generalization.
- **Custom Classification Layers**: Adds custom dense layers for the specific classification task.
- **Evaluation Metrics**: Provides detailed metrics to assess model performance.

## Usage
**Set Up Environment:**
```bash
pip install tensorflow matplotlib numpy
```
**Run the Notebook:**
- Open Transfer_learning_with_MobileNet_v1.ipynb in Jupyter Notebook and run the cells step-by-step to train and evaluate the model.

**Test the Model:**
- Use the functions provided in test_utils.py to test the trained model with new images.


## Results
The model was trained over 10 epochs, and the performance metrics were recorded at each epoch. 
**Training Accuracy:** 
- The training accuracy improved significantly over the epochs, starting from 66.03% in epoch 5 and reaching 94.66% by epoch 10.
**Validation Accuracy:**
- The validation accuracy also showed substantial improvement, with a peak at 96.92% in epoch 9.
**Loss:** 
- Both training and validation losses decreased over time, indicating that the model was learning effectively. However, a slight increase in validation loss in the last epoch suggests potential overfitting.

The final model achieved a high validation accuracy of approximately 96.92%, demonstrating its effectiveness in classifying Alpaca and Not Alpaca images.


<div align="center">
  <img src="images\result.png" style="width:400px;">

</div>