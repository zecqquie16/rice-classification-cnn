# rice-classification-cnn
Rice grain classification using neural networks in MATLAB.

dataset used for rice grain classification available on : "https://www.kaggle.com/code/rezasemyari/rice-image-classification-cnn-0-99"
# Rice Image Classification using Convolutional Neural Networks (MATLAB)

This project focuses on building a **Convolutional Neural Network (CNN)** to perform **image classification on a large rice image dataset**.  
The objective is to correctly classify rice images into **five distinct classes** using a **transfer learning approach** based on pre-trained CNN architectures available in the MATLAB Deep Learning Toolbox.

The work was completed as part of an assessed university coursework in deep learning.

---

## Project Objectives
- Design a CNN-based image classification system for rice images
- Apply transfer learning using pre-trained convolutional neural networks
- Improve model stability and accuracy through systematic optimisation
- Evaluate model performance using accuracy and loss metrics

---

## Technologies
- MATLAB
- MATLAB Deep Learning Toolbox
- Convolutional Neural Networks (CNN)
- Transfer Learning
- Image Data Augmentation

---

## Methodology

The approach is based on **transfer learning**, using pre-trained convolutional neural networks provided by MATLAB.  
The following models were explored:

- **AlexNet**
- **SqueezeNet**
- **ResNet-50**
- **VGG-16**

For each architecture:
- The appropriate feature extraction layers were selected
- New final layers were added to adapt the network to the **five-class rice classification task**
- Fully connected layers and dropout layers were tested to improve performance

### Data Processing
- Input images were normalised prior to training
- Data augmentation was applied to improve generalisation and reduce overfitting
- Augmentation parameters included pixel range variation and image reflection

### Training & Evaluation
- Models were trained using different configurations
- Performance was measured using:
  - Classification accuracy
  - Training and validation loss

---

## Experimental Design

Several experiments were conducted to identify the most effective configuration:

1. **Training / Validation Split**
   - Varied between **55% and 90%**
   - Optimal balance found to maximise learning and generalisation

2. **Data Augmentation Parameters**
   - Pixel range and reflection settings were adjusted
   - Best generalisation achieved with optimised augmentation

3. **Model Architecture Comparison**
   - AlexNet, SqueezeNet, ResNet-50, and VGG-16 were evaluated
   - Additional fully connected and dropout layers were tested

4. **Optimisation Methods**
   - Adam optimizer
   - SGDM (Stochastic Gradient Descent with Momentum)

5. **Training Hyperparameters**
   - Learning rate
   - Mini-batch size
   - Multiple combinations were tested simultaneously to identify the best-performing setup

---

## Results

- **Optimal training/validation ratio:** 88%  
- **Optimal augmentation parameters:**  
  - Pixel range: `[-30, 30]`  
  - Reflection: `True`
- **Best-performing architecture:**  
  - AlexNet base architecture with an additional fully connected layer
- **Best optimiser:**  
  - Adam (outperformed SGDM)
- **Optimal training options:**  
  - Learning rate: `1e-4`  
  - Batch size: `28`
- **Final model performance:**  
  - Achieved **100% classification accuracy**

---

## Conclusion

This project successfully achieved accurate rice image classification using a **well-tuned transfer learning approach**.  
The experiments demonstrated that careful adjustment of:
- training/validation split,
- data augmentation parameters,
- network topology,
- optimisation method,
- and training hyperparameters

plays a critical role in achieving high model performance.

The final CNN configuration achieved **100% accuracy**, confirming the effectiveness of combining a suitable pre-trained architecture with systematic optimisation.

---

## Future Improvements
- Explore additional CNN architectures
- Test alternative optimisation methods beyond Adam and SGDM
- Perform more fine-grained hyperparameter searches
- Evaluate robustness on larger or more diverse datasets

---

## Author
**Zakaria Boutarfa**  
De Montfort University  

---

## Academic Context
This project was completed as part of an assessed university coursework and is shared for educational and portfolio purposes.

