## Machine Learning for Public Policy project
Maxine Xu, Miao Li, Yujie Jiang, Jariel Yang

### Topic: Predicting Pneumonia from X-Ray Image
In this project, we aim to develop a deep learning algorithm to diagonalize pneumonia from chest x-ray images of one to five years old from Guangzhou Women and Children’s Medical Center. After conducting an in-depth literature review, we performed image augmentation to obtain a balanced dataset and then used PyTorch to build a 3-layer convolutional neural network (CNN). To further improve the accuracy of our model, we performed a series of hyperparameter tuning, including different dropout rate, training epochs, learning rate, and additional layers. During the experiments, we customized a 3-layer model, which achieved an accuracy of 93.75% and a recall of 100%, and identified overfitting as our main challenge.
