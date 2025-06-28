# Projeto-RP

This is a University Project for the Pattern Recognition Discipline. We replicated a previous studied focusing on the energetic effience of SNNs. In order to do so, we used 3 datasets to measure framework's energetic effiency. It can be seen in #Models Used. 
The First dataset is the CIFAR-10, a famous one, used in the original study, is a well-established benchmark in the field of machine learning, specifically designed for image classification. Comprising 60,000 color images, each of size 32x32 pixels, the dataset is segmented into 10 distinct classes, each representing a different object or creature. 
The second dataset is the Brain Tumor. This model takes MRI brain images as input, preprocesses them, and classifies them into two categories: "no" for no tumor and "yes" for tumor.
The third one is the Dog Breed. This model classifies the inputs images into each dog breed category. 

If you are trying to replicate it, first, run the main framework with PyTorch and Spikingjelly, the second one with Keras and Sklearn(PyTorch) and the third one with PyTorch.

This project uses the architectures Resnet20, VGG16, ResNet50. 

# Learn More
To learn more about the framework used, take a look at the following resources:
A Hybrid Neural Coding Approach for Pattern Recognition with Spiking Neural Networks: https://arxiv.org/pdf/2305.16594

# Models Used
CIFAR-10: https://www.kaggle.com/code/farzadnekouei/cifar-10-image-classification-with-cnn
Brain Tumor: https://www.kaggle.com/code/pranithchowdary/hybrid-model-for-brain-tumor
Dog Breed: https://www.kaggle.com/code/enriquecompanioni/dog-breed-classification-using-resnet-inceptionv3 

# Deploy
The easiest way to deploy is using Spyder.py. 
