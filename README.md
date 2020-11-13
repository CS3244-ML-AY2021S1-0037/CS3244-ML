# CS3244-2010-0037: The Art of Movie Posters: A Machine Learning Approach
This repository is created for the purposes of our movie poster classification ML project.

## Abstract 

We aim to predict the genre of a movie based solely on its poster, using a variety of machine learning techniques. Since a movie may belong to multiple genres, this is a multi-label classification problem. We evaluated performance of both traditional machine learning methods and modern deep learning neural networks. Predictions were made with different features using different models to gather effectiveness of each feature. From our results, we find that the raw image features are the most important, followed by emotions. Words and dominant HSV values both show subpar results. Finally, we find that Convolutional Neural Networks performed better than other traditional Machine Learning Models.

## Introduction 

A good movie poster should be able to convey important characteristics of the film like its genre to better appeal to its viewers. Given a movie poster, can we model a viewer’s perception of its genre?

Today, our world is saturated with films and movie posters. Platforms like Netflix and Hulu place great importance upon visually stunning posters. With so many works grasping at our attention, designing eye-catching posters have become an integral element in achieving success for movie producers. For us unwitting viewers, seeing a movie poster should immediately register its genre in our minds.

Our project aims to model how a viewer would perceive movie posters using machine learning. We compared and combined several traditional methods with today’s more cutting edge algorithms to produce critical insights for graphic designers to design more engaging posters.


## Feature Extraction and Results of Various Models

Check out the various sub folders to find out the different models and features used. 

### kNN 

Done by: Calvin (https://github.com/calvincxz)
  
K-Nearest Neighbors (suited for multi-label classification problems)
Preprocessing: Resized posters to 100x100x3 dimension, and flatten as a vector of 30,000 pixel intensity values
Input: 30,000 pixels intensities
Dataset size: 3000
Dataset split: 80% training, 10% validation, 10% test
Objective: Maximize F1 score
  
### CNN

Done by: Chee Yuan (https://github.com/ccyccyccy)

Preprocessing: Encode labels as a one-hot vector, normalize images as specified in PyTorch.
Input: RGB image of size 268 x 182 x 3 (Original image size)
Dataset size: 30000 images
Dataset split: 90% training, 5% validation, 5% test
Objective: Maximize F1 Score

### HSV and Emotions of Images of Posters 

Done by: Zhaung Yuan (https://github.com/Za-yn) 

Preprocessing: Encode labels as a one-hot vector, normalize images.
Input: 10 dimensional vector; 3 HSV features, 7 emotion feature
Dataset: 30,000 images
Dataset split: 70% training, 30% test
Objective: Maximise F1 Score 
Models used: Logistic Regression, Naive Bayes, AdaBoost and Random Forest 

### Emotions Confidence Scores 

Done by: Jeevan (https://github.com/Jeevz10)

Preprocessing: Extract Average Emotions Confidence Scores 
Input: 9 different emotions
Dataset size: 5,000 posters
Dataset split: 70%, 30% 
Objective: Maximise F1 Score 
Models used: Logistic Regression, Linear Regression, Naive Bayes, AdaBoost, Random Forest, Multi-kNN and Neural Networks Multi-Perceptron Layer 

### Words Found in Posters 

Done by: Sylvester (https://github.com/sylchw) and Wei Cheng (https://github.com/Weiichengg)

Preprocessing: Upscale, detect words, vectorize words, build cosine similarity matrix.
Input: 8636 values from cosine similarity ,3802 column of words 
Dataset size: 10000 images
Dataset split: 70% training , 30% testing
Objective: Maximize F1 Score
Models used: Naive Bayes, AdaBoost 


Do drop a message to any of us and let us know what you think! 
