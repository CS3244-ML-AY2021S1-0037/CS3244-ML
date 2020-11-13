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

## Discussion 

### Analysis of Common Results 

Adaboost outperforms the Naive Bayes model in both metrics.  Naive Bayes assume that the features are independent of another, which may not always be true.  The words are related to one another in the determination of genre. Additionally, Naive bayes assume equal weights for all the features when determining the genres. This may not hold all the time.  In the case of HSV value, certain Hue values (E.g Red), may indicate a higher probability of horror movies. 

The convergence of weak classifiers in Adaboost has resulted in a  better learner compared to Naive Bayes. Despite this, Adaboost is still susceptible to outlier and noisy data, which is evident from its seemingly poor F1 score. Noisy data and outliers are undesirable  since Adaboost builds each stamp based on the previous stump’s errors. Outliers have larger residual errors, hence Adaboost may spend a disproportionate amount of time on those errors, resulting in an inaccurate prediction.

### Importance of Different Features of a Movie Poster

We will determine the importance of different features of a movie poster based on the performance of the feature in predicting the genre of the movie.


#### Image Feature 

We find image features to be a good indicator of the movie genre, since CNN was able to produce good predictions with an F1 Score of 0.52. We believe that the texture, tone and colours of the image greatly influences the perceived genre of the movie.

#### Emotions 

Models using emotions as features had decent performance, achieving a best F1 score of 0.4.  However, there was some decline in performance when paired with HSV values which require further investigation. 

#### Words 

Models using words as features had the worst performance, achieving a best F1 score of 0.3. This was likely due to the incorrect words extraction caused by the low resolution of posters.


#### Choice of Machine Learning Model 

The CNN model has the best result with a F1 score of 0.52 on the test set. The CNN model was able to perform ~25% better than the second best model in terms of F1 score. Therefore, we conclude that the model has been well-trained to identify key feature maps that encapsulate the movie posters. Furthermore, the CNN model required no additional processing step unlike the extraction of emotions / words / dominant HSV which can lead to significant overhead.


## Summary 

Designing an eye-catching movie poster that succinctly conveys a movie genre is not an easy feat. There are various features ranging from colours to facial expressions of the movie cast that plays a role in the determination of a movie genre. Our CNN model achieved the best F1 Score of 0.52 . From this, we can conclude that texture, tone and colour outperforms other factors in the classification of the movie genre. This did not come as a surprise to our group, since it is known that CNN is effective for image data by preserving spatial locality of data input, automatically learning the kernels needed to recognize image features that are important in our predictions.

### Shortcomings 

One shortcoming was that we used low quality images to train our models. Since we were given a movie poster of low quality (268 x 182 pixels), it affected our ability to accurately extract other key features from the movie poster such as movie title and facial emotions. We could have seen better performance from traditional machine learning models by accurately extracting more features if we had higher resolution images.

### Future Works 

Moving on, we could explore advanced computer vision techniques to allow us to improve the quality of the image. As mentioned above, poor quality image greatly hindered our ability to extract features. We could use more advanced computer vision techniques to extract more features such as objects and fonts present in the poster. Lastly, moving on from genres, we could explore the field of IMDb film censorship classification (NC16, M18, R21). 


## Concluding Remarks from the Team

We, as a team, love this particular Machine Learning module. We have learnt a tremendous amount in Machine Learning and its numerous processes.  
Do drop a message to any of us and let us know what you think! 
