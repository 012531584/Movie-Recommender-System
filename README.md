# Movie-Recommender-System

Develop a Collaborative Filtering system to predict as accurately as possible the user item ratings.

Detailed Description:

Collaborative Filtering (CF) systems measure similarity of users by their item preferences and/or measure similarity of items by the users who like them. For this CF systems extract Item profiles and user profiles and then compute similarity of rows and columns in the Utility Matrix. (In this assignment you are given a number of ratings, from which it is possible to build a utility matrix.) In addition to using various similarity measures for finding the most similar items or users, one can use latent factor models (matrix decomposition) and other hybrid approaches to improve on the training and test data RMSE scores. We encourage you use functions available in spark libraries for similarity computation, SVD decomposition etc. However, you cannot use the spark ALS package! Performing these tasks in parallel on multiple cores is required as the dataset is quite large.

Data Description:

The training dataset consists of 85724 ratings and the test dataset consists of 2154 ratings. We provide you with the training data ratings and the test ratings are held out. The data are provided as text in train.dat and test.dat, which should be processed appropriately.

train.dat: Training set (UserID <comma separator> ItemID <tab separator> Rating (Integers 1 to 5) <tab separator> Timestamp (Unix time stamp).
  
test.dat: Testing set (RatingID<comma separator> UserID <comma separator> ItemID, no rating provided).
  
(Data is available on the Leader Board website: https://www.kaggle.com/c/cmpe-256- f2019-hw1)
