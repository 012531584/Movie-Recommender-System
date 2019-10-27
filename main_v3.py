import pandas as pd
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from pyspark import SQLContext
from pyspark.accumulators import AccumulatorParam

# -----------------------------------------   Load train.dat as RDD   ---------------------------------------------
conf = SparkConf()
sc = SparkContext(conf=conf)
train_lines = sc.textFile("train.dat")
# Remove Column's Name (UserID,ItemID,Rating,Timestamp)
header = train_lines.first()
train_lines = train_lines.filter(lambda line: line != header)
# Format Train Data (ItemID, UserID, Rating)
global train_rdd
train_rdd = train_lines.map(lambda line: line.split(',')).map(
            lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2])))

# Build Train Data Dict. with Format [(user, item)] = rating, for later check if the similar movie is rated
global train_dict
train_dict = {}
for x, y, z in train_rdd.collect():
    train_dict[(x, y)] = z

# -----------------------------------------   Build simPdsDF   -----------------------------------------------
# Form utilityMatrix to get simMat later
sqlCon = SQLContext(sc)
utilityMatrix = CoordinateMatrix(train_rdd)
# Similarity Btw. Items
simMat = utilityMatrix.toRowMatrix().columnSimilarities()
# Convert simMat to Pandas format
global simPdsDF
sparkDF = simMat.entries.map(lambda x: str(x.i)+","+str(x.j)+","+str(x.value)).map(lambda w: w.split(',')).toDF()
simPdsDF = sparkDF.toPandas()
# edit columns' name
simPdsDF.columns = ['ItemID_1', 'ItemID_2', 'Similarity']
# change data type
simPdsDF['ItemID_1'] = simPdsDF['ItemID_1'].astype(int)
simPdsDF['ItemID_2'] = simPdsDF['ItemID_2'].astype(int)
simPdsDF['Similarity'] = simPdsDF['Similarity'].astype(float)

# --------------------------------------- Used for RDD to calculate bias ---------------------------------------------
global train_pdsDF
train_pdsDF = pd.read_csv('train.dat', sep = ",")
train_pdsDF = train_pdsDF.drop("Timestamp", axis=1)
global global_avg    # overall mean rating
global_sum = train_pdsDF['Rating'].sum()
global_avg = global_sum/train_pdsDF.shape[0]

# Bias of movie & user
def getBias(user, item):
    isUser = train_pdsDF.iloc[:, 0]==user
    isItem = train_pdsDF.iloc[:, 1]==item

    user_avg =  train_pdsDF[isUser]['Rating'].sum() / train_pdsDF[isUser].shape[0]
    user_bias = user_avg - global_avg

    item_avg = train_pdsDF[isItem]['Rating'].sum() / train_pdsDF[isItem].shape[0]
    item_bias = item_avg - global_avg

    user_item_bias = user_bias + item_bias + global_avg
    return user_item_bias

# --------------------------------------- Return Similar Movie ID ---------------------------------------------
def getMovieID(idx, movie):
    if simPdsDF["ItemID_1"][idx] == movie:  # Return the one is not equal to the movie in (user,movie)
        return simPdsDF["ItemID_2"][idx]
    else:
        return simPdsDF["ItemID_1"][idx]

# --------------------------------------- Return Predicted Rating ---------------------------------------------
def getPredict(user, movie):
    # Get all similar movies
    is_ItemID_1_Sim = simPdsDF['ItemID_1'] == movie
    is_ItemID_2_Sim = simPdsDF['ItemID_2'] == movie
    is_Sim_PdsDF = simPdsDF[is_ItemID_1_Sim | is_ItemID_2_Sim]

    if len(is_Sim_PdsDF.index) < 1:    # New Item in Test.dat
        print('user: ', user, 'movie: ', movie, 'is_Sim_PdsDF.index < 1')
        return global_avg

    # Get rated index
    isRatedIdx = []
    for idx in is_Sim_PdsDF.index:
        MovieID = getMovieID(idx, movie)
        if (user, MovieID) in train_dict:
            isRatedIdx.append(idx)

    # Sorted rated movies by similarity (from high to low)
    isRatedPdsDF = simPdsDF.iloc[isRatedIdx, :]
    isRatedPdsDF = isRatedPdsDF.sort_values(by = ['Similarity'], ascending = False)

    # Compute predicted rating by top_k similar movies
    Sim_total = 0
    Up_total = 0
    Bias_movie = getBias(user, movie)
    top_k = 60
    if isRatedPdsDF.shape[0] < top_k:   # If similar movies less than top_k, just compute those similar movie
        top_k = isRatedPdsDF.shape[0]
    for k in range(top_k):
        MovieID_k = getMovieID(isRatedPdsDF.index[k], movie)
        Rating_k = train_dict[(user, MovieID_k)]
        Sim_k = isRatedPdsDF.iloc[k, 2]  # Column 2 is similairty, isRatedPdsDF is sorted by similairty
        Bias_k = getBias(user, MovieID_k)
        Up_total += Sim_k*(Rating_k - Bias_k)
        Sim_total += Sim_k
    PredRating = Bias_movie + (Up_total/Sim_total)
    # return PredRating, train_dict[(user, movie)]    # train.dat

    return PredRating  # test.dat
# ------------------------------------   Load Test.dat as RDD object   --------------------------------------------------
test_lines = sc.textFile("test.dat")
# Remove First Row (Column's Name) (Rating, UserID, ItemID)
header = test_lines.first()
test_lines = test_lines.filter(lambda line: line != header)
# Format Data (RatingID, UserID, ItemID)
test_rdd = test_lines.map(lambda line: line.split(',')).map(
            lambda tokens: (int(tokens[0]),int(tokens[1]),int(tokens[2])))

# ----------------------------------- Load Accumulator Object ------------------------------------------------
class VectorAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return [0.0]*len(value)
    def addInPlace(self, val1, val2):   #val1: list
        val1 += val2
        return val1

# ----------------------------------- Process test RDD object ------------------------------------------------
def test_result(x):  # x[0]:RatingID, x[1]:user, x[2]:item
    global ans
    PredRating = getPredict(x[1], x[2])
    ans += [[x[0], PredRating]]

# ----------------------------------- Get predicted rating of Test.dat  ------------------------------------------------
ans = sc.accumulator([], VectorAccumulatorParam())
test_rdd.foreach(test_result)
ans = np.array(ans.value)
# print("RatingID ", ans[:, 0], "Rating", ans[:, 1])

# ----------------------------------- Output File  ------------------------------------------------
predictPdsDF = pd.DataFrame({'RatingID': ans[:, 0], 'Rating': ans[:, 1]})
predictPdsDF['RatingID'] = predictPdsDF['RatingID'].astype(int)
predictPdsDF.to_csv("predict.csv", index = False)
