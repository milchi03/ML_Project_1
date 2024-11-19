import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2


# loading the data sets
trainData = pd.read_csv('loan-10k.lrn.csv')
testData = pd.read_csv('loan-10k.tes.csv')

# taking a look at the data
print("Training data head:")
print(trainData.head()) # first 5 rows
print("\nTraining Data Info:")
print(trainData.info())  # overview of the dataset's structure
print("\nTraining Data Statistics:")
print(trainData.describe())  # descriptive statistics of numerical columns

# checking for missing values
missingValues = trainData.isnull().sum()  # total missing values per column
missingValues = missingValues[missingValues > 0]  # columns with missing values
print("\nColumns with Missing Values:")
print(missingValues)

# numerical columns
numericalColumns = trainData.select_dtypes(include=['float64', 'int64']).columns
numericalColumns = numericalColumns[numericalColumns != 'ID']

# looking for OUTLIERS
Q1 = trainData[numericalColumns].quantile(0.25)
Q3 = trainData[numericalColumns].quantile(0.75)
IQR = Q3 - Q1
# defining outlier bounds
lowerBound = Q1 - 1.5 * IQR
upperBound = Q3 + 1.5 * IQR
# finding the outliers
outliers = (trainData[numericalColumns] < lowerBound) | (trainData[numericalColumns] > upperBound)
# counting the outliers per column
print("\nOutliers Count per Column:\n", outliers.sum())
# handling outliers in training data
for col in numericalColumns:
    trainData[col] = trainData[col].clip(lower=lowerBound[col], upper=upperBound[col])
# handling outliers in test data
for col in numericalColumns:
    testData[col] = testData[col].clip(lower=lowerBound[col], upper=upperBound[col])

# checking skewness of numerical columns
skewness = trainData[numericalColumns].skew()
print("\nSkewness of Numerical Features:")
print(skewness)

# identifying highly skewed columns
highlySkewed = skewness[skewness > 1]
print("\nHighly Skewed Features:")
print(highlySkewed)

# applying log transformation to highly skewed columns
for col in highlySkewed.index:
    trainData[col] = np.log1p(trainData[col])  # transforming train data
    testData[col] = np.log1p(testData[col])    # transforming test data

# checking skewness after transformation
newSkewness = trainData[highlySkewed.index].skew()
print("\nSkewness After Log Transformation:")
print(newSkewness)

# histograms for each numerical columns
# splitting columns into smaller groups for easier visualization
#batchSize = 12  # number of columns per batch
#for i in range(0, len(numericalColumns), batchSize):
 #   batch = numericalColumns[i:i + batchSize]
  #  trainData[batch].hist(figsize=(16, 12), bins=20, color='skyblue')
   # plt.tight_layout()
    #plt.suptitle(f'Batch {i // batchSize + 1}', fontsize=16)
    #plt.show()

# scaling data
# initializing scaler
scaler = StandardScaler()
# applying scaler to transformed columns
trainData[numericalColumns] = scaler.fit_transform(trainData[numericalColumns])
testData[numericalColumns] = scaler.transform(testData[numericalColumns])
# checking results
print("\nScaled Numerical Features:")
print(trainData[numericalColumns].head())

# handling CATEGORICAL FEATURES
categoricalColumns = trainData.select_dtypes(include=['object']).columns
categoricalColumns = categoricalColumns[categoricalColumns != 'grade']
print("Categorical Columns:", categoricalColumns)

# applying one-hot encoding to categorical columns
trainDataEncoded = pd.get_dummies(trainData, columns=categoricalColumns, drop_first=False)
testDataEncoded = pd.get_dummies(testData, columns=categoricalColumns, drop_first=False)

# aligning columns between train and test datasets
trainDataEncoded, testDataEncoded = trainDataEncoded.align(testDataEncoded, join='outer', axis=1)

# replacing NaNs introduced by alignment with 0s
trainDataEncoded = trainDataEncoded.fillna(0)
testDataEncoded = testDataEncoded.fillna(0)

# converting boolean columns to integers
for col in trainDataEncoded.columns:
    if trainDataEncoded[col].dtype == 'bool':
        trainDataEncoded[col] = trainDataEncoded[col].astype(int)
        testDataEncoded[col] = testDataEncoded[col].astype(int)  # applying the same to test data

# checking results
print("\nTraining data after encoding:")
print(trainDataEncoded.head())
print("\nTest data after encoding:")
print(testDataEncoded.head())

# creating a LabelEncoder
labelEncoder = LabelEncoder()

# encoding the 'grade' column
trainData['grade'] = labelEncoder.fit_transform(trainData['grade'])

# assigning the encoded 'grade' column back to encoded dataset
trainDataEncoded['grade'] = trainData['grade']

# verify the grade mapping
grade_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
print("Grade Mapping:", grade_mapping)

# separating features and response variable
featuresTrain = trainDataEncoded.drop(columns=['grade', 'ID']) # features
responseTrain = trainDataEncoded['grade']  # target variable
# same for the test data
featuresTest = testDataEncoded.drop(columns=['ID'])

print("\nhello hello hello")
print(featuresTest.head()) # first 5 rows

# splitting training data into training and validation data
featuresTrainX, featuresVal, responseTrainY, responseVal = train_test_split(featuresTrain, responseTrain, test_size=0.2, random_state=42)

# k-NN CLASSIFIER
# defining the hyperparameter grid
paramGridKnn = {
    'n_neighbors': [7, 9, 11, 13, 15],  # different number of neighbors
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
# initializing the k-NN model
knn = KNeighborsClassifier()

# performing grid search with cross-validation
gridSearch = GridSearchCV(knn, paramGridKnn, cv=5, n_jobs=-1, verbose=1)

# fitting the grid search to the data
gridSearch.fit(featuresTrainX, responseTrainY)

# best parameters found
print("\nBest Hyper-Parameters:", gridSearch.best_params_)

# best model found
best_knn = gridSearch.best_estimator_

# predicting on validation data
responsePredValKnn = best_knn.predict(featuresVal)
accuracyValKnn = accuracy_score(responseVal, responsePredValKnn)
print("\nAccuracy on Validation Set with k-NN:", accuracyValKnn)

# RANDOM FORESTS
# defining a parameter grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# initializing RandomizedSearchCV
randomForest = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # parameter combinations to try
    cv=3,
    random_state=42,
    n_jobs=-1
)
# fitting the model
randomForest.fit(featuresTrainX, responseTrainY)

# best parameters found
print("\nBest Parameters:", randomForest.best_params_)

# best model
bestRForest = randomForest.best_estimator_

# predicting on validation set
responsePredValRf = bestRForest.predict(featuresVal)

# checking accuracy
accuracyValRf = accuracy_score(responseVal, responsePredValRf)
print("\nAccuracy on Validation Set with Random Forest:", accuracyValRf)

# NAIVE BAYES
# computing correlation matrix
correlationMatrixNB = featuresTrainX.corr()
# setting a correlation threshold
threshold = 0.85
# identifying highly correlated pairs
highCorrPairs = np.where(np.abs(correlationMatrixNB) > threshold)
highCorrPairs = [(correlationMatrixNB.index[x], correlationMatrixNB.columns[y])
                 for x, y in zip(*highCorrPairs) if x != y and x < y]
print("\nHighly Correlated Pairs:")
for pair in highCorrPairs:
    print(pair)

# dropping one feature of each pair
toDrop = set()
for feature1, feature2 in highCorrPairs:
    toDrop.add(feature1)

# removing dropped features from data frames
featuresTrainXNB = featuresTrainX.drop(columns=toDrop)
featuresValNB = featuresVal.drop(columns=toDrop)

# shifting feature values to make them non-negative
min_shift = abs(featuresTrainXNB.min().min())
featuresTrainXNB_shifted = featuresTrainXNB + min_shift + 1
featuresValNB_shifted = featuresValNB + min_shift + 1
featuresTestNB_shifted = featuresTest + min_shift + 1

# feature selection
selectorNB = SelectKBest(score_func=chi2, k=20)  # selecting top 20 features
featuresTrainXNB_selected = selectorNB.fit_transform(featuresTrainXNB_shifted, responseTrainY)
featuresValNB_selected = selectorNB.transform(featuresValNB_shifted)

# getting the selected feature names for reference
selectedFeatures = featuresTrainXNB.columns[selectorNB.get_support()]
print("\nSelected Features After Feature Selection:")
print(selectedFeatures)

# initializing Gaussian Naive Bayes classifier
naiveBayes = GaussianNB()

# fitting the model with the training data
naiveBayes.fit(featuresTrainXNB_selected, responseTrainY)

# predicting on the validation data
responsePredVal_NB = naiveBayes.predict(featuresValNB_selected)

# evaluating the model on the validation set
accuracyValNB = accuracy_score(responseVal, responsePredVal_NB)
print("\nAccuracy on Validation Set with Naive Bayes:", accuracyValNB)

# SAVING PREDICTIONS
IDs = testDataEncoded['ID']
# aligning features for test set with the training set
featuresTestAligned = featuresTest.reindex(columns=featuresTrainX.columns, fill_value=0)
# making predictions with random forests
responsePredTestRf = bestRForest.predict(featuresTestAligned)
# creating a data frame for results
predictionRF = pd.DataFrame({
    'ID': testDataEncoded['ID'],
    'grade': responsePredTestRf
})
# reverting the encoded 'grade' predictions back to letter grades
predictionRF['grade'] = labelEncoder.inverse_transform(predictionRF['grade'])
# saving predictions to CSV
predictionRF.to_csv('Group33_Avendano_Skrijelj_Michlits.csv', index=False)
print("\nPredictions saved to 'Group33_Avendano_Skrijelj_Michlits.csv'.")