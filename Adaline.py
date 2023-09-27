import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #only used to split dataset

def accuracy(X, successes): #X is the predictions, successes is the groundtruth
    correctPredictions = 0
    totalPredictions = 0
    for i in range(len(X)):
        totalPredictions = totalPredictions + 1
        if X[i] == successes[i]:
            correctPredictions = correctPredictions + 1
    return correctPredictions / totalPredictions




class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        print(self.w_)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)




df = pd.read_csv('train.csv', usecols= ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
df = df.fillna(0)
train, test = train_test_split(df, test_size = 0.3)
survivedTrain = train['Survived'].tolist()
survivedTest = test['Survived'].tolist()
train['Sex'].replace(['female','male'],[0,1], inplace =True)
test['Sex'].replace(['female','male'],[0,1], inplace =True)
train = train.drop(['Survived'], axis=1).values
test = test.drop(['Survived'], axis=1).values

myAdaline = AdalineGD(0.0001, 5)
myAdaline.fit(train, survivedTrain)



predictions = myAdaline.predict(test)
print(predictions)
print(accuracy(predictions, survivedTest))
