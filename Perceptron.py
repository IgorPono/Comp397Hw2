import numpy as np
import pandas as pd


def accuracy(X, successes):
    correctPredictions = 0
    totalPredictions = 0
    for i in range(len(X)):
        totalPredictions = totalPredictions + 1
        if X[i] == successes[i]:
            correctPredictions = correctPredictions + 1
    return correctPredictions/totalPredictions

def unitSetFunc(x):
    return np.where(x > 0, 1, 0)

class Perceptron(object):

    def __init__(self, learningCoeff = 0.001, iterations = 100):
        self.learningCoeff = learningCoeff
        self.iterations = iterations
        self.activationFunc = unitSetFunc
        self.weights = None

    def fit(self, X, y):

        numSamples, numFeatures = X.shape
        self.weights = np.zeros(numFeatures)

        for _ in range(self.iterations):
            for idx, x_i in enumerate(X): #enumerate used in order to be able to index into the y array and check actual result
                linearOutput = np.dot(x_i, self.weights)
                yPred = self.activationFunc(linearOutput)
                weightUpdate = self.learningCoeff * (y[idx] - yPred)
                self.weights += weightUpdate * x_i
                #print(self.weights)


    def predict(self, X):
        linearOutput = np.dot(X, self.weights)
        y_predicted = self.activationFunc(linearOutput)
        return y_predicted


customCSVTrain = pd.read_csv('Comp379LS - Sheet1 .csv', usecols=['NumWorkers', 'Years', 'Avg. pay (thousand)']).values
#customCSVTrain = pd.read_csv('Comp379LSTrain2 - Sheet1.csv', usecols=['NumWorkers', 'Years', 'Avg. pay (thousand)']).values
successesTrain = pd.read_csv('Comp379LS - Sheet1 .csv', usecols=['Succeeded']).values
#successesTrain = pd.read_csv('Comp379LSTrain2 - Sheet1.csv', usecols=['Succeeded']).values
myPerceptron = Perceptron(0.1, 500)
myPerceptron.fit(customCSVTrain, successesTrain)
#print(myPerceptron.predict(customCSVTrain))
predictions = myPerceptron.predict(customCSVTrain)
#print(predictions)
print(accuracy(predictions, successesTrain))
