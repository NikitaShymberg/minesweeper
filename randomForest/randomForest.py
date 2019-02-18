import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from neuralNet.generateTrainingData import generateTrainingData

class RandomForestSolver():
    def __init__(self):
        self.model = RandomForestClassifier(warm_start=True, n_estimators=100) #TODO: hyperparams

    def train(self): # TODO: refactor, look into warm start again
        data, labels = [], []
        for _ in range(100):
            datum, label = generateTrainingData()
            datum = datum.reshape((datum.shape[0], datum.shape[1] * datum.shape[2])) # TODO: have this as a param to generate, along with the nn thing
            data.append(datum)
            labels.append(label)
        data = np.concatenate(data)
        labels = np.concatenate(labels)
        self.model.fit(data, labels)
    
    def train2(self): # TODO: refactor, look into warm start again
        cur_estimators = self.model.n_estimators
        self.model.set_params(n_estimators=(cur_estimators+10))
        data, labels = generateTrainingData()
        data = data.reshape((data.shape[0], data.shape[1] * data.shape[2])) # TODO: have this as a param to generate, along with the nn thing
        self.model.fit(data, labels)
    
    def test(self):
        data, labels = generateTrainingData()
        data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
        print("Predicted:", self.model.predict(data))
        print("Actual:   ", labels)

if __name__ == "__main__":
    rfs = RandomForestSolver()
    rfs.train()
    # for _ in range(100):
        # rfs.train2()
    
    rfs.test()