from imports import *
'''
Warming up to Classifiers
'''
h = dumbClassifiers.AlwaysPredictOne({})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
runClassifier.trainTestSet(h, datasets.TennisData)