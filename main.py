from imports import *

h = dumbClassifiers.AlwaysPredictOne({})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
predictAllOne = h.predictAll(datasets.TennisData.X)
print(predictAllOne)

