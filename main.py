from imports import *
import dt
'''
Warming up to Classifiers
'''
# print('Always predict one: ')
# runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictOne({}), datasets.SentimentData)
# print('Always predict most frequent class: ', )
# runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictMostFrequent({}), datasets.SentimentData)
# print('First Feature Classifier (Tennis Data):')
# runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.TennisData)
# print('First Feature Classifier (SentimentData):')
# runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.SentimentData)

h = dt.DT({'maxDepth': 5})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)
