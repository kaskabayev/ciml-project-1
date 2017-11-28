from imports import *
import dt
import knn
import perceptron

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

'''
Decision Trees
'''
# h = dt.DT({'maxDepth': 5})
# h.train(datasets.TennisData.X, datasets.TennisData.Y)
# print(h)

'''
Running training and tests set
'''
# runClassifier.trainTestSet(dt.DT({'maxDepth': 1}), datasets.SentimentData)
# runClassifier.trainTestSet(dt.DT({'maxDepth': 3}), datasets.SentimentData)
# runClassifier.trainTestSet(dt.DT({'maxDepth': 5}), datasets.SentimentData)

'''
Better in terminal
'''
# curve = runClassifier.learningCurveSet(dt.DT({'maxDepth': 9}), datasets.SentimentData)
# runClassifier.plotCurve('DT on Sentiment Data', curve)
# curve = runClassifier.hyperparamCurveSet(dt.DT({}), 'maxDepth', [1,2,4,6,8,12,16], datasets.SentimentData)
# runClassifier.plotCurve('DT on Sentiment Data (hyperparameter)', curve)


'''
KNN
'''
'''
EPS
'''
# print('EPS:')
# runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 0.5}), datasets.TennisData)
# runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 1.0}), datasets.TennisData)
# runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 2.0}), datasets.TennisData)

'''
KNN
'''
# print('KNN: ')
# runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.TennisData)
# runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.TennisData)
# runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.TennisData)

'''
The Perceptron
'''
runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.TennisData)
runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.TennisData)

runClassifier.plotData(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
h = perceptron.Perceptron({'numEpoch': 200})
h.train(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
runClassifier.plotClassifier(array([ 7.3, 18.9]), 0.0)

runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.SentimentData)
runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.SentimentData)