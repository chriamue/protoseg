# source: https://github.com/sjyk/python-segmentation-benchmark/blob/master/evaluation/Metrics.py
from sklearn.metrics import jaccard_similarity_score

def jaccard(prediction, label):
    return jaccard_similarity_score(label.flatten(), prediction.flatten())