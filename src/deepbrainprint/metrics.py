import numpy as np

from typing import Any, Optional
from collections import Counter

def average_precision(relevant_document: int,
                      relevant_documents_count: int,
                      predictions_array: np.ndarray) -> float:
    """
    Computes the average precision from the predictions array, 
    given `relevant_document` as the relevant class and `relevant_documents_count`
    as the global count of relevant documents. 
    """ 
    n_predictions = len(predictions_array)
    # If the number of relevant documents is greater than
    # the number of predictions, we cant retrieve them all, so 
    # we set the relevants to the number of predictions.  
    divisor = min(relevant_documents_count, n_predictions)
    avg_precision_acc = 0.
    for i in range(n_predictions):
        if predictions_array[i] != relevant_document: continue
        predictions_subset = predictions_array[:(i+1)]
        predictions_subset_relevant = np.sum(predictions_subset == relevant_document)
        avg_precision_acc += predictions_subset_relevant / (i+1)
    return (avg_precision_acc / divisor)


def mean_average_precision(queries_labels: np.ndarray, 
                           retrievals_labels: np.ndarray, 
                           k: Optional[int] = None, 
                           labels_count: Optional[dict] = None,
                           labels_count_contains_query: bool=True) -> np.floating[Any]:
    """
    `query_labels` is an array of dimension N contianing queries ground truths.
    `retrievals_labels` is a matrix of dimension NxK, where each row contains the 
    ground truth labels of the top-k retrievals for the corresponding query.
    The `k` parameters indicates how many retrievals we're taking into account (mAP@k).
    `labels_count` counts the occurrence of each label in queries (pre-computing the dictionary
    can speed up the process, but it isn't required).
    Computes the mAP@k. 
    """
    assert queries_labels.shape[0] == retrievals_labels.shape[0], "Queries don't match retrievals length"
    k = retrievals_labels.shape[1] if (k is None or k <= 0 or k >= retrievals_labels.shape[1]) else k
    labels_count = Counter(queries_labels) if labels_count is None else labels_count
    # if the label counter is performed on `query_labels`, the amount of scans of a specific
    # label contains also the query scan, but we have to remove it since it's not retrievable. 
    if labels_count_contains_query: labels_count = { k:(v-1) for k,v in labels_count.items() }
    return np.average([ 
        average_precision(gt, labels_count[gt], predictions[:k]) 
        for gt, predictions 
        in zip(queries_labels, retrievals_labels)
    ])
    

def recall_at_k(queries_labels: np.ndarray, 
                  retrievals_labels: np.ndarray, 
                  k: Optional[int] = None) -> np.floating[Any]:
    """
    `query_labels` is an array of dimension N contianing queries ground truths.
    `retrievals_labels` is a matrix of dimension NxK, where each row contains the 
    ground truth labels of the top-k retrievals for the corresponding query.
    The `k` parameters indicates how many retrievals we're taking into account (R@k).
    `labels_count` counts the occurrence of each label in queries (pre-computing the dictionary
    can speed up the process, but it isn't required).
    Computes the accuracy@k. 
    """
    assert queries_labels.shape[0] == retrievals_labels.shape[0], "Queries don't match retrievals length"
    k = retrievals_labels.shape[1] if (k is None or k <= 0 or k >= retrievals_labels.shape[1]) else k
    return np.average([ int(gt in preds[:k]) for gt, preds in zip(queries_labels, retrievals_labels) ])