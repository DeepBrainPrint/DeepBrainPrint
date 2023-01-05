import numpy as np

from deepbrainprint import metrics


def test_average_precision():
    ground_truth = 1
    ground_truth_count = 3    
    predictions = np.array([ 1, 2, 1, 1, 4, 5 ])
    expected_ap = (1. + 2/3 + 3/4) / 3
    obtained_ap = metrics.average_precision(ground_truth, ground_truth_count, predictions)
    assert round(expected_ap, 5) == round(obtained_ap, 5)
  

def test_mean_average_precision():
    labels_count = { 1: 3, 2: 3, 3: 3 }
    queries = np.array([ 1, 2, 3 ])
    retrievals = np.array([
        [1, 3, 1], 
        [2, 2, 1], 
        [1, 3, 3]
    ])
    # mAP@2
    exp_map2 = ( 1/2 + 1 + 1/4 ) / 3
    obt_map2 = metrics.mean_average_precision(queries, retrievals, k=2, labels_count=labels_count, labels_count_contains_query=False)
    assert round(exp_map2, 5) == round(obt_map2, 5)    
    # mAP@3
    exp_map3 = ( (1 + 2/3) / 3 + (2/3) + (1/2 + 2/3)/3 ) / 3
    obt_map3 = metrics.mean_average_precision(queries, retrievals, labels_count=labels_count, labels_count_contains_query=False)
    assert round(exp_map3, 5) == round(obt_map3, 5)    



def test_recall_at_k():    
    queries = np.array([ 1, 2, 3 ])
    retrievals = np.array([
        [1, 3, 1], 
        [1, 2, 2], 
        [1, 2, 2]
    ])
    # R@3
    exp_acc3 = 2/3
    obt_acc3 = metrics.recall_at_k(queries, retrievals)
    assert round(exp_acc3, 5) == round(obt_acc3, 5)    
    # R@1
    exp_acc1 = 1/3
    obt_acc1 = metrics.recall_at_k(queries, retrievals, k=1)
    assert round(exp_acc1, 5) == round(obt_acc1, 5)    