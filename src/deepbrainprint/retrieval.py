import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors
from brainfetch.datasets.adni import ADNI2DTFS


def images_to_embeddings(model: nn.Module, dataset: Dataset, device: str = 'cpu'):
    """ 
    Given a trained model that produces embeddings from images, this
    function applies the model to an entire dataset of images. 
    As a standard, the output should be a list / array of numpy arrays. 
    """
    model.eval()
    with torch.no_grad():
        embeddings = []
        data_loader = DataLoader(dataset, 16, False)
        for images, _ in data_loader:
            images = images.to(device)
            res = model(images).cpu().numpy()
            embeddings += list(res)
        return np.stack(embeddings, axis=0)
        


def knn(embeddings, dataset, k=1, metric='cosine'):
    """ 
    Returns k-nearest neighbors and labels for each embedding.
    `retrieved` is a matrix where the i-th row contains indices of 
    the k-nearest neighbors of the i-th embedding, while `labels`
    contains their label.  
    """
    nearest_neighbors = NearestNeighbors(n_neighbors=1, metric=metric)
    nearest_neighbors.fit(embeddings)
    results = nearest_neighbors.kneighbors(embeddings, n_neighbors=k+1, return_distance=False)
    retrieved = []
    labels = []
    # This approach also works with Subset (using patient() function
    # is not generalizable on Subset, hence cannot be used on validation). 
    for i in range(len(embeddings)):
        predictions = results[i][results[i] != i]
        predictions_labels = [ dataset[pid][-1].item() for pid in predictions ]
        retrieved.append(predictions)    
        labels.append(predictions_labels)
    return np.array(retrieved), np.array(labels)



def predicted_patients(embeddings: np.ndarray, 
                       dataset: Dataset, 
                       k: int = 1, 
                       metric: str = 'cosine'):
    """ 
    Returns two list `y_true` and `y_pred`. The i-th embedding is relative to 
    the i-th MRI in the dataset, and so it's bound to a patient with an ID. 
    The model compute good embeddings if embeddings from the same patient are near, 
    while embedding from different patients are far apart. For each embedding A, we 
    query the k nearest embeddings [B1, B2, ..., Bk], we return the patient id of 
    A in `y_true` and, at the same index, the patients id of B1, B2, ..., Bk in `y_pred`. 
    """
    nearest_neighbors = NearestNeighbors(n_neighbors=1, metric=metric)
    nearest_neighbors.fit(embeddings)
    results = nearest_neighbors.kneighbors(embeddings, n_neighbors=k+1, return_distance=False)
    y_true = [] 
    y_pred = [] 
    # This approach also works with Subset (using patient() function
    # is not generalizable on Subset, hence cannot be used on validation). 
    for i, tuple in enumerate(dataset): # type: ignore
        true_label = tuple[-1]
        y_true.append(true_label.item())
        # remove the point itself from results 
        predictions = results[i][results[i] != i]
        # ensure the predictions are exactly k
        predictions = predictions[:k]
        # map the embeddings to their labels 
        predicted_labels = [ dataset[pid][-1].item() for pid in predictions ]        
        y_pred.append(predicted_labels)
        
    return np.array(y_true), np.array(y_pred)