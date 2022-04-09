import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import hw1.dataloaders as hw1dataloaders
import torch.utils.data.sampler as sampler

import helpers.dataloader_utils as dataloader_utils
from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train, y_train = dataloader_utils.flatten(dl_train)
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = len(set(y_train.numpy()))
        return self
    




    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = self.calc_distances(x_test)

        # TODO: Implement k-NN class prediction based on distance matrix.
        # For each training sample we'll look for it's k-nearest neighbors.
        # Then we'll predict the label of that sample to be the majority
        # label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        dists = self.calc_distances(x_test)

        for i in range(n_test):
            # TODO:
            # - Find indices of k-nearest neighbors of test sample i
            # - Set y_pred[i] to the most common class among them

            # ====== YOUR CODE: ======
            ind = np.argpartition(dists[i],self.k)[:self.k]
            dict = {}
            count, itm = 0, ''
            for item in reversed(self.y_train[ind]):
                dict[item] = dict.get(item, 0) + 1
                if dict[item] >= count:
                    count, itm = dict[item], item
            y_pred[i] = itm
            #raise NotImplementedError()
            # ========================

        return y_pred

    def calc_distances(self, x_test: Tensor):
        """
        Calculates the L2 distance between each point in the given test
        samples to each point in the training samples.
        :param x_test: Test samples. Should be a tensor of shape (Ntest,D).
        :return: A distance matrix of shape (Ntrain,Ntest) where Ntrain is the
            number of training samples. The entry i, j represents the distance
            between training sample i and test sample j.
        """

        # TODO: Implement L2-distance calculation as efficiently as possible.
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - No credit will be given for an implementation with two explicit
        #   loops.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        # Hint 1: Open the expression (a-b)^2.
        # Hint 2: Use "Broadcasting Semantics".

        dists = torch.tensor([])
        # ====== YOUR CODE: ======

        sqrA = torch.sum(torch.pow(x_test, 2), 1, keepdim=True).expand(x_test.shape[0], self.x_train.shape[0])
        sqrB = torch.sum(torch.pow(self.x_train, 2), 1, keepdim=True).expand(self.x_train.shape[0], x_test.shape[0]).t()
        dists = torch.sqrt(sqrA - 2*torch.mm(x_test, self.x_train.t()) + sqrB)
        
        #raise NotImplementedError()
        # ========================

        return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1
    

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.

    accuracy = 0
    # ====== YOUR CODE: ======
    correct_examples = float(torch.eq(y,y_pred).sum())
    all_examples = float(y.shape[0])
    accuracy = correct_examples/ all_examples
    
    #raise NotImplementedError()
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """
    batch_size = 100
    num_workers = 2
    accuracies = []

    for i, k in enumerate(k_choices):
        acc_per_k = []
        model = KNNClassifier(k)

        # TODO: Train model num_folds times with different train/val data.
        # Don't use any third-party libraries.
        # You can use your train/validation splitter from part 1 (even if
        # that means that it's not really k-fold CV since it will be a
        # different split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        for k_cv in range(num_folds):
            idx = list(range(len(ds_train)))
            np.random.shuffle(idx)
            split = int(np.floor(0.2 * len(ds_train)))
            train_indices, val_indices = idx[split:], idx[:split]

            train_sampler = sampler.SubsetRandomSampler(train_indices)
            valid_sampler = sampler.SubsetRandomSampler(val_indices)

            dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, 
                                                       sampler=train_sampler)
            dl_valid = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,num_workers=num_workers, 
                                                            sampler=valid_sampler)
            #x_train, y_train = dataloader_utils.flatten(dl_train)
            x_test, y_test = dataloader_utils.flatten(dl_valid)


            model.train(dl_train)
            y_pred = model.predict(x_test)

            # Calculate accuracy
            accuracy = 0
            for i in range(y_pred.shape[0]):
                if y_test[i] == y_pred[i]:
                    accuracy += 1
            accuracy = accuracy/y_test.shape[0]
            acc_per_k.append(accuracy)
        accuracies.append(acc_per_k)
        #raise NotImplementedError()
        # ========================


    
    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
