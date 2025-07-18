from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = np.exp(X@W)
    scores_total = np.sum(scores, axis = 1)
    prob = scores/scores_total[:,np.newaxis]
    lossi = -np.log(prob[np.arange(X.shape[0]), y])
    loss = np.sum(lossi)/X.shape[0] + reg*np.sum(W*W)


    dW = X.T @ prob;
    result = np.zeros((np.max(y)+1,X.shape[1]))
    np.add.at(result, y, X)
    dW = dW - result.T
    dW = dW/X.shape[0]
    dW = dW + 2*reg*W

    




    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = np.exp(X@W)
    scores_total = np.sum(scores, axis = 1)
    prob = scores/scores_total[:,np.newaxis]
    lossi = -np.log(prob[np.arange(X.shape[0]), y])
    loss = np.sum(lossi)/X.shape[0] + reg*np.sum(W*W)


    dW = X.T @ prob;
    result = np.zeros((np.max(y)+1,X.shape[1]))
    np.add.at(result, y, X)
    dW = dW - result.T
    dW = dW/X.shape[0]
    dW = dW + 2*reg*W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
