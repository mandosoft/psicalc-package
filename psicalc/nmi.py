#Add BSD 3 Clause Here

import numpy as np
import warnings
from math import log
from scipy import sparse as sp
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.fixes import _astype_copy_false
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils.validation import _deprecate_positional_args


def check_clusterings(labels_true, labels_pred):
    """Check that the labels arrays are 1D and of same dimension.
    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        The true labels.
    labels_pred : array-like of shape (n_samples,)
        The predicted labels.
    """
    labels_true = check_array(
        labels_true, ensure_2d=False, ensure_min_samples=0, dtype=None,
    )

    labels_pred = check_array(
        labels_pred, ensure_2d=False, ensure_min_samples=0, dtype=None,
    )

    type_label = type_of_target(labels_true)
    type_pred = type_of_target(labels_pred)

    if 'continuous' in (type_pred, type_label):
        msg = f'Clustering metrics expects discrete values but received' \
              f' {type_label} values for label, and {type_pred} values ' \
              f'for target'
        warnings.warn(msg, UserWarning)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    check_consistent_length(labels_true, labels_pred)

    return labels_true, labels_pred


@_deprecate_positional_args
def contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False,
                       dtype=np.int64):
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")
    gaps = False
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    if classes[0] == 0 or clusters[0] == 0:
        gaps = True
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=dtype)
    print("\nLabels True: \n", labels_true)
    print("\nLabels Pred: \n", labels_pred)
    print("\nUnique Labels True: \n", classes)
    print("\nUnique Labels Pred: \n", clusters)
    print("\nInverse Labels True: \n", class_idx)
    print("\nInverse Labels Pred: \n", cluster_idx)
    print("\nMxN: \n", n_classes, n_clusters)
    print("\nContingency Array: \n", contingency.toarray())
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
        if gaps:
            fix_gaps(contingency)
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def fix_gaps(contingency):
    """Modifies the contingency matrix to discount MSA gaps as data"""
    a, b = contingency.nonzero()
    coords = np.array([[i, j] for i, j in zip(a, b)
                       if i == 0 or j == 0])
    for m, n in coords:
        contingency[m, n] = 0
    print(contingency.toarray())
    exit()


@_deprecate_positional_args
def mutual_info_score(labels_true, labels_pred, *, contingency=None):

    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    else:
        contingency = check_array(contingency,
                                  accept_sparse=['csr', 'csc', 'coo'],
                                  dtype=[int, np.int32, np.int64])

    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    elif sp.issparse(contingency):
        # For a sparse matrix
        nzx, nzy, nz_val = sp.find(contingency)
    else:
        raise ValueError("Unsupported type for 'contingency': %s" %
                         type(contingency))

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = (pi.take(nzx).astype(np.int64, copy=False)
             * pj.take(nzy).astype(np.int64, copy=False))
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = (contingency_nm * (log_contingency_nm - log(contingency_sum)) +
          contingency_nm * log_outer)
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)

    return np.clip(mi.sum(), 0.0, None)


def entropy(labels):
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))


def _generalized_average(U, V, average_method):
    """Return a particular mean of two numbers."""
    if average_method == "min":
        return min(U, V)
    elif average_method == "geometric":
        return np.sqrt(U * V)
    elif average_method == "arithmetic":
        return np.mean([U, V])
    elif average_method == "max":
        return max(U, V)
    else:
        raise ValueError("'average_method' must be 'min', 'geometric', "
                         "'arithmetic', or 'max'")


@_deprecate_positional_args
def normalized_mutual_info_score(labels_true, labels_pred, *,
                                 average_method='arithmetic'):

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1 or
            classes.shape[0] == clusters.shape[0] == 0):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    contingency = contingency.astype(np.float64,
                                     **_astype_copy_false(contingency))
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred,
                           contingency=contingency)
    # Calculate the expected value for the mutual information
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    normalizer = _generalized_average(h_true, h_pred, average_method)
    # Avoid 0.0 / 0.0 when either entropy is zero.
    normalizer = max(normalizer, np.finfo('float64').eps)
    nmi = mi / normalizer
    return nmi