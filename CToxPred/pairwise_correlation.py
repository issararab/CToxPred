from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator


class CorrelationThreshold(SelectorMixin, BaseEstimator):
    """
    Feature selector that removes correlated features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
    threshold : float
        Pairwise correlation threshold.
    """

    def __init__(self, threshold: float = None) -> None:
        self.threshold = threshold if threshold is not None else 1.0

    def fit(self, X, y=None):
        """
        Learn empirical correlations from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data from which to compute correlations, where `n_samples` is the
            number of samples and `n_features` is the number of features.
        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        corr = np.abs(np.corrcoef(X, rowvar=False))
        self.mask = ~(np.tril(corr, k=-1) > self.threshold).any(axis=1)
        return self

    def _get_support_mask(self):
        return self.mask