"""Is the best covariate model better than chance, once the search is counted?"""
import numpy as np

from tidewater.calibration.covariates import (LOG_TERMS, _has_k, _rows,
                                              best_score)


def permutation_test(covariates, terms, seed, n=500, max_terms=2,
                     log_terms=LOG_TERMS):
    """Permutation test of the best-of-search leave-one-out score.

    Shuffles ``k`` across the glaciers and reruns the whole search each time,
    so the null distribution is of the minimum over the candidates.

    Parameters
    ----------
    covariates : list of dict or :py:class:`pandas.DataFrame`
        one row per glacier with ``k`` in yr-1 and the terms; rows without a
        finite ``k`` are left out
    terms : sequence of str
        the candidate terms of the search
    seed : int or :py:class:`numpy.random.Generator`
        of the shuffles
    n : int
        the number of permutations
    max_terms : int
        the largest number of terms in one model
    log_terms : set of str
        the terms that enter as logs

    Returns
    -------
    dict
        ``best`` (the observed score, log units), ``null`` (the n scores of
        the shuffles), ``p_value`` = (count(null <= best) + 1) / (n + 1),
        ``n_terms`` and ``n`` (glaciers)
    """
    rng = (seed if isinstance(seed, np.random.Generator)
           else np.random.default_rng(seed))
    rows = [r for r in _rows(covariates) if _has_k(r)]
    terms = list(terms)
    obs = best_score(rows, terms, max_terms, log_terms)
    k = np.array([r['k'] for r in rows], dtype=float)
    null = np.array([best_score([dict(r, k=float(v)) for r, v in
                                 zip(rows, rng.permutation(k))],
                                terms, max_terms, log_terms)
                     for _ in range(n)])
    return dict(best=obs, null=null,
                p_value=(np.sum(null <= obs) + 1) / (n + 1),
                n_terms=len(terms), n=len(rows))
