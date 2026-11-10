"""Three-axis tail allocation without unit-scale-dependent rank truncation."""
import numpy as np


def inverse_effectiveness(matrix):
    """Return full-rank inverse after row equilibration, or reject lost authority.

    Input rows are FLU p/q/r acceleration and columns symmetric/differential/rudder
    actual angles. Weak but resolved axes must not be silently removed merely
    because pitch authority exceeds lateral authority.
    """
    matrix=np.asarray(matrix,dtype=float)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite 3x3 control effectiveness')
    scale=np.linalg.norm(matrix,axis=1)
    if np.any(scale<1e-8):raise ValueError('Unresolved control axis')
    normalized=matrix/scale[:,None]
    if np.linalg.cond(normalized)>1e6:raise ValueError('Unresolved control allocation rank')
    return np.linalg.solve(normalized,np.diag(1/scale))
