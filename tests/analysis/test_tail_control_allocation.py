import numpy as np
import pytest
from flapping_bot.analysis.tail_control_allocation import inverse_effectiveness


def test_weak_resolved_axes_are_not_discarded():
    b=np.array([[-4,1.57,1.43],[113,.38,.18],[-4.26,-.56,.03]])
    inv=inverse_effectiveness(b)
    np.testing.assert_allclose(b@inv,np.eye(3),atol=1e-12)
    assert np.linalg.matrix_rank(inv)==3


def test_missing_authority_fails_explicitly():
    with pytest.raises(ValueError):inverse_effectiveness(np.diag([1,2,0]))
