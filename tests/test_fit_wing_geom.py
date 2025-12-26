import json
from pathlib import Path

import torch

from flapping_bot.scripts.fit_wing_geom_from_points import fit_wing_geom


def test_fit_piecewise_linear_exact_on_keypoints(tmp_path: Path):
    # Keypoints define exactly linear c(x) and constant dhat; fitted midpoints should match.
    xk = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    ck = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    dk = torch.tensor([0.25, 0.25, 0.25], dtype=torch.float64)

    fitted = fit_wing_geom(xk, ck, dk, N=10, R=1.0, dtype=torch.float64)
    # Linear interpolation should match the analytical line.
    c_expected = 0.1 + 0.2 * fitted.x_mid  # 0.1 + (0.3-0.1)*x
    assert torch.allclose(fitted.c, c_expected, atol=1e-12)
    assert torch.allclose(fitted.dhat, torch.full_like(fitted.dhat, 0.25), atol=1e-12)

    # Ensure JSON serialization is valid.
    out = fitted.to_winggeom_dict()
    path = tmp_path / "wing_geom.json"
    path.write_text(json.dumps(out))
    loaded = json.loads(path.read_text())
    assert loaded["N"] == 10


def test_te_polyline_construction_is_sane():
    # Trailing-edge polyline points (x, z_te) with z_te <= 0, chord = abs(z_te).
    xk = torch.tensor([0.0, 1.0], dtype=torch.float64)
    ck = torch.tensor([0.3, 0.0], dtype=torch.float64)
    dk = torch.tensor([0.25, 0.25], dtype=torch.float64)
    fitted = fit_wing_geom(xk, ck, dk, N=4, R=1.0, dtype=torch.float64)
    assert float(fitted.R) == 1.0
    assert float(fitted.c.min()) >= 0.0
