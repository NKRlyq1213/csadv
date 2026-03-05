import numpy as np

from csadv.experiments import run_convergence_test


def test_convergence_pipeline_smoke():
    R = 1.0
    u0 = 1.0
    Ng_list = [5, 7]

    # Gaussian center at (lam0=0, lat0=0), use distance sigma_m = R*sigma_rad
    ic_kwargs = dict(
        lam0=0.0,
        lat0_or_colat0=0.0,
        sigma_m=0.4 * R,
        amp=1.0,
        background=0.0,
        use_colat=False,
    )

    res = run_convergence_test(
        Ng_list,
        R=R,
        CFL=0.1,
        u0=u0,
        alpha0=0.3,
        use_colat=False,
        ic_name="gaussian",
        ic_kwargs=ic_kwargs,
        boundary_backend="numpy",
        rhs_backend="numpy",
        n_periods=0,
    )

    assert len(res) == len(Ng_list)
    assert all("L2" in r and "Linf" in r for r in res)
    assert all(r["L2"] == 0.0 and r["Linf"] == 0.0 for r in res)