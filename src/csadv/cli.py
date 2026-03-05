from __future__ import annotations

import argparse
import json
from typing import Any

from csadv.experiments import run_convergence_test


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="csadv-conv")
    p.add_argument("--Ng", nargs="+", type=int, required=True, help="List of Ng values, e.g. --Ng 9 13 17")
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--CFL", type=float, default=0.08)
    p.add_argument("--Ne", type=int, default=1)

    p.add_argument("--u0", type=float, default=1.0)
    p.add_argument("--alpha0", type=float, default=0.3)
    p.add_argument("--use-colat", action="store_true")

    # IC (default gaussian)
    p.add_argument("--ic", type=str, default="gaussian")
    p.add_argument("--lam0", type=float, default=0.0)
    p.add_argument("--lat0", type=float, default=0.0, help="latitude if not --use-colat; colatitude if --use-colat")
    p.add_argument("--sigma-m", type=float, default=None, help="Gaussian width in great-circle distance units")
    p.add_argument("--sigma-rad", type=float, default=None, help="Gaussian width in radians (converted by R)")
    p.add_argument("--amp", type=float, default=1.0)
    p.add_argument("--background", type=float, default=0.0)

    # boundary + rhs backends
    p.add_argument("--boundary", type=str, default="sat_inflow")
    p.add_argument("--bnd-backend", type=str, default="numpy", choices=["numpy", "numba"])
    p.add_argument("--rhs-backend", type=str, default="numpy", choices=["numpy", "numba"])

    p.add_argument("--periods", type=int, default=1)
    p.add_argument("--json", type=str, default=None, help="Write results to JSON file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # IC kwargs
    ic_kwargs: dict[str, Any] = dict(
        lam0=args.lam0,
        lat0_or_colat0=args.lat0,
        amp=args.amp,
        background=args.background,
        use_colat=bool(args.use_colat),
    )
    if (args.sigma_m is None) == (args.sigma_rad is None):
        raise SystemExit("Provide exactly one of --sigma-m or --sigma-rad.")
    if args.sigma_m is not None:
        ic_kwargs["sigma_m"] = float(args.sigma_m)
    else:
        ic_kwargs["sigma_rad"] = float(args.sigma_rad)

    res = run_convergence_test(
        list(args.Ng),
        R=float(args.R),
        CFL=float(args.CFL),
        Ne=int(args.Ne),
        u0=float(args.u0),
        alpha0=float(args.alpha0),
        use_colat=bool(args.use_colat),
        ic_name=str(args.ic),
        ic_kwargs=ic_kwargs,
        boundary_name=str(args.boundary),
        boundary_backend=str(args.bnd_backend),
        rhs_backend=str(args.rhs_backend),
        n_periods=int(args.periods),
    )

    # pretty print
    print("Ng    dt         L2          Linf        order_L2")
    for r in res:
        ord2 = r["order_L2"]
        ord2_s = "-" if ord2 is None else f"{ord2:8.3f}"
        print(f"{r['Ng']:>2d}  {r['dt']:.3e}  {r['L2']:.3e}  {r['Linf']:.3e}  {ord2_s}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Wrote: {args.json}")

    return 0