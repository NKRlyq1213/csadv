from csadv.cli import main

def test_cli_smoke_runs_zero_periods(capsys):
    code = main([
        "--Ng", "5", "7",
        "--sigma-m", "0.35",
        "--periods", "0",
        "--rhs-backend", "numpy",
        "--bnd-backend", "numpy",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "Ng" in out and "L2" in out