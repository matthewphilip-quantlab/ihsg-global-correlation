import matplotlib

matplotlib.use("Agg")

from src import plots, quality as q  # noqa: E402
from src.config import sanity_ref  # noqa: E402
from tests.conftest import make_ohlc, random_walk  # noqa: E402


def test_all_phase1_plots_render(cfg):
    s = random_walk(start="2007-01-01", end="2012-12-31", level=2000)
    ref = sanity_ref(cfg)
    assert plots.plot_jkse_2008(s, ref) is not None
    _, per_year = q.open_quality(make_ohlc(s))
    assert plots.plot_open_quality(per_year, cfg["quality"]["open_checks"]) is not None
    mv = q.flag_moves(s, "log", 0.02)
    assert plots.plot_flagged_moves(s, mv, "tes") is not None


def test_phase2_plots_render():
    raw = random_walk(start="2013-01-01", end="2013-12-31", level=10000)
    clean = raw.copy()
    clean.iloc[[50, 51]] = float("nan")
    assert plots.plot_fx_cleaning(raw, clean, "tes") is not None
    jkse = random_walk(start="2013-01-01", end="2013-12-31", level=4300, seed=5)
    assert plots.plot_idr_vs_usd(jkse, jkse / raw, "tes") is not None


def test_phase3_plots_render():
    import numpy as np
    import pandas as pd
    m = pd.DataFrame(np.eye(3) * 0.5 + 0.5, index=list("ABC"), columns=list("ABC"))
    assert plots.plot_corr_heatmap(m, "tes") is not None
    tab = pd.DataFrame({"id": ["A", "B"] * 2, "periode": ["screening"] * 2 + ["pasca"] * 2,
                        "r": [0.5, -0.2, 0.4, -0.1], "ci_lo": [0.4, -0.3, 0.3, -0.2], "ci_hi": [0.6, -0.1, 0.5, 0.0]})
    assert plots.plot_corr_by_period(tab, {"A": "a"}, "tes", periods=("screening", "pasca")) is not None
    asym = pd.DataFrame({"id": ["A", "B"], "r_turun": [0.6, 0.2], "r_naik": [0.3, 0.1]})
    assert plots.plot_asymmetric(asym, {}, "tes") is not None


def test_phase4_plots_render():
    import numpy as np
    import pandas as pd
    ccf = pd.DataFrame({"id": ["A"] * 3 + ["B"] * 3, "k": [-1, 0, 1] * 2, "r": [0.0, 0.1, 0.3, 0.0, 0.5, 0.1]})
    assert plots.plot_ccf_heatmap(ccf, {"A": "a"}, {"A": "after", "B": "sync"}, "tes") is not None
    z = pd.Series(np.random.default_rng(0).normal(size=300), index=pd.bdate_range("2015-01-01", periods=300))
    assert plots.plot_spread_z(z, "tes") is not None


def test_phase5_plots_render():
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2010-01-01", periods=300, freq="W-FRI")
    r = pd.DataFrame({"A": np.linspace(0.2, 0.6, 300), "B": 0.3}, index=idx)
    assert plots.plot_rolling_corr({52: r, 156: r * 0.9}, {"A": "a"}, [(idx[10], idx[20])], "tes") is not None
    p = pd.DataFrame({"T": np.linspace(100, 80, 81), "P": 100.0}, index=pd.Index(range(-20, 61), name="hari"))
    assert plots.plot_event_paths({"2008": p, "2013": p}, {}, "tes", notes={"2013": "x"}) is not None
    q = pd.DataFrame({"kuantil": [0.1, 0.5, 0.9, "OLS"], "variabel": "B", "beta": [0.1, 0.0, -0.1, 0.0],
                      "ci_lo": [0.0, -0.1, -0.2, -0.05], "ci_hi": [0.2, 0.1, 0.0, 0.05]})
    assert plots.plot_quantile_coefs({"a": q, "b": q}, "B", "tes") is not None
