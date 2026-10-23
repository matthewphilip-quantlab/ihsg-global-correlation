import matplotlib

matplotlib.use("Agg")

from src import plots, quality as q  # noqa: E402
from src.config import sanity_ref  # noqa: E402
from tests.conftest import make_ohlc, random_walk  # noqa: E402


import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


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


def test_phase6_plot_renders():
    import pandas as pd
    c = pd.DataFrame({"p_rata": [0.4, 0.5, 0.6], "frekuensi_naik": [0.42, 0.5, 0.63], "n": [10, 10, 10]})
    assert plots.plot_calibration({"a": c, "b": c * 0.98}, "tes") is not None


def test_report_plots_render():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(rng.integers(200, 260, (3, 5)), index=list("ABC"), columns=range(2010, 2015))
    counts.iloc[0, 0] = np.nan
    assert plots.plot_availability(counts, "tes") is not None
    r = pd.Series(rng.standard_t(3, 2000) / 100, index=pd.bdate_range("2015-01-01", periods=2000))
    assert plots.plot_return_distribution(r, "tes") is not None
    coef = pd.DataFrame({"beta": [0.3, -0.1], "ci_lo": [0.1, -0.2], "ci_hi": [0.5, 0.0]}, index=["A", "B"])
    assert plots.plot_beta_forest(coef, {}, "tes", single=pd.DataFrame({"beta_1f": [0.4, 0.3]}, index=["A", "B"])) is not None
    assert plots.plot_premium(r.cumsum(), "tes") is not None
    assert plots.plot_haven_bars(pd.Series({"A": 0.1, "B": -0.05}), {}, "tes", highlight="B") is not None
    refits = pd.date_range("2020-01-01", "2022-12-01", freq="MS")
    assert plots.plot_walkforward_scheme(refits, "2010-01-01", "tes") is not None
    vf = pd.DataFrame({"r2": r.values ** 2, "s2_vix": 1e-4, "s2_tanpa_vix": 1.1e-4}, index=r.index)
    assert plots.plot_vol_forecast(vf, "tes") is not None
    plac = pd.DataFrame({"selisih_log_loss": rng.normal(0, 0.003, 10), "hit_rate": rng.uniform(0.5, 0.55, 10)})
    assert plots.plot_placebo(plac, {"selisih_log_loss": -0.003, "hit_rate": 0.58}, "tes") is not None
    assert plots.plot_bigcap_drawdown(pd.Series({"A": -0.2, "B": -0.3}), {}, "tes", reference=("IHSG", -0.18)) is not None
