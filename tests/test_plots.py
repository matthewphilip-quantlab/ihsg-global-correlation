import matplotlib

matplotlib.use("Agg")

from src import plots, quality as q  # noqa: E402
from tests.conftest import make_ohlc, random_walk  # noqa: E402


def test_all_phase1_plots_render(cfg):
    s = random_walk(start="2007-01-01", end="2012-12-31", level=2000)
    ref = cfg["sanity_checks"]["jkse_2008"]
    assert plots.plot_jkse_2008(s, ref) is not None
    _, per_year = q.open_quality(make_ohlc(s))
    assert plots.plot_open_quality(per_year, cfg["quality"]["open_checks"]) is not None
    mv = q.flag_moves(s, "log", 0.02)
    assert plots.plot_flagged_moves(s, mv, "tes") is not None
