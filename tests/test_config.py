import copy

import pytest

from src.config import series_table, validate_config


def test_core_has_ten_series_with_agreed_members(cfg):
    ids = [s["id"] for s in cfg["core"]]
    assert ids == ["STI", "KLCI", "EEM", "SPX", "NIFTY", "HSI", "VIX", "COPPER", "BRENT", "USDIDR"]


def test_overrides_and_pending_status(cfg):
    core = {s["id"]: s for s in cfg["core"]}
    assert core["HSI"]["override"] and core["BRENT"]["override"]
    assert core["USDIDR"]["status"] == "pending_phase2"


def test_dxy_kept_in_nanti_with_reason(cfg):
    nanti = {s["id"]: s for s in cfg["nanti"]}
    assert "VIX" in nanti["DXY"]["reason"]


def test_screening_window_locked_to_2010_2019(cfg):
    assert cfg["project"]["screening_window"] == ["2010-01-01", "2019-12-31"]


def test_series_table_default_groups(cfg):
    t = series_table(cfg)
    assert set(t["group"]) == {"target", "core", "tools"}
    assert len(t) == 1 + 10 + 1


def test_duplicate_ticker_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["core"].append(dict(bad["core"][0], id="DUP"))
    with pytest.raises(ValueError, match="lebih dari sekali"):
        validate_config(bad)


def test_unknown_category_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["core"][0]["category"] = "crypto"
    with pytest.raises(ValueError, match="category"):
        validate_config(bad)
