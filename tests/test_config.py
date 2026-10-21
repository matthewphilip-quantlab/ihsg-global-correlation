import copy

import pytest

from src.config import (load_config, output_dir, sanity_ref, series_table, target_ticker, target_usd_id,
                        validate_config)


def test_core_has_ten_series_with_agreed_members(cfg):
    ids = [s["id"] for s in cfg["core"]]
    assert ids == ["STI", "KLCI", "EEM", "SPX", "NIFTY", "HSI", "VIX", "COPPER", "BRENT", "USDIDR"]


def test_overrides_and_usdidr_status(cfg):
    core = {s["id"]: s for s in cfg["core"]}
    assert core["HSI"]["override"] and core["BRENT"]["override"]
    assert core["USDIDR"]["status"] == "driver"            # hasil uji ulang Phase 2


def test_phase2_rules_in_config(cfg):
    assert cfg["quality"]["fx_cleaning"] == {"max_reversal_days": 5, "max_reversal_days_old": 1}
    assert cfg["usdidr_retest"]["min_abs_corr"] == 0.300


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


# ---- target configurable -------------------------------------------------------------
def test_default_target_is_jkse(cfg):
    assert cfg["target"][0]["ticker"] == "^JKSE" and target_ticker(cfg) == "^JKSE"
    assert output_dir(cfg, "reports", "phase1").parts[-2:] == ("reports", "phase1")
    assert sanity_ref(cfg)["suspended_days"] == cfg["project"]["suspended_days"]


def test_target_switch_without_code_change():
    cfg = load_config(target_id="BBCA")
    t = series_table(cfg)
    assert t.loc[t["group"] == "target", "ticker"].tolist() == ["BBCA.JK"]
    assert len(t) == 1 + 10 + 1                             # Core & tools tidak berubah
    assert output_dir(cfg, "reports", "phase1").parts[-4:] == ("reports", "targets", "BBCA", "phase1")
    assert sanity_ref(cfg) is None and target_usd_id(cfg) == "BBCA_USD"


def test_unknown_target_rejected():
    with pytest.raises(ValueError, match="targets"):
        load_config(target_id="XXXX")


def test_target_catalog_must_be_target_session(cfg):
    bad = copy.deepcopy(cfg)
    bad["targets"][1]["session"] = "sync"
    with pytest.raises(ValueError, match="session 'target'"):
        validate_config(bad)
