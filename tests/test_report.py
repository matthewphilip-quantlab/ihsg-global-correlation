from src import report as R


def _example(a, b):
    """Docstring yang harus hilang dari potongan."""
    x = a + b

    y = x * 2
    for i in range(3):
        y += i
    return y


def test_snippet_drops_docstring_and_blank_lines_and_truncates():
    s = R.snippet(_example, 15)
    assert "Docstring" not in s and "\n\n" not in s and s.startswith("def _example")
    short = R.snippet(_example, 3).splitlines()
    assert len(short) == 3 and short[-1].strip() == "..."


def test_fmt_id_indonesian_style():
    assert R.fmt_id(0.3340, 3) == "0,334"
    assert R.fmt_id(-0.229, 2) == "−0,23"
    assert R.fmt_id(0.095, 1, sign=True, pct=True) == "+9,5%"
    assert R.fmt_id(4798, 0) == "4.798"


def test_crosscheck_accepts_ascii_or_unicode_minus():
    text = "beta −0,23 dan p = 0,77; lain -0,71"
    res = R.crosscheck([("a", "−0,23"), ("b", "0,77"), ("c", "−0,71"), ("d", "0,99")], text)
    assert res["ada_di_README"].tolist() == [True, True, True, False]
