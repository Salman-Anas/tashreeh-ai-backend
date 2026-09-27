from app.services.normalize import (
    normalize_for_matching,
    normalize_urdu,
    normalize_with_map,
    strip_diacritics,
    urdu_ratio,
)


def test_arabic_yeh_and_alef_maksura_become_urdu_yeh() -> None:
    assert normalize_urdu("علي") == "علی"
    assert normalize_urdu("مدعى") == "مدعی"


def test_arabic_kaf_becomes_keheh() -> None:
    assert normalize_urdu("كتاب") == "کتاب"


def test_arabic_heh_becomes_heh_goal_but_do_chashmi_is_kept() -> None:
    assert normalize_urdu("فيصله") == "فیصلہ"
    assert normalize_urdu("کھلا") == "کھلا"  # ھ U+06BE untouched
    assert "ھ" in normalize_urdu("کھ")


def test_arabic_indic_digits_become_urdu_digits() -> None:
    assert normalize_urdu("دفعہ ٣٠٢") == "دفعہ ۳۰۲"


def test_normalize_urdu_keeps_diacritics() -> None:
    text = "عَدالت"
    assert normalize_urdu(text) == text


def test_matching_strips_diacritics_and_tatweel() -> None:
    assert normalize_for_matching("عَدالتِ عالیہ") == "عدالت عالیہ"
    assert normalize_for_matching("عدالـــت") == "عدالت"
    assert strip_diacritics("مُلزِم") == "ملزم"


def test_matching_folds_izafat_heh_hamza() -> None:
    assert normalize_for_matching("مجموعۂ تعزیرات") == normalize_for_matching("مجموعہ تعزیرات")


def test_decomposed_alef_madda_is_composed() -> None:
    assert normalize_for_matching("آئین") == "آئین"


def test_whitespace_is_collapsed() -> None:
    assert normalize_urdu("  عدالت   عالیہ \t نے  ") == "عدالت عالیہ نے"
    assert normalize_urdu("ایک\n\n\n\nدو") == "ایک\n\nدو"
    assert normalize_urdu("ایک\nدو", preserve_newlines=False) == "ایک دو"


def test_zwnj_is_cleaned() -> None:
    zwnj = "‌"
    assert normalize_urdu(f"بیان{zwnj}{zwnj}حلفی") == f"بیان{zwnj}حلفی"
    assert normalize_urdu(f"بیان {zwnj} حلفی") == "بیان حلفی"
    assert normalize_urdu(f"{zwnj}بیان") == "بیان"
    assert normalize_for_matching(f"بیان{zwnj}حلفی") == "بیانحلفی"
    assert normalize_for_matching("بیان​حلفی") == "بیانحلفی"


def test_normalize_with_map_points_back_to_original() -> None:
    original = "  مُلزِم  کی ضمانت"
    norm, idx = normalize_with_map(original)
    assert norm == "ملزم کی ضمانت"
    assert len(idx) == len(norm)
    start = norm.index("ضمانت")
    assert original[idx[start] : idx[start + len("ضمانت") - 1] + 1] == "ضمانت"


def test_empty_input() -> None:
    assert normalize_urdu("") == ""
    assert normalize_for_matching("") == ""


def test_urdu_ratio() -> None:
    assert urdu_ratio("ملزم کی ضمانت") == 1.0
    assert urdu_ratio("bail granted") == 0.0
    assert 0 < urdu_ratio("PLD 2024 SC 337 کے مطابق") < 1
