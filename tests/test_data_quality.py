import pandas as pd

from causal_ai.data import validate_data


def valid_minimal_frame():
    return pd.DataFrame({
        "recency": [2, 5],
        "history_segment": ["1) $0 - $100", "2) $100 - $200"],
        "history": [50.0, 150.0],
        "mens": [1, 0],
        "womens": [0, 1],
        "zip_code": ["Urban", "Rural"],
        "newbie": [0, 1],
        "channel": ["Web", "Phone"],
        "segment": ["Mens E-Mail", "No E-Mail"],
        "visit": [1, 0],
        "conversion": [1, 0],
        "spend": [12.0, 0.0],
    })


def test_validation_accepts_valid_rows():
    result = validate_data(valid_minimal_frame())
    assert result["valid"]
    assert result["rows"] == 2


def test_validation_rejects_empty_frame():
    result = validate_data(pd.DataFrame())
    assert not result["valid"]
    assert "Dataset is empty" in result["issues"]


def test_validation_detects_missing_required_values():
    frame = valid_minimal_frame()
    frame.loc[0, "history"] = float("nan")
    result = validate_data(frame)
    assert not result["valid"]
    assert any("missing cells" in issue for issue in result["issues"])


def test_validation_detects_conversion_spend_inconsistency():
    frame = valid_minimal_frame()
    frame.loc[0, "conversion"] = 0
    result = validate_data(frame)
    assert not result["valid"]
    assert any("conversion must equal" in issue for issue in result["issues"])


def test_validation_rejects_negative_spend():
    frame = valid_minimal_frame()
    frame.loc[0, "spend"] = -1.0
    result = validate_data(frame)
    assert not result["valid"]
    assert any("spend must be non-negative" in issue for issue in result["issues"])
