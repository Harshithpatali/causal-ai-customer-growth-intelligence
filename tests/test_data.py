import pandas as pd
from causal_ai.data import validate_data

def test_validation_rejects_empty():
    result = validate_data(pd.DataFrame())
    assert not result["valid"]

def test_validation_detects_invalid_outcome():
    df = pd.DataFrame({
      "recency":[1],"history_segment":["1) $0 - $100"],"history":[20],
      "mens":[1],"womens":[0],"zip_code":["Urban"],"newbie":[0],
      "channel":["Web"],"segment":["No E-Mail"],"visit":[2],
      "conversion":[0],"spend":[0]
    })
    result = validate_data(df)
    assert not result["valid"]
    assert any("visit" in x for x in result["issues"])
