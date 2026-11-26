from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


EXECUTIVE_PAGES = [
    "Executive summary", "Customer insights", "Campaign planner", "AI Business Reports",
]
DATA_SCIENCE_PAGES = [
    "Dataset overview", "Data quality", "Exploratory data analysis", "Feature analysis",
    "Correlation analysis", "Statistical analysis", "Experiment evidence",
    "Uplift model diagnostics", "Budget model assumptions",
]


def _sample_data(n=240):
    rng = np.random.default_rng(2026)
    segments = np.array(["No E-Mail", "Mens E-Mail", "Womens E-Mail"] * (n // 3))
    rng.shuffle(segments)
    treatment = (segments != "No E-Mail").astype(int)
    spend = np.where(rng.random(n) < (0.008 + 0.004 * treatment), rng.gamma(2, 20, n), 0.0)
    return pd.DataFrame({
        "recency": rng.integers(1, 13, n),
        "history_segment": rng.choice(["1) $0 - $100", "2) $100 - $200", "3) $200 - $350"], n),
        "history": rng.gamma(2, 80, n),
        "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Suburban", "Rural"], n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
        "segment": segments,
        "treatment": treatment,
        "treatment_label": np.where(treatment == 1, "Email treatment", "Control"),
        "visit": rng.integers(0, 2, n),
        "conversion": (spend > 0).astype(int),
        "spend": spend,
    })


def test_every_main_dashboard_page_renders_without_streamlit_exceptions():
    frame = _sample_data()
    scored = frame.assign(
        predicted_spend_control=0.8,
        predicted_spend_treatment=1.0,
        estimated_incremental_spend=0.2,
    )

    with (
        patch("causal_ai.data.load_data", return_value=frame),
        patch("causal_ai.modeling.load_models", return_value=object()),
        patch("causal_ai.modeling.score_customers", return_value=scored),
    ):
        app = AppTest.from_file(REPO_ROOT / "app.py", default_timeout=30).run()
        assert not app.exception

        app.sidebar.radio[0].set_value("Executive view")
        app.run()
        for page in EXECUTIVE_PAGES:
            app.sidebar.radio[1].set_value(page)
            app.run()
            assert not app.exception, f"Executive page failed: {page}"

        app.sidebar.radio[0].set_value("Data science view")
        app.run()
        for page in DATA_SCIENCE_PAGES:
            app.sidebar.radio[1].set_value(page)
            app.run()
            assert not app.exception, f"Data science page failed: {page}"


def test_research_analytics_page_renders_from_published_reports():
    frame = _sample_data()
    scored = frame.assign(
        predicted_spend_control=0.8,
        predicted_spend_treatment=1.0,
        estimated_incremental_spend=0.2,
    )
    with (
        patch("causal_ai.data.load_data", return_value=frame),
        patch("causal_ai.modeling.load_models", return_value=object()),
        patch("causal_ai.modeling.score_customers", return_value=scored),
    ):
        app = AppTest.from_file(REPO_ROOT / "app.py", default_timeout=30).run()
        app.switch_page("pages/2_Research_Analytics.py").run()
        assert not app.exception
