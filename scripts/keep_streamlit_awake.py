"""Visit the public Streamlit demo and click its wake-up control if it is hibernating."""
import os
import re
import sys
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

APP_URL = os.getenv(
    "STREAMLIT_APP_URL",
    "https://causal-ai-customer-growth-intelligence.streamlit.app/",
)


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        try:
            response = page.goto(
                APP_URL,
                wait_until="domcontentloaded",
                timeout=120_000,
            )
            if response is not None and response.status >= 400:
                raise RuntimeError(f"Streamlit returned HTTP {response.status}")

            wake_button = page.get_by_role(
                "button",
                name=re.compile(r"get this app back up", re.IGNORECASE),
            )
            try:
                wake_button.first.wait_for(state="visible", timeout=5_000)
            except PlaywrightTimeoutError:
                print("No wake-up control found; waiting for the live app.")
            else:
                print("The app is hibernating. Clicking its wake-up control.")
                wake_button.first.click(timeout=15_000)

            # Keep an actual browser session open while Streamlit starts.
            page.get_by_text("CAUSAL AI", exact=False).first.wait_for(
                state="visible",
                timeout=360_000,
            )
            print(f"Streamlit app is responding: {page.url}")
        except Exception as exc:
            body = ""
            try:
                body = page.locator("body").inner_text(timeout=5_000)[:800]
            except Exception:
                pass
            print(f"Streamlit keep-alive check failed: {exc}", file=sys.stderr)
            if body:
                print(f"Page excerpt: {body}", file=sys.stderr)
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
