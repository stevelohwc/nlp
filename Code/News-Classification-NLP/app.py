"""Entrypoint for the News Classification Streamlit app.

UI rendering lives in frontend_page.py, and app logic/services live in app_core.py.
"""

from frontend_page import render_frontend_page


if __name__ == "__main__":
    render_frontend_page()
