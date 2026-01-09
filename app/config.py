from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

def get_openai_key() -> str | None:
    # 1) Streamlit secrets
    try:
        import streamlit as st
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    # 2) env var / .env
    return os.getenv("OPENAI_API_KEY")
