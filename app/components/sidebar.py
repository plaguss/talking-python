import streamlit as st

from components.faq import faq


def set_huggingface_api_token(api_key: str):
    st.session_state["HF_API_TOKEN"] = api_key


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your [🤗HuggingFace API token](https://huggingface.co/settings/tokens) below🔑\n"  # noqa: E501
            "2. Ask a question about the podcasts💬\n"
        )
        api_key_input = st.text_input(
            "HuggingFace API Token",
            type="password",
            placeholder="Paste your HuggingFace API token here (hf_...)",
            help="You can get your API token from https://huggingface.co/settings/tokens.",  # noqa: E501
            value=st.session_state.get("HF_API_TOKEN", ""),
        )

        if api_key_input:
            set_huggingface_api_token(api_key_input)

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            ":snake: Explore Talking Python allows you to ask questions about your "
            "documents and get accurate answers with instant citations. "
        )
        st.markdown(
            "This tool is a work in progress. "
            "You can contribute to the project on [GitHub](https://github.com/plaguss/talking-python) "  # noqa: E501
            "with your feedback and suggestions💡"
        )
        st.markdown(
            "Made by [Agus](https://github.com/plaguss), thanks to [Michael Kennedy](https://talkpython.fm/)"
        )
        st.markdown("---")

        faq()
