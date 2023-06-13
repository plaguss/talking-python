"""Add a section with possible Frequently Asked Questions. """

import inspect

import streamlit as st


def faq():
    st.markdown(
        inspect.cleandoc(
            """
        # FAQ
        ## How does Explore Talk Python To Me work?
        The podcast's transcripts have been embedded in a [chroma](https://docs.trychroma.com/)
        database that allows for semantic search and retrieval. When you insert
        a question, the text is embedded, and chroma is in charge of finding
        the most similar pieces of content to show.

        ## In need of inspiration? Here are some ideas
        - how to run machine learning models in production
        - rich and terminal user interfaces
        - content about the python package ecosystem

        ## Limits using the 🤗Hugging Face API token
        Please visit [Hugging Face API FAQ](https://huggingface.co/docs/api-inference/faq#more-information-about-the-api)
        for more information.

        ## Additional information
        Please visit the [repo](https://github.com/plaguss/talking-python).
        """
        )
    )
