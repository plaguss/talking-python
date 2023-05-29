import inspect

import streamlit as st


def faq():
    st.markdown(
        inspect.cleandoc(
            """
        # FAQ
        ## How does Explore Talking Python work?
        The podcast's transcripts have been embedded in a [chroma](https://docs.trychroma.com/)
        database that allows for semantic search and retrieval.
        **ADD GUIDE TO THE PROCESS**

        When you upload a document, it will be divided into smaller chunks 
        and stored in a special type of database called a vector index 
        that allows for semantic search and retrieval.

        When you ask a question, the model finds the nearest passage of text
        in the podcast's transcripts and suggests the most similar.
        **NEEDS A FUNCTION TO SORT THE RESULTS. THE MAX, SOME KIND OF AVERAGE?**

        ## Limits using the 🤗Hugging Face API token
        Please visit [Hugging Face API FAQ](https://huggingface.co/docs/api-inference/faq#more-information-about-the-api)
        for more information.

        ## What do the numbers mean under each source?
        EXPLAIN RESULTS OBTAINED FOR THE MODEL

        ## Are the answers 100% accurate?
        **EXPLAIN IN RELATION TO THE MODEL USED, STARTING WITH:**
        **[multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)**
        """
        )
    )
