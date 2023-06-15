"""Simple streamlit app to explore talk python to me podcasts. """

import streamlit as st
import components.sidebar as sb
import utils as ut
from embeddings import (
    get_chroma,
    get_embedding_function,
    query_db,
    _match_aggregating_function,
)

TALK_PYTHON_LOGO = r"https://cdn-podcast.talkpython.fm/static/img/talk_python_logo_mic.png?cache_id=dd08157a0f56a88381ec34afe167db21"

# The following check is needed for streamlit cloud to download the data,
# if starting from docker this is done prior to deploying the app
import talking_python.release as rel

if not rel.get_chroma_dir().exists():
    print("chroma database wasn't found, downloading!")
    from talking_python.utilities import prepare_chroma
    prepare_chroma()


def clear_submit():
    st.session_state["submit"] = False


st.set_page_config(
    page_title="Explore Talk Python To Me", page_icon=":snake:", layout="wide"
)
st.header(":snake: Explore Talk Python To Me")

st.markdown(
    "##### Use [semantic search](https://www.sbert.net/examples/applications/semantic-search/README.html#background)"
    " to find episodes related to your queries."
)

sb.sidebar()

query = st.text_area("Ask a question", on_change=clear_submit)
with st.expander("Advanced Options"):
    # TODO: Explain this field is not the same as the number of episodes?
    max_episodes = int(
        st.number_input(
            "Set the maximum numbers of episodes to suggest. Defaults to 20",
            value=20,
            min_value=1,
            max_value=60,
        )
    )
    aggregating_function_name = st.selectbox(
        "Aggregating_function", ("average", "minimum", "raw")
    )


button = st.button("Submit")

if button or st.session_state.get("submit"):
    if not st.session_state.get("api_key_configured"):
        st.error("Please configure your HuggingFace API Token!")
    elif not query:
        st.error("Please enter a question!")
    else:
        try:
            emb_fn = get_embedding_function()
            chroma = get_chroma(emb_fn)

            with st.spinner("Querying the database... This may take a while⏳"):
                # TODO: The results should be aggregated outside to avoid
                # hitting the model again to embed the query and finding in
                # crhom.
                episode_titles = query_db(
                    chroma,
                    query=query,
                    n_results=max_episodes,
                    aggregating_function=_match_aggregating_function(
                        aggregating_function_name
                    ),
                )

            st.session_state["submit"] = True
            # Output column to show the results
            podcasts_col = st.columns(1)[0]

            with podcasts_col:
                st.markdown("#### Suggested podcasts")
                with st.spinner(
                    "Extracting episodes table... The first time may take a while⏳"
                ):
                    ut.show_episodes_table(episode_titles=episode_titles)

        except Exception as e:
            # Unexpected error, just print it
            st.error(e)
