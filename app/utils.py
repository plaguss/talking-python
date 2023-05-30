"""General helper functions. """

from functools import lru_cache
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st

WEB_PAGE = r"https://talkpython.fm"
WEB_EPISODES = r"https://talkpython.fm/episodes/all"


def wrap_text_in_html(text: str | list[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])


def _get_episodes_table() -> pd.DataFrame:
    """Extracts the table of episodes.

    Returns:
        pd.DataFrame: _description_

    Example:
        ```console
        Show number        Date                                              Title             Guests
        0        #417  2023-05-30  Test-Driven Prompt Engineering for LLMs with P...  Maxime Beauchemin
        1        #416  2023-05-22          Open Source Sports Analytics with PySport        Koen Vossen
        2        #415  2023-05-15                     Future of Pydantic and FastAPI          Panelists
        3        #414  2023-05-07                         A Stroll Down Startup Lane          Panelists
        4        #413  2023-04-26                               Live from PyCon 2023          Panelists
        ...
        ```
    """
    return pd.read_html(WEB_EPISODES)[0]


@lru_cache
def get_episodes_table() -> dict[str, str]:
    """Grabs the table from the episodes page and creates
    a dict between the episode (transcript) number and the
    page for the episode.

    Returns:
        dict[str, str]: map from episode transcript name
            to episode page.
    """
    response = requests.get(WEB_EPISODES)
    soup = BeautifulSoup(response.content, "lxml")
    # Extract the table
    table = soup.find_all("table")[0].find("tbody")
    # Iterate over the rows, extract for the 3 column the links to the episodes
    links = [
        WEB_PAGE + row.find_all("td")[2].find("a", href=True).get("href")
        for row in table.find_all("tr")
    ]

    # Create the new column for the dataframe with the titles and the links to
    # the episodes. There must be a way to do this using pandas, but the lack of
    # internet connection made me lazy, plus the dataframe is small.
    title_urls = [
        f'<a href="{link}">{title}'
        for link, title in zip(links, episodes_table["Table"])
    ]

    episodes_table = _get_episodes_table()
    episodes_table["Title"] = title_urls

    return episodes_table


def show_episodes_table(episodes: list[int] | None = None):
    """Shows the subset of episodes in a table.

    Args:
        episodes (list[int] | None, optional):
            List of episodes to show in the table.
    """
    df = get_episodes_table()
    if episodes:
        df = df.loc[episodes]
    return st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
