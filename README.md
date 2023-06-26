<h1 align="center">Explore Talk Python To Me</h1>

If you don't already know [Talk Python To Me](https://talkpython.fm/) maybe you should visit its page first, and take a look at its [episodes](https://talkpython.fm/episodes/all).

Already done? Then you can take a look at the demo app :): [explore-talk-python-to-me](https://explore-talk-python-to-me.streamlit.app/).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://explore-talk-python-to-me.streamlit.app/)

This repository contains all the code behind the demo app *explore-talk-python-to-me*, a demo of how to look for episodes related to your preferences using natural language. You can see from the code to embed the podcast's episodes, the prefect flows run on github action to update the contents as new episodes are added, to the final streamlit app.

Table Of Content

1. [How it works](#how-it-works)

    1.1. [Deployment](#deployment)

    1.2. [UI](#ui)

    1.3. [Prefect Flows](#prefect-flows)

2. [Model behind the app](#model-behind-the-app)
3. [Running the app](#prefect-flows)
4. [Further Steps](#further-steps)

---

## How it works

Lets explore the different parts behind the app by visiting the different parts of the following figure:

![architecture](./assets/arch.png)

The first step we need is to obtain the episodes transcriptions, which Michael Kennedy is kind enough to offer on a [talk-python-transcripts](https://github.com/talkpython/talk-python-transcripts). 

---

<details>
  <summary> 🎬 Demo </summary>
  <hr>

    Video demo

https://user-images.githubusercontent.com/554369/197188237-88d3f7e4-4e5f-40b5-b996-c47b19ee2f49.mov

 </details>

### Prefect flows

The transcripts are added on a weekly basis, the frequency of new episodes. We can keep *talk-python-transcripts* as a *git submodule* and pull the contents regularly using cron on a github action, which can be seen in 
`download-transcripts.yml`.

- [`download-transcripts.yml`](./.github/workflows/download_transcripts.yml)

    This github action, which corresponds to the point 1) in the architecture figure is in charge of running the following prefect flow [clean_transcripts.py](./flows/clean_transcripts.py), which downloads the
    transcripts in the submodule, *cleans* the content (some simple preprocessing to remove unnecessary content for the final embeddings[^1]), and adds the new files in the `/flow_results` for posterior use
    (the repository itself works as storage, point 2)).

A second GitHub action runs another prefect flow after the first has finished:

- [`embed.yml`](./.github/workflows/embed.yml)

    This github action is in charge of running the prefect flow in [embed.py](./flows/embed.py), 5 minutes after *download_transcripts.yml* with the same frequency (point 3)). It downloads the latest chroma database released, embeds the new episodes that were added since the last run, and releases the new chroma database to the github repo.

[^1]: There is an error in the code and some transcripts are not being properly processed yet!.


### `talking-python` repo and the vector store

This repository works as a datastore in two ways.

- When the first flow 

HABLAR DE PUNTOS 2 Y 4, COMO FUNCIONA COMO ALMACEN PARA LOS EPISODIOS LIMPIOS Y LOS RELEASES

points 2) and 4).

Function of the repo and how we work with the vector store (chroma) and github releases.

We can [deploy chroma](https://docs.trychroma.com/deployment), but this is a simple demo, so I had to come up with something simpler (and cheaper). 

Instead, the chroma content is persisted in [parquet in disk](https://docs.trychroma.com/usage-guide#initiating-a-persistent-chroma-client), released to github actions as a compressed asset, and wherever we want to make queries to the data (when deploying on *streamlit cloud* or using *Docker*), we can download the vector store, just like we would do with any deep learning model.

### UI

The app is built using [streamlit](https://streamlit.io/), the code can be seen in the [`app`](./app/) folder. The layout is heavily inspired by [KnowledgeGPT](https://knowledgegpt.streamlit.app/).

#### Deployment

Given the app is deployed on [streamlit cloud](https://streamlit.io/cloud), we have to prepare the data in the way streamlit wants it.

Streamlit will look for a [*requirements.txt*](./requirements.txt) at the root of the repo, and we have two different internal libraries, so some unusual process must be done to go the streamlit way, vs for example installing the libraries inside a Dockerfile.

[Pip](https://pip.pypa.io/en/latest/reference/requirements-file-format/) allows us to install libraries directly from specific local distributions, so the libraries can be prebuilt and stored:

Build the UI code:

```bash
python -m build app --outdir=streamlit-dir/app
``` 

and the library:

```bash
python -m build src --outdir=streamlit-dir/src
``` 

Once the app is succesfully uploaded to streamlit, the new changes are automatically redeployed.

## Model behind the embeddings

REWRITE THIS TO HAVE A SECTION WITH ALL THE MODEL AND AGGREGATING FUNCTIONS.

The model chosen is sentence transformer's [multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) (take a look at the [intended uses](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1#intended-uses) section), apart from that is integrated with [chroma](https://docs.trychroma.com/embeddings#sentence-transformers), and [hugging face's](https://github.com/chroma-core/chroma/blob/main/chromadb/utils/embedding_functions.py#L140) inference API.

From the user perspective, the interaction with the model occurs when a query is made (point 6) in the figure). The query is embedded using the [Hugging Face inference API](https://huggingface.co/inference-api), hence you must supply the required api token.

## Running the app

Chose your preferred way: 

<details><summary> With Docker </summary><hr>

It can be built using docker directly:

```bash
docker build -t talking-python . && docker run -p 8501:8501 -it talking-python
```

Or if you prefer docker compose:

```bash
docker compose up --build
```

Visit the url that streamlit shows you in the console.

</details>

<details><summary> Without Docker </summary><hr>

```bash
python -c "import talking_python.release as rel; rel.download_release_file(rel.get_release_url(), rel.get_chroma_dir().parent); rel.untar_file(rel.get_chroma_dir().parent / 'chroma.tar.gz')"
```

Without docker, run the following command inside `/app` dir:

```bash
streamlit run app.py
```

Visit the url that streamlit shows you in the console.

</details>


## Further steps

...

