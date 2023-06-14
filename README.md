# README

Work in progress

Visit the app (deploy using streamlit?)

## Architecture

look for images and draw how everything is structured.

TODO

### Prefect flows

TODO

### Model chosen

TODO

[multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)

## Run locally

Ugly way to download the chroma database:

```bash
python -c "import talking_python.release as rel; rel.download_release_file(rel.get_release_url(), rel.get_chroma_dir().parent); rel.untar_file(rel.get_chroma_dir().parent / 'chroma.tar.gz')"
```

Without docker, run the following command inside `/app` dir:

```bash
streamlit run app.py
```

### Run with Docker

Example using docker:

```bash
docker build -t talking-python . && docker run -p 8501:8501 -it talking-python
```

Docker compose (it takes around 90 seconds on my machine).

```bash
docker compose up --build
```
