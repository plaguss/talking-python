"""_summary_

[text datasets](https://huggingface.co/docs/datasets/nlp_load)

Returns:
    _type_: _description_
"""

from pathlib import Path

from datasets import load_dataset

from dotenv import dotenv_values
from prefect.logging import get_logger
from pydantic import BaseSettings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


log = get_logger()

config = dotenv_values(".env")


class Settings(BaseSettings):
    _root: Path = Path(__file__).parent.parent.resolve()
    transcripts_folder: Path = _root / "talk-python-transcripts/transcripts"
    flow_results: Path = _root / "flow_results"
    cleaned_transcripts: Path = flow_results / "cleaned_transcripts"
    summaries: Path = flow_results / "summaries"
    flow_environ_local: bool = True if config.get("FLOW_ENVIRON") == "local" else False


settings = Settings()


if not settings.summaries.is_dir():
    settings.summaries.mkdir()


def read_clean_transcript(filename):
    with filename.open("r") as f:
        return f.read()


# dataset = load_dataset("text", data_dir=str(settings.cleaned_transcripts), sample_by="document")  # LOAD THE WHOLE DATASET

import random

random.seed(1234)

filenames = list(str(f) for f in settings.cleaned_transcripts.iterdir())
subset_filenames = random.choices(filenames, k=3)

dataset = load_dataset("text", data_files=subset_filenames, sample_by="document")
