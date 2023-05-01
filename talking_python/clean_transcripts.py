"""Prepare the transcripts to be summarized.
Create a new dataset with the data homogenized.

Apparently the format for the files changes from 079 onwards, 
clean the first ones to avoid the extra content.

Some trancripts seem to have the name of the speaker at the beginning (100.txt),
while others not (266.txt).

There are problems when running in a local DaskTaskRunner in the
ephemeral mode:
https://github.com/PrefectHQ/prefect/issues/7277
Prior to running the flow, `prefect orion start`, set
the PREFECT_API_URL as informed.
At least locally, its better to limit the maximum number of concurrent processes,
or the computer may crash.

To see the evolution of the flow:
http://127.0.0.1:8787/status

"""

import datetime as dt
import json
from pathlib import Path

import spacy
from prefect import flow, task
from prefect.logging import get_logger

from dotenv import dotenv_values
from pydantic import BaseSettings

nlp = spacy.load("en_core_web_sm")

log = get_logger()

config = dotenv_values(".env")


class Settings(BaseSettings):
    # _root: Path = Path(__file__).parent.parent.parent
    _root: Path = Path(__file__).parent.parent.resolve()
    transcripts_folder: Path = _root / "talk-python-transcripts/transcripts"
    flow_results: Path = _root / "flow_results"
    transcript_filenames: Path = flow_results / "transcript_filenames.txt"
    cleaned_transcripts: Path = flow_results / "cleaned_transcripts"
    file_lengths: Path = flow_results / "file_lengths.json"
    flow_environ_local: bool = True if config.get("FLOW_ENVIRON") == "local" else False


settings = Settings()

log.info(f"CHECK FILES FOUND IN ROOT: {list(settings._root.iterdir())}")

if not settings.flow_results.is_dir():
    settings.flow_results.mkdir()


if not settings.cleaned_transcripts.is_dir():
    settings.cleaned_transcripts.mkdir()


def transcript_filenames() -> list[Path]:
    """Get the filenames with the original transcripts."""
    with settings.transcript_filenames.open() as f:
        filenames = [settings.transcripts_folder / f for f in f.read().splitlines()]
    return filenames


if settings.flow_environ_local:
    from prefect_dask.task_runners import DaskTaskRunner

    task_runner = DaskTaskRunner(adapt_kwargs={"maximum": 4})
else:
    from prefect.task_runners import SequentialTaskRunner

    task_runner = SequentialTaskRunner()


@task
def get_filenames() -> list[Path]:
    filenames = []
    for filename in settings.transcripts_folder.iterdir():
        # Just check the filenames start with 3 digits and has .txt extension
        if filename.name[:3].isalnum() and filename.suffix == (".txt"):
            filenames.append(filename)

    # Sort the filenames acording to the 3 first digits of the filename.
    filenames = sorted(filenames, key=lambda x: int(x.stem[:3]))
    return filenames


@task
def write_filenames(filenames: list[Path]) -> None:
    with open(settings.transcript_filenames, "w") as f:
        for fname in filenames:
            f.write(fname.name + "\n")


def is_datetimelike(content: str) -> bool:
    """Check if a string resembles to a date.
    Used to remove the possible dates written in the trancripts.
    """
    content = content.strip()
    try:
        return bool(dt.datetime.strptime(content, "%M:%S"))
    except ValueError:
        try:
            return bool(dt.datetime.strptime(content, "%H:%M:%S"))
        except ValueError:
            return False


def read_transcript(filename: Path) -> list[str]:
    """Read a transcript from a .txt file.

    Some of the files contain Unusual Line Endings.
    They can be replaced with the following line,
    as per: https://stackoverflow.com/questions/33910183/how-to-omit-u2028-line-separator-in-python

    Args:
        filename (Path):

    Returns:
        list[str]: Contents of the transcript.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().replace("\u2028", " ").replace("\xa0", " ").splitlines()


@task(tags=["clean-file"])
def clean_file(filename: Path, bs: int = 10) -> list[str]:
    """Clean a single file.

    Args:
        contents (list[str]): A whole transcript file read.
        bs (int): batch size passed to nlp.pipe, defaults to 10.

    Returns:
        doc (spacy.tokens.doc.Doc): Processed file, ready to be written back.
    """
    contents = read_transcript(filename)
    cleaned = []
    for doc in nlp.pipe(contents, batch_size=bs):
        # Remove music insertions.
        if len(doc) == 0 or doc.text.startswith("[music"):
            # Remove blank lines and music lines
            continue
        if is_datetimelike(doc[0].text):
            doc = doc[1:]  # Remove the time of a turn
            if (
                doc.text.lstrip().startswith("[music")
                or doc[:4].text.lower().startswith("welcome to talk python")
                or doc[:5].text.lower().startswith("hello and welcome to talk")
            ):
                # Some cases, like in 010.txt, a line with time just informs of music.
                # Sometimes the presentation of the podcast can be easily ommited,
                # see 091.txt
                continue

        else:
            # If its not a conversation, don't use it.
            # NOTE: Also, this means losing some pieces, like in
            # transcript 311-get-inside-the-git-folder.txt
            # where some lines don't start with the time in the conversation.
            continue

        cleaned.append(doc.text.strip())

    return cleaned


@task
def count_file(contents: list[str], bs: int = 10):
    """Count the number of words in a single file.
    Do it after it has cleaning it up.
    """
    return sum([len(doc) for doc in nlp.pipe(contents, batch_size=bs)])


@task
def write_doc(contents: list[str], filename: str):
    with open(filename, "w") as f:
        for line in contents:
            f.write(line + "\n")


class FileLengths:
    """Store the file lengths, maybe loading them if there are already
    written ones.
    """

    def __init__(self, path: Path = settings.file_lengths) -> None:
        if path.exists():
            try:
                with open(path, "r") as f:
                    self._file_lengths = json.load(f)
            except Exception:
                # There exists a file, but is malformed,
                # start from a new one directly
                self._file_lengths = {}
        else:
            self._file_lengths = {}

    def __setitem__(self, filename: str | Path, length: int):
        self._file_lengths[filename] = length

    @property
    def file_lengths(self) -> dict[str, int]:
        return self._file_lengths

    def write(self, path: Path = settings.file_lengths) -> None:
        with open(path, "w") as f:
            json.dump(self._file_lengths, f, indent=2)


@flow(task_runner=task_runner)
def clean_transcripts():
    """Gets the filenames of the transcripts, cleans them and writes
    the new files.
    """
    filenames = get_filenames()
    write_filenames(filenames)

    cleaned_transcripts_dir = settings.cleaned_transcripts
    file_lenghts = FileLengths()
    lengths = {}
    for filename in filenames:
        # TODO: Check only for files that weren't previously worked.
        clean_filename = (
            cleaned_transcripts_dir / f"{filename.stem}_clean{filename.suffix}"
        )
        if clean_filename.is_file():
            log.info(f"File already exists, skip: {clean_filename}")
            continue
        log.info(f"Processing transcript: {filename}")
        cleaned = clean_file.submit(filename)
        count_future = count_file.submit(cleaned)
        lengths[str(filename.name)] = count_future  # intermediate storage
        write_doc.submit(cleaned, clean_filename)

    # We do this to allow running concurrently the previous tasks
    for fname, length in lengths.items():
        file_lenghts[fname] = length.result()

    file_lenghts.write()


if __name__ == "__main__":
    # To run a subset of the files just pass a subset of the list
    clean_transcripts()
