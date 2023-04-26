"""Prepare the transcripts to be summarized.
Create a new dataset with the data homogenized. 

Apparently the format for the files changes from 079 onwards, 
clean the first ones to avoid the extra content.

Some trancripts seem to have the name of the speaker at the beginning (100.txt),
while others not (266.txt).

ONLY CLEAN A FILE IF IT DOESN'T EXIST YET
"""

import datetime as dt
from pathlib import Path

# IMPORT LOCAL DASK EXECUTOR
import spacy
from prefect import flow, task

import talking_python.settings as setts

original_filenames = setts.transcript_filenames()

nlp = spacy.load("en_core_web_sm")


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
        except:
            return False


def read_transcript(filename: Path) -> list[str]:
    with open(filename, "r", encoding="utf-8") as f:
        # Some of the files contain Unusual Line Endings.
        # They can be replaced with the following line,
        # as per: https://stackoverflow.com/questions/33910183/how-to-omit-u2028-line-separator-in-python
        return f.read().replace("\u2028", " ").replace("\xa0", " ").splitlines()


@task
def clean_file(contents: list[str], bs: int = 10) -> list[str]:
    """Clean a single file. 

    Args:
        contents (list[str]): A whole transcript file read.
        bs (int): batch size passed to nlp.pipe, defaults to 10.

    Returns:
        doc (spacy.tokens.doc.Doc): Processed file, ready to be written back.
    """
    cleaned = []
    for doc in nlp.pipe(contents, batch_size=bs):
        # Remove music insertions.
        if len(doc) == 0:
            # Remove blank lines
            continue
        if doc.text.startswith("[music"):
            # Remove music lines
            continue
        if is_datetimelike(doc[0].text):  # Remove the time at the beginning of a block 
            if doc.text.startswith("[music"):
                # FIXME: THIS IS NOT PASSING THE FILTER
                # Some cases, like in 010.txt, a line with time just informs of music
                continue
            if doc[:4].text.lower().startswith("welcome to talk python"):
                # Sometimes the presentation of the podcast can be easily ommited,
                # see 091.txt
                continue

            # Remove the time
            doc = doc[1:]

        else:
            # If its not a conversation, don't use it.
            # NOTE: Also, this imply losing some pieces, like in 
            # transcript 311-get-inside-the-git-folder.txt
            # where some lines don't start with the time in the conversation.
            continue

        cleaned.append(doc.text)

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
        for l in contents:
            f.write(l + "\n")


@flow
def clean_transcripts(filenames: list[Path]):
    """TODO: MIRAR LocalDaskExecutor para poder limpiarlos en paralelo. """
    cleaned_transcripts_dir = setts.Settings().cleaned_transcripts
    for filename in filenames:
        # TODO: Check only for files that weren't previously worked.
        transcript = read_transcript(filename)
        cleaned = clean_file(transcript)
        print(count_file(cleaned))
        write_doc(cleaned, cleaned_transcripts_dir / f"{filename.stem}_clean{filename.suffix}")


if __name__ == "__main__":
    clean_transcripts(original_filenames[2:4])
