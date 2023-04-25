"""Prefect flow to obtain the filenames of the transcripts. """

from pathlib import Path
from prefect import flow, task

from talking_python.settings import settings


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


@flow
def filenames_flow():
    filenames = get_filenames()
    write_filenames(filenames)
    return filenames


if __name__ == "__main__":
    filenames_flow()
