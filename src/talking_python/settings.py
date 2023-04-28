"""General settings. """

from pathlib import Path
from pydantic import BaseSettings


class Settings(BaseSettings):
    _root: Path = Path(__file__).parent.parent.parent
    transcripts_folder: Path = _root / "talk-python-transcripts/transcripts"
    flow_results: Path = _root / "flow_results"
    transcript_filenames: Path = flow_results / "transcript_filenames.txt"
    cleaned_transcripts: Path = flow_results / "cleaned_transcripts"
    file_lengths: Path = flow_results / "file_lengths.json"

settings = Settings()

if not settings.flow_results.is_dir():
    settings.flow_results.mkdir()


if not settings.cleaned_transcripts.is_dir():
    settings.cleaned_transcripts.mkdir()


def transcript_filenames() -> list[Path]:
    """Get the filenames with the original transcripts. """
    with settings.transcript_filenames.open() as f:
        filenames = [settings.transcripts_folder / f for f in f.read().splitlines()]
    return filenames
