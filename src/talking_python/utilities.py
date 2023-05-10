from pathlib import Path


def read_transcript(filename: Path) -> list[str]:
    """Read a clean transcript from a .txt file.

    Args:
        filename (Path):

    Returns:
        list[str]: Contents of the transcript.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().splitlines()
