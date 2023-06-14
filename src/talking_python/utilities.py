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


def prepare_chroma() -> None:
    """Function to download chroma database, used inside docker. """
    import talking_python.release as rel
    if not rel.get_chroma_dir().parent.exists():
        rel.get_chroma_dir().parent.mkdir(parents=True)

    rel.download_release_file(rel.get_release_url(), rel.get_chroma_dir().parent)
    rel.untar_file(rel.get_chroma_dir().parent / "chroma.tar.gz")
