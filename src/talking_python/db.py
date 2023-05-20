"""This module contains the functionalities to work with the chroma
directory. It shouldn't be necessary if the data was stored in a
database.
"""

import datetime as dt
import os
import tarfile
from pathlib import Path

from ghapi.all import GhApi


def get_chroma_dir() -> Path:
    """Returns the path to the chroma content where the
    vectors are stored.

    Returns:
        Path: chroma directory.
    """
    return Path.home() / ".cache" / "talking-python" / ".chroma"


def get_repo_access_token() -> str | None:
    """Get the github api access token for the repo.

    Needed to upload content to github releases.

    Returns:
        str: token.
    """
    return os.getenv("TALKING_PYTHON_ACCESS_TOKEN", None)


def generate_release_name() -> str:
    """Generates the name for a release.
    It contains the date in isoformat to helo with 
    versioning.

    Returns:
        str: release name.
    """
    return f"talking_python_chroma_{dt.date.today().isoformat()}"


def make_tarfile(source: Path) -> None:
    """Creates a tar file from a directory and compresses it
    using gzip.

    Args:
        source (Path): Path to a directory.

    Raises:
        FileNotFoundError: If the directory doesn't exists.
    """
    if not source.is_dir():
        raise FileNotFoundError(source)
    with tarfile.open(str(source) + ".tar.gz", "w:gz") as tar:
        tar.add(str(source), arcname=source.name)
    print(f"File generated at: {str(source) + 'tar.gz'}")


class Release:
    def __init__(
        self, owner: str = "plaguss", repo: str = "talking-python", token: str = None
    ) -> None:
        self.gh: GhApi = GhApi(
            owner=owner,
            repo=repo,
            token=get_repo_access_token() if token is None else token,
        )

    def create_release(self):
        self.gh.create_release
        raise NotImplementedError

    def upload_file(self):
        raise NotImplementedError
