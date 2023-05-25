"""This module contains the functionalities to work with the chroma
directory. It shouldn't be necessary if the data was stored in a
database.
"""

import datetime as dt
import os
import tarfile
from pathlib import Path
import shutil
import requests
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import json


RELEASES_URL = r"https://github.com/plaguss/talking-python/releases"
RELEASES_ENDPOINT = r"https://api.github.com/repos/plaguss/talking-python/releases"


def get_chroma_dir() -> Path:
    """Returns the path to the chroma content where the
    vectors are stored.

    Returns:
        Path: chroma directory.
    """
    return Path.home() / ".cache" / "talking-python" / "chroma"


def get_repo_access_token() -> str:
    """Get the github api access token for the repo.

    Needed to upload content to github releases.

    Returns:
        str: token.
    """
    return os.getenv("TALKING_PYTHON_ACCESS_TOKEN", "")


def generate_release_name() -> str:
    """Generates the name for a release.
    It contains the date in isoformat to help with
    versioning.

    Returns:
        str: release name.
    """
    return f"v{dt.date.today().isoformat()}"


def make_tarfile(source: Path) -> None:
    """Creates a tar file from a directory and compresses it
    using gzip.

    Args:
        source (Path): Path to a directory.

    Raises:
        FileNotFoundError: If the directory doesn't exists.
    """
    print(f"Creating tar file from path: {source}...")
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    with tarfile.open(str(source) + ".tar.gz", "w:gz") as tar:
        tar.add(str(source), arcname=source.name)
    print(f"File generated at: {str(source) + '.tar.gz'}")


def untar_file(source: Path) -> Path:
    """Untar and decompress files which have passed by `make_tarfile`.

    Args:
        source (Path): Path pointing to a .tag.gz file.

    Returns:
        filename (Path): The filename of the file decompressed.
    """
    # It assumes the file ends with .tar.gz
    new_filename = source.parent / source.stem.replace(".tar", "")
    with tarfile.open(source, "r:gz") as f:
        f.extractall(source.parent)
    print(f"File decompressed: {new_filename}")
    return new_filename


class Release:
    def __init__(
        self, owner: str = "plaguss", repo: str = "talking-python", token: str = None
    ) -> None:
        from ghapi.all import GhApi

        self.gh: GhApi = GhApi(
            owner=owner,
            repo=repo,
            token=get_repo_access_token() if token is None else token,
        )

    def create_release(
        self,
        tag_name: str,
        files: list[Path] = None,
        branch: str = "main",
        body: str = "",
        draft: bool = False,
        prerelease: bool = False
    ):
        """Creates a release using the tag name

        Args:
            tag_name (str): Name of the tag. i.e. v2023-02-01
            files (list[Path], optional): _description_. Defaults to None.
            branch (str, optional):
                Branch where the release. Defaults to "main".
            body (str, optional):
                Description of the release. Defaults to "".
            draft (bool):
                True to create a draft (unpublished) release, False to create a published one.
            prerelease (bool):
                True to identify the release as a prerelease. False to identify the release as a full release.

        See Also:
            https://docs.github.com/en/rest/releases/releases?apiVersion=2022-11-28#create-a-release

        Examples
            ```python
            >>> rel.create_release("v2023-05-21", files=[path_to_upload], body="test release" )
            ```
        """
        if not isinstance(files, list):
            files = [files]
        assert all([f.exists() for f in files]), "All the files must exist to be released."
        self.gh.create_release(
            tag_name,
            branch=branch,
            name=tag_name,
            body=body,
            files=files,
            draft=draft,
            prerelease=prerelease
        )


def get_release_url() -> str:
    """Extracts the path to the file from the last release from
    a json response from github API.

    Args:
        response (_type_): _description_

    Returns:
        str: _description_
    """
    # TODO: Grab the filename of the last chroma release.
    try:
        with urlopen(RELEASES_ENDPOINT) as response:
            body = response.read()
    except (URLError, HTTPError) as e:
        raise URLError(
            f"Something failed, the model should be installed from: {RELEASES_URL}"
        ) from e
    else:
        releases = json.loads(body)

    return _extract_release_url(releases)


def _extract_release_url(response) -> str:
    # The releases are versioned according to calver,
    # using a `v` followed by the date in isoformat.
    # To find the most recent, we check only for the date.
    dates = [dt.datetime.fromisoformat(r["name"][1:]) for r in response]

    last_release = dates.index(max(dates))
    assets = response[last_release]["assets"]
    # Extract from the assets (there should be only one, 
    # just the chroma folder is uploaded)
    chroma_tar_gz = assets[0]["browser_download_url"]
    return chroma_tar_gz


def download_release_file(url: str, dest: Path | None = None) -> Path:
    r"""Download a file from github releases.

    Used to grab the chroma compressed data.

    Args:
        url (str): URL pointing to a file.
        dest (Path | None):
            Destination folder. Defaults to None (which corresponds
            to the current working directory).

    Returns:
        filename (Path): The name of the file downloaded

    Examples:
        ```python
        >>> url = 'https://github.com/plaguss/talking-python/releases/download/v2023-05-21/test.tar.gz'
        >>> download_release_file(url, Path.cwd())
        ```
    """
    # https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    # Extract the filename using the url as a Path object
    if dest is None:
        dest = Path.cwd()
    local_filename = dest / Path(url).name
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    print(f"File generated at: {local_filename}")
    return local_filename
