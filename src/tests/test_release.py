
import talking_python.release as rel


def test_get_release_url():
    import json
    response = json.loads("""
        [
    {
        "url": "https://api.github.com/repos/plaguss/talking-python/releases/103736184",
        "assets_url": "https://api.github.com/repos/plaguss/talking-python/releases/103736184/assets",
        "upload_url": "https://uploads.github.com/repos/plaguss/talking-python/releases/103736184/assets{?name,label}",
        "html_url": "https://github.com/plaguss/talking-python/releases/tag/v2023-05-21",
        "id": 103736184,
        "author": {
        "login": "plaguss",
        "id": 56895847,
        "node_id": "MDQ6VXNlcjU2ODk1ODQ3",
        "avatar_url": "https://avatars.githubusercontent.com/u/56895847?v=4",
        "gravatar_id": "",
        "url": "https://api.github.com/users/plaguss",
        "html_url": "https://github.com/plaguss",
        "followers_url": "https://api.github.com/users/plaguss/followers",
        "following_url": "https://api.github.com/users/plaguss/following{/other_user}",
        "gists_url": "https://api.github.com/users/plaguss/gists{/gist_id}",
        "starred_url": "https://api.github.com/users/plaguss/starred{/owner}{/repo}",
        "subscriptions_url": "https://api.github.com/users/plaguss/subscriptions",
        "organizations_url": "https://api.github.com/users/plaguss/orgs",
        "repos_url": "https://api.github.com/users/plaguss/repos",
        "events_url": "https://api.github.com/users/plaguss/events{/privacy}",
        "received_events_url": "https://api.github.com/users/plaguss/received_events",
        "type": "User",
        "site_admin": false
        },
        "node_id": "RE_kwDOJbRdo84GLuN4",
        "tag_name": "v2023-05-21",
        "target_commitish": "main",
        "name": "v2023-05-21",
        "draft": false,
        "prerelease": false,
        "created_at": "2023-05-20T23:45:34Z",
        "published_at": "2023-05-21T18:53:36Z",
        "assets": [
        {
            "url": "https://api.github.com/repos/plaguss/talking-python/releases/assets/109157723",
            "id": 109157723,
            "node_id": "RA_kwDOJbRdo84GgZ1b",
            "name": "test.tar.gz",
            "label": "",
            "uploader": {
            "login": "plaguss",
            "id": 56895847,
            "node_id": "MDQ6VXNlcjU2ODk1ODQ3",
            "avatar_url": "https://avatars.githubusercontent.com/u/56895847?v=4",
            "gravatar_id": "",
            "url": "https://api.github.com/users/plaguss",
            "html_url": "https://github.com/plaguss",
            "followers_url": "https://api.github.com/users/plaguss/followers",
            "following_url": "https://api.github.com/users/plaguss/following{/other_user}",
            "gists_url": "https://api.github.com/users/plaguss/gists{/gist_id}",
            "starred_url": "https://api.github.com/users/plaguss/starred{/owner}{/repo}",
            "subscriptions_url": "https://api.github.com/users/plaguss/subscriptions",
            "organizations_url": "https://api.github.com/users/plaguss/orgs",
            "repos_url": "https://api.github.com/users/plaguss/repos",
            "events_url": "https://api.github.com/users/plaguss/events{/privacy}",
            "received_events_url": "https://api.github.com/users/plaguss/received_events",
            "type": "User",
            "site_admin": false
            },
            "content_type": "application/x-tar",
            "state": "uploaded",
            "size": 1273,
            "download_count": 3,
            "created_at": "2023-05-21T18:53:36Z",
            "updated_at": "2023-05-21T18:53:37Z",
            "browser_download_url": "https://github.com/plaguss/talking-python/releases/download/v2023-05-21/test.tar.gz"
        }
        ],
        "tarball_url": "https://api.github.com/repos/plaguss/talking-python/tarball/v2023-05-21",
        "zipball_url": "https://api.github.com/repos/plaguss/talking-python/zipball/v2023-05-21",
        "body": "test release"
    }
    ]""")
    url = rel._extract_release_url(response)
    assert url == "https://github.com/plaguss/talking-python/releases/download/v2023-05-21/test.tar.gz"
