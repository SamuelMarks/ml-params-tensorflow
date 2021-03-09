#!/usr/bin/env python

"""
Install requirements.txt as if no anonymous http access is required (do clones instead)
"""

from subprocess import call
from os import path, environ
from urllib.parse import urlparse
from tempfile import gettempdir


def up_clone(url, clone_parent_dir=environ.get("CLONE_PARENT_DIR", gettempdir())):
    """
    Clone if not present else reuse

    :param url: URL
    :type url: ```str```

    :param clone_parent_dir: Parent directory to clone everything from
    :type clone_parent_dir: ```str```

    :returns: target_dir, filename
    :rtype: ```Tuple[str, Optional[str]]```
    """
    u = urlparse(url.replace("api.github.com/repos", "github.com"))
    zipball_idx = u.path.rfind("/zipball")
    p = u.path[:zipball_idx] if zipball_idx > -1 else "/".join(u.path.split("/")[:3])
    target_dir = path.join(clone_parent_dir, p[p.rfind("/") + 1 :])
    if not path.isdir(target_dir):
        call(
            ["git", "clone", "--depth=1", "{u.scheme}://{u.netloc}{p}".format(u=u, p=p), target_dir]
        )
    return target_dir, u.path[u.path.rfind("/") + 1 :].rstrip()


def clone_install_pip(pip_req_file):
    """
    Install requirements.txt as if no anonymous http access is required (do clones instead)

    :param pip_req_file: Filename where requirements are
    :type pip_req_file: ```str```
    """
    with open(pip_req_file) as f:
        reqs = f.readlines()

    for req in reqs:
        if req.startswith("http:") or req.startswith("https:"):
            call(["pip", "install", "."], cwd=up_clone(req)[0])
        elif req.startswith("-r"):
            clone_install_pip(path.join(*up_clone(req[2:].lstrip())))
        else:
            call(["pip", "install", req])


if __name__ == "__main__":
    clone_install_pip("requirements.txt")
