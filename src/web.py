"""Fetch an image from an internet URL, safely.

Only http(s) is allowed, hosts on private / loopback networks are refused (so the REST API can't be
abused to probe the local network), downloads are capped at 10 MB and must decode as an image.
"""
from __future__ import annotations

import io
import ipaddress
import socket
from urllib.parse import urlparse

import requests
from PIL import Image

MAX_BYTES = 10 * 1024 * 1024
TIMEOUT = 15
HEADERS = {"User-Agent": "multimodal-recsys/1.0 (student project)"}


class ImageFetchError(ValueError):
    pass


def _check_public_host(url: str):
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ImageFetchError("Only http:// or https:// image links are supported.")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except socket.gaierror:
        raise ImageFetchError(f"Couldn't resolve host '{u.hostname}'. Check the link and your internet connection.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ImageFetchError("Links to local or private network addresses are not allowed.")


def fetch_image(url: str) -> Image.Image:
    url = (url or "").strip()
    try:
        for _ in range(5):  # follow redirects manually so every hop is checked before connecting
            _check_public_host(url)
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True, allow_redirects=False)
            if r.is_redirect:
                url = requests.compat.urljoin(url, r.headers["Location"])
                r.close()
                continue
            break
        else:
            raise ImageFetchError("Too many redirects.")
        with r:
            r.raise_for_status()
            data = io.BytesIO()
            for chunk in r.iter_content(64 * 1024):
                data.write(chunk)
                if data.tell() > MAX_BYTES:
                    raise ImageFetchError("Image is larger than 10 MB.")
    except requests.RequestException as e:
        raise ImageFetchError(f"Download failed: {e.__class__.__name__}. Check the link and your connection.")
    try:
        im = Image.open(io.BytesIO(data.getvalue()))
        im.load()
        return im.convert("RGB")
    except Exception:
        raise ImageFetchError("That link didn't return an image (use a direct link to a .jpg/.png/.webp).")
