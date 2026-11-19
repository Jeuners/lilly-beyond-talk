"""Bounded, read-only Wikipedia Action API access; article content is external data."""

import asyncio
import copy
import json
import re
import time
from collections import OrderedDict
from html.parser import HTMLParser
from urllib.parse import quote

import httpx

USER_AGENT = "MartiVoice/0.1 (+https://github.com/Jeuners/lilly-beyond-talk)"
MAX_RESPONSE_BYTES = 2_000_000
MAX_TEXT_CHARS = 12_000
CACHE_TTL = 300
CACHE_SIZE = 64
_cache: OrderedDict[tuple, tuple[float, dict]] = OrderedDict()


class WikipediaError(Exception):
    """Invalid request or unavailable/unusable Wikipedia response."""


class _PlainSnippet(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _plain(value: str) -> str:
    parser = _PlainSnippet()
    parser.feed(value)
    return " ".join("".join(parser.parts).split())


def _host(language: str) -> str:
    if language not in ("de", "en"):
        raise WikipediaError("Wikipedia unterstützt hier nur Deutsch (de) und Englisch (en).")
    return f"https://{language}.wikipedia.org"


def _positive_id(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _cached(key):
    item = _cache.get(key)
    if item is None:
        return None
    if time.monotonic() - item[0] >= CACHE_TTL:
        del _cache[key]
        return None
    _cache.move_to_end(key)
    return copy.deepcopy(item[1])


def _remember(key, result):
    _cache[key] = (time.monotonic(), copy.deepcopy(result))
    _cache.move_to_end(key)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return result


async def _request(host: str, params: dict) -> dict:
    try:
        async with asyncio.timeout(20):
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT}, timeout=10, follow_redirects=False
            ) as client:
                async with client.stream(
                    "GET", f"{host}/w/api.php",
                    params={"action": "query", "format": "json", "formatversion": "2", **params},
                ) as response:
                    response.raise_for_status()
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                            raise WikipediaError("Die Wikipedia-Antwort ist zu groß.")
                        body.extend(chunk)
        data = json.loads(body)
    except (httpx.HTTPError, TimeoutError) as exc:
        raise WikipediaError("Wikipedia ist derzeit nicht erreichbar.") from exc
    except (ValueError, UnicodeError) as exc:
        raise WikipediaError("Wikipedia lieferte keine gültige JSON-Antwort.") from exc
    if not isinstance(data, dict) or "error" in data or "errors" in data:
        raise WikipediaError("Wikipedia konnte die Anfrage nicht beantworten.")
    query = data.get("query")
    if not isinstance(query, dict):
        raise WikipediaError("Die Wikipedia-Antwort enthält keine gültigen Ergebnisse.")
    return query


async def search_wikipedia(query: str, language: str = "de") -> dict:
    """Search only the encyclopedia namespace, returning up to five plain snippets."""
    host = _host(language)
    if not isinstance(query, str) or not query.strip() or len(query) > 300:
        raise WikipediaError("Bitte einen Wikipedia-Suchbegriff mit 1 bis 300 Zeichen angeben.")
    query = query.strip()
    key = ("search", language, query)
    if (cached := _cached(key)) is not None:
        return cached
    data = await _request(host, {
        "list": "search", "srsearch": query, "srnamespace": "0", "srlimit": "5",
        "srprop": "snippet",
    })
    hits = data.get("search")
    if not isinstance(hits, list):
        raise WikipediaError("Wikipedia lieferte keine gültige Trefferliste.")
    results = []
    for hit in hits[:5]:
        if (not isinstance(hit, dict) or not _positive_id(hit.get("pageid"))
                or not isinstance(hit.get("title"), str) or not hit["title"].strip()
                or not isinstance(hit.get("snippet", ""), str)):
            raise WikipediaError("Wikipedia lieferte einen ungültigen Suchtreffer.")
        results.append({
            "page_id": str(hit["pageid"]), "title": hit["title"],
            "link": f"{host}/wiki/{quote(hit['title'].replace(' ', '_'), safe='')}",
            "description": _plain(hit.get("snippet", ""))[:600],
        })
    return _remember(key, {"status": "ok", "query": query, "language": language,
                           "source": "Wikipedia", "results": results})


async def read_wikipedia(page_id: str, language: str = "de") -> dict:
    """Read a plain-text extract, retaining provenance and explicit extraction limits."""
    host = _host(language)
    if not isinstance(page_id, str) or not re.fullmatch(r"[1-9][0-9]{0,15}", page_id):
        raise WikipediaError("Bitte eine gültige Wikipedia-Seiten-ID aus der Suche verwenden.")
    key = ("read", language, page_id)
    if (cached := _cached(key)) is not None:
        return cached
    data = await _request(host, {
        "pageids": page_id, "prop": "extracts|revisions|pageprops", "explaintext": "1",
        "exsectionformat": "plain", "rvprop": "ids", "rvlimit": "1",
        "ppprop": "disambiguation",
    })
    pages = data.get("pages")
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        raise WikipediaError("Wikipedia lieferte keinen gültigen Artikel.")
    page = pages[0]
    revisions = page.get("revisions")
    props = page.get("pageprops", {})
    if ("missing" in page or page.get("pageid") != int(page_id)
            or not isinstance(page.get("title"), str) or not page["title"].strip()
            or not isinstance(page.get("extract"), str) or not page["extract"].strip()
            or not isinstance(props, dict)
            or not isinstance(revisions, list) or not revisions
            or not isinstance(revisions[0], dict)
            or not _positive_id(revisions[0].get("revid"))):
        raise WikipediaError("Für diese Wikipedia-Seite ist kein lesbarer Artikel verfügbar.")
    text = page["extract"].strip()
    revision = revisions[0]["revid"]
    url = f"{host}/wiki/{quote(page['title'].replace(' ', '_'), safe='')}"
    return _remember(key, {
        "status": "ok", "page_id": page_id, "language": language, "title": page["title"],
        "text": text[:MAX_TEXT_CHARS], "url": url, "revision": revision,
        "revision_url": f"{host}/w/index.php?oldid={revision}",
        "attribution_url": f"{host}/w/index.php?curid={page_id}&action=history",
        "source": "Wikipedia", "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "truncated": len(text) > MAX_TEXT_CHARS,
        "disambiguation": "disambiguation" in props,
        "content_kind": "plain_text_extract",
        "content_notice": "Klartextauszug; Tabellen, Formeln und andere Artikelbestandteile können fehlen.",
    })
