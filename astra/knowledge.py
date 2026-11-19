"""Persistent local knowledge store using only the Python standard library."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import logging
import os
import re
import sqlite3
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

from astra.documents import extract_pdf_text

SUPPORTED_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000
CHUNK_CHARS = 1_800
CHUNK_OVERLAP = 240


class KnowledgeError(Exception):
    """A user-facing knowledge-store error."""


class KnowledgeNotFound(KnowledgeError):
    """The requested document does not exist."""


_GERMAN_STOPWORDS = {
    "und", "oder", "aber", "auch", "noch", "schon", "mal", "bitte", "danke",
    "der", "die", "das", "des", "dem", "den", "ein", "eine", "einen", "einem", "einer",
    "ich", "du", "er", "sie", "wir", "ihr", "ist", "sind", "war", "waren",
    "habe", "hast", "hat", "haben", "wird", "werden", "kann", "kannst", "können",
    "mit", "von", "zu", "auf", "für", "im", "in", "am", "an", "bei", "nach", "vor",
    "über", "unter", "aus", "als", "wie", "was", "wer", "wo", "wann", "warum",
    "welche", "welcher", "welches", "nicht", "heute", "gerade", "denn", "dir", "dich",
    "mir", "mich", "dann", "also", "gibt", "sag", "sagen", "erzähl", "erzähle", "weißt",
    "weisst", "kennst", "kenne", "hey", "hallo", "martin",
}  # fmt: skip


def _significant_terms(text: str) -> list[str]:
    words = re.findall(r"[\wÄÖÜäöüß-]{4,}", text.lower())
    return [word for word in words if word not in _GERMAN_STOPWORDS]


def _excerpt(text: str, terms: list[str], max_chars: int = 1000) -> str:
    """Keep a matching term and surrounding context inside the source budget."""
    lowered = text.lower()
    # Prefer terms used by the automatic relevance check over short query words.
    positions = [lowered.find(term) for term in terms if len(term) >= 4]
    positions = [position for position in positions if position >= 0]
    if not positions:
        positions = [lowered.find(term) for term in terms if term in lowered]
    anchor = min(positions, default=0)
    start = min(max(0, anchor - max_chars // 3), max(0, len(text) - max_chars))
    return text[start:start + max_chars]


def find_relevant_search(store: KnowledgeStore, text: str) -> dict | None:
    """Return the search result if `text` looks answerable from stored
    documents, else None.

    A small local model calls a competing tool like search_knowledge only
    inconsistently (same failure documented in triggers.py for search_local
    and ask_martin), so this cheap keyword pre-check decides deterministically
    instead of leaving it to the model's judgement. Returns the already-fetched
    result (not just a bool) so the caller can hand it straight to the tool
    round-trip instead of searching a second time.
    """
    terms = _significant_terms(text)
    if not terms:
        return None
    try:
        result = store.search(text)
    except KnowledgeError:
        return None
    for match in result["matches"]:
        lowered = match["text"].lower()
        if any(term in lowered for term in terms):
            return result
    return None


def extract_document(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise KnowledgeError("Unterstützt werden TXT, Markdown, PDF und DOCX.")
    if len(data) > MAX_FILE_BYTES:
        raise KnowledgeError("Datei zu groß (maximal 25 MB).")
    try:
        if suffix == ".pdf":
            text = extract_pdf_text(data, max_chars=MAX_TEXT_CHARS)
        elif suffix == ".docx":
            text = _extract_docx(data)
        else:
            text = data.decode("utf-8-sig")
    except Exception as exc:
        # Broad on purpose: this is a boundary that parses untrusted uploaded
        # bytes with third-party parsers (pypdf, zipfile/ElementTree for docx)
        # whose failure modes aren't a small closed set (e.g. a docx-shaped
        # zip missing word/document.xml raises a plain KeyError). Any parse
        # failure here must become a per-file error, never a 500.
        raise KnowledgeError("Datei konnte nicht als Text gelesen werden.") from exc
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        raise KnowledgeError("Die Datei enthält keinen lesbaren Text.")
    return text[:MAX_TEXT_CHARS]


def _extract_docx(data: bytes) -> str:
    with zipfile.ZipFile(BytesIO(data)) as archive:
        info = archive.getinfo("word/document.xml")
        if info.file_size > MAX_TEXT_CHARS * 4:
            raise KnowledgeError("DOCX-Dokument enthält zu viel unkomprimierten Text.")
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        parts = [
            node.text or ""
            for node in paragraph.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
        ]
        if parts:
            paragraphs.append("".join(parts))
    return "\n\n".join(paragraphs)


def _chunks(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):
            boundary = max(text.rfind("\n", start + CHUNK_CHARS // 2, end), text.rfind(" ", start + CHUNK_CHARS // 2, end))
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks


class KnowledgeStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "knowledge.sqlite3"
        self.files = self.directory / "files"
        self.files.mkdir(exist_ok=True)
        with self._locked():
            self._initialize()
            self._recover()

    @contextlib.contextmanager
    def _locked(self):
        # Separate descriptors serialize threads and cooperating server processes,
        # including startup recovery. No lock is held during document parsing.
        with (self.directory / "knowledge.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _paths(self, row):
        target = self.files / f"{row['id']}{Path(row['filename']).suffix.lower()}"
        return target, self.files / f"{row['id']}.pending"

    def _sync_files(self):
        descriptor = os.open(self.files, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _set_status(self, document_id, status):
        with self._connect() as db:
            db.execute("UPDATE documents SET status = ? WHERE id = ?", (status, document_id))

    @staticmethod
    def _valid_file(path, row):
        try:
            if path.stat().st_size != row["size"]:
                return False
            with path.open("rb") as source:
                return hashlib.file_digest(source, "sha256").hexdigest() == row["digest"]
        except FileNotFoundError:
            return False

    def _finish_delete(self, row):
        for path in self._paths(row):
            path.unlink(missing_ok=True)
        self._sync_files()
        with self._connect() as db:
            db.execute("DELETE FROM documents WHERE id = ?", (row["id"],))

    def _recover(self):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM documents").fetchall()
        # Back up the DB before the first recovery that changes stored state.
        backup_done = False
        for row in rows:
            try:
                target, temporary = self._paths(row)
                if row["status"] == "ready" and target.is_file():
                    continue
                if row["status"] == "missing" and not target.exists() and not temporary.exists():
                    continue
                target_valid = row["status"] != "deleting" and self._valid_file(target, row)
                temporary_valid = row["status"] != "deleting" and self._valid_file(temporary, row)
                if row["status"] == "missing" and not target_valid and not temporary_valid:
                    continue
                if not backup_done:
                    backup = self.directory / f"knowledge-recovery-{uuid.uuid4().hex}.sqlite3"
                    with self._connect() as source, contextlib.closing(
                        sqlite3.connect(backup)
                    ) as destination:
                        source.backup(destination)
                    backup_done = True
                if row["status"] == "deleting":
                    self._finish_delete(row)
                    continue
                if target_valid:
                    with target.open("rb") as source:
                        os.fsync(source.fileno())
                    temporary.unlink(missing_ok=True)
                    self._sync_files()
                    self._set_status(row["id"], "ready")
                elif temporary_valid:
                    with temporary.open("rb") as source:
                        os.fsync(source.fileno())
                    temporary.replace(target)
                    self._sync_files()
                    self._set_status(row["id"], "ready")
                else:
                    # Preserve the original extracted text for repair by reupload.
                    self._set_status(row["id"], "missing")
            except (OSError, sqlite3.Error):
                logging.getLogger(__name__).exception(
                    "Wissensspeicher: Wiederherstellung von %s fehlgeschlagen", row["id"]
                )

    @contextlib.contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, media_type TEXT NOT NULL,
                    size INTEGER NOT NULL, digest TEXT NOT NULL UNIQUE, created_at REAL NOT NULL,
                    text_chars INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'ready'
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL, content TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
                """
            )

    def list_documents(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, filename, media_type, size, created_at, text_chars, status "
                "FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def add(self, filename: str, data: bytes) -> dict:
        safe_name = Path(filename or "Datei").name[:180]
        text = extract_document(safe_name, data)
        digest = hashlib.sha256(data).hexdigest()
        try:
            with self._locked():
                with self._connect() as db:
                    row = db.execute("SELECT * FROM documents WHERE digest = ?", (digest,)).fetchone()
                    if row is None:
                        document_id = uuid.uuid4().hex
                        db.execute(
                            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (document_id, safe_name, SUPPORTED_TYPES[Path(safe_name).suffix.lower()],
                             len(data), digest, time.time(), len(text), "pending"),
                        )
                        db.executemany(
                            "INSERT INTO chunks(document_id, position, content) VALUES (?, ?, ?)",
                            [(document_id, position, chunk)
                             for position, chunk in enumerate(_chunks(text))],
                        )
                        row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
                target, temporary = self._paths(row)
                if row["status"] == "deleting":
                    raise KnowledgeError("Dokument wird entfernt. Bitte Löschung erneut versuchen.")
                if row["status"] == "ready" and self._valid_file(target, row):
                    return self.get(row["id"])
                self._set_status(row["id"], "pending")
                with temporary.open("wb") as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(target)
                self._sync_files()
                self._set_status(row["id"], "ready")
                return self.get(row["id"])
        except (OSError, sqlite3.Error) as exc:
            raise KnowledgeError(
                "Dokument konnte nicht vollständig gespeichert werden. Bitte erneut hochladen."
            ) from exc

    def get(self, document_id: str) -> dict:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, filename, media_type, size, created_at, text_chars, status "
                "FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        if not row:
            raise KnowledgeNotFound("Dokument nicht gefunden.")
        return dict(row)

    def remove(self, document_id: str) -> None:
        try:
            with self._locked():
                with self._connect() as db:
                    row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
                    if not row:
                        raise KnowledgeNotFound("Dokument nicht gefunden.")
                    db.execute("UPDATE documents SET status = 'deleting' WHERE id = ?", (document_id,))
                self._finish_delete(row)
        except (OSError, sqlite3.Error) as exc:
            raise KnowledgeError("Datei konnte nicht entfernt werden. Bitte erneut versuchen.") from exc

    # Full linear scan, no FTS index — deliberate: benchmarked at ~1.3ms/80
    # chunks, ~12ms/800 chunks, ~127ms/8000 chunks (see plan/session notes).
    # Fine deep into the thousands of documents once callers thread it off
    # the event loop; revisit with SQLite FTS5 only if it grows well past
    # that and search latency itself (not just event-loop blocking) matters.
    def search(self, query: str, limit: int = 4) -> dict:
        raw_terms = [term.lower() for term in re.findall(r"[\wÄÖÜäöüß-]{2,}", query or "")]
        if not raw_terms:
            raise KnowledgeError("Suchfrage enthält keinen verwertbaren Begriff.")
        # Stopwords excluded from scoring, not just the raw term list: without
        # this, common words like "der"/"ist"/"im" dominate the count and a
        # prose-heavy unrelated document can outscore and displace the one
        # actual match once a few other documents are in the store. Falls
        # back to the raw terms if filtering would leave nothing to score.
        terms = [term for term in raw_terms if term not in _GERMAN_STOPWORDS] or raw_terms
        with self._connect() as db:
            rows = db.execute(
                "SELECT c.document_id, c.position, c.content, d.filename "
                "FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.status = 'ready'"
            ).fetchall()
        matches = []
        for row in rows:
            lowered = row["content"].lower()
            score = sum(lowered.count(term) for term in terms)
            if score:
                matches.append((score, row))
        matches.sort(key=lambda item: (-item[0], item[1]["filename"], item[1]["position"]))
        return {
            "query": query,
            "matches": [
                # Kept bounded (1000 chars, top 4): a full 6x1200-char result plus
                # system prompt and history once overflowed num_ctx (4096 tokens at
                # the time), 400ing the whole chat request and killing the pipeline
                # processor. num_ctx is now 8192, but this stays capped on principle.
                {"document_id": row["document_id"], "filename": row["filename"],
                 "chunk": row["position"] + 1, "text": _excerpt(row["content"], terms)}
                for _, row in matches[:limit]
            ],
            "total": len(matches),
        }
