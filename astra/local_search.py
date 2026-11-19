"""Bounded async access to the installed local-search CLI; no model requests."""

import asyncio
import contextlib
import json
import os
import re
import signal
from pathlib import Path
from typing import Any


class LocalSearchError(Exception):
    """A search failure that must not be mistaken for an empty result."""


async def _read(stream: asyncio.StreamReader) -> bytes:
    chunks = bytearray()
    while chunk := await stream.read(8192):
        chunks.extend(chunk)
        if len(chunks) > 131072:
            raise LocalSearchError("Suchausgabe überschreitet das Größenlimit.")
    return bytes(chunks)


async def _drain(stream: asyncio.StreamReader) -> None:
    while await stream.read(8192):
        pass


async def _execute(command: list[str], timeout: float) -> tuple[int, bytes, bytes]:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise LocalSearchError("local-search ist nicht installiert oder nicht ausführbar.") from exc
    tasks = [asyncio.create_task(_read(process.stdout)), asyncio.create_task(_read(process.stderr))]
    completed = False
    try:
        async with asyncio.timeout(timeout):
            stdout, stderr = await asyncio.gather(*tasks)
            await process.wait()
        completed = True
        return process.returncode, stdout, stderr
    finally:
        # The launcher and its one-shot children share this new process group.
        # Persistent tgrep servers create their own session and remain independent.
        if not completed:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if not completed:
            # A paused pipe transport can keep wait() pending even after SIGKILL.
            # Discard buffered output in chunks so EOF and process exit propagate.
            try:
                async with asyncio.timeout(3):
                    await asyncio.gather(
                        _drain(process.stdout), _drain(process.stderr), process.wait()
                    )
            except TimeoutError as exc:
                raise LocalSearchError(
                    "Suchprozess konnte nicht vollständig beendet werden."
                ) from exc


async def status(executable: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """Query `local-search status` for the indexing state of each registered
    root. Used only for startup diagnostics — never on the request path.
    """
    command = [str(Path(executable).expanduser()), "status"]
    try:
        code, stdout, stderr = await _execute(command, timeout)
    except TimeoutError as exc:
        raise LocalSearchError("local-search status hat das Zeitlimit überschritten.") from exc
    if code != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[:300]
        raise LocalSearchError(f"local-search status fehlgeschlagen: {detail or code}")
    try:
        data = json.loads(stdout)
        if not isinstance(data, dict):
            raise ValueError("Invalid status shape")
        return data
    except ValueError as exc:
        raise LocalSearchError("local-search status hat ungültige Daten geliefert.") from exc


async def search_local(
    executable: str,
    allowed_roots: tuple[str, ...],
    root: str,
    query: str,
    *,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Search only an enabled root; return bounded excerpts and honest status."""
    if (
        not isinstance(root, str)
        or root not in allowed_roots
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", root)
    ):
        raise LocalSearchError("Dieser Suchordner ist nicht freigegeben.")
    if (
        not isinstance(query, str)
        or not query.strip()
        or len(query) > 200
        or any(ord(char) < 32 for char in query)
    ):
        raise LocalSearchError("Suchbegriff muss 1 bis 200 Zeichen ohne Steuerzeichen enthalten.")
    query = query.strip()
    command = [str(Path(executable).expanduser()), "search", "--root", root, "--limit", "8"]
    if len(query) < 3:
        command.append("--fresh")
    command.extend(["--", query])
    try:
        code, stdout, stderr = await _execute(command, timeout)
    except TimeoutError as exc:
        raise LocalSearchError("Die lokale Suche hat ihr Zeitlimit überschritten.") from exc
    if code not in (0, 1):
        detail = stderr.decode("utf-8", errors="replace").strip()[:300]
        raise LocalSearchError(f"Lokale Suche fehlgeschlagen: {detail or code}")
    try:
        data = json.loads(stdout)
        matches, reports = data["matches"], data["reports"]
        if (
            not isinstance(matches, list)
            or not isinstance(reports, list)
            or not isinstance(data["truncated"], bool)
        ):
            raise ValueError("Invalid result fields")
        bounded = []
        size = 0
        for item in matches[:8]:
            path, line, text = item["path"], item["line"], item["text"]
            if (
                not isinstance(path, str)
                or not Path(path).is_absolute()
                or len(path) > 4096
                or type(line) is not int
                or line < 1
                or not isinstance(text, str)
            ):
                raise ValueError("Invalid source")
            source = {
                "path": path,
                "line": line,
                "text": text[:400],
                "text_truncated": bool(item.get("text_truncated")) or len(text) > 400,
            }
            size += len(json.dumps(source, ensure_ascii=False))
            if size > 6000:
                break
            bounded.append(source)
        warnings = []
        backends = set()
        for report in reports:
            if report["backend"] not in ("tgrep", "rg"):
                raise ValueError("Invalid backend")
            backends.add(report["backend"])
            if not isinstance(report["warnings"], list):
                raise ValueError("Invalid warnings")
            for warning in report["warnings"]:
                if not isinstance(warning, str):
                    raise ValueError("Invalid warning")
                if len(warnings) < 4:
                    warnings.append(warning[:300])
        return {
            "root": root,
            "matches": bounded,
            "backends": sorted(backends),
            "warnings": warnings,
            "truncated": data["truncated"] or len(bounded) < len(matches),
            "freshness": "eventually consistent" if "tgrep" in backends else "fresh scan",
        }
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise LocalSearchError("Die lokale Suche hat ungültige Ergebnisdaten geliefert.") from exc
