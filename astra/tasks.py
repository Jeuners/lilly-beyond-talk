"""Standalone local task memory: storage, voice tools and HTTP routes."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter, HTTPException
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pydantic import BaseModel, Field

from astra.routing import route_tool
from astra.triggers import trigger_tool

LOCAL_ZONE = ZoneInfo("Europe/Berlin")
LOG = logging.getLogger(__name__)
UNSET = object()
WEEKDAYS = {"montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3,
            "freitag": 4, "samstag": 5, "sonntag": 6}
HOUR_WORDS = {"eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5,
              "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
              "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14,
              "fünfzehn": 15, "sechzehn": 16, "siebzehn": 17,
              "achtzehn": 18, "neunzehn": 19, "zwanzig": 20,
              "einundzwanzig": 21, "zweiundzwanzig": 22,
              "dreiundzwanzig": 23}


class TaskError(ValueError):
    pass


class TaskClarification(TaskError):
    """The user's task request needs another detail before proposing it."""


def temporal_fragment(text: str, base_due: str | None = None) -> tuple[str | None, str | None]:
    """Resolve clear German day/hour fragments while preserving supplied parts."""
    lowered = text.lower()
    today = datetime.now(LOCAL_ZONE).date()
    day = None
    match = re.search(r"\b(\d{1,2})[.\-/](\d{1,2})(?:[.\-/](\d{4}))?\b", lowered)
    if match:
        year = int(match.group(3) or today.year)
        try:
            day = date(year, int(match.group(2)), int(match.group(1)))
        except ValueError as exc:
            raise TaskError("Ungültiges Datum.") from exc
    elif re.search(r"\bübermorgen\b", lowered):
        day = today + timedelta(days=2)
    elif re.search(r"\bmorgen\b", lowered) and not re.search(
        r"\b(?:nicht|kein\w*)\s+morgen\b", lowered
    ):
        day = today + timedelta(days=1)
    else:
        for word, weekday in WEEKDAYS.items():
            if re.search(rf"\b{word}\b", lowered):
                days = (weekday - today.weekday()) % 7
                day = today + timedelta(days=days or 7)
                break

    time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*uhr\b", lowered)
    if time_match:
        hour, minute = int(time_match.group(1)), int(time_match.group(2) or 0)
    else:
        word_match = re.search(r"\b([a-zäöüß]+)\s*uhr\b", lowered)
        hour = HOUR_WORDS.get(word_match.group(1)) if word_match else None
        minute = 0
    if hour is not None and not 0 <= hour <= 23:
        raise TaskError("Uhrzeit muss zwischen 0 und 23 Uhr liegen.")
    if minute > 59:
        raise TaskError("Ungültige Minutenangabe.")
    time_hint = f"{hour:02d}:{minute:02d}" if hour is not None else None

    if base_due:
        parsed = datetime.fromisoformat(base_due)
        base_day = parsed.date()
        base_time = None if len(base_due) == 10 else parsed.strftime("%H:%M")
    else:
        base_day = base_time = None
    day = day or base_day
    time_hint = time_hint or base_time
    if day and time_hint:
        hour, minute = map(int, time_hint.split(":"))
        return datetime(day.year, day.month, day.day, hour, minute,
                        tzinfo=LOCAL_ZONE).isoformat(timespec="minutes"), None
    return day.isoformat() if day else None, time_hint


def validate_due(value: str | None) -> str | None:
    if not value:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            day = date.fromisoformat(value)
        except ValueError as exc:
            raise TaskError("Ungültiges Datum.") from exc
        if day < datetime.now(LOCAL_ZONE).date():
            raise TaskError("Der Termin liegt in der Vergangenheit.")
        return day.isoformat()
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TaskError("Termin muss ein ISO-Datum oder Datum mit Uhrzeit sein.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_ZONE)
    if parsed <= datetime.now(UTC):
        raise TaskError("Der Termin liegt in der Vergangenheit.")
    return parsed.isoformat(timespec="minutes")


class TaskStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = self.directory / "tasks.sqlite3"
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, due_at TEXT,
                status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL,
                reminder_sent_at TEXT
            )""")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)")}
            if "reminder_sent_at" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN reminder_sent_at TEXT")

    def _connect(self):
        connection = sqlite3.connect(self.db, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def list(self, status: str = "open") -> list[dict]:
        if status not in {"open", "done", "all"}:
            raise TaskError("Ungültiger Aufgabenstatus.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE (? = 'all' OR status = ?) "
                "ORDER BY due_at IS NULL, due_at, created_at",
                (status, status),
            ).fetchall()
        return [dict(row) for row in rows]

    def add(self, title: str, due_at: str | None = None) -> dict:
        title = title.strip()
        if not title or len(title) > 300:
            raise TaskError("Aufgabe muss 1 bis 300 Zeichen enthalten.")
        due_at = validate_due(due_at)
        task = {
            "id": uuid4().hex, "title": title, "due_at": due_at,
            "status": "open", "created_at": datetime.now(UTC).isoformat(),
            "reminder_sent_at": None,
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO tasks (id, title, due_at, status, created_at, reminder_sent_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                tuple(task.values()),
            )
        return task

    def update(self, task_id: str, *, title=UNSET, due_at=UNSET) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if row is None:
                raise TaskError("Aufgabe nicht gefunden.")
            next_title = row["title"] if title is UNSET else (title or "").strip()
            if not next_title or len(next_title) > 300:
                raise TaskError("Aufgabe muss 1 bis 300 Zeichen enthalten.")
            next_due = row["due_at"] if due_at is UNSET else validate_due(due_at)
            connection.execute(
                "UPDATE tasks SET title = ?, due_at = ?, reminder_sent_at = ? WHERE id = ?",
                (next_title, next_due,
                 None if next_due != row["due_at"] else row["reminder_sent_at"], task_id),
            )
            updated = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(updated)

    def due_reminders(self, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now(UTC)
        result = []
        for task in self.list("open"):
            due = task["due_at"]
            if not due or len(due) == 10 or task["reminder_sent_at"]:
                continue
            if datetime.fromisoformat(due) <= now:
                result.append(task)
        return result

    def mark_reminded(self, task_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET reminder_sent_at = ? "
                "WHERE id = ? AND status = 'open' AND reminder_sent_at IS NULL",
                (datetime.now(UTC).isoformat(), task_id),
            )
        return bool(cursor.rowcount)

    def set_done(self, task_id: str) -> dict:
        with self._connect() as connection:
            connection.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (task_id,))
            row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise TaskError("Aufgabe nicht gefunden.")
        return dict(row)

    def delete(self, task_id: str):
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        if not cursor.rowcount:
            raise TaskError("Aufgabe nicht gefunden.")


class ProposalInput(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    due_at: str | None = None


class TaskUpdateInput(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    due_at: str | None = None


async def send_mac_notification(task: dict) -> bool:
    title = json.dumps(task["title"], ensure_ascii=False)
    script = f'display notification {title} with title "MARTI Erinnerung"'
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, error = await asyncio.wait_for(process.communicate(), timeout=10)
    except (OSError, TimeoutError) as exc:
        if process and process.returncode is None:
            process.kill()
            await process.communicate()
        LOG.warning("Desktop-Erinnerung fehlgeschlagen: %s", exc)
        return False
    if process.returncode:
        LOG.warning("Desktop-Erinnerung fehlgeschlagen: %s", error.decode(errors="replace"))
    return process.returncode == 0


class TaskExtension:
    def __init__(self, directory: str | Path, notifier=None):
        self.store = TaskStore(directory)
        self.proposals: dict[str, dict] = {}
        self.proposal_lock = threading.RLock()
        self.notifier = notifier or send_mac_notification
        self.reminder_task = None
        self.awaiting_details = False
        self.pending_request: str | None = None
        self.last_saved_id: str | None = None
        self.last_saved_at: datetime | None = None
        self.router = APIRouter(prefix="/api/tasks")
        self._install_routes()

    async def start(self):
        self.reminder_task = asyncio.create_task(self._reminder_loop())

    async def stop(self):
        if self.reminder_task:
            self.reminder_task.cancel()
            try:
                await self.reminder_task
            except asyncio.CancelledError:
                pass
            self.reminder_task = None

    async def _reminder_loop(self):
        while True:
            try:
                await self.deliver_due()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("Erinnerungsprüfung fehlgeschlagen")
            await asyncio.sleep(15)

    async def deliver_due(self, now: datetime | None = None) -> int:
        delivered = 0
        for task in await asyncio.to_thread(self.store.due_reminders, now):
            if await self.notifier(task):
                delivered += int(await asyncio.to_thread(self.store.mark_reminded, task["id"]))
        return delivered

    def _propose(self, title: str, due_at: str | None, time_hint: str | None = None) -> dict:
        title = title.strip()
        if not title or len(title) > 300:
            raise TaskError("Aufgabe muss 1 bis 300 Zeichen enthalten.")
        proposal = {"id": uuid4().hex, "title": title, "due_at": validate_due(due_at),
                    "time_hint": time_hint}
        with self.proposal_lock:
            self.proposals[proposal["id"]] = proposal
        return proposal

    def _confirm(self, proposal_id: str) -> dict:
        with self.proposal_lock:
            proposal = self.proposals.get(proposal_id)
            if proposal is None:
                raise TaskError("Aufgabenvorschlag nicht gefunden.")
            if proposal.get("time_hint") and not proposal["due_at"]:
                raise TaskError("Datum fehlt; Vorschlag noch nicht gespeichert.")
            task = self.store.add(proposal["title"], proposal["due_at"])
            del self.proposals[proposal_id]
            self.last_saved_id = task["id"]
            self.last_saved_at = datetime.now(UTC)
        return task

    def _revise(self, proposal_id: str, *, title=UNSET, due_at=UNSET) -> dict:
        with self.proposal_lock:
            proposal = self.proposals.get(proposal_id)
            if proposal is None:
                raise TaskError("Aufgabenvorschlag nicht gefunden.")
            next_title = proposal["title"] if title is UNSET else (title or "").strip()
            if not next_title or len(next_title) > 300:
                raise TaskError("Aufgabe muss 1 bis 300 Zeichen enthalten.")
            next_due = proposal["due_at"] if due_at is UNSET else validate_due(due_at)
            proposal.update(title=next_title, due_at=next_due,
                            time_hint=None if due_at is None or (next_due and len(next_due) > 10)
                            else proposal.get("time_hint"))
            return dict(proposal)

    def _install_routes(self):
        @self.router.get("")
        async def list_tasks(status: str = "open"):
            try:
                tasks = await asyncio.to_thread(self.store.list, status)
            except TaskError as exc:
                raise HTTPException(422, str(exc)) from exc
            with self.proposal_lock:
                proposals = list(self.proposals.values())
            return {"tasks": tasks, "proposals": proposals}

        @self.router.post("/proposals")
        async def propose(body: ProposalInput):
            try:
                return {"proposal": self._propose(body.title, body.due_at)}
            except TaskError as exc:
                raise HTTPException(422, str(exc)) from exc

        @self.router.post("/proposals/{proposal_id}/confirm")
        async def confirm(proposal_id: str):
            try:
                return {"task": await asyncio.to_thread(self._confirm, proposal_id)}
            except TaskError as exc:
                raise HTTPException(404, str(exc)) from exc

        @self.router.patch("/proposals/{proposal_id}")
        async def revise(proposal_id: str, body: TaskUpdateInput):
            changes = {}
            if "title" in body.model_fields_set:
                changes["title"] = body.title
            if "due_at" in body.model_fields_set:
                changes["due_at"] = body.due_at
            try:
                return {"proposal": self._revise(proposal_id, **changes)}
            except TaskError as exc:
                raise HTTPException(422, str(exc)) from exc

        @self.router.delete("/proposals/{proposal_id}")
        async def discard(proposal_id: str):
            with self.proposal_lock:
                removed = self.proposals.pop(proposal_id, None)
            if removed is None:
                raise HTTPException(404, "Aufgabenvorschlag nicht gefunden.")
            return {"ok": True}

        @self.router.post("/{task_id}/done")
        async def done(task_id: str):
            try:
                return {"task": await asyncio.to_thread(self.store.set_done, task_id)}
            except TaskError as exc:
                raise HTTPException(404, str(exc)) from exc

        @self.router.patch("/{task_id}")
        async def update(task_id: str, body: TaskUpdateInput):
            changes = {}
            if "title" in body.model_fields_set:
                changes["title"] = body.title
            if "due_at" in body.model_fields_set:
                changes["due_at"] = body.due_at
            try:
                return {"task": await asyncio.to_thread(self.store.update, task_id, **changes)}
            except TaskError as exc:
                raise HTTPException(422, str(exc)) from exc

        @self.router.delete("/{task_id}")
        async def delete(task_id: str):
            try:
                await asyncio.to_thread(self.store.delete, task_id)
            except TaskError as exc:
                raise HTTPException(404, str(exc)) from exc
            return {"ok": True}

        @self.router.post("/reset")
        async def reset():
            # awaiting_details/pending_request live on this one process-wide
            # instance (see create_app()), not per conversation - a single
            # misfired "Rückfrage" can wedge every session (and survives a
            # page reload, since the flag isn't client state). This is the
            # manual unstick for that.
            was_stuck = self.awaiting_details
            self.awaiting_details = False
            self.pending_request = None
            return {"ok": True, "was_stuck": was_stuck}

    async def _extract(self, config, utterance: str) -> tuple[str, str | None]:
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "due_at": {"type": ["string", "null"]},
                "clarification": {"type": ["string", "null"]},
            }, "required": ["title", "due_at", "clarification"],
        }
        now = datetime.now(LOCAL_ZONE)
        payload = {
            "model": config.model, "stream": False, "think": False, "format": schema,
            "options": {"temperature": 0, "num_predict": 160},
            "messages": [{"role": "system", "content": (
                "Extrahiere aus dem deutschen Aufgabensatz einen kurzen Aufgabentitel und "
                "optional due_at als ISO-8601-Termin. Bei einem Tag ohne Uhrzeit "
                "(z.B. morgen oder am Freitag) verwende nur YYYY-MM-DD. "
                "Nur bei ausdrücklich genannter Uhrzeit verwende Datum, Uhrzeit und Zeitzone. "
                "Ohne klaren Termin setze due_at auf null. Erfinde keinen Termin. "
                "Wenn die konkrete Tätigkeit fehlt oder unverständlich ist, setze clarification "
                "auf eine kurze Rückfrage und title auf einen leeren String. Sonst clarification "
                "auf null. Bei einer ergänzenden Nutzerantwort übernimm den vorher genannten "
                "Termin, sofern die Ergänzung ihn nicht korrigiert. Gesprochene Buchstaben "
                "wie 'A R D und Z D F schauen' ergeben 'ARD und ZDF schauen'. "
                "clarification darf sich NUR auf eine fehlende konkrete Tätigkeit beziehen, "
                "niemals auf Datum oder Uhrzeit. 'Freitag' bedeutet den nächsten kommenden "
                "Freitag; frage dafür NICHT nach einer Datumsbestätigung. Ohne Uhrzeit ist "
                "ein Datum ausreichend. Ohne Termin ist due_at=null zulässig. "
                "Beispiel: 'für Freitag' plus 'A R D und Z D F schauen' ergibt title="
                "'ARD und ZDF schauen', clarification=null und das nächste Freitagsdatum. "
                f"Jetzt ist {now.isoformat(timespec='minutes')}."
            )}, {"role": "user", "content": utterance}],
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{config.ollama_url}/api/chat", json=payload)
            response.raise_for_status()
        result = json.loads(response.json()["message"]["content"])
        if not isinstance(result, dict) or not isinstance(result.get("title"), str):
            raise TaskError("Aufgabe konnte nicht sicher erkannt werden.")
        clarification = result.get("clarification")
        if isinstance(clarification, str) and clarification.strip():
            # The extraction model occasionally invents date ambiguities in
            # its question. Dates are resolved locally; only ask for the task.
            raise TaskClarification("Was soll ich konkret als Aufgabe festhalten?")
        due = result.get("due_at")
        if due is not None and not isinstance(due, str):
            raise TaskError("Termin konnte nicht sicher erkannt werden.")
        if due is None and re.search(r"\bmorgen\b", utterance.lower()) and not re.search(
            r"\b(?:nicht|kein\w*)\s+morgen\b", utterance.lower()
        ):
            due = (datetime.now(LOCAL_ZONE).date() + timedelta(days=1)).isoformat()
        return result["title"], due

    def tools(self, config, notify):
        self.session_notify = notify
        self.session_config = config

        async def failure(params, exc):
            if isinstance(exc, sqlite3.Error):
                LOG.exception("Aufgaben-Datenbankzugriff fehlgeschlagen")
                message = "Datenbank nicht verfügbar oder nicht beschreibbar. Nichts geändert."
            else:
                message = str(exc)
            await params.result_callback({"status": "error", "error": message})
            notify({"type": "tool_error", "text": message})

        async def propose(params):
            utterance = params.arguments.get("text", "")
            notify({"type": "tool_start", "text": "Aufgabe und Termin werden geprüft."})
            try:
                title, due = await self._extract(config, utterance)
                original, separator, correction = utterance.rpartition(
                    "\nErgänzung/Korrektur des Nutzers: "
                )
                if separator:
                    original_due, original_time = temporal_fragment(original)
                    clear_due, clear_time = temporal_fragment(correction, original_due or due)
                    if original_time and clear_due and len(clear_due) == 10:
                        clear_due, clear_time = temporal_fragment(original_time + " Uhr", clear_due)
                else:
                    clear_due, clear_time = temporal_fragment(utterance)
                if clear_due or clear_time:
                    due, time_hint = clear_due, clear_time
                else:
                    due, time_hint = temporal_fragment(utterance, due)
                proposal = self._propose(title, due, time_hint)
            except TaskClarification as exc:
                self.awaiting_details = True
                self.pending_request = utterance
                await params.result_callback({"status": "needs_details", "question": str(exc)})
                return
            except (TaskError, httpx.HTTPError, ValueError, KeyError) as exc:
                await params.result_callback({"error": str(exc) or "Aufgabe nicht erkannt."})
                notify({"type": "tool_error", "text": "Aufgabe nicht erkannt. Bitte neu formulieren."})
                return
            self.awaiting_details = False
            self.pending_request = None
            notify({"type": "task_proposal", "proposal": proposal})
            await params.result_callback({"status": "pending", "proposal": proposal,
                                          "message": "Bitte ausdrücklich bestätigen; noch nicht gespeichert."})

        async def confirm(params):
            try:
                task = await asyncio.to_thread(
                    self._confirm, params.arguments.get("proposal_id", "")
                )
            except (TaskError, sqlite3.Error) as exc:
                await failure(params, exc)
                return
            notify({"type": "task_saved", "task": task})
            await params.result_callback({"status": "ok", "task": task})

        async def list_open(params):
            try:
                tasks = await asyncio.to_thread(
                    self.store.list, params.arguments.get("status", "open")
                )
            except (TaskError, sqlite3.Error) as exc:
                await failure(params, exc)
                return
            with self.proposal_lock:
                proposals = [dict(item) for item in self.proposals.values()]
            await params.result_callback({"status": "ok", "tasks": tasks,
                                          "proposals": proposals})
            notify({"type": "task_changed"})

        async def update(params):
            try:
                changes = TaskUpdateInput.model_validate(params.arguments)
                fields = changes.model_dump(exclude_unset=True)
                if not fields:
                    raise TaskError("Keine Änderung angegeben.")
                task = await asyncio.to_thread(
                    self.store.update, params.arguments.get("task_id", ""), **fields
                )
            except (ValueError, sqlite3.Error) as exc:
                await failure(params, exc)
                return
            notify({"type": "task_changed", "task": task})
            await params.result_callback({"status": "ok", "task": task})

        async def finish(params):
            try:
                task = await asyncio.to_thread(
                    self.store.set_done, params.arguments.get("task_id", "")
                )
            except (TaskError, sqlite3.Error) as exc:
                await failure(params, exc)
                return
            notify({"type": "task_changed", "task": task})
            await params.result_callback({"status": "ok", "task": task})

        async def delete(params):
            task_id = params.arguments.get("task_id", "")
            try:
                await asyncio.to_thread(self.store.delete, task_id)
            except (TaskError, sqlite3.Error) as exc:
                await failure(params, exc)
                return
            notify({"type": "task_changed", "deleted_id": task_id})
            await params.result_callback({"status": "ok", "deleted_id": task_id})

        async def revise(params):
            try:
                changes = TaskUpdateInput.model_validate(params.arguments)
                fields = changes.model_dump(exclude_unset=True)
                if not fields:
                    raise TaskError("Keine Änderung angegeben.")
                proposal = self._revise(params.arguments.get("proposal_id", ""), **fields)
            except ValueError as exc:
                await failure(params, exc)
                return
            notify({"type": "task_proposal", "proposal": proposal})
            await params.result_callback({"status": "pending", "proposal": proposal,
                                          "message": "Korrigiert, noch nicht gespeichert."})

        task_id_property = {"type": "string", "description": "Exakte Aufgaben-ID aus list_tasks."}
        edit_properties = {
            "title": {"type": "string", "minLength": 1, "maxLength": 300,
                      "description": (
                          "Neuer Titel. Weglassen, wenn unverändert. Bei 'Benenne X in Y um' "
                          "ist der Titel nur Y; das abschließende 'um' gehört zum Befehl."
                      )},
            "due_at": {"type": ["string", "null"], "description": (
                "Neuer ISO-Termin in Europe/Berlin. Bei Verschieben nur des Tages die bisherige "
                "Uhrzeit erhalten; bei Änderung nur der Uhrzeit den Tag erhalten. "
                "YYYY-MM-DD für Tag ohne Uhrzeit, null entfernt den Termin. "
                "Weglassen, wenn unverändert."
            )},
        }
        selection_rule = (
            " Zuerst list_tasks lesen und die exakte ID verwenden. Bei mehreren passenden "
            "Aufgaben nachfragen, niemals eine ID erfinden. Nur auf ausdrücklichen Nutzerauftrag."
        )

        return [
            FunctionSchema(name="propose_task", description=(
                "Erstelle einen Vorschlag für eine persönliche Aufgabe, wenn der Nutzer "
                "ausdrücklich etwas merken, notieren oder als Aufgabe anlegen will. "
                "Der Vorschlag wird noch NICHT gespeichert; bitte um Bestätigung."
            ), properties={"text": {"type": "string", "description": "Der vollständige Nutzersatz."}},
                required=["text"], handler=propose),
            FunctionSchema(name="confirm_task", description=(
                "Speichere nur einen bereits sichtbaren Aufgabenvorschlag nach einer "
                "ausdrücklichen Bestätigung des Nutzers."
            ), properties={"proposal_id": {"type": "string", "description": "Vorschlags-ID."}},
                required=["proposal_id"], handler=confirm),
            FunctionSchema(name="list_tasks", description=(
                "Lies gespeicherte Aufgaben und ungespeicherte Vorschläge mit ihren echten IDs. "
                "Auch vor Änderungen, Löschen, Erledigen und beim erneuten Zeigen eines Vorschlags."
            ), properties={"status": {"type": "string", "enum": ["open", "done", "all"]}},
                required=[], handler=list_open),
            FunctionSchema(name="update_task", description=(
                "Bearbeite oder verschiebe eine gespeicherte Aufgabe; ändere Titel und/oder Termin."
                + selection_rule
            ), properties={"task_id": task_id_property, **edit_properties},
                required=["task_id"], handler=update),
            FunctionSchema(name="complete_task", description=(
                "Markiere eine gespeicherte Aufgabe als erledigt." + selection_rule
            ), properties={"task_id": task_id_property}, required=["task_id"], handler=finish),
            FunctionSchema(name="delete_task", description=(
                "Lösche eine gespeicherte Aufgabe dauerhaft." + selection_rule
            ), properties={"task_id": task_id_property}, required=["task_id"], handler=delete),
            FunctionSchema(name="revise_task_proposal", description=(
                "Korrigiere Titel oder Termin eines ungespeicherten Vorschlags, z.B. "
                "'Mittagessen statt ARD schauen'. Das speichert noch keine Aufgabe. "
                "Verwende die Vorschlags-ID aus list_tasks, keine Aufgaben-ID."
            ), properties={"proposal_id": {"type": "string"}, **edit_properties},
                required=["proposal_id"], handler=revise),
        ]

    @staticmethod
    def describe(item: dict) -> str:
        due = item.get("due_at")
        if not due:
            hint = item.get("time_hint")
            return (f"{item['title']} um {hint} Uhr, Datum fehlt" if hint else
                    item["title"] + " ohne Termin")
        parsed = datetime.fromisoformat(due)
        day = parsed.strftime("%d.%m.%Y")
        if len(due) == 10:
            return f"{item['title']} am {day}, ohne Uhrzeit"
        return f"{item['title']} am {day} um {parsed.strftime('%H:%M')} Uhr"

    def emit(self, event: dict):
        notify = getattr(self, "session_notify", None)
        if notify:
            notify(event)

    async def _manage(self, context) -> str:
        listed = await trigger_tool(context, "list_tasks", {"status": "all"})
        if not listed or listed.get("status") != "ok":
            return "Aufgaben konnten nicht gelesen werden. Nichts geändert."
        names = {"update_task", "complete_task", "delete_task", "revise_task_proposal"}
        tools = [{"type": "function", "function": tool.to_default_dict()}
                 for tool in context.tools.standard_tools if tool.name in names]
        try:
            selected = await route_tool(self.session_config, context, tools, notify=self.emit)
        except (httpx.HTTPError, RuntimeError, ValueError):
            LOG.exception("Aufgabenänderung konnte nicht ausgewählt werden")
            return "Die Änderung konnte nicht sicher erkannt werden. Bitte formuliere sie erneut."
        if selected is None:
            return "Nichts geändert. Bitte nenne die Aufgabe und die gewünschte Änderung eindeutig."
        result = json.loads(context.get_messages()[-1]["content"])
        if result.get("status") not in {"ok", "pending"}:
            return "Nicht geändert: " + result.get("error", "Änderung fehlgeschlagen.")
        name, arguments = selected
        if name == "delete_task":
            deleted = next(task for task in listed["tasks"] if task["id"] == arguments["task_id"])
            return "Gelöscht: " + deleted["title"] + "."
        if name == "revise_task_proposal":
            return "Vorschlag korrigiert: " + self.describe(result["proposal"]) + ". Noch nicht gespeichert."
        prefix = "Erledigt: " if name == "complete_task" else "Geändert: "
        return prefix + self.describe(result["task"]) + "."

    async def _task_reference(self, text: str) -> bool:
        if re.search(r"\b(?:aufgabe\w*|vorschlag\w*|termin\w*|erinnerung\w*|titel)\b", text):
            return True
        try:
            items = await asyncio.to_thread(self.store.list, "all")
        except sqlite3.Error:
            return False
        with self.proposal_lock:
            items += [dict(item) for item in self.proposals.values()]
        ignored = {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
                   "bei", "mit", "für", "von", "und", "oder", "zum", "zur", "auf"}
        words = {word for item in items for word in re.findall(r"\w+", item["title"].lower())
                 if len(word) > 2 and word not in ignored}
        return bool(words.intersection(re.findall(r"\w+", text)))

    async def on_user_text(self, context, text: str) -> str | None:
        lowered = text.lower().strip()
        new_task = bool(re.search(
            r"\b(?:neue aufgabe|merk(?:e|t)?\s+(?:dir|ihr)|notiere|als aufgabe|"
            r"erinnere mich|ich muss|ich habe.*(?:anrufen|telefonieren)|"
            r"(?:muss|soll|will).*?(?:anrufen|telefonieren|erledigen))\b|"
            r"\b(?:leg\w*|erstell\w*|mach\w*)\b.{0,80}\baufgabe\b.*\ban\b|"
            r"\b(?:leg\w*|erstell\w*)\b.{0,80}\baufgabe\b|"
            r"\baufgabe\b.{0,80}\b(?:anlegen|erstellen)\b", lowered
        ))
        # Management requests need real IDs and model interpretation, not the
        # short date-correction shortcut (which could change the wrong task).
        management = re.search(
            r"\b(?:lösch\w*|entfern\w*|verschieb\w*|verleg\w*|erledigt|abhaken|"
            r"bearbeit\w*|editier\w*|änder\w*|umbenenn\w*|benenn\w*|korrigier\w*)\b", lowered
        )
        if management and not new_task:
            if await self._task_reference(lowered):
                return await self._manage(context)
            return None
        show_request = re.search(
            r"\b(?:zeig(?:e|en)?|zeug(?:e|en)?|öffne|öffnen)\b.{0,60}\b"
            r"(?:aufgaben|termine|erinnerungen)\b|"
            r"\b(?:aufgaben|termine|erinnerungen)\b.{0,30}\b(?:anzeigen|einblenden|zeigen)\b",
            lowered,
        )
        if show_request and not new_task and not re.search(r"\b(?:nicht|keine)\b", lowered):
            try:
                tasks = await asyncio.to_thread(self.store.list)
            except (TaskError, sqlite3.Error):
                LOG.exception("Aufgabenanzeige konnte nicht geladen werden")
                return "Aufgaben konnten nicht geladen werden."
            with self.proposal_lock:
                proposals = [dict(item) for item in self.proposals.values()]
            self.emit({"type": "task_show", "tasks": tasks, "proposals": proposals})
            return "Bitteschön."
        list_request = re.search(
            r"\b(?:welche (?:aufgaben|termine|erinnerungen)|was ist (?:noch )?offen|"
            r"meine (?:aufgaben|erinnerungen)|erinner mich.*was offen)\b", lowered
        )
        if list_request:
            result = await trigger_tool(context, "list_tasks", {})
            if not result or result.get("status") != "ok":
                return "Aufgaben konnten nicht gelesen werden."
            tasks = result.get("tasks", []) if result else []
            return ("Es sind keine Aufgaben offen." if not tasks else
                    "Offen sind: " + "; ".join(self.describe(task) for task in tasks) + ".")

        if re.search(r"\b(?:aktiv|automatisch|von selbst)\s+erinner", lowered):
            return ("Ja, bei Aufgaben mit Uhrzeit zeigt MARTI auf diesem Mac zum Termin "
                    "eine Desktop-Benachrichtigung, solange der Server läuft und macOS "
                    "Benachrichtigungen zulässt. Aufgaben ohne Uhrzeit stehen nur in der Liste.")

        with self.proposal_lock:
            proposal_id = next(reversed(self.proposals)) if self.proposals else None
            proposal = dict(self.proposals[proposal_id]) if proposal_id else None

        reminder_idea = bool(re.search(r"\b(?:erinnerung|erinnern)\b", lowered))
        action = bool(re.search(r"\b(?:anrufen|telefonieren|erledigen|testen|"
                                r"abholen|bezahlen|machen|schreiben|schauen|ansehen)\b", lowered))
        if reminder_idea and not action and not new_task and not proposal:
            self.awaiting_details = True
            self.pending_request = text
            return "Woran soll ich dich erinnern? Nenne Aufgabe und möglichst einen Termin."

        confirmation = bool(re.search(r"\b(?:bestätig\w*|speicher\w*|ja bitte)\b", lowered))
        if self.awaiting_details and confirmation and not new_task:
            return "Es fehlt noch die konkrete Aufgabe. Was soll ich als Vorschlag anlegen?"

        if proposal and not new_task and not self.awaiting_details:
            if re.search(r"\b(?:zeig\w*|vorschlag.*nochmal)\b", lowered):
                listed = await trigger_tool(context, "list_tasks", {})
                if not listed or listed.get("status") != "ok":
                    return "Vorschläge konnten nicht gelesen werden."
                for item in listed["proposals"]:
                    self.emit({"type": "task_proposal", "proposal": item})
                return "Noch nicht gespeichert: " + "; ".join(
                    self.describe(item) for item in listed["proposals"]
                ) + "."
            if re.search(r"\b(?:statt|nicht|nein)\b", lowered):
                if await self._task_reference(lowered):
                    return await self._manage(context)
                return None
            try:
                due, time_hint = temporal_fragment(text, proposal["due_at"])
            except TaskError as exc:
                return str(exc)
            temporal = bool(re.search(
                r"\b(?:morgen|übermorgen|montag|dienstag|mittwoch|donnerstag|"
                r"freitag|samstag|sonntag|uhr)\b|\d{1,2}[.\-/]\d{1,2}", lowered
            ))
            # Only bare day/time corrections are safe to apply implicitly.
            short_correction = bool(re.fullmatch(
                r"(?:ja|bitte|am|um|auf|doch|dann|ist|der|den|es|"
                r"morgen|übermorgen|montag|dienstag|mittwoch|donnerstag|freitag|"
                r"samstag|sonntag|uhr|" + "|".join(HOUR_WORDS) + r"|[\s\d.:,!?/\-])+",
                lowered,
            ))
            if temporal and not short_correction:
                return await self._manage(context)
            if temporal and (due or time_hint):
                if due and len(due) == 10 and proposal.get("time_hint"):
                    due, _ = temporal_fragment(proposal["time_hint"] + " Uhr", due)
                with self.proposal_lock:
                    revised = self._revise(proposal_id, due_at=due) if due else self.proposals[proposal_id]
                    revised["time_hint"] = time_hint
                    self.proposals[proposal_id]["time_hint"] = time_hint
                self.emit({"type": "task_proposal", "proposal": revised})
                return "Vorschlag korrigiert: " + self.describe(revised) + ". Noch nicht gespeichert."

            confirm = bool(re.search(r"\b(?:bestätig\w*|speicher\w*|ja bitte)\b", lowered))
            negated = bool(re.search(r"\b(?:nicht|kein\w*)\s+(?:bestätig|speicher)", lowered))
            if confirm and not negated:
                result = await trigger_tool(context, "confirm_task", {"proposal_id": proposal_id})
                if result and result.get("status") == "ok":
                    task = result["task"]
                    return "Gespeichert: " + self.describe(task) + "."
                return "Nicht gespeichert: " + (result or {}).get("error", "Bestätigung fehlgeschlagen.")

            if action:
                return await self._manage(context)

        if (not proposal and not new_task and not self.awaiting_details
                and self.last_saved_id and self.last_saved_at):
            recent = datetime.now(UTC) - self.last_saved_at < timedelta(minutes=10)
            temporal = bool(re.search(
                r"\b(?:morgen|übermorgen|montag|dienstag|mittwoch|donnerstag|"
                r"freitag|samstag|sonntag|uhr)\b|\d{1,2}[.\-/]\d{1,2}", lowered
            ))
            if recent and temporal:
                if not re.fullmatch(
                    r"(?:ja|bitte|am|um|auf|doch|dann|ist|der|den|es|"
                    r"morgen|übermorgen|montag|dienstag|mittwoch|donnerstag|freitag|"
                    r"samstag|sonntag|uhr|" + "|".join(HOUR_WORDS) + r"|[\s\d.:,!?/\-])+",
                    lowered,
                ):
                    return await self._manage(context)
                listed = await trigger_tool(context, "list_tasks", {})
                if not listed or listed.get("status") != "ok":
                    return "Termin nicht geändert: Aufgaben konnten nicht gelesen werden."
                task = next((item for item in listed["tasks"]
                             if item["id"] == self.last_saved_id), None)
                if task:
                    try:
                        due, hint = temporal_fragment(text, task["due_at"])
                        if not due:
                            return f"Ich habe {hint} Uhr verstanden, aber ohne Datum nichts geändert."
                        result = await trigger_tool(context, "update_task", {
                            "task_id": task["id"], "due_at": due,
                        })
                    except TaskError as exc:
                        return "Termin nicht geändert: " + str(exc)
                    if not result or result.get("status") != "ok":
                        return "Termin nicht geändert: " + (result or {}).get(
                            "error", "Änderung fehlgeschlagen."
                        )
                    return "Termin korrigiert: " + self.describe(result["task"]) + "."

        if self.awaiting_details and not new_task and re.search(
            r"\b(?:abbrechen|vergiss es|lass es|doch nicht)\b", lowered
        ):
            self.awaiting_details = False
            self.pending_request = None
            return "Abgebrochen. Es wurde keine Aufgabe angelegt."

        detail_reply = self.awaiting_details and not confirmation and bool(lowered) and not (
            "?" in lowered or re.match(r"(?:wer|wie|was|wo|warum|wann)\b", lowered)
        )
        if new_task or detail_reply:
            if not action and len(lowered) < 25:
                if new_task:
                    self.awaiting_details = True
                    self.pending_request = text
                    return "Woran soll ich dich erinnern? Nenne die konkrete Aufgabe."
            utterance = text
            if detail_reply and not new_task and self.pending_request:
                utterance = ("Ursprünglicher Aufgabenwunsch: " + self.pending_request
                             + "\nErgänzung/Korrektur des Nutzers: " + text)
            result = await trigger_tool(context, "propose_task", {"text": utterance})
            if result and result.get("status") == "needs_details":
                return result["question"]
            if result and result.get("status") == "pending":
                proposal = result["proposal"]
                suffix = (" Das Datum fehlt noch." if proposal.get("time_hint") and
                          not proposal.get("due_at") else "")
                return ("Vorschlag: " + self.describe(proposal) + "." + suffix +
                        " Er ist sichtbar. Sag 'Ja, speichern' oder drücke Speichern.")
            return "Kein Vorschlag angelegt: " + (result or {}).get(
                "error", "Aufgabe nicht sicher erkannt."
            )

        if re.search(r"\b(?:bestätig\w*|speicher\w*)\b", lowered) and not proposal:
            return "Es liegt kein Aufgabenvorschlag zur Bestätigung vor."
        return None
