"""Async client for the agentclaw A2A gateway. Knows nothing about the voice pipeline."""

import uuid

import httpx


class AgentClawError(Exception):
    """Raised when agentclaw is unreachable or a task fails."""


async def ask_martin(base_url: str, message: str, *, timeout: float = 60.0) -> str:
    """Send one message to Martin via agentclaw's A2A gateway and return his reply text."""
    task_id = uuid.uuid4().hex
    body = {"id": task_id, "message": {"parts": [{"text": message}]}}

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(f"{base_url}/a2a/tasks/send", json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AgentClawError(f"agentclaw ist nicht erreichbar: {exc}") from exc

    data = response.json()
    if data.get("status", {}).get("state") != "completed":
        parts = data.get("artifacts", [{}])[0].get("parts", [{}])
        reason = parts[0].get("text", "unbekannter Fehler") if parts else "unbekannter Fehler"
        raise AgentClawError(f"Martin konnte die Aufgabe nicht abschließen: {reason}")

    try:
        return data["artifacts"][0]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AgentClawError("Antwort von Martin hatte kein lesbares Format.") from exc
