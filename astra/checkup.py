"""Non-blocking startup diagnostics.

Targets one specific failure class: a tool silently vanishing instead of
failing loud. `ASTRA_LOCAL_SEARCH_ROOTS` pointing at a root local-search
doesn't actually have, or a broken/inaccessible local-search binary, both
leave search_local unregistered or unusable with zero visible error — the
model just picks a different tool (ask_martin) and nobody notices why.
`run_checkup` never raises: a broken check must not block the app from
becoming ready, it only means one less warning gets logged.
"""

from astra.core import Settings
from astra.local_search import LocalSearchError, status


async def run_checkup(config: Settings) -> list[str]:
    if not config.local_search_roots:
        return []
    try:
        current = await status(config.local_search_bin)
    except LocalSearchError as exc:
        roots = ", ".join(config.local_search_roots)
        return [f"Lokale Suche konfiguriert ({roots}), aber Status-Abfrage fehlgeschlagen: {exc}"]
    warnings = []
    for root in config.local_search_roots:
        entry = current.get(root)
        if entry is None:
            warnings.append(f'Lokale Suche: Root "{root}" ist bei local-search nicht registriert.')
        elif not entry.get("available") or not entry.get("indexed"):
            warnings.append(f'Lokale Suche: Root "{root}" ist nicht verfügbar oder noch nicht indiziert.')
    return warnings
