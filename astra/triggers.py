"""Deterministic keyword triggers that call an already-registered tool's
handler directly, bypassing the LLM's own decision to call it. Exists
because read_news, measured, is only called by the LLM in ~15-20% of
matching requests once generate_image and read_article also compete for
its attention — a simple keyword match is ~100% reliable for the same job.
ask_martin shows the same failure once it's one of four competing tools:
the model narrates ("Ich frage Martin ab …") instead of calling it.
"""

import json
import re
import uuid

from astra.feeds import TOPICS

_NEWS_WORDS = ("nachrichten", "news")
_TOPIC_ALIASES = {
    "hilden": "hilden",
    "technik": "tech",
    "tech": "tech",
    "ki": "tech",
    "wirtschaft": "wirtschaft",
}

_ARTICLE_WORDS = ("artikel", "schlagzeile")
_NUMBER_WORDS = {
    "eins": 1, "eine": 1, "einen": 1, "erste": 1, "ersten": 1, "erster": 1,
    "zwei": 2, "zweite": 2, "zweiten": 2, "zweiter": 2,
    "drei": 3, "dritte": 3, "dritten": 3, "dritter": 3,
    "vier": 4, "vierte": 4, "vierten": 4, "vierter": 4,
    "fünf": 5, "fünfte": 5, "fünften": 5, "fünfter": 5,
    "sechs": 6, "sechste": 6, "sechsten": 6, "sechster": 6,
    "sieben": 7, "siebte": 7, "siebten": 7, "siebter": 7,
    "acht": 8, "achte": 8, "achten": 8, "achter": 8,
    "neun": 9, "neunte": 9, "neunten": 9, "neunter": 9,
    "zehn": 10, "zehnte": 10, "zehnten": 10, "zehnter": 10,
}  # fmt: skip


def detect_news_topic(text: str) -> str | None:
    """Return a configured feed topic if `text` looks like a news request."""
    lowered = text.lower()
    if not any(word in lowered for word in _NEWS_WORDS):
        return None
    for alias, topic in _TOPIC_ALIASES.items():
        if alias in lowered and topic in TOPICS:
            return topic
    return "nachrichten" if "nachrichten" in TOPICS else None


def detect_article_reference(text: str) -> int | None:
    """Return the 1-based article number if `text` asks for one, e.g.
    "Artikel 2", "Artikel Nummer drei", "zweiter Artikel", "Detail zu
    Schlagzeile 4". Resolution against the last shown headlines is the
    caller's job — this only extracts the number.
    """
    lowered = text.lower()
    if not any(word in lowered for word in _ARTICLE_WORDS):
        return None
    match = re.search(r"(?:artikel|schlagzeile)\D{0,15}(\d+)", lowered)
    if match:
        return int(match.group(1))
    for word, number in _NUMBER_WORDS.items():
        if re.search(rf"(?:artikel|schlagzeile)\D{{0,15}}{word}\b", lowered):
            return number
        if re.search(rf"\b{word}\D{{0,15}}(?:artikel|schlagzeile)", lowered):
            return number
    return None


def detect_martin_reference(text: str) -> str | None:
    """Return the message to forward to Martin if `text` addresses him by name."""
    if re.search(r"\bmartin\b", text.lower()):
        return text
    return None


def detect_local_search(text: str, roots: tuple[str, ...]) -> dict[str, str] | None:
    """Recognize explicit local 'suche … nach …' commands without inventing scope."""
    if not roots:
        return None
    match = re.match(
        r"^\s*(?:(?:hey|hallo)\s+)?(?:martin[,.]?\s+)?(?:bitte\s+)?"
        r"(?:suche|such|durchsuche)\b(.{0,140}?)\bnacht?\s+(.+)",
        text,
        re.I,
    )
    if not match:
        return None
    prefix = text[: match.start(2)].lower()
    if re.search(r"\b(?:nicht|kein\w*|internet|web|google)\b", prefix):
        return None
    scope = match.group(1).lower()
    named = re.search(r"\b(?:projekt|ordner|verzeichnis|wurzel)\s+[\"'„]?([\w-]+)", scope)
    if named and named.group(1) not in [root.lower() for root in roots]:
        return None
    selected = [
        root
        for root in roots
        if re.search(rf"(?<![\w-]){re.escape(root.lower())}(?:-projekte[n]?)?(?![\w-])", scope)
    ]
    if len(selected) > 1:
        return None
    if not selected:
        if len(roots) != 1 or not re.fullmatch(
            r"\s*(?:in\s+(?:meinen|den)\s+(?:lokalen\s+)?(?:projekten|dateien)|lokal)\s*",
            scope,
        ):
            return None
        selected = [roots[0]]
    query = match.group(2).strip()
    quoted = re.match(r"^[\"'„“](.+?)[\"'“”](?:\s|[.!?]|$)", query)
    if quoted:
        query = quoted.group(1)
    else:
        query = re.split(r"\s+und\s+(?:erklär\w*|zeig\w*|fass\w*)\b", query, flags=re.I)[0]
        query = query.rstrip(" .!?")
    return {"root": selected[0], "query": query} if 0 < len(query) <= 200 else None


def detect_wikipedia_search(text: str) -> dict | None:
    """Only explicit search imperatives; never capability or negated requests."""
    match = re.fullmatch(
        r"\s*(?:bitte\s+)?(?:suche|such)\s+(?:auf|in)\s+(?:der\s+)?"
        r"(?:(deutschen|englischen)\s+)?wikipedia\s+nach\s+(.+?)\s*[.!?]?\s*",
        text, re.I,
    )
    if not match:
        return None
    query = match.group(2).strip().rstrip(".!?")
    if not query or re.search(r"\b(?:nicht|keine?)\b", query, re.I):
        return None
    return {"query": query, "language": "en" if (match.group(1) or "").lower() == "englischen" else "de"}


def latest_reference_source(messages: list[dict]) -> str | None:
    calls = {}
    source = None
    for message in messages:
        for call in message.get("tool_calls", []):
            calls[call.get("id")] = call.get("function", {}).get("name")
        if message.get("role") == "tool":
            name = calls.get(message.get("tool_call_id"), message.get("name"))
            if name in {"search_wikipedia", "read_news"}:
                source = name
    return source


def wikipedia_reference_number(text: str, messages: list[dict]) -> int | None:
    lowered = text.lower()
    if re.search(r"\b(?:nicht|kein\w*)\b", lowered):
        return None
    if not re.match(r"\s*(?:bitte\s+)?(?:lies|lese|zeig\w*|fass\w*|details|mehr\s+zu)\b", lowered):
        return None
    explicit = "wikipedia" in lowered
    if not explicit and latest_reference_source(messages) != "search_wikipedia":
        return None
    normalized = re.sub(r"\btreffer\b", "artikel", lowered)
    return detect_article_reference(normalized)


def wikipedia_read_arguments(text: str, messages: list[dict]) -> dict | None:
    number = wikipedia_reference_number(text, messages)
    evidence = latest_tool_result(messages, "search_wikipedia")
    if number is None or not evidence or evidence.get("status") != "ok":
        return None
    results = evidence.get("results", [])
    if not 1 <= number <= len(results):
        return None
    return {"page_id": str(results[number - 1]["page_id"]),
            "language": evidence.get("language", "de")}


async def trigger_tool(
    context, tool_name: str, arguments: dict, *, params_extra: dict | None = None
) -> dict | None:
    """Call an already-registered tool's handler directly, then inject the
    round-trip into `context` so the LLM's next completion already sees it
    as answered instead of having to decide to call the tool itself.
    Returns the tool's result dict (or None if the tool wasn't found).
    """
    tools = getattr(context.tools, "standard_tools", None) or []
    schema = next((t for t in tools if t.name == tool_name), None)
    if schema is None:
        return None

    call_id = f"trigger_{uuid.uuid4().hex[:8]}"
    captured: dict = {}

    class _DirectParams:
        pass

    params = _DirectParams()
    params.arguments = arguments
    for key, value in (params_extra or {}).items():
        setattr(params, key, value)

    async def result_callback(result, *, properties=None):
        captured["result"] = result
        context.add_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": arguments},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, ensure_ascii=False),
                },
            ]
        )

    params.result_callback = result_callback
    await schema.handler(params)
    return captured.get("result")


async def dispatch_deterministic_trigger(config, context, text: str, last_headlines: list[dict]):
    """Bypass the LLM's own decision for requests a keyword match can resolve
    reliably (see astra/triggers.py's module docstring for why: read_news alone
    was only called by the model in ~15-20% of matching requests). Runs for
    every model, not just ones without native tool support - even models with
    native tools narrate ("Ich frage Martin ab ...") instead of calling them
    often enough that the deterministic path is worth it unconditionally.
    `last_headlines` is mutated in place so a later article-number reference in
    the same session can resolve against the most recently shown list.
    """
    wiki_query = detect_wikipedia_search(text)
    if wiki_query:
        return "search_wikipedia", await trigger_tool(context, "search_wikipedia", wiki_query)
    wiki_read = wikipedia_read_arguments(text, context.get_messages())
    if wiki_read:
        return "read_wikipedia", await trigger_tool(context, "read_wikipedia", wiki_read)
    wiki_number = wikipedia_reference_number(text, context.get_messages())
    if wiki_number is not None:
        evidence = latest_tool_result(context.get_messages(), "search_wikipedia")
        count = len(evidence.get("results", [])) if evidence and evidence.get("status") == "ok" else 0
        reply = (f"Die Wikipedia-Suche hat {count} Treffer. Welche Nummer von 1 bis {count} möchtest du lesen?"
                 if count else "Es liegt keine Wikipedia-Trefferliste vor. Wonach soll ich auf Wikipedia suchen?")
        return "wikipedia_selection", {"reply": reply}
    # Wikipedia references must never fall through to the RSS article list.
    if "wikipedia" in text.lower() or (
        latest_reference_source(context.get_messages()) == "search_wikipedia"
        and re.search(r"\b(?:treffer|artikel)\b", text, re.I)
    ):
        return
    local_query = detect_local_search(text, config.local_search_roots)
    if local_query:
        # NativeOllamaService.get_chat_completions detects and runs this itself,
        # for every model, before building its request - detecting it here only
        # suppresses the checks below so a local-search phrasing isn't also read
        # as a news/article/Martin request.
        return
    martin_message = detect_martin_reference(text)
    if martin_message:
        result = await trigger_tool(context, "ask_martin", {"message": martin_message})
        return "ask_martin", result
    number = detect_article_reference(text)
    if number is not None:
        match = next((h for h in last_headlines if h.get("number") == number), None)
        if match:
            result = await trigger_tool(context, "read_article", {"url": match["link"]})
            return "read_article", result
    topic = detect_news_topic(text)
    if topic:
        result = await trigger_tool(context, "read_news", {"topic": topic})
        if result and result.get("status") == "ok":
            last_headlines[:] = result.get("headlines", [])
        return "read_news", result


def latest_tool_result(messages: list[dict], name: str, *, current_turn: bool = False):
    """Read actual tool evidence, associating results with their call IDs."""
    if current_turn:
        start = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i].get("role") == "user"), 0)
        messages = messages[start:]
    calls = {}
    found = None
    for message in messages:
        for call in message.get("tool_calls", []):
            calls[call.get("id")] = call.get("function", {}).get("name")
        if message.get("role") == "tool" and (
            calls.get(message.get("tool_call_id")) == name or message.get("name") == name
        ):
            try:
                value = json.loads(message.get("content", ""))
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                found = value
    return found


def wants_article_summary(text: str) -> bool:
    if re.search(r"\b(?:nicht|keine)\s+zusammenfass\w*\b|\bfass\w*\b.*\bnicht\s+zusammen\b", text, re.I):
        return False
    return bool(re.search(r"\b(?:zusammenfass\w*|kurzfassung|fass\w*\b.*\bzusammen)\b", text, re.I))


def references_previous_article(text: str) -> bool:
    if re.search(r"https?://|\b(?:dokument\w*|datei\w*|projekt\w*|lokal\w*|pdf)\b", text, re.I):
        return False
    return bool(re.search(
        r"\bfass\w*\s+(?:mir\s+)?(?:das|ihn)(?:\s+(?:bitte|kurz|nochmal))*\s+zusammen\b|"
        r"\b(?:den|diesen|vorigen|letzten)\s+artikel\b|"
        r"^\s*(?:bitte\s+)?(?:eine\s+)?(?:kurze\s+)?zusammenfassung(?:\s+bitte)?[.!?]?\s*$",
        text, re.I,
    ))
