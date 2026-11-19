"""Small integration contract for independent MARTI capabilities."""

from collections.abc import Iterable
from typing import Protocol

from fastapi import APIRouter
from pipecat.adapters.schemas.function_schema import FunctionSchema


class Extension(Protocol):
    router: APIRouter

    def tools(self, config, notify) -> Iterable[FunctionSchema]: ...

    async def on_user_text(self, context, text: str) -> str | None: ...


class ExtensionRegistry:
    def __init__(self, extensions: Iterable[Extension] = ()):
        self.extensions = tuple(extensions)

    def mount(self, app):
        for extension in self.extensions:
            app.include_router(extension.router)

    async def start(self):
        for extension in self.extensions:
            start = getattr(extension, "start", None)
            if start:
                await start()

    async def stop(self):
        for extension in reversed(self.extensions):
            stop = getattr(extension, "stop", None)
            if stop:
                await stop()

    def tools(self, config, notify):
        return [schema for extension in self.extensions for schema in extension.tools(config, notify)]

    async def on_user_text(self, context, text: str) -> str | None:
        for extension in self.extensions:
            reply = await extension.on_user_text(context, text)
            if reply:
                return reply
        return None
