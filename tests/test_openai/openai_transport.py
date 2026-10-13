"""Stub OpenAI answers on whichever HTTP library the installed openai sends through.

openai 2 sends through httpx and openai 3 through httpx2. LangChain builds its
clients with openai's defaults, so a stub patched onto the httpx transports
alone lets an openai 3 request leave the process.
"""
import httpx

try:
    import httpx2
except ImportError:
    httpx2 = None

HTTP_LIBRARIES = tuple(library for library in (httpx, httpx2) if library is not None)


def http_library_of(request):
    return httpx2 if httpx2 is not None and isinstance(request, httpx2.Request) else httpx


def answer_every_transport(monkeypatch, respond):
    async def respond_async(self, request):
        await request.aread()
        return respond(request)

    for library in HTTP_LIBRARIES:
        monkeypatch.setattr(library.HTTPTransport, "handle_request", lambda self, request: respond(request))
        monkeypatch.setattr(library.AsyncHTTPTransport, "handle_async_request", respond_async)


def _chunked_stream_class(library):
    class ChunkedStream(library.SyncByteStream, library.AsyncByteStream):
        """A body served chunk by chunk; httpx2 refuses a response stream of another library's type."""

        def __init__(self, chunks):
            self._chunks = chunks

        def __iter__(self):
            yield from self._chunks

        async def __aiter__(self):
            for chunk in self._chunks:
                yield chunk

    return ChunkedStream


CHUNKED_STREAMS = {library: _chunked_stream_class(library) for library in HTTP_LIBRARIES}
