"""One MCP tool, ordinary JSON HTTP responses, no frontend required."""
import contextlib
import os
from pathlib import Path

import anyio
import uvicorn
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route

from search import Catalog, INPUT_SCHEMA, OUTPUT_SCHEMA

server = Server('polaris-product-search', version='1.0.0')
catalog = None


@server.list_tools()
async def list_tools():
    return [types.Tool(name='search_products',
        description='Search the product catalog with filters, ranking, and optional complementary recommendations.',
        inputSchema=INPUT_SCHEMA, outputSchema=OUTPUT_SCHEMA)]


@server.call_tool()
async def call_tool(name, arguments):
    if name != 'search_products':
        raise ValueError('Unknown tool')
    # Keep synchronous baseline search off the HTTP event loop.
    return await anyio.to_thread.run_sync(catalog.search, arguments)


ALLOWED_HOSTS = os.getenv('MCP_ALLOWED_HOSTS', 'localhost:*,127.0.0.1:*,[::1]:*').strip()
security = TransportSecuritySettings(
    # MCP_ALLOWED_HOSTS='*' is for hosting platforms that assign the public hostname at deploy
    # time: it turns off only the Host/Origin allowlist (DNS-rebinding protection for browsers).
    enable_dns_rebinding_protection=ALLOWED_HOSTS != '*',
    allowed_hosts=[h.strip() for h in ALLOWED_HOSTS.split(',') if h.strip() and h.strip() != '*'],
    allowed_origins=[o.strip() for o in os.getenv('MCP_ALLOWED_ORIGINS', '').split(',') if o.strip()],
)
sessions = StreamableHTTPSessionManager(server, json_response=True, stateless=True, security_settings=security)


@contextlib.asynccontextmanager
async def lifespan(app):
    global catalog
    path = os.getenv('CATALOG_PATH', str(Path(__file__).parent / 'data/products.json'))
    catalog = await anyio.to_thread.run_sync(Catalog.load, path)
    await anyio.to_thread.run_sync(warm_up, catalog)
    async with sessions.run():
        yield


def warm_up(catalog):
    """Pay one-time costs (ONNX session, lazily built masks and sort ranks) before serving."""
    for sort_by in ('id', 'price', 'rating', 'review_count'):
        for sort_order in ('asc', 'desc'):
            catalog.engine.run({'sort_by': sort_by, 'sort_order': sort_order})
    for query in ('black jeans and white t-shirts under 2000', 'dress with a matching bag', 'lazy sunday',
                  'wedding guest outfit'):
        catalog.engine.run({'query': query, 'sort_by': 'price', 'include_recommendations': True})


class MCPApplication:
    async def __call__(self, scope, receive, send):
        if scope['method'] == 'GET':
            # Stateless JSON transport offers no standalone server-to-client stream.
            await PlainTextResponse('Method Not Allowed', status_code=405,
                                    headers={'Allow': 'POST, DELETE'})(scope, receive, send)
            return
        await sessions.handle_request(scope, receive, send)


async def health(request):
    return JSONResponse({'status': 'ok', 'products': len(catalog.products),
                         'dense': catalog.engine.dense is not None})


async def lab(request):
    return FileResponse(Path(__file__).parent / 'static/lab.html')


async def explain(request):
    """Search Lab only: the same search as the MCP tool, plus its plan and score breakdown."""
    try:
        arguments = await request.json()
        out = await anyio.to_thread.run_sync(catalog.explain, arguments)
    except ValueError as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)
    return JSONResponse(out)


app = Starlette(lifespan=lifespan, routes=[
    Route('/mcp', endpoint=MCPApplication(), methods=['POST', 'GET', 'DELETE']),
    Route('/healthz', endpoint=health),
    Route('/', endpoint=lab),
    Route('/api/explain', endpoint=explain, methods=['POST']),
])

if __name__ == '__main__':
    # Keep idle connections open longer than the load balancer in front of us does (AWS ALB
    # default 60 s). If the app closes an idle keep-alive first, the balancer can forward the
    # next request down a dead connection and answer 502 without the app ever seeing it.
    uvicorn.run(app, host=os.getenv('HOST', '0.0.0.0'), port=int(os.getenv('PORT', '8000')),
                timeout_keep_alive=int(os.getenv('KEEP_ALIVE_SECONDS', '120')))
