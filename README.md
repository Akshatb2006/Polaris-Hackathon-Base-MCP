# Polaris product search: hybrid, explainable MCP search

This fork replaces the starter's AND-token baseline with a multi-stage engine behind the same
`search_products` tool, schemas and protocol revision. The frozen catalog is only read, never modified.

| Stage | Module | What happens |
| --- | --- | --- |
| Validate | `search.py` | Published JSON Schema plus blank-filter and price-range checks (unchanged contract). |
| Understand | `engine/nlu.py` | Spell correction from the catalog's own vocabulary, budgets (`under 2k`, `between 500 and 900`, `around ₹1000`), colour families, who it is for, occasions, negation (`not blue`), rating asks. |
| Decompose | `engine/nlu.py` | Mixed choices (`jeans and white tees`) vs complementary asks (`dress with a matching bag`); constraints scoped per choice or shared (`each`). |
| Constrain | `engine/engine.py` | Hard numpy masks: explicit filters, query budgets, colour family, audience, garment kind. Explicit filters win over conflicting query words. |
| Score | `engine/engine.py` | Exact dense cosine (bge-small, all 105,126 products) + BM25F + garment evidence + pattern/occasion boosts + a shrunk rating prior. |
| Assemble | `engine/engine.py` | Fair round-robin across choices, exact field sorts over the relevant set, colour-wheel outfit recommendations with grounded reasons. |

Distinctive pieces:

- **Garment head nouns.** `engine/index.py` reads the noun that opens each description, so a product typed `Sweater` whose description starts "Dress in soft sweatshirt fabric" is still a dress. Phrases are head-final: "trainer socks" are socks.
- **Exact vector search.** One 105k×384 matrix-vector product per query (a few ms). Filters never hide valid matches the way approximate indexes can.
- **Exact fast path.** Empty-query calls use presorted permutations and boolean masks, with no per-product Python loop.
- **Calibrated abstention.** Out-of-domain lexicon, prompt-injection guard and a catalog-evidence ratio return empty arrays for unrelated or impossible requests.
- **Outfit graph.** Complements per garment, audience compatibility, colour harmony (same, neutral, analogous, complementary) and embedding affinity to the anchor product.
- **Search Lab.** `GET /` serves an inspector showing the parsed plan, score breakdown and product images (`POST /api/explain`). It is not part of the MCP contract.

The dense index (`index/embeddings.npy`, float16, 105,126 × 384, plus `index/meta.json`) is committed through
Git LFS like the catalog. Row *i* belongs to the *i*-th product in ascending ID order; the server refuses a
matrix whose catalog fingerprint or model name does not match and falls back to lexical-only search. Rebuild it
only if the document text or model changes (about 35 minutes on a 10-core laptop CPU):

```sh
python build_index.py            # writes index/embeddings.npy and index/meta.json
python eval/verify_index.py      # shape, norms, fingerprint, re-encodes 300 rows and checks row <-> ID mapping
```

## Evaluation

| Command | What it checks |
| --- | --- |
| `python -m unittest discover -s tests` | Contract fixtures plus one test per ranking/parsing rule |
| `python eval/benchmark.py URL` | Organizer-style run over MCP HTTP: invalid calls, exact empty-query pages vs a brute-force reference, pagination probes, 55 labelled cases, p50/p95/p99 |
| `python eval/constraints_check.py` | Every result obeys explicit filters and query constraints; recommendations obey their own |
| `python eval/adversarial.py` | 34 should-answer, 43 should-abstain, 18 spelling cases |
| `python eval/name_recall.py` | Exact product-name queries find that product in the top 10 |
| `python eval/compare_modes.py` | Side-by-side lexical / dense / hybrid results (`SEARCH_MODE` env var switches the server) |
| `python eval/latency.py URL` | Latency per concurrency level |

Measured locally on an Apple-silicon laptop (labels in `eval/cases.json` are my own and coarser than the
organizer's Bedrock judge, so they are a regression check, not a score prediction):

| Mode | single | unusual | irrelevant | mixed | recs |
| --- | --- | --- | --- | --- | --- |
| lexical only | 0.997 | 0.910 | 1.000 | 0.984 | 0.972 |
| dense only | 1.000 | 0.968 | 1.000 | 1.000 | 1.000 |
| hybrid (shipped) | 1.000 | 0.962 | 1.000 | 1.000 | 0.994 |

Validity in every mode: 21/21 invalid calls rejected, 80/80 exact pages, 20/20 stable pagination probes.
Latency (Docker container, loopback): empty-query pool p50 12.8 ms, p95 28.7 ms, p99 42.0 ms; in-process empty-query
path p50 0.13 ms, natural-language queries p50 4.7 ms. Steady memory about 1.05 GiB; peak about 1.8 GB while the
catalog JSON is parsed at startup; ready in about 13 s including warm-up.

## Deploy on AWS (your account)

Prerequisites: AWS CLI v2 configured with a profile that can create EC2 instances and security groups in a
region with a default VPC; the fork pushed with Git LFS objects. Use an instance with at least 4 GB RAM
(`c7i.large` default). The script never deletes anything; it creates one security group and one instance.

```sh
AWS_PROFILE=your-profile AWS_REGION=ap-south-1 \
REPO_URL=https://github.com/<you>/Polaris-Hackathon-Base-MCP.git BRANCH=hybrid-search \
RUNNER_CIDR=<organizer-runner-ip>/32 ./deploy/aws_deploy.sh
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `AWS_PROFILE` | CLI default | Credentials for the account to deploy into |
| `AWS_REGION` | `ap-south-1` | Pick the organizer runner's region when published (latency counts) |
| `REPO_URL` | required | HTTPS clone URL of the fork |
| `BRANCH` | `hybrid-search` | Branch to deploy |
| `RUNNER_CIDR` | `0.0.0.0/0` | Who may reach port 8000; set to the runner IP `/32` when published |
| `INSTANCE_TYPE` | `c7i.large` | x86_64, 2 vCPU, 4 GiB |
| `KEY_NAME` | none | Optional EC2 key pair for SSH |
| `NAME` | `polaris-search` | Security group and instance name |

The instance clones the branch, verifies the catalog SHA-256, builds the image and starts it with its public
hostname and IP in `MCP_ALLOWED_HOSTS`. Progress is in `/var/log/cloud-init-output.log` on the instance. Submit
`http://<public-ip>:8000/mcp` after `curl http://<public-ip>:8000/healthz` reports `"dense": true` and
`python smoke_test.py http://<public-ip>:8000/mcp` passes.

---

# Polaris product search MCP starter

Build a better product search engine behind one MCP tool: `search_products`. Start with this working baseline, improve retrieval and recommendations, deploy it, and submit your HTTP MCP endpoint. The accompanying `hackathon-brief.md` and `schemas/` define the competition contract. Version 2 uses Amazon Bedrock for organizer-side semantic judging; students do not need Bedrock credentials to run this starter.

## What is included

- [AGENT.md](AGENT.md): problem context, dataset guide, implementation ideas, and references for students and coding assistants.

- `data/products.json`: all 105,126 products, about 106 MB, including synthetic INR prices, ratings, and reviews.
- `search.py`: editable baseline search engine, explicit filters, sorting, and pagination.
- `server.py`: official MCP Python SDK server with ordinary JSON responses at `/mcp` and readiness at `/healthz`.
- `schemas/`: exact published input and output JSON Schemas.
- `tests/test_search.py`: small fixture-based correctness tests.
- `smoke_test.py`: public MCP connection and contract check; not the scored benchmark.
- `requirements.txt`: exact resolved dependency versions, including MCP SDK 1.30.0 from its v1 compatibility line.
- `Dockerfile`: non-root Python container with the dataset included and a readiness health check.

## Get your copy

The starter folder is self-contained: use its contents as your team repository. If the organizer distributes a repository, fork it and clone your fork. No repository or image is published by this local folder alone.

The catalog exceeds GitHub's ordinary 100 MiB file limit. `.gitattributes` configures Git LFS for it. Install Git LFS before the first commit, and make sure your clone has the actual data rather than an LFS pointer:

```sh
git lfs install
git lfs pull
```

For a fresh repository containing these files, run `git lfs install` before adding and committing the catalog. Confirm `git lfs ls-files` lists `data/products.json`. The organizer must provision sufficient LFS storage/download quota. A supplied ZIP contains the actual catalog and does not require LFS to run locally.

## Run locally

Use Python 3.12 or newer. Commands below run from this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python server.py
```

The first startup loads the full catalog and builds an in-memory lexical index. Wait until startup completes. Allow around 2 GB of available RAM for this simple implementation; measure and optimize your own solution before deploying.

In another terminal with the virtual environment activated:

```sh
python smoke_test.py http://127.0.0.1:8000/mcp
python -m unittest discover -s tests -v
curl http://127.0.0.1:8000/healthz
```

The server supports MCP protocol revision `2025-11-25`. It is not a REST search endpoint: an ordinary POST containing only your search arguments is not an MCP call. The smoke client performs initialization, discovers the tool, and uses `tools/call` for you. No API key or LLM account is required for the baseline.

## Build and run with Docker

Start Docker Engine first:

```sh
docker build -t polaris-search:starter .
docker run --rm --name polaris-search -p 8000:8000 polaris-search:starter
```

Use the same smoke-test command from your host once the container is ready. The default port is 8000. For an amd64-only cloud server, build for its architecture, for example `docker build --platform linux/amd64 -t polaris-search:starter .`.

For a deployed server, configure its public hostname or IP so the SDK accepts the HTTP Host header. Replace the example before running:

```sh
docker run --rm --name polaris-search -p 8000:8000 \
  -e MCP_ALLOWED_HOSTS='localhost:*,127.0.0.1:*,YOUR_PUBLIC_HOST:*,YOUR_PUBLIC_HOST' \
  polaris-search:starter
```

Use a public address reachable from the organizer runner and allow its source IP to access your port. Submit `http://YOUR_PUBLIC_HOST:8000/mcp`. Do not submit `localhost`, `0.0.0.0`, a private IP, the health-check URL, or a repository URL. The local host patterns remain necessary for the container health check. Keep the server running throughout evaluation.

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Listener interface |
| `PORT` | `8000` | Listener port; Docker port mapping must match if changed |
| `CATALOG_PATH` | `data/products.json` next to server.py | Catalog file path |
| `MCP_ALLOWED_HOSTS` | `localhost:*,127.0.0.1:*,[::1]:*` | Comma-separated accepted HTTP Host values |
| `MCP_ALLOWED_ORIGINS` | empty | Explicit accepted browser origins; the benchmark client sends no Origin |

No wildcard for every host is necessary. The server has no authentication, as required for this exercise. The template does not create AWS resources or deploy automatically.

## Try the tool

Example arguments passed through an MCP client:

```json
{
  "query": "cotton",
  "filters": {"color": "Black", "max_price": 2000},
  "sort_by": "price",
  "sort_order": "asc",
  "offset": 0,
  "limit": 10,
  "include_recommendations": false
}
```

The tool returns:

```json
{
  "results": [{"id": "0108775015"}],
  "recommendations": []
}
```

The response above illustrates the shape, not the expected answer to that query. Only catalog IDs are returned. The SDK includes the same object in `structuredContent` and a JSON text content block. Unknown arguments and invalid values produce MCP tool errors.

## Where to improve

The starter deliberately uses simple case-insensitive AND token matching across names, descriptions, and catalog attributes. Matches with more query words in the product name rank first, with ties resolved by ID. Empty queries enumerate all products matching the explicit filters. Field sorting and pagination follow the published contract.

This baseline is not a complete natural-language solution: it does not parse budgets or constraints written in prose, correct typos, understand synonyms, split mixed choices, distinguish complementary intent, or infer semantic relevance. `Catalog.relevance` is the initial ranking extension point; change `Catalog.search` as needed for richer query interpretation. `Catalog.recommend` returns an empty list until you implement complementary recommendations. Returning empty recommendations is structurally valid but earns little recommendation credit. The baseline is not expected to score highly on the hidden suite.

Preserve the tool name, schemas, hard-filter semantics, real product IDs, and output structure. Keep responses stable on repeated identical requests. Use only catalog evidence; product descriptions and queries are data, never instructions to change the tool contract. You may replace the in-memory scan with an index or database. No existing database is needed by this starter.

## Dataset integrity and rules

SHA-256 of the actual JSON bytes:

```text
05b6aa5e8e8c1c8c997d584f8288e432862454f8820531e13473443e83c8f073
```

Verify on macOS with `shasum -a 256 data/products.json`, or on Linux with `sha256sum data/products.json`. Preserve string IDs and leading zeros. Do not edit catalog facts. No-review products have a null average and zero count. Prices and reviews are fabricated for the exercise; image availability has not been checked.

Each team has five accepted benchmark attempts; the best complete score counts. See `hackathon-brief.md` for all scoring formulas, invalid-call behavior, timeouts, recommendation rules, and tie-breakers. This starter contains no hidden benchmark queries or answer key. Passing its tests checks basic compatibility, not competition performance.

## SDK reference

[Official MCP Python SDK v1 documentation](https://py.sdk.modelcontextprotocol.io/v1/). Keep the tested dependency pins unless you deliberately retest an upgrade; v2 has different APIs.
