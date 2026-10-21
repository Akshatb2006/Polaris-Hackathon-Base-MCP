#!/usr/bin/env bash
# Pre-submission checks against a deployed server, run from a machine outside AWS:
#
#   ./deploy/verify_endpoint.sh http://<public-ip>:8000
#
# 1. /healthz reports the full catalog with the dense index loaded
# 2. standalone GET /mcp is refused with 405, as the brief requires
# 3. the starter smoke test: MCP 2025-11-25 handshake, exact schemas, structured results, invalid calls
# 4. the local organizer-style benchmark (validity, labelled cases, latency incl. network)
# None of this uses a competition benchmark attempt.
set -uo pipefail

BASE="${1:?usage: verify_endpoint.sh http://<host>:8000}"
BASE="${BASE%/}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3
fail=0
step() { printf '%-34s' "$1"; }

step "healthz"
health=$(curl -fsS -m 10 "$BASE/healthz") || health=""
if [[ "$health" == *'"products":105126'* && "$health" == *'"dense":true'* ]]; then echo "PASS $health"
else echo "FAIL ${health:-unreachable (still building? check /var/log/cloud-init-output.log)}"; fail=1; fi

step "GET /mcp returns 405"
code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H 'accept: text/event-stream' "$BASE/mcp")
[[ "$code" == "405" ]] && echo "PASS" || { echo "FAIL (got $code)"; fail=1; }

step "MCP smoke test"
if out=$(cd "$ROOT" && "$PY" smoke_test.py "$BASE/mcp" 2>&1); then echo "PASS"; else echo "FAIL"; echo "$out" | tail -5; fail=1; fi

if [[ $fail -eq 0 && "${SKIP_BENCHMARK:-}" != "1" ]]; then
  echo "Benchmark (includes network latency):"
  (cd "$ROOT" && "$PY" eval/benchmark.py "$BASE/mcp" 2>&1 | grep -E "^ ! |invalid rejected|mean|p50|latency max") || fail=1
fi

[[ $fail -eq 0 ]] && echo "All pre-submission checks passed. Endpoint to submit: $BASE/mcp" \
  || echo "Checks failed: do not submit yet."
exit $fail
