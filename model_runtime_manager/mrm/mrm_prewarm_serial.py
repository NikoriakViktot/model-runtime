#!/usr/bin/env python3
import json
import time
import sys
import urllib.request
import urllib.error
import socket
from typing import Dict, List, Tuple

# --- КОНФІГУРАЦІЯ ТАЙМ-АУТІВ ---
# Час очікування на відповідь від POST /ensure (в секундах)
HTTP_TIMEOUT_SEC = 3600

# Час очікування в циклі перевірки статусу (в секундах)
POLLING_TIMEOUT_SEC = 3600

# Пауза між моделями (в секундах), щоб GPU встигла очистити пам'ять
GPU_COOLDOWN_SEC = 10
# -------------------------------

MRM_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8010"
MODELS = sys.argv[2:]


def _http_json(method: str, url: str, payload: Dict | None = None, timeout: int = 30) -> Dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8") if e.fp else ""
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {"raw": raw}
        return {"_http_error": True, "status": e.code, "body": body}
    except (urllib.error.URLError, socket.timeout) as e:
        return {"_http_error": True, "status": 0, "body": {"detail": f"Network/Timeout error: {str(e)}"}}
    except Exception as e:
        return {"_http_error": True, "status": 0, "body": {"detail": str(e)}}


def list_models_from_registry() -> List[str]:
    print(f"🔍 Discovery: Fetching models from {MRM_URL}/models/status ...")
    out = _http_json("GET", f"{MRM_URL}/models/status", timeout=10)
    if isinstance(out, list):
        return [x["base_model"] for x in out if "base_model" in x]
    print(f"⚠️ Warning: Could not list models. Response: {out}")
    return []


def wait_ready(base_model: str, timeout_sec: int) -> Tuple[bool, Dict]:
    """
    Цикл, який довбає статус, поки модель не стане READY або не вийде час.
    """
    deadline = time.time() + timeout_sec
    last = {}
    print(f"⏳ Polling status for '{base_model}' (Timeout: {timeout_sec}s)...")

    start_time = time.time()

    while time.time() < deadline:
        st = _http_json("GET", f"{MRM_URL}/models/status/{base_model}", timeout=10)
        last = st if isinstance(st, dict) else {}

        state = last.get("state", "UNKNOWN")

        if state == "READY":
            elapsed = int(time.time() - start_time)
            print(f"✅ Model '{base_model}' is READY! (Took {elapsed}s)")
            return True, last

        # Логуємо прогрес кожні 10 секунд
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0:
            print(f"   ... still waiting for '{base_model}'. Current state: {state}. Elapsed: {elapsed}s")

        time.sleep(2.0)

    return False, last


def main():
    models = MODELS[:] if MODELS else list_models_from_registry()

    if not models:
        print("❌ No models found to prewarm.")
        sys.exit(1)

    print(f"MRM_URL={MRM_URL}")
    print(f"Models to prewarm ({len(models)}):")
    for m in models:
        print(f"  - {m}")

    results: Dict[str, Dict] = {}

    print("\n=== STARTING PREWARM SEQUENCE (Ensure -> Wait -> Stop) ===")

    for i, bm in enumerate(models):
        # Пауза перед наступною моделлю (окрім першої)
        if i > 0:
            print(f"\n❄️  Cooling down GPU for {GPU_COOLDOWN_SEC}s before next model...")
            time.sleep(GPU_COOLDOWN_SEC)

        print(f"\n---------------------------------------------------")
        print(f"🚀 Processing: {bm}")
        print(f"---------------------------------------------------")

        # 1. ENSURE
        print(f"[1/3] Calling ensure() - This triggers download if needed...")
        ensure_resp = _http_json("POST", f"{MRM_URL}/models/ensure", {"base_model": bm}, timeout=HTTP_TIMEOUT_SEC)

        if ensure_resp.get("_http_error"):
            print(f"⚠️ Ensure request returned error or timeout: {ensure_resp.get('body')}")
            print("      -> Switching to polling mode anyway, just in case download started.")
        else:
            print(f"      Ensure call accepted. MRM says: {ensure_resp.get('state', 'UNKNOWN')}")

        results[bm] = {"ensure": ensure_resp}

        # 2. WAIT
        print(f"[2/3] Waiting for READY state...")
        ok, st = wait_ready(bm, timeout_sec=POLLING_TIMEOUT_SEC)
        results[bm]["status_after_ensure"] = st

        if not ok:
            print(f"❌ TIMEOUT: Model {bm} did not become READY within {POLLING_TIMEOUT_SEC}s.")
            print(f"   Last status: {st}")
            continue

        print(f"      Container info: {st.get('container')} | GPU: {st.get('gpu')}")

        # 3. STOP
        print(f"[3/3] Stopping model to free GPU...")
        stop_resp = _http_json("POST", f"{MRM_URL}/models/stop", {"base_model": bm}, timeout=60)
        results[bm]["stop"] = stop_resp

        if stop_resp.get("_http_error"):
            print(f"⚠️ Error stopping model: {stop_resp.get('body')}")
        else:
            print("✅ Model STOPPED successfully.")

    # --- ЗВІТ ---
    print("\n================ SUMMARY ================")
    success_count = 0
    for bm in models:
        st = results.get(bm, {}).get("status_after_ensure", {})
        state = st.get("state") if isinstance(st, dict) else "UNKNOWN"

        if state == "READY":
            success_count += 1
            print(f"✅ PASS: {bm}")
        else:
            print(f"❌ FAIL: {bm} (Final state: {state})")

    print(f"-----------------------------------------")
    print(f"Total Success: {success_count} / {len(models)}")

    if success_count != len(models):
        sys.exit(1)


if __name__ == "__main__":
    main()