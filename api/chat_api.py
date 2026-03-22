import os
import logging
import requests
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .neo4j_rag import Neo4jRAG
from .chat_rag import extract_triggers
from .context_builder import build_context
from .adapter_materializer import _materialize_adapter_by_run_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dispatcher.chat")

router = APIRouter()

CONTROL_PLANE_URL = os.getenv("CONTROL_PLANE_URL", "http://control_plane:8004")
MRM_URL = os.getenv("MRM_URL", "http://model_runtime_manager:8010")
VLLM_API_BASE_FALLBACK = os.getenv("VLLM_API_BASE", "http://vllm_qwen_1p8b_chat:8000")


class ChatReq(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    system_prompt: Optional[str] = None
    use_rag: bool = False
    rag_k: int = 3
    route: str = "direct"

def resolve_latest_run_for_epitaph(epitaph_id: str) -> str:
    r = requests.get(
        f"{CONTROL_PLANE_URL}/runs-filter",
        params={
            "epitaph_id": epitaph_id,
            "run_type": "train.qlora.v1",
            "status": "DONE",
            "limit": 1,
        },
        timeout=5,
    )
    r.raise_for_status()
    runs = r.json()
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=f"No trained adapter found for epitaph {epitaph_id}",
        )
    return runs[0]["id"]

def _ensure_system_prompt(messages: List[Dict], target_persona: str) -> None:
    if any(m.get("role") == "system" for m in messages):
        return
    messages.insert(0, {
        "role": "system",
        "content": f"You are {target_persona}. Reply briefly and naturally. No fluff."
    })


def _apply_system_prompt(req: ChatReq, target_persona: str) -> None:
    if any(m.get("role") == "system" for m in req.messages):
        return
    if req.system_prompt:
        req.messages.insert(0, {"role": "system", "content": req.system_prompt})
        return
    _ensure_system_prompt(req.messages, target_persona)


def _normalize_keywords(keywords: Any) -> list[str]:
    if not keywords:
        return []
    out: list[str] = []
    for k in keywords:
        if isinstance(k, str):
            s = k.strip()
            if s:
                out.append(s)
        elif isinstance(k, dict):
            for key in ("name", "text", "value", "keyword"):
                v = k.get(key)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
                    break
    seen = set()
    uniq = []
    for x in out:
        lx = x.lower()
        if lx not in seen:
            seen.add(lx)
            uniq.append(x)
    return uniq


def _inject_rag_context(
    messages: List[Dict],
    model_alias_for_extraction: str,
    target_name: str,
    k: int,
    dataset_run_id: Optional[str] = None,
):
    last_user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)
    if not last_user_msg:
        return

    user_text = last_user_msg["content"]

    try:
        keywords = extract_triggers(user_text, model_name=f"hosted_vllm/{model_alias_for_extraction}")
    except Exception as e:
        logger.error(f"RAG Extraction Error: {e}")
        return

    keywords = _normalize_keywords(keywords)
    if not keywords:
        return

    rag = Neo4jRAG()
    try:
        snippets = rag.retrieve(
            epitaph_id=target_name,
            keywords=keywords,
            dataset_run_id=dataset_run_id,
            limit=k,
        )
    except Exception as e:
        logger.error(f"Neo4j Connection/Query Error: {e}")
        snippets = []
    finally:
        rag.close()

    if not snippets:
        return

    context_str = build_context(snippets)

    system_prompt_addition = (
        f"\n\n### RELEVANT MEMORY CONTEXT ###\n"
        f"Use the following past memories/facts to inform your answer. "
        f"If the context is irrelevant, ignore it.\n\n"
        f"{context_str}\n"
        f"#################################\n"
    )

    system_msg = next((m for m in messages if m["role"] == "system"), None)
    if system_msg:
        system_msg["content"] += system_prompt_addition
    else:
        messages.insert(0, {
            "role": "system",
            "content": f"You are {target_name}. Reply briefly and naturally. No fluff.\n{system_prompt_addition}"
        })


def _cp_get_run(run_id: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{CONTROL_PLANE_URL}/runs/{run_id}", timeout=5)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"CP Error: {e}")
        return None


def _mrm_ensure(base_model: str) -> Dict[str, Any]:
    try:
        r = requests.post(f"{MRM_URL}/models/ensure", json={"base_model": base_model}, timeout=600)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MRM Ensure Failed: {e}")


def _mrm_touch(base_model: str) -> None:
    try:
        requests.post(f"{MRM_URL}/models/touch", json={"base_model": base_model}, timeout=5)
    except:
        pass


def _call_vllm_direct(api_base: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    base = api_base.rstrip("/")
    url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"vLLM Error: {r.text}")
    return r.json()


def _resolve_run(req: ChatReq) -> Dict[str, Any]:
    dataset_run_id = None

    if req.model.startswith("base:"):
        base_model_raw = req.model.split("base:", 1)[1]
        target_persona = "Assistant"
        is_lora = False
        run_id = None
    else:
        run_id = req.model
        is_lora = True
        run = _cp_get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found in DB")

        contract = run.get("contract", {}).get("payload", {})
        base_model_raw = contract.get("base_model")
        target_persona = contract.get("target_slug") or contract.get("target_name") or "Unknown"
        dataset_run_id = contract.get("parent_run_id")

        if not base_model_raw:
            raise HTTPException(status_code=400, detail="Corrupt run contract: missing base_model")

    m = _mrm_ensure(base_model_raw)
    _mrm_touch(base_model_raw)

    return {
        "dataset_run_id": dataset_run_id,
        "target_persona": target_persona,
        "is_lora": is_lora,
        "run_id": run_id,
        "model_alias": m["model_alias"],
        "vllm_api_base": m.get("api_base") or VLLM_API_BASE_FALLBACK,
    }


def _vllm_load_adapter(api_base: str, run_id: str, lora_path: str) -> bool:
    base = api_base.rstrip("/")
    url = f"{base}/load_lora_adapter" if base.endswith("/v1") else f"{base}/v1/load_lora_adapter"

    payload = {"lora_name": run_id, "lora_path": lora_path}

    try:
        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            logger.info(f"Successfully loaded LoRA adapter: {run_id}")
            return True

        if response.status_code == 400:
            try:
                body = response.json()
            except ValueError:
                body = {}

            msg = ""
            if isinstance(body.get("error"), dict):
                msg = body["error"].get("message", "")
            elif isinstance(body.get("message"), str):
                msg = body["message"]
            else:
                msg = response.text or ""

            s = msg.lower()

            # <-- ОЦЕ КЛЮЧОВЕ: “already been loaded” теж вважаємо OK
            if ("already" in s and "loaded" in s) or ("already" in s and "exists" in s):
                logger.info(f"LoRA adapter {run_id} already loaded. Proceeding.")
                return True

            logger.error(f"Failed to load LoRA adapter {run_id}: {response.text}")
            return False

        logger.error(f"Unexpected error loading LoRA {run_id}: Status {response.status_code}, {response.text}")
        return False

    except Exception as e:
        logger.error(f"Exception during vLLM load adapter request: {e}")
        return False


@router.post("/chat")
def chat(req: ChatReq):
    if req.route != "direct":
        raise HTTPException(status_code=400, detail="Only route=direct (vLLM) is supported.")

    info = _resolve_run(req)
    _apply_system_prompt(req, info["target_persona"])

    # RAG логіка (відновлено з твого коду, щоб не загубилась)
    if req.use_rag:
        _inject_rag_context(
            req.messages,
            model_alias_for_extraction=info["model_alias"],
            target_name=info["target_persona"],
            k=req.rag_k,
            dataset_run_id=info["dataset_run_id"]
        )

    # Логіка для LoRA
    if info["is_lora"]:
        run_id = info["run_id"]

        # 1. Матеріалізація (скачування файлів на спільний диск)
        try:
            _materialize_adapter_by_run_id(run_id)
        except Exception as e:
            logger.error(f"Materialization failed for {run_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to materialize adapter files.")

        # 2. Динамічне завантаження в vLLM
        # ВАЖЛИВО: Переконайся, що volume mapping у vLLM відповідає цьому шляху!
        container_lora_path = f"/lora/{run_id}"

        is_loaded = _vllm_load_adapter(info["vllm_api_base"], run_id, container_lora_path)

        if not is_loaded:
            # Якщо не вдалося завантажити (і це не помилка "already loaded"), перериваємо
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load LoRA adapter '{run_id}' into inference engine."
            )

        # 3. Target Model = назва адаптера
        target_model = run_id
    else:
        target_model = info["model_alias"]

    payload: Dict[str, Any] = {
        "model": target_model,
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "stream": False,
        # Додаємо stream=False явно, або беремо з req якщо планується стрімінг
    }

    return _call_vllm_direct(info["vllm_api_base"], payload)