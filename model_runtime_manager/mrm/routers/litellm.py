# mrm/routers/litellm.py
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, JSONResponse
import yaml

router = APIRouter(tags=["litellm"])

@router.get("/litellm/config", response_class=PlainTextResponse)
def litellm_config(request: Request):
    mrm = request.app.state.mrm

    models = []
    for _, spec in mrm.registry.items():
        models.append({
            "model_name": spec.served_model_name,
            "litellm_params": {
                "model": f"hosted_vllm/{spec.served_model_name}",
                "api_base": f"http://{spec.container_name}:{spec.port}/v1",
                "api_key": "none",
            },
        })

    out = {"model_list": models}
    return yaml.safe_dump(out, sort_keys=False)

@router.post("/litellm/materialize")
def litellm_materialize(request: Request):
    mrm = request.app.state.mrm
    return mrm.materialize_litellm_config()

