from fastapi import APIRouter
from . import contracts, events, runs, prompts

router = APIRouter()

router.include_router(contracts.router, tags=["Contracts"])
router.include_router(events.router, tags=["Events"])
router.include_router(runs.router, tags=["Runs"])
router.include_router(prompts.router, tags=["Prompts"])
