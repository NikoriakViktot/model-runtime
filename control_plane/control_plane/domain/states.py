#domain/states.py

from enum import Enum

class JobState(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RunState(str, Enum):
    CREATED = "CREATED"
    DATASET_RUNNING = "DATASET_RUNNING"
    DATASET_READY = "DATASET_READY"
    TRAIN_RUNNING = "TRAIN_RUNNING"
    TRAIN_READY = "TRAIN_READY"
    EVAL_RUNNING = "EVAL_RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
