
from .states import RunState
from .events import EventType


def transition(current_state: RunState, event: EventType) -> RunState:
    """
    Pure function defining state transitions based on events.
    """
    if event == EventType.FAILED:
        return RunState.FAILED

    if current_state == RunState.CREATED and event == EventType.CONTRACT_ACCEPTED:
        # Immediate transition before dispatching build
        return RunState.DATASET_RUNNING

    if current_state == RunState.DATASET_RUNNING and event == EventType.DATASET_BUILT:
        return RunState.DATASET_READY

    if current_state == RunState.TRAIN_RUNNING and event == EventType.TRAIN_COMPLETED:
        return RunState.DONE

    if current_state == RunState.EVAL_RUNNING and event == EventType.EVAL_COMPLETED:
        return RunState.DONE

    # No transition defined, stay same
    return current_state


def next_action(state: RunState, event: EventType) -> str | None:
    """
    Pure function determining the next side-effect (Action) to dispatch.
    Returns an Action string or None.
    """
    if state == RunState.FAILED:
        return None

    # Trigger Dataset Build immediately after creation
    if state == RunState.DATASET_RUNNING and event == EventType.CONTRACT_ACCEPTED:
        return "DATASET_BUILD"

    # Trigger Training after Dataset is ready
    if state == RunState.DATASET_READY and event == EventType.DATASET_BUILT:
        return "TRAIN"

    # Trigger Eval after Training is ready
    # if state == RunState.TRAIN_READY and event == EventType.TRAIN_COMPLETED:
    #     return "EVAL"

    return None