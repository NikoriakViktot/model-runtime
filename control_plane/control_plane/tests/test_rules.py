import pytest
from control_plane.domain.states import RunState
from control_plane.domain.events import EventType
from control_plane.domain.rules import transition, next_action


def test_happy_path_transitions():
    """Перевірка ідеального сценарію переходу станів"""
    # 1. Створено -> Прийнято контракт
    state = transition(RunState.CREATED, EventType.CONTRACT_ACCEPTED)
    assert state == RunState.DATASET_RUNNING

    # 2. Датасет готовий -> Тренування
    # Зверніть увагу: за вашою логікою DATASET_RUNNING + BUILT -> DATASET_READY
    state = transition(RunState.DATASET_RUNNING, EventType.DATASET_BUILT)
    assert state == RunState.DATASET_READY


def test_next_action_dispatch():
    """Перевірка, яку дію система каже виконати (Side Effects)"""
    # Коли датасет збілдився, система має сказати "TRAIN"
    action = next_action(RunState.DATASET_READY, EventType.DATASET_BUILT)
    assert action == "TRAIN"


def test_failure_handling():
    """Якщо сталась помилка, стан має стати FAILED"""
    state = transition(RunState.DATASET_RUNNING, EventType.FAILED)
    assert state == RunState.FAILED