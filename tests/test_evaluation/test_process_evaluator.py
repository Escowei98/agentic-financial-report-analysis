import pytest

from src.evaluation.process_evaluator import (
    calculate_cost_per_correct_answer,
    evaluate_tool_selection,
)


def test_evaluate_tool_selection_perfect():
    expected = ["search_section", "calculate"]
    actual = ["search_section", "calculate", "search_section"]  # Duplicate is fine
    result = evaluate_tool_selection(expected, actual)

    assert result.tool_precision == 1.0
    assert result.tool_recall == 1.0
    assert result.tool_f1 == 1.0

def test_evaluate_tool_selection_missing():
    expected = ["search_section", "calculate"]
    actual = ["search_section"]
    result = evaluate_tool_selection(expected, actual)

    # 1 correct, 0 false pos, 1 false neg
    assert result.tool_precision == 1.0
    assert result.tool_recall == 0.5
    assert result.tool_f1 == pytest.approx(0.666, 0.01)

def test_evaluate_tool_selection_extra():
    expected = ["search_section"]
    actual = ["search_section", "calculate"]
    result = evaluate_tool_selection(expected, actual)

    # 1 correct, 1 false pos, 0 false neg
    assert result.tool_precision == 0.5
    assert result.tool_recall == 1.0
    assert result.tool_f1 == pytest.approx(0.666, 0.01)

def test_evaluate_tool_selection_none_expected():
    expected = []
    actual = []
    result = evaluate_tool_selection(expected, actual)
    assert result.tool_precision == 1.0
    assert result.tool_f1 == 1.0

    actual_extra = ["search_section"]
    result2 = evaluate_tool_selection(expected, actual_extra)
    assert result2.tool_precision == 0.0
    assert result2.tool_f1 == 0.0

def test_calculate_cost_per_correct_answer():
    cost = 1.0  # $1.0 total
    scores = [1.0, 0.9, 0.5, 0.0]

    # threshold 0.8 -> 2 correct
    result = calculate_cost_per_correct_answer(cost, scores, threshold=0.8)
    assert result == 0.5

    # threshold 0.95 -> 1 correct
    result2 = calculate_cost_per_correct_answer(cost, scores, threshold=0.95)
    assert result2 == 1.0

    # threshold 2.0 -> 0 correct
    result3 = calculate_cost_per_correct_answer(cost, scores, threshold=2.0)
    assert result3 == 0.0
