"""
Unit tests for the calculate tool.
"""

import pytest

from src.common.tools.calculate import calculate


class TestCalculate:
    """Tests for safe math evaluator."""

    def test_basic_addition(self):
        result = calculate.invoke({"expression": "1 + 2"})
        assert result == "3"

    def test_basic_subtraction(self):
        result = calculate.invoke({"expression": "100 - 37"})
        assert result == "63"

    def test_multiplication(self):
        result = calculate.invoke({"expression": "150 * 3"})
        assert result == "450"

    def test_division(self):
        result = calculate.invoke({"expression": "100 / 4"})
        assert result == "25"

    def test_float_division(self):
        result = calculate.invoke({"expression": "10 / 3"})
        assert float(result) == pytest.approx(3.33333, rel=1e-3)

    def test_percentage_calculation(self):
        """YoY growth: (new - old) / old * 100"""
        result = calculate.invoke({"expression": "(391000 - 365000) / 365000 * 100"})
        assert float(result) == pytest.approx(7.1233, rel=1e-3)

    def test_parentheses(self):
        result = calculate.invoke({"expression": "(10 + 20) * 3"})
        assert result == "90"

    def test_power(self):
        result = calculate.invoke({"expression": "2 ** 10"})
        assert result == "1024"

    def test_abs_function(self):
        result = calculate.invoke({"expression": "abs(-42)"})
        assert result == "42"

    def test_round_function(self):
        result = calculate.invoke({"expression": "round(3.14159, 2)"})
        assert result == "3.14"

    def test_min_function(self):
        result = calculate.invoke({"expression": "min(10, 20, 5)"})
        assert result == "5"

    def test_max_function(self):
        result = calculate.invoke({"expression": "max(10, 20, 5)"})
        assert result == "20"

    def test_division_by_zero(self):
        result = calculate.invoke({"expression": "10 / 0"})
        assert "Error" in result
        assert "zero" in result.lower()

    def test_empty_expression(self):
        result = calculate.invoke({"expression": ""})
        assert "Error" in result

    def test_invalid_expression(self):
        result = calculate.invoke({"expression": "import os"})
        assert "Error" in result

    def test_no_variable_access(self):
        """Should not allow access to Python builtins or variables."""
        result = calculate.invoke({"expression": "__import__('os').system('echo hack')"})
        assert "Error" in result

    def test_large_financial_numbers(self):
        """Revenue in millions."""
        result = calculate.invoke({"expression": "391000000000 - 365817000000"})
        assert result == "25183000000"

    def test_average(self):
        result = calculate.invoke({"expression": "(100 + 200 + 300) / 3"})
        assert result == "200"
