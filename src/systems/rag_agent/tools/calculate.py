"""
Calculate Tool for System 2 (Agent RAG).

Safe mathematical expression evaluator using simpleeval.
Supports basic arithmetic, percentages, and common financial calculations
WITHOUT allowing arbitrary Python code execution.

Addresses Exposé H2: Math hallucination testing — the agent should use
this tool instead of computing numbers in its head.
"""

import logging

from langchain_core.tools import tool
from simpleeval import EvalWithCompoundTypes, InvalidExpression

logger = logging.getLogger(__name__)

# Configure safe evaluator with financial math functions
_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
}


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely. Use for ALL financial calculations.

    NEVER compute numbers mentally — always use this tool for arithmetic.

    Supported operations:
    - Basic: +, -, *, /, ** (power), % (modulo)
    - Parentheses for grouping
    - Functions: abs(), round(), min(), max(), sum()

    Examples:
    - Revenue growth %: '(391000 - 365000) / 365000 * 100'
    - YoY change: '391000 / 365000 - 1'
    - Average: '(391000 + 245000 + 574000) / 3'
    - Rounding: 'round((391000 - 365000) / 365000 * 100, 2)'

    Args:
        expression: Mathematical expression as string. Use ONLY numbers and operators,
                    no variable names. Extract numbers from retrieved data first.
    """
    if not expression or not expression.strip():
        return "Error: Empty expression. Provide a mathematical expression."

    expression = expression.strip()

    try:
        evaluator = EvalWithCompoundTypes(
            functions=_SAFE_FUNCTIONS,
        )
        result = evaluator.eval(expression)

        # Format result
        if isinstance(result, float):
            # Avoid excessive decimal places
            if result == int(result):
                formatted = str(int(result))
            else:
                formatted = f"{result:.6g}"
        else:
            formatted = str(result)

        logger.info("calculate: %r → %s", expression, formatted)
        return formatted

    except InvalidExpression as e:
        logger.warning("calculate: invalid expression %r: %s", expression, e)
        return (
            f"Error: Invalid expression '{expression}'. "
            f"Use only numbers, operators (+, -, *, /, **), "
            f"parentheses, and functions (abs, round, min, max, sum). "
            f"Detail: {e}"
        )
    except ZeroDivisionError:
        logger.warning("calculate: division by zero in %r", expression)
        return "Error: Division by zero."
    except Exception as e:
        logger.warning("calculate: unexpected error for %r: %s", expression, e)
        return f"Error evaluating expression: {e}"
