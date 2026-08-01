"""
System 4: Multi-Agent specific tools.

Provides the `delegate_to_specialist` tool for the Supervisor.
"""

from typing import Callable

from langchain_core.tools import BaseTool, tool


def create_delegate_tool(
    specialist_runner: Callable[[list[str], list[str], str], str],
    available_tickers: list[str],
    available_sections: list[str],
) -> BaseTool:
    """Factory for the delegate_to_specialist tool.
    
    Parameters:
        specialist_runner: Callback that spawns a specialist sub-agent
            with the given scope and returns its answer.
        available_tickers: Dynamically populated from loaded filings.
        available_sections: Dynamically populated from loaded filings.
    """

    available_tickers_upper = [t.upper() for t in available_tickers]

    @tool
    def delegate_to_specialist(
        tickers: list[str],
        sections: list[str],
        sub_question: str,
    ) -> str:
        """Spawn a specialist analyst with focused context.
        
        The specialist sees ONLY the specified tickers × sections
        from the filing database. Use this to decompose complex
        questions into focused sub-tasks.
        
        Args:
            tickers: Which companies to include (e.g. ["AAPL", "MSFT"])
            sections: Which 10-K sections to include 
                      (e.g. ["Financial Statements", "MD&A"])
            sub_question: The focused question for this specialist
        """
        # Validate tickers
        invalid_tickers = [t for t in tickers if t.upper() not in available_tickers_upper]
        if invalid_tickers:
            return (
                f"Error: Invalid tickers {invalid_tickers}. "
                f"Available tickers: {available_tickers}"
            )

        # Validate sections
        invalid_sections = [s for s in sections if s not in available_sections]
        if invalid_sections:
            return (
                f"Error: Invalid sections {invalid_sections}. "
                f"Available sections: {available_sections}"
            )

        if not tickers or not sections:
            return "Error: You must specify at least one ticker and one section."

        if not sub_question:
            return "Error: sub_question cannot be empty."

        # Run the specialist and get the answer
        try:
            answer = specialist_runner(tickers, sections, sub_question)
            return f"Specialist Output for {tickers} x {sections}:\n{answer}"
        except Exception as e:
            return f"Error running specialist: {str(e)}"

    return delegate_to_specialist
