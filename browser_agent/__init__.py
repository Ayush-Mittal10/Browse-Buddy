"""A browser automation agent.

Give it a task in plain English; it drives a real Chromium through Playwright,
reading each page as numbered interactive elements plus visible text, and
deciding the next action with an LLM.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
