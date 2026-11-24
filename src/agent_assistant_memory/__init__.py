"""agent-assistant-memory: how an AI assistant maintains memory and sessions."""

from .agent import AssistantAgent
from .compaction import Compactor
from .llm import make_llm
from .retrieval import search_memory
from .sessions import ContextWindow, SessionManager
from .store import MemoryStore

__all__ = [
    "AssistantAgent",
    "Compactor",
    "ContextWindow",
    "MemoryStore",
    "SessionManager",
    "make_llm",
    "search_memory",
]

__version__ = "0.1.0"
