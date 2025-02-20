"""Define the configurable parameters for the Gmail assistant agent."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, List

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the Gmail assistant agent."""

    # Ollama model configuration
    model_name: str = "llama3"
    
    # Gmail search parameters
    days_back: int = 3
    max_emails: int = 10
    search_query: str = "in:inbox -label:replied -from:me"
    
    # Priority thresholds
    high_priority_threshold: int = 8
    low_priority_threshold: int = 3
    
    # Response generation parameters
    response_style: str = "professional"  # Options: professional, casual, formal
    max_response_length: int = 500
    include_signature: bool = True
    signature: str = "Best regards,\n[Your Name]"
    
    # Auto-reply configuration
    enable_auto_replies: bool = False
    auto_reply_threshold: int = 2  # Priority level below which to auto-reply
    
    # Filter options
    ignore_newsletters: bool = True
    priority_senders: List[str] = None  # List of email addresses to prioritize
    
    def __post_init__(self):
        if self.priority_senders is None:
            self.priority_senders = []

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        configurable = (config.get("configurable") or {}) if config else {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})