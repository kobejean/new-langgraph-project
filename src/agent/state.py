"""Define the state structures for the Gmail assistant agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class EmailInfo:
    """Information about a single email."""
    
    thread_id: str
    message_id: str
    subject: str
    sender: str
    date: str
    body: str
    priority_score: Optional[int] = None
    ask_user: bool = True
    reasoning: str = ""
    proposed_response_options: List[str] = field(default_factory=list)
    selected_option: Optional[str] = None
    generated_response: Optional[str] = None
    draft_id: Optional[str] = None


@dataclass
class State:
    """Defines the state for the Gmail assistant agent.
    
    This tracks the full workflow from email retrieval through prioritization,
    response generation, and draft creation.
    """
    
    # Email processing state
    emails: List[EmailInfo] = field(default_factory=list)
    current_email_index: int = 0
    
    # Workflow control
    next: str = "generate_response"  # Controls the next state in the workflow
    
    tools: List[Any] = field(default_factory=list)

    # Processing status
    emails_retrieved: bool = False
    emails_prioritized: bool = False
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # User decisions
    user_decisions: Dict[str, str] = field(default_factory=dict)
    
    # Additional fields for LangGraph interaction
    human_feedback_needed: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize any additional state after dataclass initialization."""
        # Make sure user_decisions is initialized
        if self.user_decisions is None:
            self.user_decisions = {}
    
    def get_current_email(self) -> Optional[EmailInfo]:
        """Get the current email being processed."""
        if not self.emails or self.current_email_index >= len(self.emails):
            return None
        return self.emails[self.current_email_index]
    
    def move_to_next_email(self) -> None:
        """Move to the next email in the queue."""
        self.current_email_index += 1
        
    def add_error(self, step: str, error_message: str, details: Any = None) -> None:
        """Track an error that occurred during processing."""
        self.errors.append({
            "step": step,
            "error_message": error_message,
            "details": details,
            "timestamp": str(datetime.now())
        })
        
    def get_high_priority_emails(self, threshold: int = 7) -> List[EmailInfo]:
        """Get emails above the priority threshold."""
        return [email for email in self.emails if email.priority_score and email.priority_score >= threshold]
    
    def get_emails_requiring_decisions(self) -> List[EmailInfo]:
        """Get emails that require user decisions."""
        return [email for email in self.emails if email.ask_user]
        
    def __setattr__(self, name, value):
        """Allow setting arbitrary attributes on the state object."""
        # This makes the state object more flexible for LangGraph integration
        object.__setattr__(self, name, value)
        
    def to_dict(self):
        """Convert state to a serializable dictionary, excluding non-serializable fields."""
        return {
            "emails": [email.__dict__ for email in self.emails],
            "current_email_index": self.current_email_index,
            "next": self.next,
            "errors": self.errors,
            "user_decisions": self.user_decisions,
            "human_feedback_needed": self.human_feedback_needed
        }