"""
Conversation memory management for the chat agent.
"""

from typing import Dict, List, Optional
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema  import HumanMessage, AIMessage, BaseMessage
import logging
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Data structure for a chat session."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    memory: ConversationBufferWindowMemory = field(default=None)
    message_count: int = 0
    
    def __post_init__(self):
        if self.memory is None:
            self.memory = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 exchanges
                return_messages=True,
                memory_key="chat_history"
            )


class ConversationMemoryManager:
    """
    Manages conversation memory across multiple sessions.
    Thread-safe implementation for concurrent access.
    """
    
    _instance: Optional['ConversationMemoryManager'] = None
    _lock = Lock()
    
    def __new__(cls) -> 'ConversationMemoryManager':
        """Singleton pattern for memory manager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._sessions: Dict[str, SessionData] = {}
                cls._instance._sessions_lock = Lock()
        return cls._instance
    
    def get_or_create_session(self, session_id: str) -> SessionData:
        """Get existing session or create a new one."""
        with self._sessions_lock:
            if session_id not in self._sessions:
                logger.info(f"Creating new session: {session_id}")
                self._sessions[session_id] = SessionData(session_id=session_id)
            return self._sessions[session_id]
    
    def get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get memory for a specific session."""
        session = self.get_or_create_session(session_id)
        session.last_activity = datetime.utcnow()
        return session.memory
    
    def add_messages(
        self, 
        session_id: str, 
        human_message: str, 
        ai_message: str
    ):
        """Add a message pair to session memory."""
        session = self.get_or_create_session(session_id)
        session.memory.save_context(
            {"input": human_message},
            {"output": ai_message}
        )
        session.message_count += 1
        session.last_activity = datetime.utcnow()
        logger.debug(f"Added messages to session {session_id}. Total: {session.message_count}")
    
    def get_chat_history(self, session_id: str) -> List[BaseMessage]:
        """Get chat history for a session."""
        session = self.get_or_create_session(session_id)
        return session.memory.chat_memory.messages
    
    def get_chat_history_as_text(self, session_id: str) -> str:
        """Get chat history as formatted text."""
        messages = self.get_chat_history(session_id)
        history_parts = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(history_parts)
    
    def clear_session(self, session_id: str):
        """Clear memory for a specific session."""
        with self._sessions_lock:
            if session_id in self._sessions:
                self._sessions[session_id].memory.clear()
                self._sessions[session_id].message_count = 0
                logger.info(f"Cleared session: {session_id}")
    
    def delete_session(self, session_id: str):
        """Delete a session entirely."""
        with self._sessions_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
    
    def get_session_info(self, session_id: str) -> dict:
        """Get information about a session."""
        session = self.get_or_create_session(session_id)
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": session.message_count
        }
    
    def get_all_sessions(self) -> List[dict]:
        """Get information about all sessions."""
        with self._sessions_lock:
            return [self.get_session_info(sid) for sid in self._sessions.keys()]
    
    def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """Remove sessions that have been inactive for too long."""
        cutoff = datetime.utcnow()
        sessions_to_remove = []
        
        with self._sessions_lock:
            for session_id, session in self._sessions.items():
                age = (cutoff - session.last_activity).total_seconds() / 3600
                if age > max_age_hours:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self._sessions[session_id]
                logger.info(f"Cleaned up inactive session: {session_id}")
        
        return len(sessions_to_remove)


# Singleton instance
memory_manager = ConversationMemoryManager()