"""
Streamlit frontend for the Titanic Chat Agent.
"""

import streamlit as st
import requests
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.styles import get_custom_css
from frontend.components import (
    render_header,
    render_chat_message,
    render_visualization,
    render_reasoning_panel,
    render_suggested_questions,
    render_sidebar_stats,
    render_sample_questions,
    render_error_message,
    render_execution_time
)

# ===========================================
# Configuration
# ===========================================

st.set_page_config(
    page_title="Titanic Dataset Explorer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Backend API URL
API_URL = os.getenv("API_URL", "https://titanic-chat-agent-0log.onrender.com")


# ===========================================
# Session State Initialization
# ===========================================

def init_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "dataset_stats" not in st.session_state:
        st.session_state.dataset_stats = None
    
    if "show_reasoning" not in st.session_state:
        st.session_state.show_reasoning = False
    
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


# ===========================================
# API Interaction Functions
# ===========================================

def fetch_dataset_stats() -> Optional[Dict[str, Any]]:
    """Fetch dataset statistics from the API."""
    try:
        response = requests.get(f"{API_URL}/api/dataset/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        st.sidebar.error(f"Failed to fetch stats: {e}")
    return None


def send_chat_message(message: str, include_reasoning: bool = False) -> Optional[Dict[str, Any]]:
    """Send a chat message to the API and get the response."""
    try:
        payload = {
            "message": message,
            "session_id": st.session_state.session_id,
            "include_reasoning": include_reasoning
        }
        
        response = requests.post(
            f"{API_URL}/api/chat",
            json=payload,
            timeout=120  # Longer timeout for complex queries
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.RequestException as e:
        st.error(f"Connection error: {e}")
        return None


def clear_chat_history():
    """Clear the chat history."""
    try:
        requests.delete(
            f"{API_URL}/api/session/{st.session_state.session_id}",
            timeout=10
        )
    except:
        pass
    
    st.session_state.chat_history = []
    st.session_state.session_id = str(uuid.uuid4())


# ===========================================
# Main Application
# ===========================================

def main():
    """Main application entry point."""
    init_session_state()
    
    # ===========================================
    # Sidebar
    # ===========================================
    
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg", use_column_width=True)
        
        # Fetch and display stats
        if st.session_state.dataset_stats is None:
            st.session_state.dataset_stats = fetch_dataset_stats()
        
        if st.session_state.dataset_stats:
            render_sidebar_stats(st.session_state.dataset_stats)
        
        st.markdown("---")
        
        # Sample questions
        sample_question = render_sample_questions()
        if sample_question:
            st.session_state.pending_question = sample_question
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        st.session_state.show_reasoning = st.checkbox(
            "Show agent reasoning",
            value=st.session_state.show_reasoning,
            help="Display the agent's thought process and tool usage"
        )
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            clear_chat_history()
            st.rerun()
        
        st.markdown("---")
        
        # Session info
        st.markdown("### 📋 Session Info")
        st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
        st.caption(f"Messages: {len(st.session_state.chat_history)}")
    
    # ===========================================
    # Main Content
    # ===========================================
    
    render_header()
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for msg in st.session_state.chat_history:
            render_chat_message(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp")
            )
            
            # Display visualizations if present
            if msg.get("visualizations"):
                for viz in msg["visualizations"]:
                    render_visualization(
                        image_base64=viz["image_base64"],
                        title=viz["title"],
                        description=viz["description"]
                    )
            
            # Display reasoning if present
            if msg.get("reasoning") and st.session_state.show_reasoning:
                with st.expander("🧠 View Agent Reasoning"):
                    render_reasoning_panel(msg["reasoning"])
            
            # Display execution time
            if msg.get("execution_time"):
                render_execution_time(msg["execution_time"])
        
        # Display suggested questions from last assistant message
        if st.session_state.chat_history:
            last_msg = st.session_state.chat_history[-1]
            if last_msg["role"] == "assistant" and last_msg.get("suggested_questions"):
                st.markdown("---")
                selected = render_suggested_questions(
                    last_msg["suggested_questions"],
                    key_prefix=f"sugg_{len(st.session_state.chat_history)}"
                )
                if selected:
                    st.session_state.pending_question = selected
    
    # ===========================================
    # Input Area
    # ===========================================
    
    st.markdown("---")
    
    # Handle pending questions (from suggestions or sample questions)
    if st.session_state.pending_question:
        pending = st.session_state.pending_question
        st.session_state.pending_question = None
        process_user_message(pending)
        st.rerun()
    
    # Chat input
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask me about the Titanic dataset...",
            key="user_input",
            placeholder="e.g., What was the survival rate for women in first class?",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 🚀", use_container_width=True)
    
    # Process input
    if send_button and user_input:
        process_user_message(user_input)
        st.rerun()
    
    # Quick action buttons
    st.markdown("### 🚀 Quick Actions")
    
    quick_cols = st.columns(4)
    
    quick_actions = [
        ("📊 Survival Overview", "Give me a comprehensive overview of survival patterns with visualizations"),
        ("👥 Demographics", "Analyze passenger demographics including age and gender distributions"),
        ("💰 Fare Analysis", "Show me how ticket prices affected survival chances"),
        ("🎯 Key Insights", "What are the most interesting and surprising findings in this dataset?")
    ]
    
    for col, (label, query) in zip(quick_cols, quick_actions):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.pending_question = query
                st.rerun()


def process_user_message(message: str):
    """Process a user message and get the agent's response."""
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now()
    })
    
    # Show loading state
    with st.spinner("🤔 Thinking..."):
        # Get response from API
        response = send_chat_message(
            message=message,
            include_reasoning=st.session_state.show_reasoning
        )
    
    if response:
        # Add assistant message to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.get("message", "I couldn't generate a response."),
            "timestamp": datetime.now(),
            "visualizations": response.get("visualizations", []),
            "reasoning": response.get("reasoning"),
            "suggested_questions": response.get("suggested_questions", []),
            "execution_time": response.get("execution_time")
        })
    else:
        # Add error message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "I'm sorry, I encountered an error. Please try again.",
            "timestamp": datetime.now()
        })


if __name__ == "__main__":
    main()