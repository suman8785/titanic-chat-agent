"""
Reusable UI components for the Streamlit frontend.
"""

import streamlit as st
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime


def render_header():
    """Render the main header."""
    st.markdown("""
        <div class="main-header">
            <h1>🚢 Titanic Dataset Explorer</h1>
            <p>Ask me anything about the Titanic passengers - I can analyze data, create visualizations, and find insights!</p>
        </div>
    """, unsafe_allow_html=True)


def render_chat_message(role: str, content: str, timestamp: Optional[datetime] = None):
    """Render a chat message."""
    role_class = "user" if role.lower() == "user" else "assistant"
    role_icon = "👤" if role.lower() == "user" else "🤖"
    role_label = "You" if role.lower() == "user" else "Titanic AI"
    
    time_str = ""
    if timestamp:
        time_str = f" • {timestamp.strftime('%H:%M')}"
    
    st.markdown(f"""
        <div class="chat-message {role_class}">
            <div class="role">{role_icon} {role_label}{time_str}</div>
            <div class="content">{content}</div>
        </div>
    """, unsafe_allow_html=True)


def render_visualization(image_base64: str, title: str, description: str):
    """Render a visualization with its metadata."""
    st.markdown(f"""
        <div class="viz-container">
            <h4>📊 {title}</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Decode and display image
    image_bytes = base64.b64decode(image_base64)
    st.image(image_bytes, use_column_width=True)
    
    # Show description in an expander
    with st.expander("📝 Chart Description"):
        st.write(description)


def render_reasoning_panel(reasoning: str):
    """Render the agent reasoning panel."""
    st.markdown("""
        <div class="reasoning-panel">
            <h4>🧠 Agent Reasoning</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Display reasoning steps
    for line in reasoning.split('\n'):
        if line.strip():
            if line.startswith('🔧'):
                st.markdown(f"**{line}**")
            elif line.startswith('   '):
                st.code(line.strip(), language=None)
            else:
                st.write(line)


def render_suggested_questions(questions: List[str], key_prefix: str = "suggestion"):
    """Render suggested follow-up questions."""
    if not questions:
        return None
    
    st.markdown("**💡 Suggested questions:**")
    
    cols = st.columns(len(questions))
    selected = None
    
    for i, (col, question) in enumerate(zip(cols, questions)):
        with col:
            if st.button(
                question[:50] + "..." if len(question) > 50 else question,
                key=f"{key_prefix}_{i}",
                use_container_width=True
            ):
                selected = question
    
    return selected


def render_stats_card(label: str, value: Any, delta: Optional[str] = None):
    """Render a statistics card."""
    delta_html = f'<div class="delta">{delta}</div>' if delta else ''
    
    st.markdown(f"""
        <div class="stat-card">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def render_sidebar_stats(stats: Dict[str, Any]):
    """Render dataset statistics in the sidebar."""
    st.sidebar.markdown("### 📊 Dataset Overview")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("Total Passengers", stats.get('total_passengers', 'N/A'))
        st.metric("Survival Rate", f"{stats.get('survival_rate', 0):.1f}%")
    
    with col2:
        st.metric("Survivors", stats.get('total_survived', 'N/A'))
        st.metric("Avg Age", f"{stats.get('age_stats', {}).get('mean', 0):.1f}")
    
    # Class distribution
    st.sidebar.markdown("### 🎫 Class Distribution")
    class_dist = stats.get('class_distribution', {})
    for cls, count in sorted(class_dist.items()):
        pct = count / stats.get('total_passengers', 1) * 100
        st.sidebar.progress(pct / 100, text=f"Class {cls}: {count} ({pct:.1f}%)")
    
    # Gender distribution
    st.sidebar.markdown("### ⚥ Gender Distribution")
    gender_dist = stats.get('gender_distribution', {})
    for gender, count in gender_dist.items():
        pct = count / stats.get('total_passengers', 1) * 100
        st.sidebar.progress(pct / 100, text=f"{gender.capitalize()}: {count} ({pct:.1f}%)")


def render_sample_questions():
    """Render sample questions section."""
    st.sidebar.markdown("### 💬 Sample Questions")
    
    questions = [
        "What was the overall survival rate?",
        "How did survival differ by gender?",
        "Show me survival by passenger class",
        "What was the age distribution?",
        "Did fare price affect survival?",
        "Give me insights about the data",
        "Show me a comprehensive analysis",
        "How many children survived?"
    ]
    
    selected = None
    for q in questions:
        if st.sidebar.button(q, key=f"sample_{hash(q)}", use_container_width=True):
            selected = q
    
    return selected


def render_loading_animation():
    """Render a loading animation."""
    st.markdown("""
        <div class="loading">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
    """, unsafe_allow_html=True)


def render_error_message(error: str):
    """Render an error message."""
    st.error(f"""
        ❌ **Error occurred**
        
        {error}
        
        Please try again or rephrase your question.
    """)


def render_info_box(title: str, content: str):
    """Render an info box."""
    st.markdown(f"""
        <div class="info-box">
            <strong>{title}</strong><br>
            {content}
        </div>
    """, unsafe_allow_html=True)


def render_execution_time(seconds: float):
    """Render execution time badge."""
    if seconds < 1:
        color = "green"
    elif seconds < 3:
        color = "orange"
    else:
        color = "red"
    
    st.caption(f"⏱️ Response generated in {seconds:.2f}s")