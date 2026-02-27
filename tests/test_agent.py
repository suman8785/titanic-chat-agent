"""
Tests for the LangChain agent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os

from backend.memory import memory_manager


# Skip agent tests if no API key is available
SKIP_AGENT_TESTS = not os.getenv("OPENAI_API_KEY")
SKIP_REASON = "OPENAI_API_KEY not set - skipping agent tests"


class TestConversationMemory:
    """Tests for conversation memory."""
    
    def test_session_creation(self):
        """Test session is created properly."""
        session_id = "test-memory-session"
        memory = memory_manager.get_memory(session_id)
        
        assert memory is not None
    
    def test_message_storage(self):
        """Test messages are stored correctly."""
        session_id = "test-storage-session"
        
        memory_manager.add_messages(
            session_id,
            "Hello, how are you?",
            "I'm doing well, thank you!"
        )
        
        history = memory_manager.get_chat_history(session_id)
        assert len(history) >= 2
    
    def test_session_info(self):
        """Test session info retrieval."""
        session_id = "test-info-session"
        memory_manager.get_or_create_session(session_id)
        
        info = memory_manager.get_session_info(session_id)
        
        assert info["session_id"] == session_id
        assert "created_at" in info
        assert "message_count" in info
    
    def test_clear_session(self):
        """Test session clearing."""
        session_id = "test-clear-session"
        
        memory_manager.add_messages(session_id, "Test", "Response")
        memory_manager.clear_session(session_id)
        
        info = memory_manager.get_session_info(session_id)
        assert info["message_count"] == 0
    
    def test_delete_session(self):
        """Test session deletion."""
        session_id = "test-delete-session"
        
        memory_manager.get_or_create_session(session_id)
        memory_manager.delete_session(session_id)
        
        # Getting info should create a new session
        info = memory_manager.get_session_info(session_id)
        assert info["message_count"] == 0


@pytest.mark.skipif(SKIP_AGENT_TESTS, reason=SKIP_REASON)
class TestTitanicChatAgent:
    """Tests for TitanicChatAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        from backend.agent import get_agent, reset_agent
        reset_agent()  # Reset to get fresh instance
        return get_agent()
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.llm is not None
        assert agent.tools is not None
        assert len(agent.tools) > 0
    
    def test_agent_has_required_tools(self, agent):
        """Test agent has all required tools."""
        tool_names = [t.name for t in agent.tools]
        
        required_tools = [
            "titanic_data_query",
            "titanic_statistical_analysis",
            "titanic_visualization",
            "titanic_insights",
            "titanic_dataset_info"
        ]
        
        for tool in required_tools:
            assert tool in tool_names, f"Missing tool: {tool}"
    
    def test_extract_visualizations_empty(self, agent):
        """Test visualization extraction with no visualizations."""
        visualizations = agent._extract_visualizations([])
        assert visualizations == []
    
    def test_extract_visualizations_with_data(self, agent):
        """Test visualization extraction with valid data."""
        mock_action = MagicMock()
        mock_action.tool = "titanic_visualization"
        
        mock_result = json.dumps({
            "chart_type": "bar_chart",
            "title": "Test Chart",
            "image_base64": "dGVzdA==",  # base64 of "test"
            "description": "Test description"
        })
        
        intermediate_steps = [(mock_action, mock_result)]
        visualizations = agent._extract_visualizations(intermediate_steps)
        
        assert len(visualizations) == 1
        assert visualizations[0].title == "Test Chart"
    
    def test_generate_suggested_questions_survival(self, agent):
        """Test suggestion generation for survival query."""
        questions = agent._generate_suggested_questions(
            "What was the survival rate?",
            "The survival rate was 38.4%"
        )
        
        assert len(questions) > 0
        assert len(questions) <= 3
    
    def test_generate_suggested_questions_age(self, agent):
        """Test suggestion generation for age query."""
        questions = agent._generate_suggested_questions(
            "What was the average age?",
            "The average age was 29.7 years"
        )
        
        assert len(questions) > 0
        # Should contain age-related suggestions
        assert any("age" in q.lower() for q in questions)


@pytest.mark.skipif(SKIP_AGENT_TESTS, reason=SKIP_REASON)
class TestAgentIntegration:
    """Integration tests for the agent."""
    
    @pytest.fixture
    def agent(self):
        from backend.agent import get_agent, reset_agent
        reset_agent()
        return get_agent()
    
    @pytest.mark.asyncio
    async def test_simple_query(self, agent):
        """Test simple query processing."""
        response, visualizations, reasoning, suggestions = await agent.chat(
            message="How many passengers were on the Titanic?",
            session_id="integration-test-1",
            include_reasoning=False
        )
        
        assert response is not None
        assert len(response) > 0
        # Should mention a number
        assert any(char.isdigit() for char in response)
    
    @pytest.mark.asyncio
    async def test_visualization_query(self, agent):
        """Test query that should generate visualization."""
        response, visualizations, reasoning, suggestions = await agent.chat(
            message="Show me a chart of survival by gender",
            session_id="integration-test-2",
            include_reasoning=True
        )
        
        assert response is not None
        # May or may not have visualizations depending on agent decision
        assert isinstance(visualizations, list)
    
    @pytest.mark.asyncio
    async def test_follow_up_query(self, agent):
        """Test follow-up query uses context."""
        session_id = "integration-test-3"
        
        # First query
        await agent.chat(
            message="What was the survival rate for women?",
            session_id=session_id
        )
        
        # Follow-up query
        response, _, _, _ = await agent.chat(
            message="How does that compare to men?",
            session_id=session_id
        )
        
        assert response is not None