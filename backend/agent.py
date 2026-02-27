"""
LangChain agent setup for the Titanic Chat Agent.
"""

import json
import logging
from typing import Optional, Tuple, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.callbacks.base import BaseCallbackHandler

from backend.config import settings
from backend.tools import get_all_tools
from backend.memory import memory_manager
from backend.schemas import VisualizationData, VisualizationType

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert data analyst assistant specializing in the Titanic dataset. 
Your role is to help users explore, understand, and gain insights from this famous dataset about 
the passengers aboard the RMS Titanic.

## Your Capabilities:
1. **Data Queries**: You can filter, search, and retrieve specific data from the dataset
2. **Statistical Analysis**: You can perform calculations, find correlations, and compute metrics
3. **Visualizations**: You can create charts, graphs, and visual representations
4. **Insights**: You can identify patterns, trends, and interesting findings

## Guidelines:
- Always be accurate and precise with numbers and statistics
- When asked about survival rates or comparisons, use the statistical analysis or visualization tools
- Create visualizations when they would help explain data patterns or when specifically requested
- Provide context and explanations alongside raw data
- If you're not sure about something, say so and explain your reasoning
- Reference the specific numbers from the data to support your answers

## Dataset Context:
The Titanic dataset contains information about {total_passengers} passengers including:
- Survival status (survived: 0 = No, 1 = Yes)
- Passenger class (pclass: 1 = First, 2 = Second, 3 = Third)
- Demographics (sex, age)
- Family relations (sibsp = siblings/spouses, parch = parents/children)
- Ticket information (fare, embarked port)
- Derived features (family_size, is_alone, age_group, fare_category)

## Important:
- When the user asks for visual representation, comparisons, or distributions, USE the visualization tool
- When the user asks for specific numbers or calculations, USE the statistical analysis tool
- When the user wants to see raw data or specific passengers, USE the data query tool
- Always explain your findings in a clear, conversational manner
"""


class ReasoningCallbackHandler(BaseCallbackHandler):
    """Callback handler to capture agent reasoning."""
    
    def __init__(self):
        self.reasoning_steps: List[str] = []
        self.tool_calls: List[Dict[str, Any]] = []
    
    def on_agent_action(self, action, **kwargs):
        """Capture agent actions."""
        self.reasoning_steps.append(f"🔧 Using tool: {action.tool}")
        self.reasoning_steps.append(f"   Input: {action.tool_input}")
        self.tool_calls.append({
            "tool": action.tool,
            "input": action.tool_input
        })
    
    def on_tool_end(self, output, **kwargs):
        """Capture tool outputs."""
        # Truncate long outputs for reasoning display
        output_preview = str(output)[:200] + "..." if len(str(output)) > 200 else str(output)
        self.reasoning_steps.append(f"   Result preview: {output_preview}")
    
    def on_agent_finish(self, finish, **kwargs):
        """Capture final agent output."""
        self.reasoning_steps.append("✅ Agent completed reasoning")
    
    def reset(self):
        """Reset the handler for a new query."""
        self.reasoning_steps = []
        self.tool_calls = []


class TitanicChatAgent:
    """
    Main agent class for handling Titanic dataset queries.
    Implements LangChain agent with custom tools and memory.
    """
    
    def __init__(self):
        """Initialize the chat agent."""
        self.llm = self._create_llm()
        self.tools = get_all_tools()
        self.agent_executor = self._create_agent()
        self.reasoning_handler = ReasoningCallbackHandler()
        logger.info("TitanicChatAgent initialized successfully")
    
    def _create_llm(self) -> ChatOpenAI:
        """Create the LLM instance."""
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
            streaming=True
        )
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor with tools and prompts."""
        from backend.data_loader import get_data_loader
        
        # Get dataset stats for the system prompt
        loader = get_data_loader()
        stats = loader.get_statistics()
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT.format(
                total_passengers=stats['total_passengers']
            )),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent using create_openai_functions_agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=settings.debug,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=True
        )
    
    def _extract_visualizations(
        self, 
        intermediate_steps: List
    ) -> List[VisualizationData]:
        """Extract visualization data from intermediate steps."""
        visualizations = []
        
        for step in intermediate_steps:
            if len(step) >= 2:
                action, result = step[0], step[1]
                
                if action.tool == "titanic_visualization":
                    try:
                        result_data = json.loads(result)
                        if "image_base64" in result_data and "error" not in result_data:
                            # Map string to enum safely
                            chart_type_str = result_data.get("chart_type", "bar_chart")
                            try:
                                chart_type = VisualizationType(chart_type_str)
                            except ValueError:
                                chart_type = VisualizationType.BAR_CHART
                            
                            viz = VisualizationData(
                                chart_type=chart_type,
                                title=result_data.get("title", "Visualization"),
                                image_base64=result_data["image_base64"],
                                description=result_data.get("description", ""),
                                data_summary=None
                            )
                            visualizations.append(viz)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse visualization result: {e}")
        
        return visualizations
    
    def _generate_suggested_questions(
        self, 
        user_message: str, 
        response: str
    ) -> List[str]:
        """Generate follow-up question suggestions based on the conversation."""
        suggestions = []
        
        message_lower = user_message.lower()
        
        # Context-aware suggestions
        if "survival" in message_lower or "survived" in message_lower:
            suggestions.extend([
                "How did survival rates differ by passenger class?",
                "What was the survival rate for children?",
                "Show me a visualization of survival by gender"
            ])
        elif "age" in message_lower:
            suggestions.extend([
                "What was the average age of survivors vs non-survivors?",
                "Show me the age distribution",
                "How did age affect survival chances?"
            ])
        elif "class" in message_lower or "pclass" in message_lower:
            suggestions.extend([
                "What were the fare prices for each class?",
                "How many passengers were in each class?",
                "Compare survival rates across all classes"
            ])
        elif "fare" in message_lower or "price" in message_lower or "ticket" in message_lower:
            suggestions.extend([
                "What was the most expensive ticket?",
                "Show fare distribution by class",
                "Did paying more increase survival chances?"
            ])
        elif "gender" in message_lower or "male" in message_lower or "female" in message_lower:
            suggestions.extend([
                "Why did women have higher survival rates?",
                "Compare survival by gender and class",
                "How many women were in first class?"
            ])
        else:
            # General suggestions
            suggestions.extend([
                "Give me an overview of survival patterns",
                "What are the most interesting insights from the data?",
                "Show me a comprehensive survival analysis"
            ])
        
        # Return unique suggestions (max 3)
        unique_suggestions = list(dict.fromkeys(suggestions))[:3]
        return unique_suggestions
    
    async def chat(
        self, 
        message: str, 
        session_id: str,
        include_reasoning: bool = False
    ) -> Tuple[str, List[VisualizationData], Optional[str], List[str]]:
        """
        Process a chat message and return the response.
        
        Args:
            message: User's message
            session_id: Unique session identifier
            include_reasoning: Whether to include agent reasoning
            
        Returns:
            Tuple of (response_text, visualizations, reasoning, suggested_questions)
        """
        logger.info(f"Processing message for session {session_id}: {message[:50]}...")
        
        # Reset reasoning handler
        self.reasoning_handler.reset()
        
        # Get chat history
        memory = memory_manager.get_memory(session_id)
        chat_history = memory.chat_memory.messages
        
        # Prepare callbacks
        callbacks = [self.reasoning_handler] if include_reasoning else []
        
        try:
            # Invoke the agent
            result = await self.agent_executor.ainvoke(
                {
                    "input": message,
                    "chat_history": chat_history
                },
                config={"callbacks": callbacks}
            )
            
            # Extract response
            response_text = result.get("output", "I apologize, but I couldn't generate a response.")
            
            # Extract visualizations from intermediate steps
            intermediate_steps = result.get("intermediate_steps", [])
            visualizations = self._extract_visualizations(intermediate_steps)
            
            # Generate reasoning summary
            reasoning = None
            if include_reasoning and self.reasoning_handler.reasoning_steps:
                reasoning = "\n".join(self.reasoning_handler.reasoning_steps)
            
            # Generate suggested questions
            suggested_questions = self._generate_suggested_questions(message, response_text)
            
            # Save to memory
            memory_manager.add_messages(session_id, message, response_text)
            
            logger.info(f"Response generated with {len(visualizations)} visualizations")
            
            return response_text, visualizations, reasoning, suggested_questions
            
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            error_message = f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question."
            return error_message, [], None, ["What data is available in the Titanic dataset?"]
    
    def chat_sync(
        self, 
        message: str, 
        session_id: str,
        include_reasoning: bool = False
    ) -> Tuple[str, List[VisualizationData], Optional[str], List[str]]:
        """Synchronous version of chat for non-async contexts."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.chat(message, session_id, include_reasoning)
        )


# Singleton instance
_agent_instance: Optional[TitanicChatAgent] = None


def get_agent() -> TitanicChatAgent:
    """Get or create the singleton agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TitanicChatAgent()
    return _agent_instance


def reset_agent():
    """Reset the agent instance (useful for testing)."""
    global _agent_instance
    _agent_instance = None