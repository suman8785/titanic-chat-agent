"""
Custom CSS styles for the Streamlit frontend.
"""

CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .chat-message.assistant {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    
    .chat-message .role {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #000000 
    }
    
    .chat-message.user .role {
        color: #1976d2;
    }
    
    .chat-message.assistant .role {
        color: #388e3c;
    }
    
    .chat-message {
    color: #000000 !important;
    }
    .chat-message .content {
        line-height: 1.6;
        color: #000000 !important;
    }

    .chat-message .role {
    color: #000000 !important;
    }

    
    /* Sidebar styling */
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .sidebar-section h3 {
        margin-top: 0;
        color: #333;
        font-size: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .stat-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-card .label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
    }
    
    /* Suggestion buttons */
    .suggestion-btn {
        background-color: #e8eaf6;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.85rem;
    }
    
    .suggestion-btn:hover {
        background-color: #c5cae9;
        transform: translateY(-1px);
    }
    
    /* Reasoning panel */
    .reasoning-panel {
        background-color: #fff3e0;
        border: 1px solid #ffcc80;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-family: monospace;
        font-size: 0.85rem;
    }
    
    .reasoning-panel h4 {
        margin-top: 0;
        color: #e65100;
    }
    
    /* Visualization container */
    .viz-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .viz-container h4 {
        margin-top: 0;
        color: #333;
    }
    
    /* Loading animation */
    .loading {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }
    
    .loading-dot {
        width: 10px;
        height: 10px;
        background-color: #667eea;
        border-radius: 50%;
        margin: 0 5px;
        animation: bounce 0.6s infinite alternate;
    }
    
    .loading-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .loading-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes bounce {
        to {
            transform: translateY(-10px);
        }
    }
    
    /* Input area */
    .stTextInput input {
        border-radius: 25px !important;
        padding: 0.75rem 1.25rem !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f5f5f5;
        border-radius: 8px;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.75rem;
        }
        
        .stat-card .value {
            font-size: 1.25rem;
        }
    }
</style>
"""


def get_custom_css() -> str:
    """Return the custom CSS styles."""
    return CUSTOM_CSS