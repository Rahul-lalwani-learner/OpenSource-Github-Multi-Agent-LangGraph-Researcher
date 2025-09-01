"""
Search Tools Module
This module provides structured search tools using Serper and Tavily APIs.
Both tools are designed to work with LangGraph agents and can be used independently.
"""

from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_tavily import TavilySearch
from config import settings
import os
from typing import Dict, Any
from utils import logger

# Set API keys from settings
os.environ["SERPER_API_KEY"] = settings.SERPER_API_SECRET
os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_SECRET

# Initialize search wrappers
try:
    serper_search = GoogleSerperAPIWrapper()
    logger.info("Serper search wrapper initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Serper: {e}")
    serper_search = None

try:
    tavily_search = TavilySearch(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False
    )
    logger.info("Tavily search wrapper initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Tavily: {e}")
    tavily_search = None


@tool("serper_search", return_direct=False)
def serper_search_tool(query: str) -> str:
    """
    Use Serper (Google-like search) to get latest and comprehensive search results.
    
    Best for:
    - Broad discovery and exploration
    - Latest trends and news
    - General web search queries
    - Real-time information
    - Technical information
    - New Technology
    
    Args:
        query (str): The search query string
        
    Returns:
        str: Search results formatted as text
        
    Example:
        serper_search_tool("latest AI frameworks 2025")
        serper_search_tool("new frontend technologies trends")
    """
    if not serper_search:
        return "Error: Serper search is not available. Please check API key configuration."
    
    try:
        results = serper_search.run(query)
        logger.success(f"Serper search successful: {query}")
        return f"Serper Search Results for '{query}':\n\n{results}"
    except Exception as e:
        logger.error(f"Serper search error: {e}")
        return f"Error performing Serper search: {str(e)}"


@tool("tavily_search", return_direct=False)
def tavily_search_tool(query: str) -> str:
    """
    Use Tavily search for factual, research-oriented, and in-depth queries.
    
    Best for:
    - Academic and research queries
    - Technical documentation
    - Fact-checking and verification
    - Deep dives into specific topics
    - Complex explanations
    
    Args:
        query (str): The research query string
        
    Returns:
        str: Research results formatted as text
        
    Example:
        tavily_search_tool("explain quantum computing algorithms")
        tavily_search_tool("technical details of transformer architecture")
    """
    if not tavily_search:
        return "Error: Tavily search is not available. Please check API key configuration."
    
    try:
        results = tavily_search.run(query)
        # Format results nicely
        if isinstance(results, list):
            formatted_results = []
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', 'No URL')
                    formatted_results.append(f"{i}. {title}\n   {content}\n   Source: {url}\n")
                else:
                    formatted_results.append(f"{i}. {str(result)}\n")
            logger.success(f"Tavily search successful: {query}")
            return f"Tavily Research Results for '{query}':\n\n" + "\n".join(formatted_results)
        else:
            return f"Tavily Research Results for '{query}':\n\n{str(results)}"
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Error performing Tavily search: {str(e)}"


def get_available_search_tools() -> Dict[str, Any]:
    """
    Get information about available search tools and their status.
    
    Returns:
        Dict[str, Any]: Dictionary containing tool availability and descriptions
    """
    return {
        "serper": {
            "available": serper_search is not None,
            "description": "Google-like search for broad discovery and latest information",
            "best_for": ["trends", "news", "general search", "real-time info"]
        },
        "tavily": {
            "available": tavily_search is not None,
            "description": "Research-oriented search for factual and technical queries",
            "best_for": ["research", "technical docs", "fact-checking", "deep analysis"]
        }
    }


# Example usage and testing
if __name__ == "__main__":
    print("🔧 Testing Search Tools\n")
    
    # Check tool availability
    tools_status = get_available_search_tools()
    print("📊 Tool Status:")
    for tool_name, info in tools_status.items():
        status = "✅ Available" if info["available"] else "❌ Unavailable"
        print(f"  {tool_name.capitalize()}: {status}")
    print()
    
    # Test queries
    test_query = "latest developments in Large Language Models 2025"
    
    if tools_status["serper"]["available"]:
        print("🔎 Testing Serper Search:")
        serper_result = serper_search_tool.invoke(test_query)
        print(f"Result length: {len(serper_result)} characters")
        print(f"Preview: {serper_result[:200]}...\n")
    
    if tools_status["tavily"]["available"]:
        print("📚 Testing Tavily Search:")
        tavily_result = tavily_search_tool.invoke(test_query)
        print(f"Result length: {len(tavily_result)} characters")
        print(f"Preview: {tavily_result[:200]}...\n")
    
    print("✨ Search tools testing completed!")
