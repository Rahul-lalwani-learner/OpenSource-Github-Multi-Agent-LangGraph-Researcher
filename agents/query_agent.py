"""
Query Agent Module

This agent analyzes user queries to determine the appropriate action:
1. Direct repo info extraction (if URL is provided)
2. Repository search (if search query is provided)

The agent uses LangChain Google GenAI and returns structured Pydantic outputs.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from utils.logging import logger
import re

class QueryDecision(BaseModel):
    """Pydantic model for LLM decision making."""
    action_type: Literal["get_repo_info", "search_repos"] = Field(
        ..., 
        description="Whether to get info for a specific repo or search for repositories"
    )
    repo_url: Optional[str] = Field(
        None, 
        description="GitHub repository URL if action_type is get_repo_info"
    )
    search_query: Optional[str] = Field(
        None, 
        description="Search query if action_type is search_repos"
    )
    search_method: Optional[Literal["pygithub", "graphql"]] = Field(
        None, 
        description="Search method to use - pygithub for simple searches, graphql for complex queries needing detailed repo info"
    )
    top_k: Optional[int] = Field(
        default=5, 
        description="Number of repositories to return for search"
    )
    max_depth: Optional[int] = Field(
        default=3, 
        description="Maximum depth for file structure traversal in repo info"
    )
    reasoning: str = Field(
        ..., 
        description="Brief explanation of why this decision was made"
    )

class RepoInfoRequest(BaseModel):
    """Pydantic model for direct repository info extraction requests."""
    action_type: Literal["get_repo_info"] = "get_repo_info"
    repo_url: str = Field(..., description="The GitHub repository URL")
    max_depth: Optional[int] = Field(default=3, description="Maximum depth for file structure traversal")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "get_repo_info",
                "repo_url": "https://github.com/owner/repo",
                "max_depth": 3
            }
        }

class RepoSearchRequest(BaseModel):
    """Pydantic model for repository search requests."""
    action_type: Literal["search_repos"] = "search_repos"
    search_query: str = Field(..., description="The search query for finding repositories")
    top_k: Optional[int] = Field(default=5, description="Number of top repositories to return")
    search_method: Optional[Literal["pygithub", "graphql"]] = Field(default="pygithub", description="Search method to use based on the query if needed result of complex query then graphql otherwise pygithub")

    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "search_repos",
                "search_query": "python machine learning",
                "top_k": 5,
                "search_method": "pygithub"
            }
        }

class QueryAgent:
    """
    Query Agent that analyzes user input and determines the appropriate action.
    
    This agent uses LangChain Google GenAI to classify user queries and return
    structured Pydantic outputs for downstream processing.
    """
    
    def __init__(self):
        """Initialize the Query Agent with Google GenAI model."""
        logger.info("Initializing QueryAgent with Google GenAI")
        
        try:
            self.model = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0,
                max_tokens=500
            )
            # Bind the model with structured output
            self.structured_model = self.model.with_structured_output(QueryDecision)
            logger.success("QueryAgent initialized successfully with structured output")
        except Exception as e:
            logger.error(f"Failed to initialize QueryAgent: {e}")
            raise
    
    def _is_github_url(self, query: str) -> bool:
        """
        Check if the query contains a GitHub repository URL.
        
        Args:
            query (str): The user query
            
        Returns:
            bool: True if query contains a GitHub URL, False otherwise
        """
        github_url_pattern = r"https?://github\.com/[^/]+/[^/]+/?(?:\s|$)"
        return bool(re.search(github_url_pattern, query))
    
    def _extract_github_url(self, query: str) -> Optional[str]:
        """
        Extract GitHub URL from the query.
        
        Args:
            query (str): The user query
            
        Returns:
            Optional[str]: Extracted GitHub URL or None
        """
        github_url_pattern = r"(https?://github\.com/[^/]+/[^/]+/?)"
        match = re.search(github_url_pattern, query)
        return match.group(1) if match else None

    def analyze_query(self, user_query: str) -> Union[RepoInfoRequest, RepoSearchRequest]:
        """
        Analyze user query and determine the appropriate action using LLM with structured output.
        
        Args:
            user_query (str): The user's input query
            
        Returns:
            Union[RepoInfoRequest, RepoSearchRequest]: Structured output for the determined action
        """
        logger.info(f"Analyzing query: '{user_query}'")
        
        try:
            prompt = f"""
            You are an intelligent GitHub repository analysis router. Analyze the user query and decide:

            1. **Action Type**: 
               - Use "get_repo_info" if the user provides a specific GitHub repository URL or asks for detailed analysis of a specific repo
               - Use "search_repos" if the user wants to find/search for repositories based on criteria

            2. **For get_repo_info**:
               - Extract the GitHub URL from the query
               - Set appropriate max_depth (1-5) based on how detailed they want file structure

            3. **For search_repos**:
               - Generate GitHub search syntax for the query (use GitHub search qualifiers)
               - Choose search_method:
                 * "pygithub" for simple searches (basic repo info, popular repos, language-based searches)
                 * "graphql" for complex searches needing detailed info (repos with specific issue counts, PR counts, contributor info, license details, language statistics)
               - Set appropriate top_k (typically 5-10)

            **GitHub Search Syntax Examples:**
            - "find python machine learning repositories" → "python machine learning language:python"
            - "react projects with open issues" → "react language:javascript"
            - "most starred javascript repositories" → "language:javascript"
            - "python repos with good issues" → "language:python good-first-issues:>1"
            - Use qualifiers like: language:, stars:, forks:, created:, pushed:, topic:

            **Examples:**
            - "https://github.com/microsoft/vscode" → get_repo_info
            - "analyze this repo: https://github.com/openai/gpt-3" → get_repo_info  
            - "find python machine learning repositories" → search_repos (pygithub), query: "python machine learning language:python"
            - "search for JavaScript repos with open issues" → search_repos (pygithub), query: "language:javascript"
            - "popular AI projects" → search_repos (pygithub), query: "AI machine learning"
            - "most recent react repositories" → search_repos (pygithub), query: "react language:javascript"

            **User Query:** {user_query}

            Provide structured output with your decision and reasoning.
            """
            
            logger.debug("Sending query to LLM for structured decision making")
            response = self.structured_model.invoke(prompt)
            
            # Handle the response properly
            if isinstance(response, QueryDecision):
                decision = response
            else:
                # Fallback if structured output fails
                logger.warning("Structured output failed, using fallback logic")
                if self._is_github_url(user_query):
                    github_url = self._extract_github_url(user_query)
                    if github_url:
                        return RepoInfoRequest(
                            action_type="get_repo_info",
                            repo_url=github_url,
                            max_depth=3
                        )
                return RepoSearchRequest(
                    action_type="search_repos",
                    search_query=user_query,
                    top_k=5,
                    search_method="pygithub"
                )
            
            logger.info(f"LLM Decision: {decision.action_type} - {decision.reasoning}")
            
            # Convert LLM decision to appropriate response model
            if decision.action_type == "get_repo_info":
                if not decision.repo_url:
                    # Try to extract URL from query as fallback
                    extracted_url = self._extract_github_url(user_query)
                    if not extracted_url:
                        logger.warning("No GitHub URL found for get_repo_info action, falling back to search")
                        return RepoSearchRequest(
                            action_type="search_repos",
                            search_query=user_query,
                            top_k=5,
                            search_method="pygithub"
                        )
                    decision.repo_url = extracted_url
                
                logger.success(f"Routing to get_repo_info for URL: {decision.repo_url}")
                return RepoInfoRequest(
                    action_type="get_repo_info",
                    repo_url=decision.repo_url,
                    max_depth=decision.max_depth or 3
                )
            
            else:  # search_repos
                search_query = decision.search_query or user_query
                search_method = decision.search_method or "pygithub"
                top_k = decision.top_k or 5
                
                logger.success(f"Routing to search_repos with method: {search_method}, query: '{search_query}'")
                return RepoSearchRequest(
                    action_type="search_repos",
                    search_query=search_query,
                    top_k=top_k,
                    search_method=search_method
                )
                
        except Exception as e:
            logger.error(f"Error in LLM query analysis: {e}")
            # Fallback logic
            if self._is_github_url(user_query):
                github_url = self._extract_github_url(user_query)
                if github_url:
                    logger.info("Fallback: GitHub URL detected, using get_repo_info")
                    return RepoInfoRequest(
                        action_type="get_repo_info",
                        repo_url=github_url,
                        max_depth=3
                    )
            
            logger.info("Fallback: Using search_repos with pygithub")
            return RepoSearchRequest(
                action_type="search_repos",
                search_query=user_query,
                top_k=5,
                search_method="pygithub"
            )

    def process_query(self, user_query: str) -> dict:
        """
        Process user query and return structured output.
        
        Args:
            user_query (str): The user's input query
            
        Returns:
            dict: Structured output ready for workflow processing
        """
        try:
            result = self.analyze_query(user_query)
            logger.success(f"Query processed successfully. Action: {result.action_type}")
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "action_type": "error"
            }

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing QueryAgent with LLM decision making")
    
    agent = QueryAgent()
    
    # Test cases covering different scenarios
    test_queries = [
        # Direct URL cases
        "https://github.com/microsoft/vscode",
        "analyze this repo: https://github.com/openai/gpt-3",
        
        # Simple search cases (should use pygithub)
        "find python machine learning repositories",
        "popular AI projects on GitHub",
        "react frontend frameworks",
        
        # Complex search cases (should use graphql)
        "search for JavaScript repos with more than 100 open issues",
        "Python repositories with MIT license and active contributors",
        "repos with detailed language statistics and pull request activity",
        "find repositories with specific issue counts and contributor details"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        result = agent.process_query(query)
        print(f"📊 Action: {result.get('action_type')}")
        if result.get('action_type') == 'search_repos':
            print(f"🔧 Method: {result.get('search_method')}")
            print(f"📝 Search Query: {result.get('search_query')}")
        elif result.get('action_type') == 'get_repo_info':
            print(f"🔗 Repo URL: {result.get('repo_url')}")
            print(f"📁 Max Depth: {result.get('max_depth')}")
        print("─" * 50)
    
    logger.info("QueryAgent testing completed")
