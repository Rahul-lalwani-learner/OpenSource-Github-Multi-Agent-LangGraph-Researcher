"""
Entry Point and Issue Breakdown Agent Module

This agent analyzes repository information and specific issue details to provide:
1. Entry points for the repository (main files, important directories)
2. Detailed issue breakdown and analysis
3. Project summary and architecture overview
4. Actionable insights for working on the issue

The agent uses LangChain Google GenAI and returns structured Pydantic outputs.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, cast
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from utils.logging import logger
import json

class EntryPoint(BaseModel):
    """Model for repository entry points."""
    file_path: str = Field(..., description="Path to the entry point file")
    file_type: Literal["main", "config", "test", "documentation", "build"] = Field(
        ..., description="Type of entry point"
    )
    importance: Literal["critical", "high", "medium", "low"] = Field(
        ..., description="Importance level of this entry point"
    )
    description: str = Field(..., description="Brief description of what this file does")
    related_to_issue: bool = Field(
        default=False, description="Whether this entry point is related to the issue"
    )

class IssueAnalysis(BaseModel):
    """Model for detailed issue analysis."""
    issue_type: str = Field(..., description="Type of issue (bug, feature, enhancement, etc.)")
    complexity: Literal["low", "medium", "high", "very_high"] = Field(
        ..., description="Estimated complexity of the issue"
    )
    affected_components: List[str] = Field(
        default=[], description="List of components/modules affected by this issue"
    )
    technologies_involved: List[str] = Field(
        default=[], description="Technologies/languages involved in solving this issue"
    )
    estimated_effort: str = Field(..., description="Estimated time/effort to resolve")
    prerequisites: List[str] = Field(
        default=[], description="Prerequisites or dependencies to understand before working on this issue"
    )
    related_files: List[str] = Field(
        default=[], description="Files likely to be modified for this issue"
    )

class ProjectSummary(BaseModel):
    """Model for project overview."""
    project_type: str = Field(..., description="Type of project (web app, library, CLI tool, etc.)")
    main_technologies: List[str] = Field(..., description="Primary technologies used")
    architecture_pattern: str = Field(..., description="Overall architecture pattern")
    key_directories: str = Field(
        default='', description="Important directories and their purposes as a key-value pair string, e.g. 'src: Source code directory, tests: Test files directory'"
    )
    setup_instructions: str = Field(..., description="How to set up the project locally")
    testing_approach: str = Field(..., description="How testing is structured in the project")

class EntryPointAnalysisResult(BaseModel):
    """Complete analysis result from the Entry Point and Issue Breakdown Agent."""
    entry_points: List[EntryPoint] = Field(..., description="List of repository entry points")
    issue_analysis: IssueAnalysis = Field(..., description="Detailed analysis of the selected issue")
    project_summary: ProjectSummary = Field(..., description="Overall project summary")
    issue_summary: str = Field(..., description="Concise summary of the issue")
    recommended_approach: str = Field(..., description="Recommended approach to tackle the issue")
    next_steps: List[str] = Field(..., description="Actionable next steps for working on the issue")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score of the analysis (0.0 to 1.0)"
    )

class EntryPointAndIssueBreakdownAgent:
    """
    Agent that analyzes repository information and issue details to provide comprehensive insights.
    
    This agent takes repository data and a specific issue, then provides detailed analysis
    including entry points, issue breakdown, and actionable recommendations.
    """
    
    def __init__(self):
        """Initialize the Entry Point and Issue Breakdown Agent with Google GenAI model."""
        logger.info("Initializing EntryPointAndIssueBreakdownAgent with Google GenAI")
        
        try:
            self.model = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.3,
                max_tokens=3000
            )
            # Bind the model with structured output
            self.structured_model = self.model.with_structured_output(EntryPointAnalysisResult)
            logger.success("EntryPointAndIssueBreakdownAgent initialized successfully with structured output")
        except Exception as e:
            logger.error(f"Failed to initialize EntryPointAndIssueBreakdownAgent: {e}")
            raise
    
    def analyze_repo_and_issue(self, repo_data: Dict[str, Any], selected_issue: Dict[str, Any]) -> EntryPointAnalysisResult:
        """
        Analyze repository data and selected issue to provide comprehensive insights.
        
        Args:
            repo_data (Dict[str, Any]): Complete repository information from get_all_repo_details
            selected_issue (Dict[str, Any]): Selected issue details
            
        Returns:
            EntryPointAnalysisResult: Structured analysis result
        """
        logger.info(f"Analyzing repository and issue #{selected_issue.get('number', 'unknown')}")
        
        try:
            # Prepare the analysis prompt
            prompt = self._create_analysis_prompt(repo_data, selected_issue)
            
            logger.debug("Sending analysis request to LLM")
            
            # Try structured output first
            try:
                response = self.structured_model.invoke(prompt)
                logger.debug(f"Structured response type: {type(response)}")
                
                if isinstance(response, EntryPointAnalysisResult):
                    logger.success(f"Analysis completed successfully with confidence score: {response.confidence_score}")
                    return response
                else:
                    logger.warning(f"Unexpected response type: {type(response)}")
                    raise ValueError("Invalid response type")
                    
            except Exception as struct_error:
                logger.debug(f"Structured output failed (expected): {struct_error}")
                
                # Try with regular model and manual parsing
                logger.info("Attempting manual parsing with regular model")
                regular_response = self.model.invoke(prompt)
                
                # Try to parse the response manually
                content = regular_response.content
                if isinstance(content, list):
                    content = str(content)
                elif not isinstance(content, str):
                    content = str(content)
                    
                parsed_result = self._parse_llm_response(content)
                if parsed_result:
                    logger.success("Manual parsing successful")
                    return parsed_result
                else:
                    logger.warning("Manual parsing also failed")
                    raise struct_error
                
        except Exception as e:
            logger.error(f"Error in repository and issue analysis: {e}")
            logger.debug(f"Full error details: {str(e)}")
            return self._create_fallback_response(repo_data, selected_issue)
    
    def _parse_llm_response(self, response_content: str) -> Optional[EntryPointAnalysisResult]:
        """
        Manually parse LLM response and fix common issues.
        
        Args:
            response_content (str): Raw LLM response content
            
        Returns:
            Optional[EntryPointAnalysisResult]: Parsed result or None if parsing fails
        """
        try:
            # Try to extract JSON from response
            import re
            import json
            
            # Look for JSON content between ```json and ``` or just the JSON object
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', response_content, re.DOTALL)
            
            if not json_match:
                logger.warning("No JSON found in response")
                return None
            
            json_str = json_match.group(1)
            
            # Simple JSON cleanup - remove trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Parse JSON and create result
            data = json.loads(json_str)
            result = EntryPointAnalysisResult(**data)
            return result
            
        except Exception as e:
            logger.error(f"Manual parsing failed: {e}")
            logger.debug(f"Response content preview: {response_content[:500]}...")
            return None
    
    def _create_analysis_prompt(self, repo_data: Dict[str, Any], selected_issue: Dict[str, Any]) -> str:
        """Create the analysis prompt for the LLM."""
        
        # Extract key information from repo data
        basic_info = repo_data.get("basic_info", {})
        technologies = repo_data.get("technologies", {}).get("technologies", {})
        file_structure = repo_data.get("file_structure", {}).get("file_structure", {})
        dependencies = repo_data.get("dependencies", {})
        readme = repo_data.get("readme", {}).get("readme_content", "No README available")
        
        # Limit README length for prompt
        readme_preview = readme[:2000] + "..." if len(readme) > 2000 else readme
        
        prompt = f"""
        You are an expert software architect and issue analyst. Analyze the provided repository information and issue details to provide comprehensive insights.

        **Repository Information:**
        - Name: {basic_info.get('name', 'Unknown')}
        - Description: {basic_info.get('description', 'No description')}
        - Technologies: {list(technologies.keys())[:10]}  # Top 10 technologies
        - Stars: {basic_info.get('stargazer_count', 0)}
        - Forks: {basic_info.get('fork_count', 0)}
        
        **File Structure (Top Level):**
        {json.dumps(list(file_structure.keys())[:20], indent=2)}  # Top 20 files/dirs
        
        **Dependencies Found:**
        {list(dependencies.get('found_files', []))}
        
        **README Preview:**
        {readme_preview}
        
        **Selected Issue:**
        - Number: #{selected_issue.get('number', 'N/A')}
        - Title: {selected_issue.get('title', 'No title')}
        - Body: {selected_issue.get('body', 'No description')[:1500]}...
        - Labels: {selected_issue.get('labels', [])}
        - State: {selected_issue.get('state', 'unknown')}
        
        **Instructions:**
        1. **Entry Points**: Identify 5-8 key entry points in the repository (main files, config files, etc.)
        2. **Issue Analysis**: Provide detailed analysis of the issue including complexity, affected components, and effort estimation
        3. **Project Summary**: Summarize the project architecture, technologies, and structure
        4. **Recommendations**: Provide actionable recommendations and next steps
        
        **Focus on:**
        - Identifying files most relevant to the issue
        - Understanding the project architecture
        - Estimating complexity and effort realistically
        - Providing actionable insights
        
        **IMPORTANT**: For the field "key_directories" in project_summary, output a string as key-value pairs, e.g. 'src: Source code directory, tests: Test files directory'. Do NOT output a dictionary or JSON object for this field.
        
        Provide structured analysis following the defined schema exactly.
        """
        
        return prompt
    
    def _create_fallback_response(self, repo_data: Dict[str, Any], selected_issue: Dict[str, Any]) -> EntryPointAnalysisResult:
        """Create a fallback response if LLM analysis fails."""
        logger.info("Creating fallback analysis response")
        
        basic_info = repo_data.get("basic_info", {})
        technologies = repo_data.get("technologies", {}).get("technologies", {})
        file_structure = repo_data.get("file_structure", {}).get("file_structure", {})
        
        # Basic entry points detection
        entry_points = []
        common_entry_files = [
            ("README.md", "documentation", "Documentation and project overview"),
            ("package.json", "config", "Node.js project configuration"),
            ("requirements.txt", "config", "Python dependencies"),
            ("main.py", "main", "Main Python application entry point"),
            ("app.py", "main", "Flask/Django application entry point"),
            ("index.js", "main", "Main JavaScript entry point"),
            ("src/", "main", "Source code directory"),
            ("lib/", "main", "Library code directory"),
            ("test/", "test", "Test directory"),
            ("tests/", "test", "Test directory")
        ]
        
        for file_path, file_type, description in common_entry_files:
            if file_path in file_structure:
                # Ensure file_type is properly typed
                valid_file_type = cast(Literal["main", "config", "test", "documentation", "build"], 
                                     file_type if file_type in ["main", "config", "test", "documentation", "build"] else "main")
                entry_points.append(EntryPoint(
                    file_path=file_path,
                    file_type=valid_file_type,
                    importance="medium",
                    description=description,
                    related_to_issue=False
                ))
        
        return EntryPointAnalysisResult(
            entry_points=entry_points,
            issue_analysis=IssueAnalysis(
                issue_type="unknown",
                complexity="medium",
                affected_components=[],
                technologies_involved=list(technologies.keys())[:5],
                estimated_effort="Unable to estimate without detailed analysis",
                prerequisites=["Repository setup", "Understanding project structure"],
                related_files=[]
            ),
            project_summary=ProjectSummary(
                project_type="Software Project",
                main_technologies=list(technologies.keys())[:5],
                architecture_pattern="Unknown - requires detailed analysis",
                key_directories="src: Source code directory, tests: Test directory, docs: Documentation",
                setup_instructions="Please refer to README.md for setup instructions",
                testing_approach="Unknown - requires investigation"
            ),
            issue_summary=f"Issue #{selected_issue.get('number')}: {selected_issue.get('title', 'No title')}",
            recommended_approach="Manual analysis required - LLM analysis failed",
            next_steps=[
                "Review repository structure manually",
                "Read project documentation",
                "Understand the issue requirements",
                "Identify relevant files and components"
            ],
            confidence_score=0.3
        )
    
    def process_analysis(self, repo_data: Dict[str, Any], selected_issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process repository and issue analysis and return structured output.
        
        Args:
            repo_data (Dict[str, Any]): Complete repository information
            selected_issue (Dict[str, Any]): Selected issue details
            
        Returns:
            Dict[str, Any]: Structured analysis result as dictionary
        """
        try:
            result = self.analyze_repo_and_issue(repo_data, selected_issue)
            logger.success(f"Analysis processed successfully for issue #{selected_issue.get('number')}")
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error processing analysis: {e}")
            return {
                "error": str(e),
                "analysis_type": "error"
            }

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing EntryPointAndIssueBreakdownAgent")
    
    # Mock data for testing
    sample_repo_data = {
        "basic_info": {
            "name": "sample-project",
            "description": "A sample project for testing",
            "stargazer_count": 100,
            "fork_count": 20
        },
        "technologies": {
            "technologies": {"Python": {"percentage": 80}, "JavaScript": {"percentage": 20}}
        },
        "file_structure": {
            "file_structure": {"README.md": {}, "src/": {}, "tests/": {}}
        },
        "readme": {"readme_content": "# Sample Project\nThis is a sample project for testing."}
    }
    
    sample_issue = {
        "number": 123,
        "title": "Add authentication feature",
        "body": "We need to implement user authentication with JWT tokens",
        "labels": ["enhancement", "backend"],
        "state": "open"
    }
    
    agent = EntryPointAndIssueBreakdownAgent()
    result = agent.process_analysis(sample_repo_data, sample_issue)
    
    print("🔍 Analysis Result:")
    print(f"📊 Entry Points: {len(result.get('entry_points', []))}")
    print(f"🎯 Issue Type: {result.get('issue_analysis', {}).get('issue_type', 'N/A')}")
    print(f"📈 Confidence: {result.get('confidence_score', 0)}")
    print(f"🛠️ Complexity: {result.get('issue_analysis', {}).get('complexity', 'N/A')}")
    
    # Show more detailed results
    if result.get('entry_points'):
        print(f"\n📁 Entry Points Found:")
        for i, ep in enumerate(result['entry_points'][:3], 1):  # Show first 3
            print(f"  {i}. {ep.get('file_path', 'N/A')} ({ep.get('file_type', 'N/A')}) - {ep.get('importance', 'N/A')}")
    
    if result.get('issue_analysis', {}).get('technologies_involved'):
        print(f"\n🔧 Technologies Involved: {', '.join(result['issue_analysis']['technologies_involved'][:3])}")
    
    if result.get('next_steps'):
        print(f"\n📝 Next Steps:")
        for i, step in enumerate(result['next_steps'][:3], 1):  # Show first 3
            print(f"  {i}. {step}")
    
    print(f"\n💡 Issue Summary: {result.get('issue_summary', 'N/A')[:100]}...")
    print(f"🎯 Recommended Approach: {result.get('recommended_approach', 'N/A')[:100]}...")
    
    logger.info("EntryPointAndIssueBreakdownAgent testing completed")