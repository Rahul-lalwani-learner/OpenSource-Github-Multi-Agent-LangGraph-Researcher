"""
Solver Agent Module

This agent is responsible for solving GitHub issues by:
1. Analyzing the issue and repository context
2. Using search tools to find relevant solutions and documentation
3. Fetching specific file contents when needed
4. Providing detailed step-by-step solutions with proper citations

The agent has access to:
- Serper search tool for general web search
- Tavily search tool for specialized technical search
- GitHub file content retrieval tool
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from config.settings import settings
from utils.logging import logger
from core.tools.search_tools import serper_search_tool, tavily_search_tool
from core.tools.get_file_content import get_file_content_tool
import json

class SolutionStep(BaseModel):
    """Model for individual solution steps."""
    step_number: int = Field(..., description="Sequential step number")
    title: str = Field(..., description="Brief title of the step")
    description: str = Field(..., description="Detailed description of what to do")
    file_path: Optional[str] = Field(default=None, description="File path if this step involves modifying a specific file")
    code_snippet: Optional[str] = Field(default=None, description="Code snippet to add/modify if applicable")
    commands: List[str] = Field(default=[], description="Terminal commands to run if any")
    rationale: str = Field(..., description="Why this step is necessary")
    citations: List[str] = Field(default=[], description="Sources/references used for this step")

class SolutionAnalysis(BaseModel):
    """Model for overall solution analysis."""
    issue_understanding: str = Field(..., description="Summary of how the issue was understood")
    approach_rationale: str = Field(..., description="Why this approach was chosen")
    complexity_assessment: Literal["simple", "moderate", "complex", "very_complex"] = Field(
        ..., description="Assessed complexity of the solution"
    )
    estimated_time: str = Field(..., description="Estimated time to implement the solution")
    potential_risks: List[str] = Field(default=[], description="Potential risks or challenges")
    prerequisites: List[str] = Field(default=[], description="Prerequisites before implementing the solution")

class SolverResponse(BaseModel):
    """Complete response from the Solver Agent."""
    solution_analysis: SolutionAnalysis = Field(..., description="Overall analysis of the solution")
    solution_steps: List[SolutionStep] = Field(..., description="Step-by-step solution")
    testing_strategy: str = Field(..., description="How to test the implemented solution")
    additional_resources: List[str] = Field(default=[], description="Additional helpful resources")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the solution (0.0 to 1.0)")
    tools_used: List[str] = Field(default=[], description="List of tools used during analysis")

class SolverAgent:
    """
    ReAct Agent that solves GitHub issues by reasoning through problems and using tools strategically.
    
    This agent uses the ReAct (Reasoning and Acting) pattern to:
    - Think through problems step by step
    - Decide when and which tools to use
    - Reason about tool outputs before proceeding
    - Provide comprehensive solutions with proper citations
    """
    
    def __init__(self):
        """Initialize the Solver Agent with Google GenAI model and ReAct framework."""
        logger.info("Initializing SolverAgent with ReAct framework and Google GenAI")
        
        try:
            self.model = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.2,  # Lower temperature for more focused reasoning
                max_tokens=4000
            )
            
            # Initialize available tools
            self.tools = [
                serper_search_tool,
                tavily_search_tool,
                get_file_content_tool
            ]
            
            # Create ReAct agent prompt
            self.react_prompt = self._create_react_prompt()
            
            # Create ReAct agent
            self.agent = create_react_agent(self.model, self.tools, self.react_prompt)
            self.agent_executor = AgentExecutor(
                agent=self.agent, 
                tools=self.tools, 
                verbose=True,
                max_iterations=10,
                handle_parsing_errors=True
            )
            
            logger.success("SolverAgent with ReAct framework initialized successfully")
            logger.info(f"Available tools: {[tool.name for tool in self.tools]}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SolverAgent: {e}")
            raise
    
    def _create_react_prompt(self) -> PromptTemplate:
        """Create the ReAct prompt template for the agent."""
        template = """You are an expert software engineer solving GitHub issues using a step-by-step reasoning approach.

Available tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (must be valid JSON format)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT TOOL INPUT FORMAT:
- For get_file_content: {{"repo_url": "https://github.com/owner/repo", "file_path": "src/file.js"}}
- For serper_search: {{"query": "your search query here"}}
- For tavily_search: {{"query": "your search query here"}}

Examples of correct tool usage:
Action: get_file_content
Action Input: {{"repo_url": "https://github.com/facebook/react", "file_path": "packages/react/src/ReactHooks.js"}}

Action: serper_search
Action Input: {{"query": "React useEffect cleanup function best practices"}}

When solving GitHub issues:
1. THINK through the problem systematically
2. Search for relevant information when needed
3. Fetch specific file contents if they might help
4. Reason about the information gathered
5. Provide a comprehensive solution with proper citations

Remember:
- Always reason before taking action
- Use tools strategically, not excessively
- Cite sources from your research
- Provide step-by-step solutions
- Consider complexity and risks
- Format tool inputs as valid JSON

Question: {input}
{agent_scratchpad}"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )
    
    def solve_issue(self, 
                   issue_data: Dict[str, Any], 
                   repo_analysis: Dict[str, Any], 
                   entry_point_analysis: Dict[str, Any]) -> SolverResponse:
        """
        Solve a GitHub issue using ReAct reasoning and available tools.
        
        Args:
            issue_data (Dict[str, Any]): Issue information (title, body, labels, etc.)
            repo_analysis (Dict[str, Any]): Repository analysis from previous agents
            entry_point_analysis (Dict[str, Any]): Entry point and issue breakdown analysis
            
        Returns:
            SolverResponse: Complete solution with steps and analysis
        """
        logger.info(f"Starting to solve issue #{issue_data.get('number', 'unknown')}: {issue_data.get('title', 'No title')}")
        
        try:
            # Create comprehensive input for the ReAct agent
            agent_input = self._create_agent_input(issue_data, repo_analysis, entry_point_analysis)
            
            # Use ReAct agent to solve the issue
            logger.debug("Running ReAct agent to solve the issue")
            result = self.agent_executor.invoke({"input": agent_input})
            
            # Parse the agent's output into structured response
            solution = self._parse_agent_output(result, issue_data)
            
            logger.success(f"Successfully generated solution with {len(solution.solution_steps)} steps")
            return solution
            
        except Exception as e:
            logger.error(f"Error solving issue with ReAct agent: {e}")
            return self._create_fallback_solution(issue_data, str(e))
    
    def _create_agent_input(self, issue_data: Dict[str, Any], repo_analysis: Dict[str, Any], entry_point_analysis: Dict[str, Any]) -> str:
        """Create comprehensive input for the ReAct agent."""
        
        # Extract key information
        issue_title = issue_data.get('title', 'No title')
        issue_body = issue_data.get('body', 'No description')
        issue_labels = issue_data.get('labels', [])
        issue_number = issue_data.get('number', 'unknown')
        
        # Repository context
        repo_name = repo_analysis.get('basic_info', {}).get('name', 'Unknown')
        repo_desc = repo_analysis.get('basic_info', {}).get('description', 'No description')
        technologies = entry_point_analysis.get('project_summary', {}).get('main_technologies', [])
        
        # Issue analysis
        issue_type = entry_point_analysis.get('issue_analysis', {}).get('issue_type', 'unknown')
        complexity = entry_point_analysis.get('issue_analysis', {}).get('complexity', 'medium')
        related_files = entry_point_analysis.get('issue_analysis', {}).get('related_files', [])
        
        # Entry points
        entry_points = entry_point_analysis.get('entry_points', [])
        important_files = [ep.get('file_path', '') for ep in entry_points if ep.get('importance') in ['critical', 'high']]
        
        agent_input = f"""
Please solve the following GitHub issue using step-by-step reasoning:

**ISSUE DETAILS:**
- Repository: {repo_name}
- Issue #{issue_number}: {issue_title}
- Description: {issue_body[:800]}...
- Labels: {issue_labels}
- Type: {issue_type}
- Complexity: {complexity}

**REPOSITORY CONTEXT:**
- Description: {repo_desc}
- Technologies: {technologies}
- Important Files: {important_files[:5]}
- Files Likely to Change: {related_files[:5]}

**YOUR TASK:**
1. Analyze the issue thoroughly
2. Research relevant solutions and best practices
3. Examine relevant files if needed
4. Provide a comprehensive step-by-step solution
5. Include proper citations and rationale
6. Assess risks and provide testing strategy

**REQUIREMENTS:**
- Think through each step carefully
- Use tools strategically to gather information
- Provide actionable implementation steps
- Include code snippets where helpful
- Cite all sources used
- Consider edge cases and potential issues

Begin your analysis and solution:
"""
        return agent_input
    
    def _parse_agent_output(self, result: Dict[str, Any], issue_data: Dict[str, Any]) -> SolverResponse:
        """Parse the ReAct agent's output into a structured SolverResponse."""
        
        try:
            # Get the final output from the agent
            output = result.get('output', '')
            
            # Try to extract structured information from the output
            solution_steps = self._extract_solution_steps(output)
            analysis = self._extract_solution_analysis(output, issue_data)
            testing_strategy = self._extract_testing_strategy(output)
            
            # Get tools used from intermediate steps if available
            tools_used = self._extract_tools_used(result)
            
            return SolverResponse(
                solution_analysis=analysis,
                solution_steps=solution_steps,
                testing_strategy=testing_strategy,
                additional_resources=self._extract_resources(output),
                confidence_score=self._assess_confidence(output, len(solution_steps)),
                tools_used=tools_used
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse agent output: {e}")
            return self._create_manual_solution_from_output(result.get('output', ''), issue_data)
    
    def _extract_solution_steps(self, output: str) -> List[SolutionStep]:
        """Extract solution steps from the agent output."""
        import re
        
        # Look for numbered steps or bullet points
        step_patterns = [
            r'(\d+)\.\s*([^\n]+)\n([^0-9]*?)(?=\d+\.|$)',
            r'[-*]\s*([^\n]+)\n([^-*]*?)(?=[-*]|$)',
            r'Step\s*(\d+):\s*([^\n]+)\n([^S]*?)(?=Step\s*\d+|$)'
        ]
        
        steps = []
        step_number = 1
        
        for pattern in step_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if matches:
                for match in matches:
                    if len(match) >= 2:
                        title = match[1] if len(match) > 2 else match[0]
                        description = match[2] if len(match) > 2 else match[1]
                        
                        steps.append(SolutionStep(
                            step_number=step_number,
                            title=title.strip(),
                            description=description.strip()[:500],
                            rationale=f"Step identified from ReAct agent reasoning",
                            citations=[]
                        ))
                        step_number += 1
                break
        
        # If no structured steps found, create a single step from the output
        if not steps:
            steps.append(SolutionStep(
                step_number=1,
                title="Solution Implementation",
                description=output[:500] + "..." if len(output) > 500 else output,
                rationale="Generated from ReAct agent analysis",
                citations=[]
            ))
        
        return steps
    
    def _extract_solution_analysis(self, output: str, issue_data: Dict[str, Any]) -> SolutionAnalysis:
        """Extract solution analysis from the agent output."""
        
        # Try to find complexity and time estimates in the output
        complexity = "moderate"
        if any(word in output.lower() for word in ["simple", "easy", "straightforward"]):
            complexity = "simple"
        elif any(word in output.lower() for word in ["complex", "difficult", "challenging"]):
            complexity = "complex"
        elif any(word in output.lower() for word in ["very complex", "extremely difficult"]):
            complexity = "very_complex"
        
        # Extract time estimate
        time_estimate = "2-4 hours"
        import re
        time_matches = re.findall(r'(\d+[-\s]*(?:hours?|days?|minutes?))', output.lower())
        if time_matches:
            time_estimate = time_matches[0]
        
        return SolutionAnalysis(
            issue_understanding=f"Issue: {issue_data.get('title', 'Unknown issue')}",
            approach_rationale="Approach determined through ReAct reasoning process",
            complexity_assessment=complexity,
            estimated_time=time_estimate,
            potential_risks=["Implementation complexity", "Testing requirements"],
            prerequisites=["Repository setup", "Understanding of codebase"]
        )
    
    def _extract_testing_strategy(self, output: str) -> str:
        """Extract testing strategy from the agent output."""
        
        test_keywords = ["test", "testing", "unit test", "integration test", "verify"]
        
        # Look for testing-related content
        import re
        for keyword in test_keywords:
            pattern = rf'{keyword}[^.]*\.'
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                return " ".join(matches[:2])  # Return first 2 test-related sentences
        
        return "Implement comprehensive unit tests and integration tests to verify the solution works correctly."
    
    def _extract_tools_used(self, result: Dict[str, Any]) -> List[str]:
        """Extract tools used from the agent result."""
        
        tools_used = []
        
        # Check intermediate steps if available
        if 'intermediate_steps' in result:
            for step in result['intermediate_steps']:
                if isinstance(step, tuple) and len(step) > 0:
                    action = step[0]
                    if hasattr(action, 'tool'):
                        tool_name = action.tool
                        if tool_name not in tools_used:
                            tools_used.append(tool_name)
        
        # Fallback: check output for tool mentions
        output = result.get('output', '').lower()
        for tool in self.tools:
            if tool.name.lower() in output:
                if tool.name not in tools_used:
                    tools_used.append(tool.name)
        
        return tools_used
    
    def _extract_resources(self, output: str) -> List[str]:
        """Extract additional resources from the agent output."""
        
        import re
        
        # Look for URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', output)
        
        # Look for references to documentation
        doc_patterns = [
            r'documentation[^.]*\.',
            r'official\s+guide[^.]*\.',
            r'reference[^.]*\.'
        ]
        
        resources = urls[:3]  # Limit URLs
        
        for pattern in doc_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            resources.extend(matches[:2])
        
        return resources[:5]  # Limit total resources
    
    def _assess_confidence(self, output: str, num_steps: int) -> float:
        """Assess confidence based on output quality and completeness."""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on output length and detail
        if len(output) > 500:
            confidence += 0.1
        if len(output) > 1000:
            confidence += 0.1
        
        # Increase confidence based on number of steps
        if num_steps >= 3:
            confidence += 0.1
        if num_steps >= 5:
            confidence += 0.1
        
        # Check for quality indicators
        quality_indicators = ["code", "implementation", "test", "example", "step"]
        for indicator in quality_indicators:
            if indicator.lower() in output.lower():
                confidence += 0.05
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _create_manual_solution_from_output(self, output: str, issue_data: Dict[str, Any]) -> SolverResponse:
        """Create a solution response from raw agent output when parsing fails."""
        
        return SolverResponse(
            solution_analysis=SolutionAnalysis(
                issue_understanding=f"Issue: {issue_data.get('title', 'Unknown issue')}",
                approach_rationale="Analysis completed through ReAct reasoning",
                complexity_assessment="moderate",
                estimated_time="2-4 hours",
                potential_risks=["Implementation complexity"],
                prerequisites=["Repository setup"]
            ),
            solution_steps=[
                SolutionStep(
                    step_number=1,
                    title="Implementation based on ReAct analysis",
                    description=output[:800] + "..." if len(output) > 800 else output,
                    rationale="Generated through ReAct reasoning process",
                    citations=[]
                )
            ],
            testing_strategy="Test the implementation thoroughly with appropriate test cases",
            additional_resources=[],
            confidence_score=0.6,
            tools_used=[]
        )
    
    def _create_solution_plan(self, issue_data: Dict[str, Any], repo_analysis: Dict[str, Any], entry_point_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create an initial solution plan based on available context."""
        logger.debug("Creating initial solution plan")
        
        # Extract key information
        issue_type = entry_point_analysis.get('issue_analysis', {}).get('issue_type', 'unknown')
        complexity = entry_point_analysis.get('issue_analysis', {}).get('complexity', 'medium')
        technologies = entry_point_analysis.get('issue_analysis', {}).get('technologies_involved', [])
        
        plan = {
            "issue_type": issue_type,
            "complexity": complexity,
            "technologies": technologies,
            "search_queries": self._generate_search_queries(issue_data, technologies),
            "files_to_examine": entry_point_analysis.get('issue_analysis', {}).get('related_files', []),
            "research_needed": self._determine_research_needs(issue_type, complexity, technologies)
        }
        
        logger.debug(f"Solution plan created: {plan}")
        return plan
    
    def _generate_search_queries(self, issue_data: Dict[str, Any], technologies: List[str]) -> List[str]:
        """Generate search queries based on issue and technologies."""
        title = issue_data.get('title', '')
        body = issue_data.get('body', '')
        labels = issue_data.get('labels', [])
        
        queries = []
        
        # Basic issue-based queries
        if title:
            queries.append(f"{title} solution")
            for tech in technologies[:2]:  # Limit to top 2 technologies
                queries.append(f"{title} {tech} implementation")
        
        # Label-based queries
        for label in labels[:2]:  # Limit to top 2 labels
            if isinstance(label, dict):
                label_name = label.get('name', '')
            else:
                label_name = str(label)
            
            if label_name and len(label_name) > 2:
                queries.append(f"{label_name} best practices")
        
        # Technology-specific queries
        for tech in technologies[:3]:  # Limit to top 3 technologies
            queries.append(f"{tech} common issues solutions")
        
        return queries[:5]  # Limit total queries
    
    def _determine_research_needs(self, issue_type: str, complexity: str, technologies: List[str]) -> Dict[str, bool]:
        """Determine what type of research is needed."""
        needs = {
            "general_search": True,
            "technical_search": len(technologies) > 0,
            "documentation_search": complexity in ['high', 'very_high'],
            "code_examples": issue_type in ['feature', 'enhancement', 'bug'],
            "best_practices": complexity in ['medium', 'high', 'very_high']
        }
        return needs
    
    def _conduct_research(self, issue_data: Dict[str, Any], repo_analysis: Dict[str, Any], solution_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Conduct research using available search tools."""
        logger.debug("Conducting research using search tools")
        
        research_results = []
        search_queries = solution_plan.get('search_queries', [])
        research_needs = solution_plan.get('research_needed', {})
        
        # Use Serper for general searches
        if research_needs.get('general_search', False):
            for query in search_queries[:2]:  # Limit to 2 queries per tool
                try:
                    logger.debug(f"Searching with Serper: {query}")
                    result = serper_search_tool.invoke({"query": query})
                    research_results.append({
                        "tool": "serper",
                        "query": query,
                        "result": result,
                        "type": "general"
                    })
                except Exception as e:
                    logger.warning(f"Serper search failed for '{query}': {e}")
        
        # Use Tavily for technical searches
        if research_needs.get('technical_search', False):
            for query in search_queries[2:4]:  # Different queries for technical search
                try:
                    logger.debug(f"Searching with Tavily: {query}")
                    result = tavily_search_tool.invoke({"query": query})
                    research_results.append({
                        "tool": "tavily",
                        "query": query,
                        "result": result,
                        "type": "technical"
                    })
                except Exception as e:
                    logger.warning(f"Tavily search failed for '{query}': {e}")
        
        logger.debug(f"Research completed with {len(research_results)} results")
        return research_results
    
    def _fetch_relevant_files(self, entry_point_analysis: Dict[str, Any], research_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch relevant file contents using the file content tool."""
        logger.debug("Fetching relevant file contents")
        
        file_contents = []
        
        # Get files from entry point analysis
        related_files = entry_point_analysis.get('issue_analysis', {}).get('related_files', [])
        entry_points = entry_point_analysis.get('entry_points', [])
        
        # Prioritize files from entry points that are related to the issue
        priority_files = []
        for ep in entry_points:
            if ep.get('related_to_issue', False):
                priority_files.append(ep.get('file_path', ''))
        
        # Add related files
        priority_files.extend(related_files)
        
        # Fetch contents for priority files (limit to avoid too many requests)
        for file_path in priority_files[:3]:  # Limit to 3 files
            if file_path and file_path.strip():
                try:
                    logger.debug(f"Fetching content for: {file_path}")
                    # Note: This assumes we have repo URL available in the context
                    # In a real implementation, you'd pass the repo URL
                    result = get_file_content_tool.invoke({
                        "repo_url": "placeholder_repo_url",  # This should be passed from context
                        "file_path": file_path
                    })
                    file_contents.append({
                        "file_path": file_path,
                        "content": result,
                        "source": "entry_point_analysis"
                    })
                except Exception as e:
                    logger.warning(f"Failed to fetch content for '{file_path}': {e}")
        
        logger.debug(f"Fetched contents for {len(file_contents)} files")
        return file_contents
    
    def _generate_solution(self, 
                          issue_data: Dict[str, Any], 
                          repo_analysis: Dict[str, Any], 
                          entry_point_analysis: Dict[str, Any],
                          research_results: List[Dict[str, Any]], 
                          file_contents: List[Dict[str, Any]], 
                          solution_plan: Dict[str, Any]) -> SolverResponse:
        """Generate the comprehensive solution using LLM."""
        logger.debug("Generating comprehensive solution using LLM")
        
        # Create the prompt with all gathered information
        prompt = self._create_solution_prompt(
            issue_data, repo_analysis, entry_point_analysis,
            research_results, file_contents, solution_plan
        )
        
        try:
            # Get response from LLM
            response = self.model.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Handle different content types
            if isinstance(content, list):
                content = str(content)
            elif not isinstance(content, str):
                content = str(content)
            
            # Parse the response into structured format
            solution = self._parse_solution_response(content, research_results, file_contents)
            
            return solution
            
        except Exception as e:
            logger.error(f"Failed to generate solution with LLM: {e}")
            return self._create_fallback_solution(issue_data, str(e))
    
    def _create_solution_prompt(self, issue_data, repo_analysis, entry_point_analysis, research_results, file_contents, solution_plan):
        """Create the prompt for solution generation."""
        
        # Format research results
        research_summary = "\n".join([
            f"- {res['tool'].upper()}: {res['query']} -> {str(res['result'])[:200]}..."
            for res in research_results
        ])
        
        # Format file contents
        files_summary = "\n".join([
            f"- {fc['file_path']}: {str(fc['content'])[:300]}..."
            for fc in file_contents
        ])
        
        prompt = f"""
        You are an expert software engineer tasked with solving a GitHub issue. Based on the provided context and research, create a comprehensive step-by-step solution.

        **ISSUE INFORMATION:**
        - Title: {issue_data.get('title', 'No title')}
        - Body: {issue_data.get('body', 'No description')[:1000]}...
        - Labels: {issue_data.get('labels', [])}
        - Number: #{issue_data.get('number', 'N/A')}

        **REPOSITORY CONTEXT:**
        - Project Type: {entry_point_analysis.get('project_summary', {}).get('project_type', 'Unknown')}
        - Technologies: {entry_point_analysis.get('project_summary', {}).get('main_technologies', [])}
        - Architecture: {entry_point_analysis.get('project_summary', {}).get('architecture_pattern', 'Unknown')}

        **RESEARCH RESULTS:**
        {research_summary}

        **RELEVANT FILE CONTENTS:**
        {files_summary}

        **REQUIREMENTS:**
        1. Provide a detailed solution analysis
        2. Create step-by-step implementation instructions
        3. Include code snippets where applicable
        4. Cite sources from the research results
        5. Assess complexity and risks
        6. Provide testing strategy

        **RESPONSE FORMAT:**
        Return a JSON object with the following structure:
        {{
            "solution_analysis": {{
                "issue_understanding": "Clear summary of the issue",
                "approach_rationale": "Why this approach was chosen",
                "complexity_assessment": "simple|moderate|complex|very_complex",
                "estimated_time": "Time estimate",
                "potential_risks": ["risk1", "risk2"],
                "prerequisites": ["prereq1", "prereq2"]
            }},
            "solution_steps": [
                {{
                    "step_number": 1,
                    "title": "Step title",
                    "description": "Detailed description",
                    "file_path": "optional/file/path",
                    "code_snippet": "optional code",
                    "commands": ["optional commands"],
                    "rationale": "Why this step is needed",
                    "citations": ["source1", "source2"]
                }}
            ],
            "testing_strategy": "How to test the solution",
            "additional_resources": ["resource1", "resource2"],
            "confidence_score": 0.8,
            "tools_used": ["serper", "tavily", "get_file_content"]
        }}

        Provide a comprehensive, actionable solution with proper citations.
        """
        
        return prompt
    
    def _parse_solution_response(self, content: str, research_results: List[Dict], file_contents: List[Dict]) -> SolverResponse:
        """Parse the LLM response into a structured SolverResponse."""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                # Clean up JSON
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                data = json.loads(json_str)
                
                # Add tools used based on what was actually used
                tools_used = []
                for res in research_results:
                    if res['tool'] not in tools_used:
                        tools_used.append(res['tool'])
                if file_contents:
                    tools_used.append('get_file_content')
                
                data['tools_used'] = tools_used
                
                return SolverResponse(**data)
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.warning(f"Failed to parse structured response: {e}")
            return self._create_manual_solution(content, research_results, file_contents)
    
    def _create_manual_solution(self, content: str, research_results: List[Dict], file_contents: List[Dict]) -> SolverResponse:
        """Create a solution response manually when JSON parsing fails."""
        
        tools_used = list(set([res['tool'] for res in research_results]))
        if file_contents:
            tools_used.append('get_file_content')
        
        return SolverResponse(
            solution_analysis=SolutionAnalysis(
                issue_understanding="Analysis completed based on available context",
                approach_rationale="Approach determined from research and file analysis",
                complexity_assessment="moderate",
                estimated_time="1-3 hours",
                potential_risks=["Implementation complexity", "Testing requirements"],
                prerequisites=["Repository setup", "Understanding of codebase"]
            ),
            solution_steps=[
                SolutionStep(
                    step_number=1,
                    title="Analyze the solution approach",
                    description=content[:500] + "...",
                    rationale="Based on research and analysis",
                    citations=[res['query'] for res in research_results]
                )
            ],
            testing_strategy="Test the implementation thoroughly with unit and integration tests",
            additional_resources=[res['query'] for res in research_results],
            confidence_score=0.6,
            tools_used=tools_used
        )
    
    def _create_fallback_solution(self, issue_data: Dict[str, Any], error_msg: str) -> SolverResponse:
        """Create a fallback solution when the main process fails."""
        logger.info("Creating fallback solution due to processing failure")
        
        return SolverResponse(
            solution_analysis=SolutionAnalysis(
                issue_understanding=f"Issue: {issue_data.get('title', 'Unknown issue')}",
                approach_rationale="Fallback analysis due to processing error",
                complexity_assessment="moderate",
                estimated_time="Manual analysis required",
                potential_risks=["Requires manual investigation"],
                prerequisites=["Manual code review", "Understanding of issue context"]
            ),
            solution_steps=[
                SolutionStep(
                    step_number=1,
                    title="Manual Investigation Required",
                    description=f"Automated analysis failed: {error_msg}. Manual investigation needed.",
                    rationale="Automated tools were unable to complete analysis",
                    citations=[]
                )
            ],
            testing_strategy="Manual testing required after manual implementation",
            additional_resources=[],
            confidence_score=0.2,
            tools_used=[]
        )
    
    def process_solution(self, issue_data: Dict[str, Any], repo_analysis: Dict[str, Any], entry_point_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the solution and return as dictionary.
        
        Args:
            issue_data (Dict[str, Any]): Issue information
            repo_analysis (Dict[str, Any]): Repository analysis
            entry_point_analysis (Dict[str, Any]): Entry point analysis
            
        Returns:
            Dict[str, Any]: Solution response as dictionary
        """
        try:
            solution = self.solve_issue(issue_data, repo_analysis, entry_point_analysis)
            logger.success(f"Solution processed successfully with {len(solution.solution_steps)} steps")
            return solution.model_dump()
        except Exception as e:
            logger.error(f"Error processing solution: {e}")
            return {
                "error": str(e),
                "solution_type": "error"
            }

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing SolverAgent")
    
    # Mock data for testing
    sample_issue = {
        "number": 123,
        "title": "Add user authentication",
        "body": "We need to implement JWT-based authentication for the API",
        "labels": ["enhancement", "backend", "security"]
    }
    
    sample_repo_analysis = {
        "basic_info": {"name": "test-api", "description": "Test API project"},
        "technologies": {"technologies": {"Python": {"percentage": 80}, "JavaScript": {"percentage": 20}}}
    }
    
    sample_entry_point_analysis = {
        "issue_analysis": {
            "issue_type": "enhancement",
            "complexity": "high",
            "technologies_involved": ["Python", "Flask", "JWT"],
            "related_files": ["app.py", "auth.py", "models.py"]
        },
        "entry_points": [
            {"file_path": "app.py", "related_to_issue": True},
            {"file_path": "auth.py", "related_to_issue": True}
        ]
    }
    
    agent = SolverAgent()
    result = agent.process_solution(sample_issue, sample_repo_analysis, sample_entry_point_analysis)
    
    print("🔧 Solver Agent Result:")
    print(f"📊 Solution Steps: {len(result.get('solution_steps', []))}")
    print(f"🎯 Complexity: {result.get('solution_analysis', {}).get('complexity_assessment', 'N/A')}")
    print(f"📈 Confidence: {result.get('confidence_score', 0)}")
    print(f"🛠️ Tools Used: {result.get('tools_used', [])}")
    
    logger.info("SolverAgent testing completed")
