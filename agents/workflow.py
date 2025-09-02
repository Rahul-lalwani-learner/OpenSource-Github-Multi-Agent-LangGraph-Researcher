"""
GitHub Issue Multi-Agent Workflow

This module implements a comprehensive LangGraph workflow that orchestrates multiple agents
to analyze GitHub repositories and solve issues through a structured pipeline:

1. Query Analysis -> Repository Search/Selection -> Human Input -> Issue Selection
2. Entry Point Analysis -> Solution Generation (Solver + Evaluator Loop) -> Final Response

The workflow uses LangGraph state management and human-in-the-loop interactions.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated, cast
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from utils.logging import logger
import json

# Import all agents and functions
from agents.query_agent import QueryAgent, QueryDecision
from agents.entry_point_issue_agent import EntryPointAndIssueBreakdownAgent
from agents.solver_agent import SolverAgent
from agents.evaluator_agent import EvaluatorAgent
from core.functions.get_repo_info import get_all_repo_details
from core.functions.graphql_search_repo import search_github_repositories, pygithub_search_repositories

class WorkflowState(TypedDict):
    """
    LangGraph state that maintains all workflow information across nodes.
    """
    # User input and query analysis
    user_query: str
    query_decision: Optional[Dict[str, Any]]
    
    # Repository search and selection
    search_results: Optional[List[Dict[str, Any]]]
    selected_repo_url: Optional[str]
    
    # Repository information and issues
    repo_data: Optional[Dict[str, Any]]
    available_issues: Optional[List[Dict[str, Any]]]
    selected_issue: Optional[Dict[str, Any]]
    
    # Entry point analysis
    entry_point_analysis: Optional[Dict[str, Any]]
    
    # Solution generation loop
    solver_solution: Optional[Dict[str, Any]]
    evaluator_feedback: Optional[Dict[str, Any]]
    solution_iterations: int
    max_iterations: int
    solution_approved: bool
    
    # Final response and metadata
    final_response: Optional[Dict[str, Any]]
    workflow_status: str
    error_message: Optional[str]
    human_inputs: List[Dict[str, Any]]

class GitHubIssueWorkflow:
    """
    Main workflow orchestrator that coordinates all agents using LangGraph.
    """
    
    def __init__(self):
        """Initialize the workflow with all agents and state graph."""
        logger.info("Initializing GitHubIssueWorkflow")
        
        # Initialize agents
        self.query_agent = QueryAgent()
        self.entry_point_agent = EntryPointAndIssueBreakdownAgent()
        self.solver_agent = SolverAgent()
        self.evaluator_agent = EvaluatorAgent()
        
        # Create the state graph
        self.workflow = self._create_workflow()
        self.memory = MemorySaver()
        # Compile workflow with checkpointer and interrupts for human-in-the-loop
        self.app = self.workflow.compile(
            checkpointer=self.memory,
            interrupt_before=["human_select_repo", "human_select_issue"]
        )
        
        logger.success("GitHubIssueWorkflow initialized successfully")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with all nodes and edges."""
        logger.info("Creating workflow graph")
        
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("init_agents", self._init_agents_node)
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("search_repos", self._search_repos_node)
        workflow.add_node("human_select_repo", self._human_select_repo_node)
        workflow.add_node("get_repo_info", self._get_repo_info_node)
        workflow.add_node("human_select_issue", self._human_select_issue_node)
        workflow.add_node("analyze_entry_points", self._analyze_entry_points_node)
        workflow.add_node("solve_issue", self._solve_issue_node)
        workflow.add_node("evaluate_solution", self._evaluate_solution_node)
        workflow.add_node("finalize_response", self._finalize_response_node)
        workflow.add_node("end_workflow", self._end_workflow_node)
        
        # Set entry point
        workflow.set_entry_point("init_agents")
        
        # Add edges based on the flow you described
        workflow.add_edge("init_agents", "analyze_query")
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_after_query_analysis,
            {
                "search_repos": "search_repos",
                "get_repo_info": "get_repo_info", 
                "end": "end_workflow"
            }
        )
        workflow.add_edge("search_repos", "human_select_repo")
        workflow.add_edge("human_select_repo", "get_repo_info")
        workflow.add_edge("get_repo_info", "human_select_issue")
        workflow.add_conditional_edges(
            "human_select_issue",
            self._route_after_issue_selection,
            {
                "analyze_entry_points": "analyze_entry_points",
                "end": "end_workflow"
            }
        )
        workflow.add_edge("analyze_entry_points", "solve_issue")
        workflow.add_edge("solve_issue", "evaluate_solution")
        workflow.add_conditional_edges(
            "evaluate_solution",
            self._route_after_evaluation,
            {
                "solve_issue": "solve_issue",  # Loop back for iterations
                "finalize_response": "finalize_response"
            }
        )
        workflow.add_edge("finalize_response", "end_workflow")
        workflow.add_edge("end_workflow", END)
        
        logger.success("Workflow graph created successfully")
        return workflow
    
    def _init_agents_node(self, state: WorkflowState) -> WorkflowState:
        """Initialize all agents in the state."""
        logger.info("Initializing agents in workflow state")
        
        # Don't store agent instances in state - they cause serialization issues
        # Agents are available through self.* properties
        state["solution_iterations"] = 0
        state["max_iterations"] = 3
        state["solution_approved"] = False
        state["workflow_status"] = "initialized"
        state["human_inputs"] = []
        
        logger.success("Agents initialized in workflow state")
        return state
    
    def _analyze_query_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze user query to determine the action type."""
        logger.info(f"Analyzing user query: {state['user_query']}")
        
        try:
            decision = self.query_agent.analyze_query(state["user_query"])
            # Convert Pydantic model to dict
            state["query_decision"] = decision.model_dump() if hasattr(decision, 'model_dump') else decision.__dict__
            state["workflow_status"] = "query_analyzed"
            
            action_type = getattr(decision, 'action_type', 'unknown')
            logger.success(f"Query analysis complete. Action: {action_type}")
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
        
        return state
    
    def _search_repos_node(self, state: WorkflowState) -> WorkflowState:
        """Search for repositories based on query decision."""
        logger.info("Searching for repositories")
        
        try:
            decision = state["query_decision"]
            if decision is None:
                raise ValueError("Query decision not available")
            
            search_query = decision.get("search_query")
            search_method = decision.get("search_method", "pygithub")
            top_k = decision.get("top_k", 5)
            
            if not search_query:
                raise ValueError("Search query is required")
            
            if search_method == "graphql":
                raw_results = search_github_repositories(str(search_query), int(top_k))
                
                # Handle type checking for GraphQL results
                results = []
                if isinstance(raw_results, dict):
                    # Check for error in GraphQL result
                    if "error" in raw_results:
                        raise ValueError(f"GraphQL search error: {raw_results['error']}")
                    
                    # Extract repositories from GraphQL result structure
                    search_data = raw_results.get("search", {})
                    if isinstance(search_data, dict):
                        edges = search_data.get("edges", [])
                        for edge in edges:
                            if isinstance(edge, dict):
                                repo_node = edge.get("node", {})
                                if repo_node:
                                    # Flatten the GraphQL response to match expected format
                                    repo_dict = {
                                        "id": repo_node.get("id"),
                                        "name": repo_node.get("name"),
                                        "full_name": repo_node.get("nameWithOwner"),
                                        "url": repo_node.get("url"),
                                        "description": repo_node.get("description"),
                                        "created_at": repo_node.get("createdAt"),
                                        "updated_at": repo_node.get("updatedAt"),
                                        "pushed_at": repo_node.get("pushedAt"),
                                        "stargazer_count": repo_node.get("stargazerCount", 0),
                                        "fork_count": repo_node.get("forkCount", 0),
                                        "watchers_count": repo_node.get("watchers", {}).get("totalCount", 0),
                                        "open_issues_count": repo_node.get("issues", {}).get("totalCount", 0),
                                        "open_pulls_count": repo_node.get("pullRequests", {}).get("totalCount", 0),
                                        "language": repo_node.get("primaryLanguage", {}).get("name") if repo_node.get("primaryLanguage") else None,
                                        "license": repo_node.get("licenseInfo", {}).get("name") if repo_node.get("licenseInfo") else None,
                                        "is_fork": repo_node.get("isFork", False),
                                        "is_archived": repo_node.get("isArchived", False),
                                        "is_private": repo_node.get("isPrivate", False),
                                        "owner": {
                                            "login": repo_node.get("owner", {}).get("login"),
                                            "url": repo_node.get("owner", {}).get("url"),
                                            "type": repo_node.get("owner", {}).get("__typename")
                                        }
                                    }
                                    results.append(repo_dict)
                else:
                    logger.warning(f"Unexpected GraphQL result type: {type(raw_results)}")
                    results = []
            else:
                results = pygithub_search_repositories(str(search_query), int(top_k))
            
            # Ensure results is a list of dictionaries
            if isinstance(results, list):
                state["search_results"] = results
            else:
                state["search_results"] = []
            state["workflow_status"] = "repos_searched"
            
            logger.success(f"Repository search complete. Found {len(state['search_results'])} repositories")
        except Exception as e:
            logger.error(f"Error in repository search: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
        
        return state
    
    def _human_select_repo_node(self, state: WorkflowState) -> WorkflowState:
        """Pause for human repository selection (interrupt node)."""
        logger.info("Pausing for human repository selection. Present search_results externally.")
        state["workflow_status"] = "awaiting_repo_selection"
        # Do not block or prompt; just return state
        # The workflow will be interrupted before this node due to the interrupt configuration
        return state
    
    def _get_repo_info_node(self, state: WorkflowState) -> WorkflowState:
        """Get detailed repository information and issues."""
        logger.info(f"Getting repository information for: {state.get('selected_repo_url')}")
        
        try:
            repo_url = state.get("selected_repo_url")
            if not repo_url:
                # If coming from direct repo info request
                decision = state["query_decision"]
                if decision is not None:
                    repo_url = decision.get("repo_url")
                    state["selected_repo_url"] = repo_url
            
            if not repo_url:
                raise ValueError("Repository URL is required")
            
            repo_data = get_all_repo_details(str(repo_url))
            state["repo_data"] = repo_data
            
            # Extract available issues from both open and closed
            open_issues = repo_data.get("open_issues", {}).get("issues", [])
            closed_issues = repo_data.get("closed_issues", {}).get("issues", [])
            all_issues = open_issues + closed_issues
            state["available_issues"] = all_issues
            state["workflow_status"] = "repo_info_fetched"
            
            logger.success(f"Repository information fetched. Found {len(all_issues)} issues ({len(open_issues)} open, {len(closed_issues)} closed)")
        except Exception as e:
            logger.error(f"Error fetching repository info: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
        
        return state
    
    def _human_select_issue_node(self, state: WorkflowState) -> WorkflowState:
        """Pause for human issue selection (interrupt node)."""
        logger.info("Pausing for human issue selection. Present available_issues externally.")
        state["workflow_status"] = "awaiting_issue_selection"
        # Do not block or prompt; just return state
        # The workflow will be interrupted before this node due to the interrupt configuration
        return state
    
    def _analyze_entry_points_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze repository entry points and issue breakdown."""
        logger.info("Analyzing entry points and issue breakdown")
        
        try:
            repo_data = state["repo_data"]
            selected_issue = state["selected_issue"]
            
            if repo_data is None:
                raise ValueError("Repository data not available")
            if selected_issue is None:
                raise ValueError("Selected issue not available")
            
            analysis = self.entry_point_agent.process_analysis(repo_data, selected_issue)
            state["entry_point_analysis"] = analysis
            state["workflow_status"] = "entry_points_analyzed"
            
            logger.success("Entry point analysis complete")
        except Exception as e:
            logger.error(f"Error in entry point analysis: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
        
        return state
    
    def _solve_issue_node(self, state: WorkflowState) -> WorkflowState:
        """Generate solution using the Solver Agent."""
        logger.info(f"Generating solution (iteration {state['solution_iterations'] + 1})")
        
        try:
            issue_data = state["selected_issue"]
            repo_analysis = state["repo_data"]
            entry_point_analysis = state["entry_point_analysis"]
            
            if issue_data is None:
                raise ValueError("Issue data not available")
            if repo_analysis is None:
                raise ValueError("Repository analysis not available")
            if entry_point_analysis is None:
                raise ValueError("Entry point analysis not available")
            
            # Use self.solver_agent instead of storing in state
            solution = self.solver_agent.solve_issue(issue_data, repo_analysis, entry_point_analysis)
            # Convert Pydantic model to dict
            state["solver_solution"] = solution.model_dump() if hasattr(solution, 'model_dump') else solution.__dict__
            state["solution_iterations"] += 1
            state["workflow_status"] = "solution_generated"
            
            logger.success(f"Solution generated (iteration {state['solution_iterations']})")
        except Exception as e:
            logger.error(f"Error in solution generation: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
        
        return state
    
    def _evaluate_solution_node(self, state: WorkflowState) -> WorkflowState:
        """Evaluate the solution using the Evaluator Agent."""
        logger.info("Evaluating solution quality and methodology")
        
        try:
            solver_solution = state["solver_solution"]
            issue_data = state["selected_issue"]
            repo_data = state["repo_data"]
            
            if solver_solution is None:
                raise ValueError("Solver solution not available")
            if issue_data is None:
                raise ValueError("Issue data not available")
            if repo_data is None:
                raise ValueError("Repository data not available")
            
            evaluation = self.evaluator_agent.evaluate_solution(solver_solution, issue_data, repo_data)
            # Convert Pydantic model to dict
            evaluation_dict = evaluation.model_dump() if hasattr(evaluation, 'model_dump') else evaluation.__dict__
            state["evaluator_feedback"] = evaluation_dict
            
            # Check if solution is approved
            overall_assessment = evaluation_dict.get("overall_assessment", {})
            overall_score = overall_assessment.get("overall_score", 0.0)
            approval_threshold = 0.75  # 75% approval threshold
            
            if overall_score >= approval_threshold or state["solution_iterations"] >= state["max_iterations"]:
                state["solution_approved"] = True
                state["workflow_status"] = "solution_approved"
                logger.success(f"Solution approved with score: {overall_score}")
            else:
                state["solution_approved"] = False
                state["workflow_status"] = "solution_needs_improvement"
                logger.info(f"Solution needs improvement. Score: {overall_score}")
            
        except Exception as e:
            logger.error(f"Error in solution evaluation: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
        
        return state
    
    def _finalize_response_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize the complete response with all analysis and solution."""
        logger.info("Finalizing complete response")
        
        try:
            repo_data = state["repo_data"] or {}
            evaluator_feedback = state["evaluator_feedback"] or {}
            
            final_response = {
                "user_query": state["user_query"],
                "selected_repository": {
                    "url": state["selected_repo_url"],
                    "basic_info": repo_data.get("basic_info", {})
                },
                "selected_issue": state["selected_issue"],
                "entry_point_analysis": state["entry_point_analysis"],
                "final_solution": state["solver_solution"],
                "evaluation_feedback": state["evaluator_feedback"],
                "solution_iterations": state["solution_iterations"],
                "solution_approved": state["solution_approved"],
                "workflow_metadata": {
                    "total_iterations": state["solution_iterations"],
                    "max_iterations": state["max_iterations"],
                    "approval_score": evaluator_feedback.get("overall_assessment", {}).get("overall_score", 0.0)
                }
            }
            
            state["final_response"] = final_response
            state["workflow_status"] = "completed"
            
            logger.success("Workflow completed successfully")
        except Exception as e:
            logger.error(f"Error finalizing response: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
        
        return state
    
    def _end_workflow_node(self, state: WorkflowState) -> WorkflowState:
        """Final node to end the workflow."""
        logger.info(f"Workflow ended with status: {state['workflow_status']}")
        return state
    
    def _route_after_query_analysis(self, state: WorkflowState) -> str:
        """Route after query analysis based on decision."""
        decision = state.get("query_decision")
        if decision is None:
            return "end"
        
        action_type = decision.get("action_type")
        
        if action_type == "search_repos":
            return "search_repos"
        elif action_type == "get_repo_info":
            return "get_repo_info"
        else:
            return "end"
    
    def _route_after_issue_selection(self, state: WorkflowState) -> str:
        """Route after issue selection."""
        selected_issue = state.get("selected_issue")
        available_issues = state.get("available_issues", [])
        
        if selected_issue and available_issues:
            return "analyze_entry_points"
        else:
            return "end"
    
    def _route_after_evaluation(self, state: WorkflowState) -> str:
        """Route after solution evaluation."""
        if state.get("solution_approved") or state.get("solution_iterations", 0) >= state.get("max_iterations", 3):
            return "finalize_response"
        else:
            return "solve_issue"
    
    def run_workflow(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Run the complete workflow with human-in-the-loop interactions.
        
        Args:
            user_query (str): The user's input query
            thread_id (str): Thread ID for conversation persistence
            
        Returns:
            Dict[str, Any]: Complete workflow result
        """
        logger.info(f"Starting workflow for query: {user_query}")
        
        initial_state = {
            "user_query": user_query,
            "query_decision": None,
            "search_results": None,
            "selected_repo_url": None,
            "repo_data": None,
            "available_issues": None,
            "selected_issue": None,
            "entry_point_analysis": None,
            "solver_solution": None,
            "evaluator_feedback": None,
            "solution_iterations": 0,
            "max_iterations": 3,
            "solution_approved": False,
            "final_response": None,
            "workflow_status": "starting",
            "error_message": None,
            "human_inputs": []
        }
        
        config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
        
        try:
            # Run the workflow
            logger.info("Invoking LangGraph workflow...")
            final_state = self.app.invoke(initial_state, config)
            
            logger.info(f"Workflow invoke completed. Status: {final_state.get('workflow_status')}")
            logger.success("Workflow completed successfully")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "error": str(e),
                "workflow_status": "failed",
                "user_query": user_query
            }
    
    def handle_human_input(self, thread_id: str, input_type: str, user_input: Any) -> Dict[str, Any]:
        """
        Handle human input during workflow execution.
        
        Args:
            thread_id (str): Thread ID for the conversation
            input_type (str): Type of input ('repo_selection' or 'issue_selection')
            user_input (Any): The human input (repo URL or issue number)
            
        Returns:
            Dict[str, Any]: Updated state after human input
        """
        logger.info(f"Handling human input: {input_type}")
        
        config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
        
        try:
            # Get current state
            current_state = self.app.get_state(config)
            state_values = current_state.values
            
            # Update state based on input type
            if input_type == "repo_selection":
                if isinstance(user_input, dict) and "selected_repo_url" in user_input:
                    # Direct dictionary input with URL
                    state_values["selected_repo_url"] = user_input["selected_repo_url"]
                elif isinstance(user_input, int) and state_values.get("search_results"):
                    # User selected by index
                    selected_repo = state_values["search_results"][user_input]
                    state_values["selected_repo_url"] = selected_repo.get("url")
                elif isinstance(user_input, str):
                    # User provided direct URL
                    state_values["selected_repo_url"] = user_input
                
                state_values["human_inputs"].append({
                    "type": "repo_selection",
                    "input": user_input,
                    "selected_url": state_values.get("selected_repo_url")
                })
            
            elif input_type == "issue_selection":
                if isinstance(user_input, int) and state_values.get("available_issues"):
                    # User selected by index
                    selected_issue = state_values["available_issues"][user_input]
                    state_values["selected_issue"] = selected_issue
                
                state_values["human_inputs"].append({
                    "type": "issue_selection", 
                    "input": user_input,
                    "selected_issue": state_values.get("selected_issue")
                })
            
            # Continue workflow execution from interrupt point
            # Update the state first
            self.app.update_state(config, state_values)
            
            # Then stream the continuation
            final_result = state_values  # Default fallback
            for chunk in self.app.stream(None, config):
                if chunk:  # Update with each non-empty chunk
                    final_result = chunk
            
            logger.success(f"Human input processed: {input_type}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error handling human input: {e}")
            return {"error": str(e)}
    
    def get_workflow_status(self, thread_id: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow execution.
        
        Args:
            thread_id (str): Thread ID for the conversation
            
        Returns:
            Dict[str, Any]: Current workflow status and relevant information
        """
        config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
        
        try:
            current_state = self.app.get_state(config)
            state_values = current_state.values
            
            status_info = {
                "workflow_status": state_values.get("workflow_status", "unknown"),
                "current_step": current_state.next,
                "search_results_count": len(state_values.get("search_results") or []),
                "available_issues_count": len(state_values.get("available_issues") or []),
                "solution_iterations": state_values.get("solution_iterations", 0),
                "solution_approved": state_values.get("solution_approved", False),
                "error_message": state_values.get("error_message"),
                "human_inputs": state_values.get("human_inputs", [])
            }
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing GitHubIssueWorkflow")
    
    # Initialize workflow
    workflow = GitHubIssueWorkflow()
    
    # Test query
    test_query = "Find repositories for machine learning in Python"
    thread_id = "test_thread_1"
    
    # Run workflow
    result = workflow.run_workflow(test_query, thread_id)
    
    print("🔄 Workflow Result:")
    print(f"Status: {result.get('workflow_status')}")
    print(f"Query: {result.get('user_query')}")
    if result.get('search_results'):
        print(f"Found {len(result['search_results'])} repositories")
    
    logger.info("GitHubIssueWorkflow testing completed")
