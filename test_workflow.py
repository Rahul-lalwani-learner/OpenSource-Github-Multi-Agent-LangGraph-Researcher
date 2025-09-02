"""
Simple test script for the GitHub Issue Multi-Agent Workflow

This script tests the basic functionality of the workflow without the CLI interface.
"""

from agents.workflow import GitHubIssueWorkflow
from utils.logging import logger
import json

def test_workflow():
    """Test the basic workflow functionality."""
    logger.info("Starting GitHub Issue Workflow Test")
    
    try:
        # Initialize workflow
        workflow = GitHubIssueWorkflow()
        logger.success("Workflow initialized successfully")
        
        # Test query
        # test_query = "https://github.com/kamihi-org/kamihi"
        test_query = "https://github.com/calcom/cal.com"
        thread_id = "test_thread"
        
        logger.info(f"Testing with query: {test_query}")
        
        # Run workflow (this should stop at human input points)
        result = workflow.run_workflow(test_query, thread_id)
        
        # Check if workflow is interrupted and needs human input
        logger.info(f"Workflow result status: {result.get('workflow_status')}")
        
        # Get current workflow state to check what step we're at
        current_status = workflow.get_workflow_status(thread_id)
        current_step = current_status.get('current_step')
        logger.info(f"Current workflow step: {current_step}")
        
        # Handle different types of human input needed
        if result.get('search_results') and len(result['search_results']) > 0:
            # Case 1: We have search results and need to select a repo
            logger.info("Search results found, simulating human repository selection...")
            selected_repo = result['search_results'][0]
            logger.info(f"Selecting repository: {selected_repo.get('full_name', selected_repo.get('name'))}")
            
            # Continue workflow with human input
            continuation_result = workflow.handle_human_input(
                thread_id, 
                "repo_selection", 
                {"selected_repo_url": selected_repo.get('url')}
            )
            
            logger.info(f"After repo selection, status: {continuation_result.get('workflow_status') if continuation_result else 'None'}")
            
            # Update result with continuation if it's not None
            if continuation_result:
                result.update(continuation_result)
                # Update current status after repo selection
                current_status = workflow.get_workflow_status(thread_id)
                current_step = current_status.get('current_step')
        
        # Case 2: Check if we need to select an issue (regardless of how we got here)
        if current_step and 'human_select_issue' in str(current_step):
            available_issues_count = current_status.get('available_issues_count', 0)
            logger.info(f"At issue selection step with {available_issues_count} issues available")
            
            if available_issues_count > 0:
                logger.info(f"Found {available_issues_count} issues, simulating human issue selection...")
                
                # Get the actual state to access issues
                from langchain_core.runnables import RunnableConfig
                from typing import cast
                config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
                current_state = workflow.app.get_state(config)
                actual_issues = current_state.values.get('available_issues', [])
                
                if actual_issues:
                    # Select the first issue
                    selected_issue_idx = 0
                    selected_issue = actual_issues[selected_issue_idx]
                    logger.info(f"Selecting issue #{selected_issue.get('number')}: {selected_issue.get('title', 'No title')[:50]}...")
                    
                    # Continue workflow with issue selection
                    issue_continuation = workflow.handle_human_input(
                        thread_id,
                        "issue_selection",
                        selected_issue_idx
                    )
                    
                    logger.info(f"After issue selection, status: {issue_continuation.get('workflow_status') if issue_continuation else 'None'}")
                    
                    # Update result with issue continuation
                    if issue_continuation:
                        result.update(issue_continuation)
                else:
                    logger.warning("Available issues count > 0 but no actual issues found in state")
            else:
                logger.info("No issues available for selection")
        
        # Check status
        status = workflow.get_workflow_status(thread_id)
        
        print("\n" + "="*50)
        print("WORKFLOW TEST RESULTS")
        print("="*50)
        print(f"Query: {test_query}")
        print(f"Status: {result.get('workflow_status', 'unknown')}")
        print(f"Current Step: {status.get('current_step', 'unknown')}")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
        
        if result.get('query_decision'):
            decision = result['query_decision']
            print(f"Query Decision: {decision.get('action_type', 'unknown')}")
        
        if result.get('search_results'):
            print(f"Search Results: {len(result['search_results'])} repositories found")
        
        if result.get('repo_data'):
            repo_info = result['repo_data'].get('basic_info', {})
            print(f"Repository: {repo_info.get('name', 'unknown')}")
            print(f"Issues: {len(result.get('available_issues', []))} found")
        
        print("="*50)
        
        # Save result for inspection
        with open('workflow_test_result.json', 'w') as f:
            # Remove non-serializable agent instances
            clean_result = {k: v for k, v in result.items() if not k.startswith('_')}
            json.dump(clean_result, f, indent=2, default=str)
        
        logger.success("Test completed - results saved to workflow_test_result.json")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_workflow()
