#!/usr/bin/env python3
"""Test script for the SolverAgent with ReAct pattern."""

import asyncio
import json
from agents.solver_agent import SolverAgent

def test_solver_agent():
    """Test the SolverAgent with a sample issue."""
    
    print("Testing SolverAgent with ReAct pattern...")
    
    # Sample issue data for testing
    test_issue = {
        "title": "Add user authentication to login endpoint",
        "body": "The login endpoint needs proper user authentication mechanism. Currently it accepts any credentials without validation.",
        "labels": ["bug", "authentication"],
        "repo_url": "https://github.com/example/test-repo"
    }
    
    # Sample entry point analysis
    test_entry_point_analysis = {
        "key_files": ["app.py", "auth/login.py", "models/user.py"],
        "key_directories": "auth/, models/, tests/",
        "relevant_functions": ["login", "authenticate_user", "validate_credentials"],
        "analysis_notes": "The authentication logic should be implemented in the auth module"
    }
    
    # Sample repo analysis
    test_repo_analysis = {
        "structure": "Flask web application",
        "main_tech_stack": "Python, Flask, SQLAlchemy",
        "testing_framework": "pytest",
        "key_patterns": "MVC architecture"
    }
    
    try:
        # Initialize the SolverAgent
        solver = SolverAgent()
        
        print("SolverAgent initialized successfully!")
        print("ReAct agent configured with tools:")
        for tool in solver.tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Test the solve_issue method
        print("\nTesting issue solving...")
        solution = solver.solve_issue(test_issue, test_repo_analysis, test_entry_point_analysis)
        
        print("\n" + "="*50)
        print("SOLUTION ANALYSIS:")
        print("="*50)
        print(f"Issue Understanding: {solution.solution_analysis.issue_understanding}")
        print(f"Approach: {solution.solution_analysis.approach_rationale}")
        print(f"Complexity: {solution.solution_analysis.complexity_assessment}")
        print(f"Estimated Time: {solution.solution_analysis.estimated_time}")
        
        print("\nSOLUTION STEPS:")
        for step in solution.solution_steps:
            print(f"{step.step_number}. {step.title}")
            print(f"   {step.description[:200]}...")
        
        print(f"\nTesting Strategy: {solution.testing_strategy}")
        print(f"Confidence Score: {solution.confidence_score}")
        print(f"Tools Used: {solution.tools_used}")
        
        print("\n✅ SolverAgent test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing SolverAgent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_solver_agent()
