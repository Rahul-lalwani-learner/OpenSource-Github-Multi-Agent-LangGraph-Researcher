"""
Workflow Visualization Script

This script creates a visual representation of the GitHub Issue Multi-Agent Workflow
using LangGraph's built-in visualization capabilities.
"""

from agents.workflow import GitHubIssueWorkflow
from utils.logging import logger
import io
from PIL import Image
import matplotlib.pyplot as plt

def visualize_workflow():
    """Create and display a visual representation of the workflow."""
    logger.info("Creating workflow visualization")
    
    try:
        # Initialize the workflow
        workflow = GitHubIssueWorkflow()
        logger.success("Workflow initialized for visualization")
        
        # Get the compiled application (which has visualization methods)
        app = workflow.app
        
        # Generate Mermaid diagram
        logger.info("Generating Mermaid diagram...")
        try:
            # Try to get PNG representation
            mermaid_png = app.get_graph().draw_mermaid_png()
            
            # Display the image
            image = Image.open(io.BytesIO(mermaid_png))
            plt.figure(figsize=(16, 12))
            plt.imshow(image)
            plt.axis('off')
            plt.title("GitHub Issue Multi-Agent Workflow", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Save the image
            image.save('workflow_diagram.png')
            logger.success("Workflow diagram saved as 'workflow_diagram.png'")
            
        except Exception as png_error:
            logger.warning(f"PNG generation failed: {png_error}")
            
            # Fallback to ASCII representation
            logger.info("Generating ASCII representation...")
            try:
                ascii_graph = app.get_graph().draw_ascii()
                print("\n" + "="*80)
                print("GITHUB ISSUE MULTI-AGENT WORKFLOW - ASCII REPRESENTATION")
                print("="*80)
                print(ascii_graph)
                print("="*80)
                
                # Save ASCII to file
                with open('workflow_ascii.txt', 'w') as f:
                    f.write(ascii_graph)
                logger.success("ASCII workflow saved as 'workflow_ascii.txt'")
                
            except Exception as ascii_error:
                logger.error(f"ASCII generation also failed: {ascii_error}")
                
                # Manual workflow description
                print_manual_workflow_description()
        
        # Print workflow summary
        print_workflow_summary(workflow)
        
    except Exception as e:
        logger.error(f"Error creating workflow visualization: {e}")
        print_manual_workflow_description()

def print_manual_workflow_description():
    """Print a manual description of the workflow when automatic visualization fails."""
    print("\n" + "="*80)
    print("GITHUB ISSUE MULTI-AGENT WORKFLOW - MANUAL DESCRIPTION")
    print("="*80)
    
    workflow_steps = [
        ("START", "Workflow begins"),
        ("init_agents", "Initialize all agents (Query, EntryPoint, Solver, Evaluator)"),
        ("analyze_query", "QueryAgent analyzes user input"),
        ("Route Decision", "Based on query: search_repos OR get_repo_info"),
        ("search_repos", "Search GitHub for repositories (if search needed)"),
        ("🛑 INTERRUPT", "human_select_repo - User selects repository"),
        ("get_repo_info", "Fetch repository details, structure, issues"),
        ("🛑 INTERRUPT", "human_select_issue - User selects issue"),
        ("analyze_entry_points", "EntryPointAgent analyzes repo and issue"),
        ("solve_issue", "SolverAgent generates solution"),
        ("evaluate_solution", "EvaluatorAgent reviews solution"),
        ("Route Decision", "Continue iterating OR finalize"),
        ("finalize_response", "Prepare final response"),
        ("end_workflow", "Workflow completion"),
        ("END", "Final state")
    ]
    
    for i, (step, description) in enumerate(workflow_steps, 1):
        prefix = "🛑" if "INTERRUPT" in step else "📍" if "Route" in step else "⚡"
        print(f"{i:2d}. {prefix} {step:20} | {description}")
    
    print("\n" + "="*80)
    print("HUMAN-IN-THE-LOOP POINTS:")
    print("- Repository Selection: After search results are found")
    print("- Issue Selection: After repository analysis is complete")
    print("="*80)

def print_workflow_summary(workflow):
    """Print a summary of the workflow configuration."""
    print("\n" + "="*80)
    print("WORKFLOW CONFIGURATION SUMMARY")
    print("="*80)
    
    print("🔧 AGENTS:")
    print("   • QueryAgent: Analyzes user queries and routes actions")
    print("   • EntryPointAgent: Analyzes repository structure and issues")
    print("   • SolverAgent: Generates solutions using ReAct framework")
    print("   • EvaluatorAgent: Reviews and validates solutions")
    
    print("\n🛠️  TOOLS AVAILABLE:")
    print("   • Serper Search: Web search for additional context")
    print("   • Tavily Search: Research-oriented search")
    print("   • File Content: Retrieve specific files from repositories")
    
    print("\n⚡ WORKFLOW FEATURES:")
    print("   • Interrupt-based human-in-the-loop")
    print("   • Persistent state with MemorySaver checkpointer")
    print("   • Multi-agent collaboration")
    print("   • Iterative solution refinement")
    print("   • Comprehensive repository analysis")
    
    print("\n🔄 EXECUTION FLOW:")
    print("   1. Query Analysis → Route Decision")
    print("   2. Repository Search/Analysis → Human Selection")
    print("   3. Issue Analysis → Human Selection")
    print("   4. Solution Generation → Evaluation Loop")
    print("   5. Final Response Generation")
    
    print("="*80)

if __name__ == "__main__":
    logger.info("Starting workflow visualization")
    visualize_workflow()
    logger.info("Workflow visualization completed")
