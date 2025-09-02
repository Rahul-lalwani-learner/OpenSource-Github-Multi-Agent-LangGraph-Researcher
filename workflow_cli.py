"""
Command Line Interface for GitHub Issue Multi-Agent Workflow

This module provides an interactive CLI for running the complete GitHub issue analysis workflow
with human-in-the-loop interactions for repository and issue selection.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.markdown import Markdown
from utils.logging import logger
from agents.workflow import GitHubIssueWorkflow

console = Console()

class WorkflowCLI:
    """
    Command-line interface for the GitHub Issue Multi-Agent Workflow.
    """
    
    def __init__(self):
        """Initialize the CLI with workflow instance."""
        self.workflow = GitHubIssueWorkflow()
        self.current_thread_id = None
        self.current_state = None
    
    def display_welcome(self):
        """Display welcome message and instructions."""
        welcome_text = """
# 🤖 GitHub Issue Multi-Agent Workflow

Welcome to the intelligent GitHub repository analysis system!

## What this system does:
- **Analyzes your query** to understand your intent
- **Searches repositories** or gets direct repo information
- **Lets you select** the repository and issue to analyze
- **Analyzes entry points** and project structure
- **Generates solutions** using advanced AI reasoning
- **Evaluates solutions** for quality and methodology
- **Provides comprehensive results** with citations

## How to use:
1. Enter your query (e.g., "Find Python ML repos" or "https://github.com/owner/repo")
2. Select repository from search results (if searching)
3. Select issue to analyze from the repository
4. Wait for AI agents to analyze and solve the issue
5. Review the comprehensive solution and evaluation

Let's get started! 🚀
        """
        console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))
    
    def get_user_query(self) -> str:
        """Get the initial user query."""
        console.print("\n[bold blue]Step 1: Enter your query[/bold blue]")
        console.print("Examples:")
        console.print("  • 'Find machine learning repositories in Python'")
        console.print("  • 'Search for web development projects'")
        console.print("  • 'https://github.com/username/repository-name'")
        console.print("")
        
        query = Prompt.ask("[bold green]Enter your query[/bold green]")
        return query
    
    def display_search_results(self, search_results: List[Dict[str, Any]]) -> int:
        """Display repository search results and get user selection."""
        console.print(f"\n[bold blue]Step 2: Select Repository[/bold blue]")
        console.print(f"Found {len(search_results)} repositories:")
        
        # Create table for repositories
        table = Table(title="Repository Search Results")
        table.add_column("Index", style="cyan", no_wrap=True)
        table.add_column("Name", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Stars", style="yellow", justify="right")
        table.add_column("Language", style="blue")
        
        for idx, repo in enumerate(search_results):
            table.add_row(
                str(idx),
                repo.get("name", "N/A"),
                (repo.get("description", "")[:80] + "...") if len(repo.get("description", "")) > 80 else repo.get("description", ""),
                str(repo.get("stars", 0)),
                repo.get("language", "N/A")
            )
        
        console.print(table)
        
        # Get user selection
        max_index = len(search_results) - 1
        selection = IntPrompt.ask(
            f"[bold green]Select repository (0-{max_index})[/bold green]",
            default=0
        )
        
        if 0 <= selection <= max_index:
            selected_repo = search_results[selection]
            console.print(f"✅ Selected: [bold]{selected_repo.get('name')}[/bold]")
            return selection
        else:
            console.print("[red]Invalid selection. Using first repository.[/red]")
            return 0
    
    def display_repository_info(self, repo_data: Dict[str, Any]):
        """Display repository information."""
        basic_info = repo_data.get("basic_info", {})
        
        repo_panel = f"""
# 📁 Repository Information

**Name:** {basic_info.get('name', 'N/A')}
**Description:** {basic_info.get('description', 'N/A')}
**Language:** {basic_info.get('language', 'N/A')}
**Stars:** {basic_info.get('stars', 0)}
**Forks:** {basic_info.get('forks', 0)}
**Open Issues:** {basic_info.get('open_issues', 0)}
        """
        
        console.print(Panel(Markdown(repo_panel), title="Repository Details", border_style="green"))
    
    def display_issues(self, available_issues: List[Dict[str, Any]]) -> int:
        """Display available issues and get user selection."""
        console.print(f"\n[bold blue]Step 3: Select Issue[/bold blue]")
        
        if not available_issues:
            console.print("[red]No issues found in this repository.[/red]")
            return -1
        
        console.print(f"Found {len(available_issues)} issues:")
        
        # Create table for issues
        table = Table(title="Available Issues")
        table.add_column("Index", style="cyan", no_wrap=True)
        table.add_column("Number", style="magenta", no_wrap=True)
        table.add_column("Title", style="green")
        table.add_column("State", style="yellow")
        table.add_column("Labels", style="blue")
        
        for idx, issue in enumerate(available_issues):
            labels = ", ".join([label.get("name", "") for label in issue.get("labels", [])])
            table.add_row(
                str(idx),
                str(issue.get("number", "N/A")),
                (issue.get("title", "")[:60] + "...") if len(issue.get("title", "")) > 60 else issue.get("title", ""),
                issue.get("state", "N/A"),
                labels[:30] + "..." if len(labels) > 30 else labels
            )
        
        console.print(table)
        
        # Get user selection
        max_index = len(available_issues) - 1
        selection = IntPrompt.ask(
            f"[bold green]Select issue (0-{max_index})[/bold green]",
            default=0
        )
        
        if 0 <= selection <= max_index:
            selected_issue = available_issues[selection]
            console.print(f"✅ Selected Issue #{selected_issue.get('number')}: [bold]{selected_issue.get('title')}[/bold]")
            return selection
        else:
            console.print("[red]Invalid selection. Using first issue.[/red]")
            return 0
    
    def display_entry_point_analysis(self, analysis: Dict[str, Any]):
        """Display entry point analysis results."""
        console.print(f"\n[bold blue]Step 4: Entry Point Analysis[/bold blue]")
        
        analysis_text = f"""
# 🔍 Entry Point Analysis

## Key Entry Points
{analysis.get('entry_points', 'N/A')}

## Issue Analysis
{analysis.get('issue_analysis', 'N/A')}

## Project Summary
{analysis.get('project_summary', 'N/A')}
        """
        
        console.print(Panel(Markdown(analysis_text), title="Analysis Results", border_style="cyan"))
    
    def display_solution_progress(self, iteration: int, max_iterations: int):
        """Display solution generation progress."""
        console.print(f"\n[bold blue]Step 5: Solution Generation (Iteration {iteration}/{max_iterations})[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Generating solution (attempt {iteration})...", total=None)
            # Simulate some processing time
            import time
            time.sleep(2)
    
    def display_solution_and_evaluation(self, solution: Dict[str, Any], evaluation: Dict[str, Any]):
        """Display the solution and evaluation results."""
        console.print(f"\n[bold blue]Step 6: Solution & Evaluation[/bold blue]")
        
        # Display solution
        solution_text = f"""
# 💡 Solution Analysis

## Problem Understanding
{solution.get('solution_analysis', {}).get('problem_understanding', 'N/A')}

## Proposed Solution
{solution.get('solution_analysis', {}).get('proposed_solution', 'N/A')}

## Implementation Steps
"""
        
        steps = solution.get('solution_steps', [])
        for i, step in enumerate(steps, 1):
            solution_text += f"\n{i}. {step}"
        
        solution_text += f"""

## Testing Strategy
{solution.get('testing_strategy', 'N/A')}
        """
        
        console.print(Panel(Markdown(solution_text), title="Generated Solution", border_style="green"))
        
        # Display evaluation
        overall_assessment = evaluation.get('overall_assessment', {})
        score = overall_assessment.get('overall_score', 0.0)
        recommendation = overall_assessment.get('recommendation', 'N/A')
        
        evaluation_text = f"""
# 📊 Solution Evaluation

**Overall Score:** {score:.2%}
**Recommendation:** {recommendation}

## Quality Metrics
"""
        
        quality_metrics = evaluation.get('quality_metrics', {})
        for metric, value in quality_metrics.items():
            if isinstance(value, (int, float)):
                evaluation_text += f"\n- **{metric.replace('_', ' ').title()}:** {value:.2%}"
            else:
                evaluation_text += f"\n- **{metric.replace('_', ' ').title()}:** {value}"
        
        evaluation_text += f"""

## Key Strengths
"""
        strengths = overall_assessment.get('strengths', [])
        for strength in strengths:
            evaluation_text += f"\n- {strength}"
        
        evaluation_text += f"""

## Areas for Improvement
"""
        improvements = overall_assessment.get('areas_for_improvement', [])
        for improvement in improvements:
            evaluation_text += f"\n- {improvement}"
        
        color = "green" if score >= 0.75 else "yellow" if score >= 0.5 else "red"
        console.print(Panel(Markdown(evaluation_text), title="Evaluation Results", border_style=color))
        
        return score >= 0.75
    
    def display_final_results(self, final_response: Dict[str, Any]):
        """Display the final comprehensive results."""
        console.print(f"\n[bold blue]🎉 Final Results[/bold blue]")
        
        metadata = final_response.get('workflow_metadata', {})
        
        summary_text = f"""
# 📋 Workflow Summary

**Repository:** {final_response.get('selected_repository', {}).get('url', 'N/A')}
**Issue:** #{final_response.get('selected_issue', {}).get('number', 'N/A')} - {final_response.get('selected_issue', {}).get('title', 'N/A')}
**Solution Iterations:** {metadata.get('total_iterations', 0)}
**Final Score:** {metadata.get('approval_score', 0.0):.2%}
**Status:** {'✅ Approved' if final_response.get('solution_approved') else '⚠️ Needs Improvement'}

## Next Steps
1. Review the generated solution above
2. Apply the implementation steps to your repository
3. Run the suggested tests to validate the solution
4. Consider the evaluation feedback for improvements
        """
        
        console.print(Panel(Markdown(summary_text), title="Workflow Complete", border_style="blue"))
    
    def run_interactive_workflow(self, query: Optional[str] = None):
        """Run the complete interactive workflow."""
        self.display_welcome()
        
        # Get user query
        if not query:
            query = self.get_user_query()
        
        # Generate unique thread ID
        import uuid
        thread_id = f"cli_session_{uuid.uuid4().hex[:8]}"
        self.current_thread_id = thread_id
        
        try:
            # Start workflow
            console.print("\n[bold yellow]🔄 Starting workflow...[/bold yellow]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Analyzing query...", total=None)
                
                # This will run until human input is needed
                initial_result = self.workflow.run_workflow(query, thread_id)
                self.current_state = initial_result
            
            # Check what human input is needed
            status = self.workflow.get_workflow_status(thread_id)
            current_status = status.get('workflow_status')
            
            # Handle repository selection if needed
            if current_status == 'awaiting_repo_selection' and initial_result.get('search_results'):
                selection = self.display_search_results(initial_result['search_results'])
                
                # Continue workflow with repo selection
                console.print("\n[bold yellow]🔄 Processing repository selection...[/bold yellow]")
                result = self.workflow.handle_human_input(thread_id, "repo_selection", selection)
                self.current_state = result
            
            # Display repository info
            if self.current_state.get('repo_data'):
                self.display_repository_info(self.current_state['repo_data'])
            
            # Handle issue selection if needed
            status = self.workflow.get_workflow_status(thread_id)
            if status.get('workflow_status') == 'awaiting_issue_selection' and self.current_state.get('available_issues'):
                selection = self.display_issues(self.current_state['available_issues'])
                
                if selection >= 0:
                    # Continue workflow with issue selection
                    console.print("\n[bold yellow]🔄 Processing issue selection...[/bold yellow]")
                    result = self.workflow.handle_human_input(thread_id, "issue_selection", selection)
                    self.current_state = result
                else:
                    console.print("[red]No valid issue selected. Workflow terminated.[/red]")
                    return
            
            # Display entry point analysis
            if self.current_state.get('entry_point_analysis'):
                self.display_entry_point_analysis(self.current_state['entry_point_analysis'])
            
            # Monitor solution generation and evaluation loop
            max_iterations = self.current_state.get('max_iterations', 3)
            current_iteration = self.current_state.get('solution_iterations', 0)
            
            while (not self.current_state.get('solution_approved') and 
                   current_iteration < max_iterations and
                   self.current_state.get('workflow_status') not in ['completed', 'error']):
                
                current_iteration += 1
                self.display_solution_progress(current_iteration, max_iterations)
                
                # Check for updates
                status = self.workflow.get_workflow_status(thread_id)
                current_iteration = status.get('solution_iterations', current_iteration)
            
            # Display final solution and evaluation
            if self.current_state.get('solver_solution') and self.current_state.get('evaluator_feedback'):
                approved = self.display_solution_and_evaluation(
                    self.current_state['solver_solution'], 
                    self.current_state['evaluator_feedback']
                )
            
            # Display final results
            if self.current_state.get('final_response'):
                self.display_final_results(self.current_state['final_response'])
            
            console.print("\n[bold green]✅ Workflow completed successfully![/bold green]")
            
        except KeyboardInterrupt:
            console.print("\n[red]Workflow interrupted by user.[/red]")
        except Exception as e:
            console.print(f"\n[red]Error during workflow execution: {e}[/red]")
            logger.error(f"CLI workflow error: {e}")
    
    def save_results(self, filename: Optional[str] = None):
        """Save the current workflow results to a file."""
        if not self.current_state:
            console.print("[red]No workflow results to save.[/red]")
            return
        
        if not filename:
            filename = f"workflow_results_{self.current_thread_id}.json"
        
        try:
            # Remove non-serializable agents from state
            state_to_save = {k: v for k, v in self.current_state.items() 
                           if not k.startswith('_')}
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, indent=2, default=str)
            
            console.print(f"[green]Results saved to: {filename}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving results: {e}[/red]")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="GitHub Issue Multi-Agent Workflow CLI")
    parser.add_argument("--query", "-q", help="Initial query to process")
    parser.add_argument("--save", "-s", help="Save results to specified file")
    
    args = parser.parse_args()
    
    cli = WorkflowCLI()
    
    try:
        cli.run_interactive_workflow(args.query)
        
        if args.save:
            cli.save_results(args.save)
        elif Confirm.ask("\n[bold blue]Save results to file?[/bold blue]", default=True):
            cli.save_results()
        
    except Exception as e:
        console.print(f"[red]CLI Error: {e}[/red]")
        logger.error(f"CLI main error: {e}")

if __name__ == "__main__":
    main()
