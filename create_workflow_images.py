"""
Enhanced Workflow Visualization with Image Export

This script creates a professional visual diagram of the GitHub Issue Multi-Agent Workflow
that can be saved as an image file.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from utils.logging import logger

def create_workflow_diagram():
    """Create a professional workflow diagram and save as image."""
    logger.info("Creating workflow diagram")
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define colors
    colors = {
        'start_end': '#2E7D32',      # Dark green
        'agent': '#1976D2',          # Blue
        'human': '#F57C00',          # Orange
        'decision': '#7B1FA2',       # Purple
        'process': '#455A64',        # Blue grey
        'arrow': '#424242'           # Dark grey
    }
    
    # Define node positions and information
    nodes = [
        # (x, y, width, height, text, color_key, text_size)
        (5, 13, 1.5, 0.6, 'START', 'start_end', 10),
        (5, 12, 2, 0.6, 'init_agents', 'process', 9),
        (5, 11, 2.5, 0.6, 'analyze_query\n(QueryAgent)', 'agent', 9),
        (2.5, 9.5, 2, 0.8, 'search_repos', 'process', 9),
        (7.5, 9.5, 2, 0.8, 'get_repo_info', 'process', 9),
        (5, 8, 3, 0.8, '🛑 human_select_repo\n(INTERRUPT)', 'human', 9),
        (5, 6.5, 3, 0.8, '🛑 human_select_issue\n(INTERRUPT)', 'human', 9),
        (5, 5, 3.5, 0.8, 'analyze_entry_points\n(EntryPointAgent)', 'agent', 9),
        (5, 3.5, 3, 0.8, 'solve_issue\n(SolverAgent)', 'agent', 9),
        (5, 2, 3.5, 0.8, 'evaluate_solution\n(EvaluatorAgent)', 'agent', 9),
        (8, 2, 2, 0.6, 'finalize_response', 'process', 8),
        (8, 0.8, 1.5, 0.6, 'END', 'start_end', 10),
        (2, 2.5, 1.8, 1, 'Loop?\nIterate', 'decision', 8)
    ]
    
    # Draw nodes
    for x, y, w, h, text, color_key, text_size in nodes:
        # Create rounded rectangle
        if 'INTERRUPT' in text:
            # Special styling for interrupt nodes
            rect = FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle="round,pad=0.1",
                facecolor=colors[color_key],
                edgecolor='red',
                linewidth=3,
                alpha=0.9
            )
        else:
            rect = FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle="round,pad=0.1",
                facecolor=colors[color_key],
                edgecolor='white',
                linewidth=2,
                alpha=0.9
            )
        
        ax.add_patch(rect)
        
        # Add text
        text_color = 'white' if color_key != 'human' else 'black'
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=text_size, fontweight='bold', color=text_color)
    
    # Define arrows (from_node_index, to_node_index, style)
    arrows = [
        (0, 1, 'normal'),    # START -> init_agents
        (1, 2, 'normal'),    # init_agents -> analyze_query
        (2, 3, 'branch'),    # analyze_query -> search_repos (left branch)
        (2, 4, 'branch'),    # analyze_query -> get_repo_info (right branch)
        (3, 5, 'normal'),    # search_repos -> human_select_repo
        (4, 5, 'normal'),    # get_repo_info -> human_select_repo
        (5, 6, 'normal'),    # human_select_repo -> human_select_issue
        (6, 7, 'normal'),    # human_select_issue -> analyze_entry_points
        (7, 8, 'normal'),    # analyze_entry_points -> solve_issue
        (8, 9, 'normal'),    # solve_issue -> evaluate_solution
        (9, 10, 'normal'),   # evaluate_solution -> finalize_response
        (10, 11, 'normal'),  # finalize_response -> END
        (9, 12, 'branch'),   # evaluate_solution -> Loop decision
        (12, 8, 'loop')      # Loop -> solve_issue (back)
    ]
    
    # Draw arrows
    for from_idx, to_idx, style in arrows:
        from_node = nodes[from_idx]
        to_node = nodes[to_idx]
        
        from_x, from_y = from_node[0], from_node[1]
        to_x, to_y = to_node[0], to_node[1]
        
        arrow = None  # Initialize arrow variable
        
        if style == 'branch':
            # Curved arrows for branching
            if from_idx == 2 and to_idx == 3:  # analyze_query -> search_repos
                arrow = ConnectionPatch((from_x - 0.5, from_y - 0.3), (to_x + 0.5, to_y + 0.4),
                                       "data", "data", arrowstyle="->", 
                                       shrinkA=5, shrinkB=5, mutation_scale=20,
                                       fc=colors['arrow'], ec=colors['arrow'],
                                       connectionstyle="arc3,rad=-0.3")
            elif from_idx == 2 and to_idx == 4:  # analyze_query -> get_repo_info
                arrow = ConnectionPatch((from_x + 0.5, from_y - 0.3), (to_x - 0.5, to_y + 0.4),
                                       "data", "data", arrowstyle="->",
                                       shrinkA=5, shrinkB=5, mutation_scale=20,
                                       fc=colors['arrow'], ec=colors['arrow'],
                                       connectionstyle="arc3,rad=0.3")
            elif from_idx == 9 and to_idx == 12:  # evaluate_solution -> Loop
                arrow = ConnectionPatch((to_x + 0.9, from_y), (to_x + 0.9, to_y + 0.5),
                                       "data", "data", arrowstyle="->",
                                       shrinkA=5, shrinkB=5, mutation_scale=20,
                                       fc=colors['arrow'], ec=colors['arrow'])
        elif style == 'loop':
            # Loop back arrow
            arrow = ConnectionPatch((from_x - 0.9, from_y), (to_x - 1.5, to_y),
                                   "data", "data", arrowstyle="->",
                                   shrinkA=5, shrinkB=5, mutation_scale=20,
                                   fc='red', ec='red', linewidth=2,
                                   connectionstyle="arc3,rad=-0.5")
        else:
            # Straight arrows
            arrow = ConnectionPatch((from_x, from_y - from_node[3]/2), 
                                   (to_x, to_y + to_node[3]/2),
                                   "data", "data", arrowstyle="->",
                                   shrinkA=5, shrinkB=5, mutation_scale=20,
                                   fc=colors['arrow'], ec=colors['arrow'])
        
        if arrow is not None:
            ax.add_patch(arrow)
    
    # Add decision labels
    ax.text(5, 10, 'Route Decision:\nsearch_repos OR get_repo_info', 
            ha='center', va='center', fontsize=8, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax.text(1, 1, 'Solution\nIteration\nLoop', 
            ha='center', va='center', fontsize=8, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    # Add title and legend
    plt.title('GitHub Issue Multi-Agent LangGraph Workflow\nwith Human-in-the-Loop Interrupts', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color=colors['agent'], label='AI Agents'),
        mpatches.Patch(color=colors['human'], label='Human Interaction (Interrupts)'),
        mpatches.Patch(color=colors['process'], label='Processing Nodes'),
        mpatches.Patch(color=colors['decision'], label='Decision Points'),
        mpatches.Patch(color=colors['start_end'], label='Start/End')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add workflow summary box
    summary_text = """
Key Features:
• Interrupt-based human-in-the-loop
• Multi-agent collaboration
• Persistent state management
• Iterative solution refinement
• Comprehensive repo analysis
    """
    
    ax.text(9.5, 12, summary_text, 
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the diagram
    output_files = [
        ('workflow_diagram.png', 'PNG'),
        ('workflow_diagram.pdf', 'PDF'),
        ('workflow_diagram.svg', 'SVG')
    ]
    
    for filename, format_name in output_files:
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.success(f"Workflow diagram saved as '{filename}' ({format_name})")
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
    
    plt.show()
    
    return fig

def create_detailed_flow_chart():
    """Create a detailed flow chart showing all states and transitions."""
    logger.info("Creating detailed flow chart")
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Add detailed state information
    states = [
        "🏁 START\n• User provides query\n• Initialize thread",
        "⚙️ INIT AGENTS\n• QueryAgent\n• EntryPointAgent\n• SolverAgent\n• EvaluatorAgent",
        "🔍 ANALYZE QUERY\n• Parse user input\n• Determine action type\n• Extract parameters",
        "🔍 SEARCH REPOS\n• GitHub API search\n• Filter results\n• Return top matches",
        "📊 GET REPO INFO\n• Fetch repo details\n• Analyze structure\n• Extract issues",
        "👤 SELECT REPO\n• Present options\n• Wait for selection\n• Update state",
        "👤 SELECT ISSUE\n• Present issues\n• Wait for selection\n• Update state", 
        "🎯 ANALYZE ENTRY POINTS\n• Identify key files\n• Analyze issue context\n• Generate insights",
        "🛠️ SOLVE ISSUE\n• Generate solution\n• Use ReAct framework\n• Access file contents",
        "✅ EVALUATE SOLUTION\n• Review quality\n• Check completeness\n• Provide feedback",
        "📝 FINALIZE RESPONSE\n• Compile results\n• Format output\n• Prepare delivery",
        "🏁 END\n• Return final result\n• Clean up resources"
    ]
    
    # Position states in a flow
    positions = [
        (6, 15), (6, 13.5), (6, 12), (3, 10.5), (9, 10.5),
        (6, 9), (6, 7.5), (6, 6), (6, 4.5), (6, 3),
        (6, 1.5), (6, 0.5)
    ]
    
    # Draw detailed states
    for i, (state_text, (x, y)) in enumerate(zip(states, positions)):
        width = 3.5 if i in [3, 4] else 3
        height = 1.2
        
        color = '#FF6B35' if '👤' in state_text else '#4ECDC4' if i in [0, 11] else '#45B7D1'
        
        rect = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='white',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        ax.text(x, y, state_text, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white')
    
    plt.title('Detailed GitHub Issue Workflow States', fontsize=18, fontweight='bold', pad=20)
    
    # Save detailed chart
    plt.savefig('detailed_workflow.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    logger.success("Detailed workflow saved as 'detailed_workflow.png'")
    
    plt.show()
    
    return fig

if __name__ == "__main__":
    logger.info("Creating workflow visualizations")
    
    # Create main workflow diagram
    main_fig = create_workflow_diagram()
    
    # Create detailed flow chart
    detail_fig = create_detailed_flow_chart()
    
    logger.success("All workflow visualizations completed!")
    print("\n📁 Generated Files:")
    print("  • workflow_diagram.png (Main workflow)")
    print("  • workflow_diagram.pdf (PDF version)")  
    print("  • workflow_diagram.svg (Vector version)")
    print("  • detailed_workflow.png (Detailed states)")
