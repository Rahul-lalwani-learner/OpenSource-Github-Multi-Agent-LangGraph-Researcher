"""
Evaluator Agent Module

This agent evaluates and validates solutions provided by the Solver Agent by:
1. Checking the quality and completeness of the solution
2. Validating citations and external tool usage
3. Assessing the methodology and approach
4. Providing feedback and recommendations for improvement
5. Ensuring the solution follows best practices

The agent acts as a quality control layer for the Solver Agent's output.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from utils.logging import logger
import json
import re

class CitationValidation(BaseModel):
    """Model for citation validation results."""
    citation: str = Field(..., description="The citation being validated")
    is_valid: bool = Field(..., description="Whether the citation is valid and relevant")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0.0 to 1.0)")
    validation_notes: str = Field(..., description="Notes about the citation validation")

class MethodologyAssessment(BaseModel):
    """Model for methodology assessment."""
    approach_quality: Literal["excellent", "good", "fair", "poor"] = Field(
        ..., description="Overall quality of the approach"
    )
    step_completeness: float = Field(..., ge=0.0, le=1.0, description="Completeness of solution steps (0.0 to 1.0)")
    technical_accuracy: float = Field(..., ge=0.0, le=1.0, description="Technical accuracy assessment (0.0 to 1.0)")
    best_practices_adherence: float = Field(..., ge=0.0, le=1.0, description="Adherence to best practices (0.0 to 1.0)")
    implementation_feasibility: Literal["very_feasible", "feasible", "challenging", "not_feasible"] = Field(
        ..., description="How feasible the implementation is"
    )

class QualityMetrics(BaseModel):
    """Model for solution quality metrics."""
    clarity_score: float = Field(..., ge=0.0, le=1.0, description="Clarity of instructions (0.0 to 1.0)")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Completeness of solution (0.0 to 1.0)")
    actionability_score: float = Field(..., ge=0.0, le=1.0, description="How actionable the solution is (0.0 to 1.0)")
    code_quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality of code snippets (0.0 to 1.0)")
    testing_adequacy: float = Field(..., ge=0.0, le=1.0, description="Adequacy of testing strategy (0.0 to 1.0)")

class ImprovementSuggestion(BaseModel):
    """Model for improvement suggestions."""
    category: Literal["methodology", "implementation", "testing", "documentation", "citations"] = Field(
        ..., description="Category of improvement"
    )
    priority: Literal["high", "medium", "low"] = Field(..., description="Priority of the improvement")
    suggestion: str = Field(..., description="Specific suggestion for improvement")
    rationale: str = Field(..., description="Why this improvement is needed")

class EvaluationResult(BaseModel):
    """Complete evaluation result from the Evaluator Agent."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall evaluation score (0.0 to 1.0)")
    quality_metrics: QualityMetrics = Field(..., description="Detailed quality metrics")
    methodology_assessment: MethodologyAssessment = Field(..., description="Assessment of the methodology")
    citation_validations: List[CitationValidation] = Field(..., description="Validation of citations")
    tool_usage_assessment: str = Field(..., description="Assessment of tool usage effectiveness")
    strengths: List[str] = Field(..., description="Strengths of the solution")
    weaknesses: List[str] = Field(..., description="Weaknesses identified")
    improvement_suggestions: List[ImprovementSuggestion] = Field(..., description="Suggestions for improvement")
    final_recommendation: Literal["approve", "approve_with_changes", "reject", "needs_revision"] = Field(
        ..., description="Final recommendation for the solution"
    )
    evaluator_confidence: float = Field(..., ge=0.0, le=1.0, description="Evaluator's confidence in assessment")

class EvaluatorAgent:
    """
    Agent that evaluates and validates solutions from the Solver Agent.
    
    This agent performs comprehensive quality control including:
    - Solution quality assessment
    - Citation validation
    - Methodology evaluation
    - Best practices compliance
    - Improvement recommendations
    """
    
    def __init__(self):
        """Initialize the Evaluator Agent with Google GenAI model."""
        logger.info("Initializing EvaluatorAgent with Google GenAI")
        
        try:
            self.model = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.1,  # Very low temperature for consistent evaluation
                max_tokens=3500
            )
            
            # Bind the model with structured output
            self.structured_model = self.model.with_structured_output(EvaluationResult)
            
            logger.success("EvaluatorAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EvaluatorAgent: {e}")
            raise
    
    def evaluate_solution(self, 
                         solver_response: Dict[str, Any], 
                         original_issue: Dict[str, Any],
                         repo_context: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate a solution provided by the Solver Agent.
        
        Args:
            solver_response (Dict[str, Any]): Response from the Solver Agent
            original_issue (Dict[str, Any]): Original issue data
            repo_context (Dict[str, Any]): Repository context and analysis
            
        Returns:
            EvaluationResult: Comprehensive evaluation of the solution
        """
        logger.info("Starting evaluation of solver response")
        
        try:
            # Step 1: Validate citations
            citation_results = self._validate_citations(solver_response)
            logger.debug(f"Validated {len(citation_results)} citations")
            
            # Step 2: Assess tool usage
            tool_assessment = self._assess_tool_usage(solver_response)
            logger.debug("Assessed tool usage effectiveness")
            
            # Step 3: Evaluate methodology
            methodology_eval = self._evaluate_methodology(solver_response, original_issue)
            logger.debug("Completed methodology evaluation")
            
            # Step 4: Generate comprehensive evaluation
            evaluation = self._generate_evaluation(
                solver_response, original_issue, repo_context,
                citation_results, tool_assessment, methodology_eval
            )
            
            logger.success(f"Evaluation completed with overall score: {evaluation.overall_score}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return self._create_fallback_evaluation(str(e))
    
    def _validate_citations(self, solver_response: Dict[str, Any]) -> List[CitationValidation]:
        """Validate citations used in the solver response."""
        logger.debug("Validating citations in solver response")
        
        citations = []
        solution_steps = solver_response.get('solution_steps', [])
        
        # Extract all citations from solution steps
        for step in solution_steps:
            step_citations = step.get('citations', [])
            citations.extend(step_citations)
        
        # Add additional resources as citations
        additional_resources = solver_response.get('additional_resources', [])
        citations.extend(additional_resources)
        
        # Remove duplicates
        unique_citations = list(set(citations))
        
        validation_results = []
        for citation in unique_citations:
            if citation and citation.strip():
                validation = self._validate_single_citation(citation, solver_response)
                validation_results.append(validation)
        
        logger.debug(f"Validated {len(validation_results)} unique citations")
        return validation_results
    
    def _validate_single_citation(self, citation: str, solver_response: Dict[str, Any]) -> CitationValidation:
        """Validate a single citation for relevance and quality."""
        
        # Simple heuristic-based validation
        is_valid = True
        relevance_score = 0.8  # Default score
        notes = "Citation appears relevant"
        
        # Check if citation is too generic
        generic_terms = ['google', 'stackoverflow', 'documentation', 'tutorial', 'guide']
        if any(term in citation.lower() for term in generic_terms):
            relevance_score = 0.6
            notes = "Generic citation - could be more specific"
        
        # Check if citation matches the technologies mentioned
        technologies = solver_response.get('solution_analysis', {}).get('prerequisites', [])
        if any(tech.lower() in citation.lower() for tech in technologies):
            relevance_score = min(1.0, relevance_score + 0.2)
            notes = "Citation relevant to project technologies"
        
        # Check if citation is actually informative
        if len(citation) < 5 or citation.lower() in ['todo', 'tbd', 'placeholder']:
            is_valid = False
            relevance_score = 0.1
            notes = "Citation is placeholder or too generic"
        
        return CitationValidation(
            citation=citation,
            is_valid=is_valid,
            relevance_score=relevance_score,
            validation_notes=notes
        )
    
    def _assess_tool_usage(self, solver_response: Dict[str, Any]) -> str:
        """Assess the effectiveness of tool usage in the solution."""
        logger.debug("Assessing tool usage effectiveness")
        
        tools_used = solver_response.get('tools_used', [])
        solution_steps = solver_response.get('solution_steps', [])
        
        assessment_points = []
        
        # Check if appropriate tools were used
        if not tools_used:
            assessment_points.append("No tools were used - may indicate lack of research")
        else:
            assessment_points.append(f"Used {len(tools_used)} tools: {', '.join(tools_used)}")
        
        # Check for search tool usage
        search_tools = ['serper', 'tavily']
        used_search = any(tool in tools_used for tool in search_tools)
        if used_search:
            assessment_points.append("Search tools used for research - good practice")
        else:
            assessment_points.append("No search tools used - may have missed relevant information")
        
        # Check for file content tool usage
        if 'get_file_content' in tools_used:
            assessment_points.append("File content tool used - shows code analysis")
        
        # Check if tool usage is reflected in citations
        citations_present = any(
            step.get('citations', []) for step in solution_steps
        )
        if used_search and not citations_present:
            assessment_points.append("Search tools used but citations missing - poor documentation")
        
        return "; ".join(assessment_points)
    
    def _evaluate_methodology(self, solver_response: Dict[str, Any], original_issue: Dict[str, Any]) -> MethodologyAssessment:
        """Evaluate the methodology used in the solution."""
        logger.debug("Evaluating solution methodology")
        
        solution_analysis = solver_response.get('solution_analysis', {})
        solution_steps = solver_response.get('solution_steps', [])
        
        # Assess approach quality
        issue_understanding = solution_analysis.get('issue_understanding', '')
        approach_rationale = solution_analysis.get('approach_rationale', '')
        
        approach_quality = "good"  # Default
        if len(issue_understanding) > 100 and len(approach_rationale) > 50:
            approach_quality = "excellent"
        elif len(issue_understanding) < 50 or len(approach_rationale) < 20:
            approach_quality = "fair"
        
        # Assess step completeness
        required_elements = ['title', 'description', 'rationale']
        total_elements = len(solution_steps) * len(required_elements)
        present_elements = 0
        
        for step in solution_steps:
            for element in required_elements:
                if step.get(element) and len(str(step.get(element))) > 10:
                    present_elements += 1
        
        step_completeness = present_elements / max(total_elements, 1)
        
        # Assess technical accuracy (heuristic-based)
        technical_accuracy = 0.8  # Default
        
        # Check for code snippets presence and quality
        has_code = any(step.get('code_snippet') for step in solution_steps)
        if has_code:
            technical_accuracy = 0.9
        
        # Check for testing strategy
        testing_strategy = solver_response.get('testing_strategy', '')
        if len(testing_strategy) > 50:
            technical_accuracy = min(1.0, technical_accuracy + 0.1)
        
        # Assess best practices adherence
        best_practices_score = 0.7  # Default
        
        # Check for proper error handling mentions
        error_handling_mentions = sum(
            1 for step in solution_steps 
            if 'error' in step.get('description', '').lower() or 'exception' in step.get('description', '').lower()
        )
        if error_handling_mentions > 0:
            best_practices_score += 0.1
        
        # Check for security considerations
        security_mentions = sum(
            1 for step in solution_steps 
            if any(term in step.get('description', '').lower() 
                  for term in ['security', 'authentication', 'authorization', 'validation'])
        )
        if security_mentions > 0:
            best_practices_score += 0.1
        
        best_practices_score = min(1.0, best_practices_score)
        
        # Assess implementation feasibility
        complexity = solution_analysis.get('complexity_assessment', 'moderate')
        estimated_time = solution_analysis.get('estimated_time', '')
        
        feasibility = "feasible"  # Default
        if complexity == "simple" and 'hour' in estimated_time.lower():
            feasibility = "very_feasible"
        elif complexity in ["complex", "very_complex"]:
            feasibility = "challenging"
        
        return MethodologyAssessment(
            approach_quality=approach_quality,
            step_completeness=step_completeness,
            technical_accuracy=technical_accuracy,
            best_practices_adherence=best_practices_score,
            implementation_feasibility=feasibility
        )
    
    def _generate_evaluation(self, 
                           solver_response: Dict[str, Any], 
                           original_issue: Dict[str, Any],
                           repo_context: Dict[str, Any],
                           citation_results: List[CitationValidation],
                           tool_assessment: str,
                           methodology_eval: MethodologyAssessment) -> EvaluationResult:
        """Generate comprehensive evaluation using LLM."""
        logger.debug("Generating comprehensive evaluation using LLM")
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(
            solver_response, original_issue, repo_context,
            citation_results, tool_assessment, methodology_eval
        )
        
        try:
            # Try structured output first
            response = self.structured_model.invoke(prompt)
            
            if isinstance(response, EvaluationResult):
                logger.success("Generated structured evaluation successfully")
                return response
            else:
                logger.warning("Structured output failed, creating manual evaluation")
                return self._create_manual_evaluation(
                    solver_response, citation_results, tool_assessment, methodology_eval
                )
                
        except Exception as e:
            logger.error(f"Failed to generate evaluation with LLM: {e}")
            return self._create_manual_evaluation(
                solver_response, citation_results, tool_assessment, methodology_eval
            )
    
    def _create_evaluation_prompt(self, solver_response, original_issue, repo_context, citation_results, tool_assessment, methodology_eval):
        """Create the evaluation prompt for the LLM."""
        
        # Format citation results
        citation_summary = "\n".join([
            f"- {cv.citation}: {'Valid' if cv.is_valid else 'Invalid'} (Score: {cv.relevance_score}) - {cv.validation_notes}"
            for cv in citation_results
        ])
        
        prompt = f"""
        You are an expert code reviewer and solution evaluator. Evaluate the following solution for a GitHub issue.

        **ORIGINAL ISSUE:**
        - Title: {original_issue.get('title', 'No title')}
        - Body: {original_issue.get('body', 'No description')[:500]}...

        **SOLVER'S RESPONSE:**
        - Confidence Score: {solver_response.get('confidence_score', 'N/A')}
        - Solution Steps: {len(solver_response.get('solution_steps', []))}
        - Tools Used: {solver_response.get('tools_used', [])}
        - Complexity Assessment: {solver_response.get('solution_analysis', {}).get('complexity_assessment', 'N/A')}

        **CITATION ANALYSIS:**
        {citation_summary}

        **TOOL USAGE ASSESSMENT:**
        {tool_assessment}

        **METHODOLOGY SCORES:**
        - Approach Quality: {methodology_eval.approach_quality}
        - Step Completeness: {methodology_eval.step_completeness}
        - Technical Accuracy: {methodology_eval.technical_accuracy}
        - Best Practices: {methodology_eval.best_practices_adherence}
        - Feasibility: {methodology_eval.implementation_feasibility}

        **EVALUATION REQUIREMENTS:**
        1. Assess overall solution quality (0.0 to 1.0)
        2. Evaluate clarity, completeness, and actionability
        3. Identify strengths and weaknesses
        4. Provide improvement suggestions
        5. Make final recommendation

        **RESPONSE FORMAT:**
        Provide a comprehensive evaluation following the EvaluationResult schema.
        Focus on constructive feedback and specific improvement suggestions.
        """
        
        return prompt
    
    def _create_manual_evaluation(self, solver_response, citation_results, tool_assessment, methodology_eval) -> EvaluationResult:
        """Create evaluation manually when LLM fails."""
        logger.info("Creating manual evaluation")
        
        # Calculate overall score based on methodology assessment
        overall_score = (
            methodology_eval.step_completeness * 0.3 +
            methodology_eval.technical_accuracy * 0.3 +
            methodology_eval.best_practices_adherence * 0.2 +
            (len(citation_results) / max(1, len(citation_results))) * 0.2
        )
        
        # Calculate quality metrics
        quality_metrics = QualityMetrics(
            clarity_score=methodology_eval.step_completeness,
            completeness_score=methodology_eval.step_completeness,
            actionability_score=methodology_eval.technical_accuracy,
            code_quality_score=methodology_eval.best_practices_adherence,
            testing_adequacy=0.7  # Default
        )
        
        # Generate basic strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if methodology_eval.approach_quality in ["excellent", "good"]:
            strengths.append("Well-structured approach to the problem")
        
        if methodology_eval.step_completeness > 0.8:
            strengths.append("Comprehensive step-by-step instructions")
        else:
            weaknesses.append("Steps could be more detailed")
        
        if len(citation_results) > 0:
            strengths.append("Includes citations and references")
        else:
            weaknesses.append("Lacks proper citations")
        
        # Generate improvement suggestions
        suggestions = []
        if methodology_eval.step_completeness < 0.8:
            suggestions.append(ImprovementSuggestion(
                category="implementation",
                priority="high",
                suggestion="Provide more detailed step-by-step instructions",
                rationale="Current steps lack sufficient detail for implementation"
            ))
        
        if not citation_results:
            suggestions.append(ImprovementSuggestion(
                category="citations",
                priority="medium",
                suggestion="Add proper citations and references",
                rationale="Citations improve credibility and help users find additional resources"
            ))
        
        # Determine final recommendation
        if overall_score >= 0.8:
            recommendation = "approve"
        elif overall_score >= 0.6:
            recommendation = "approve_with_changes"
        elif overall_score >= 0.4:
            recommendation = "needs_revision"
        else:
            recommendation = "reject"
        
        return EvaluationResult(
            overall_score=overall_score,
            quality_metrics=quality_metrics,
            methodology_assessment=methodology_eval,
            citation_validations=citation_results,
            tool_usage_assessment=tool_assessment,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions,
            final_recommendation=recommendation,
            evaluator_confidence=0.7
        )
    
    def _create_fallback_evaluation(self, error_msg: str) -> EvaluationResult:
        """Create fallback evaluation when evaluation fails."""
        logger.info("Creating fallback evaluation due to processing error")
        
        return EvaluationResult(
            overall_score=0.5,
            quality_metrics=QualityMetrics(
                clarity_score=0.5,
                completeness_score=0.5,
                actionability_score=0.5,
                code_quality_score=0.5,
                testing_adequacy=0.5
            ),
            methodology_assessment=MethodologyAssessment(
                approach_quality="fair",
                step_completeness=0.5,
                technical_accuracy=0.5,
                best_practices_adherence=0.5,
                implementation_feasibility="feasible"
            ),
            citation_validations=[],
            tool_usage_assessment=f"Evaluation failed: {error_msg}",
            strengths=["Unable to assess due to evaluation error"],
            weaknesses=["Evaluation process failed - manual review required"],
            improvement_suggestions=[
                ImprovementSuggestion(
                    category="methodology",
                    priority="high",
                    suggestion="Manual evaluation required",
                    rationale=f"Automated evaluation failed: {error_msg}"
                )
            ],
            final_recommendation="needs_revision",
            evaluator_confidence=0.1
        )
    
    def process_evaluation(self, solver_response: Dict[str, Any], original_issue: Dict[str, Any], repo_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the evaluation and return as dictionary.
        
        Args:
            solver_response (Dict[str, Any]): Solver's response to evaluate
            original_issue (Dict[str, Any]): Original issue data
            repo_context (Dict[str, Any]): Repository context
            
        Returns:
            Dict[str, Any]: Evaluation result as dictionary
        """
        try:
            evaluation = self.evaluate_solution(solver_response, original_issue, repo_context)
            logger.success(f"Evaluation completed with overall score: {evaluation.overall_score}")
            return evaluation.model_dump()
        except Exception as e:
            logger.error(f"Error processing evaluation: {e}")
            return {
                "error": str(e),
                "evaluation_type": "error"
            }

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing EvaluatorAgent")
    
    # Mock solver response for testing
    sample_solver_response = {
        "solution_analysis": {
            "issue_understanding": "Need to implement JWT authentication",
            "approach_rationale": "Using Flask-JWT-Extended for secure authentication",
            "complexity_assessment": "moderate",
            "estimated_time": "2-3 hours"
        },
        "solution_steps": [
            {
                "step_number": 1,
                "title": "Install JWT library",
                "description": "Install Flask-JWT-Extended using pip",
                "rationale": "Required for JWT token handling",
                "citations": ["Flask-JWT-Extended documentation"]
            }
        ],
        "tools_used": ["serper", "get_file_content"],
        "confidence_score": 0.8
    }
    
    sample_issue = {
        "title": "Add JWT authentication",
        "body": "Implement secure authentication using JWT tokens"
    }
    
    sample_repo_context = {
        "technologies": ["Python", "Flask"]
    }
    
    agent = EvaluatorAgent()
    result = agent.process_evaluation(sample_solver_response, sample_issue, sample_repo_context)
    
    print("📊 Evaluator Agent Result:")
    print(f"🎯 Overall Score: {result.get('overall_score', 0)}")
    print(f"📋 Final Recommendation: {result.get('final_recommendation', 'N/A')}")
    print(f"✅ Strengths: {len(result.get('strengths', []))}")
    print(f"⚠️ Weaknesses: {len(result.get('weaknesses', []))}")
    print(f"💡 Suggestions: {len(result.get('improvement_suggestions', []))}")
    
    logger.info("EvaluatorAgent testing completed")
