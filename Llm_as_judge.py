import json
import sys
import ollama
from typing import Dict, Any, List, Tuple
import re
import time
from dataclasses import dataclass

@dataclass
class CodeMetrics:
    """Extracted metrics from the evaluation results"""
    passrate: float
    weighted_score: float
    memory_score: float
    robustness_score: float
    tests_passed: int
    tests_failed: int
    total_tests: int
    execution_time_ms: int
    failed_tests: List[str]
    potential_edge_cases: List[str]
    program_type: str
    difficulty_level: str

class AdvancedCodeAnalyzer:
    def __init__(self, model_name="codellama:13b-instruct"):
        self.model_name = model_name
        self.analysis_conversations = []
        
        # Multi-stage analysis system prompts
        self.code_understanding_prompt = """You are an expert C code analyzer with deep understanding of algorithms, data structures, and software engineering principles.

Your task is to perform DEEP CODE ANALYSIS. Analyze the given C code to understand:

1. **Algorithm Analysis**: What algorithm/approach is used? Is it optimal?
2. **Code Structure**: How is the code organized? Is it well-structured?
3. **Logic Flow**: Trace through the execution flow and identify potential issues
4. **Implementation Quality**: Are there better ways to implement this?
5. **Potential Bugs**: What could go wrong that tests might not catch?

Provide your analysis in this JSON format:
{
  "algorithm_analysis": {
    "algorithm_type": "description of algorithm used",
    "time_complexity": "Big O notation",
    "space_complexity": "Big O notation", 
    "optimality": "is this optimal? why/why not?",
    "alternative_approaches": ["list of better approaches if any"]
  },
  "code_structure_analysis": {
    "organization_quality": "assessment of code organization",
    "readability_score": "1-10",
    "maintainability_issues": ["list of maintainability concerns"],
    "style_violations": ["coding style issues"]
  },
  "logic_flow_analysis": {
    "execution_path": "description of main execution flow",
    "branch_coverage": "assessment of different code paths",
    "potential_infinite_loops": ["any loop conditions that could cause infinite loops"],
    "unreachable_code": ["code that might never execute"]
  },
  "implementation_quality": {
    "code_smells": ["bad practices or code smells detected"],
    "design_patterns": ["design patterns used or should be used"],
    "refactoring_suggestions": ["specific refactoring recommendations"]
  },
  "potential_hidden_bugs": [
    {
      "bug_type": "type of potential bug",
      "location": "where in code this could occur", 
      "scenario": "specific scenario that could trigger this bug",
      "severity": "low|medium|high|critical"
    }
  ]
}

Be extremely thorough and look for subtle issues that automated tests might miss."""

        self.failure_analysis_prompt = """You are a debugging expert specializing in test failure analysis.

Analyze the failed tests and provide insights into WHY they failed and what this reveals about the code quality.

For each failed test, determine:
1. **Root Cause**: What exactly caused this test to fail?
2. **Code Issue**: What specific problem in the code led to this?
3. **Fix Complexity**: How difficult would this be to fix?
4. **Related Issues**: Could this failure indicate other problems?
5. **Pattern Analysis**: Do multiple failures show a pattern?

Provide analysis in JSON format:
{
  "failure_pattern_analysis": {
    "common_root_causes": ["patterns seen across failures"],
    "severity_assessment": "how serious are these failures overall",
    "fix_priority": ["which failures to fix first and why"]
  },
  "detailed_failure_analysis": [
    {
      "test_description": "what the test was checking",
      "failure_reason": "technical reason for failure", 
      "root_cause": "underlying code issue",
      "fix_complexity": "trivial|easy|moderate|complex|major_rewrite",
      "fix_suggestion": "specific steps to fix this issue",
      "related_risks": ["other issues this might indicate"]
    }
  ],
  "cascading_effects": {
    "issues_that_could_cause_multiple_failures": ["problems that affect multiple areas"],
    "hidden_dependencies": ["dependencies between different parts of code"]
  }
}"""

        self.edge_case_discovery_prompt = """You are an expert at finding edge cases and potential vulnerabilities that could break code in production.

Based on the code analysis and test results, identify:
1. **Missing Critical Edge Cases**: What scenarios weren't tested that should be?
2. **Security Vulnerabilities**: What could an attacker exploit?
3. **Production Risks**: What could fail in real-world usage?
4. **Stress Test Scenarios**: What happens under extreme conditions?
5. **Integration Issues**: How could this code fail when used with other systems?

Provide comprehensive edge case analysis in JSON:
{
  "critical_missing_edge_cases": [
    {
      "case_description": "specific edge case scenario",
      "risk_level": "low|medium|high|critical",
      "failure_probability": "how likely this is to occur",
      "impact_assessment": "what happens if this case occurs",
      "test_suggestion": "specific test case to add",
      "mitigation_strategy": "how to handle this case in code"
    }
  ],
  "security_vulnerabilities": [
    {
      "vulnerability_type": "buffer overflow|injection|etc",
      "attack_vector": "how an attacker could exploit this",
      "severity": "low|medium|high|critical",
      "affected_code": "specific code sections vulnerable",
      "mitigation": "how to secure against this"
    }
  ],
  "production_risks": [
    {
      "risk_scenario": "what could go wrong in production",
      "trigger_conditions": "what conditions trigger this risk",
      "business_impact": "impact on users/business",
      "monitoring_needed": "what to monitor to detect this",
      "prevention_strategy": "how to prevent this risk"
    }
  ],
  "stress_test_scenarios": [
    {
      "stress_type": "memory|cpu|concurrency|volume",
      "scenario": "specific stress test scenario",
      "expected_failure_mode": "how the code would likely fail",
      "resilience_improvements": "how to make code more resilient"
    }
  ]
}"""

        self.comprehensive_feedback_prompt = """You are a senior software engineering mentor providing comprehensive feedback to help a programmer improve.

Synthesize all previous analysis into actionable, educational feedback that:
1. **Prioritizes Issues**: What should be fixed first and why?
2. **Explains Concepts**: Teach the underlying principles, don't just point out problems
3. **Provides Learning Path**: What should the student study to improve?
4. **Gives Specific Examples**: Show exactly how to improve with code examples
5. **Builds Confidence**: Acknowledge what was done well while being constructive about issues

Create a comprehensive learning-focused report in JSON:
{
  "executive_summary": {
    "overall_assessment": "comprehensive assessment in 2-3 sentences",
    "key_strengths": ["main things done well"],
    "critical_issues": ["most important problems to address"],
    "learning_level": "beginner|intermediate|advanced assessment"
  },
  "prioritized_improvements": [
    {
      "priority": "1-5 (1=highest)",
      "improvement_area": "specific area to improve",
      "why_important": "explanation of why this matters",
      "learning_concepts": ["underlying CS concepts to study"],
      "code_example": "specific code improvement example",
      "resources": ["suggested study materials or topics"]
    }
  ],
  "educational_insights": {
    "concepts_demonstrated": ["CS concepts the student shows understanding of"],
    "concepts_missing": ["fundamental concepts the student needs to learn"],
    "common_mistakes": ["typical beginner mistakes being made"],
    "advanced_techniques": ["more sophisticated approaches they could learn"]
  },
  "mentorship_guidance": {
    "immediate_next_steps": ["what to work on right now"],
    "medium_term_goals": ["what to focus on over next few weeks"],
    "long_term_development": ["areas for ongoing improvement"],
    "confidence_builders": ["positive aspects to build upon"]
  },
  "detailed_explanations": {
    "why_tests_failed": "educational explanation of test failures",
    "algorithmic_thinking": "assessment of problem-solving approach",
    "coding_maturity": "assessment of overall coding skill development",
    "industry_readiness": "how close is this to production-quality code"
  }
}"""

    def extract_metrics(self, eval_results: Dict[str, Any]) -> CodeMetrics:
        """Extract and structure metrics from evaluation results"""
        return CodeMetrics(
            passrate=eval_results.get('passrate', 0),
            weighted_score=eval_results.get('weighted_score', 0),
            memory_score=eval_results.get('memory_score', 0),
            robustness_score=eval_results.get('robustness_score', 0),
            tests_passed=eval_results.get('tests_passed', 0),
            tests_failed=eval_results.get('tests_failed', 0),
            total_tests=eval_results.get('total_tests', 0),
            execution_time_ms=eval_results.get('execution_time_ms', 0),
            failed_tests=eval_results.get('failed_test_details', []),
            potential_edge_cases=eval_results.get('potential_edge_cases', []),
            program_type=eval_results.get('program_type', 'unknown'),
            difficulty_level=eval_results.get('difficulty_level', 'unknown')
        )

    def analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Deep structural analysis of the code"""
        analysis = {}
        
        # Function analysis
        functions = re.findall(r'(\w+\s+)*(\w+)\s*\([^)]*\)\s*{', code)
        analysis['function_count'] = len(functions)
        analysis['has_custom_functions'] = len([f for f in functions if f[1] != 'main']) > 0
        
        # Complexity indicators
        analysis['nested_loops'] = len(re.findall(r'for\s*\([^}]*for\s*\(|while\s*\([^}]*while\s*\(', code))
        analysis['conditional_complexity'] = len(re.findall(r'if\s*\(|else\s+if\s*\(|switch\s*\(', code))
        analysis['loop_count'] = len(re.findall(r'for\s*\(|while\s*\(|do\s*{', code))
        
        # Memory usage patterns
        analysis['dynamic_allocation'] = 'malloc' in code or 'calloc' in code or 'realloc' in code
        analysis['array_usage'] = '[' in code and ']' in code
        analysis['pointer_usage'] = '*' in code and ('char *' in code or 'int *' in code)
        
        # Error handling patterns
        analysis['error_checking'] = bool(re.search(r'if\s*\([^)]*==\s*NULL|if\s*\([^)]*!=\s*1\)', code))
        analysis['return_value_checking'] = 'scanf' in code and 'if' in code
        
        # Code style metrics
        analysis['line_count'] = len(code.split('\n'))
        analysis['comment_lines'] = len([line for line in code.split('\n') if '//' in line or '/*' in line])
        analysis['average_function_length'] = analysis['line_count'] // max(analysis['function_count'], 1)
        
        return analysis

    def call_llm_with_retry(self, messages: List[Dict], max_retries: int = 3) -> str:
        """Call LLM with retry logic and error handling"""
        for attempt in range(max_retries):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": 0.1,  # Very low temperature for consistent analysis
                        "top_p": 0.8,
                        "num_predict": 4096,  # Allow very long responses
                        "repeat_penalty": 1.1
                    }
                )
                return response['message']['content'].strip()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response with error handling"""
        # Try to find JSON in the response
        json_patterns = [
            r'\{.*\}',  # Basic JSON pattern
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            r'```\s*(\{.*?\})\s*```'  # JSON in generic code blocks
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                json_text = matches[0] if isinstance(matches[0], str) else matches[0]
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    continue
        
        # If no JSON found, try the whole response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON from response: {e}")
            print(f"Response text: {response_text[:500]}...")
            return {"error": "Failed to parse LLM response", "raw_response": response_text}

    def stage_1_code_understanding(self, source_code: str, metrics: CodeMetrics) -> Dict[str, Any]:
        """Stage 1: Deep code understanding and algorithm analysis"""
        print("🔍 Stage 1: Deep Code Understanding & Algorithm Analysis...")
        
        # Add structural analysis context
        structure_analysis = self.analyze_code_structure(source_code)
        
        context = f"""
CODE STRUCTURAL ANALYSIS:
- Function count: {structure_analysis['function_count']}
- Has custom functions: {structure_analysis['has_custom_functions']}
- Nested loops: {structure_analysis['nested_loops']}
- Conditional complexity: {structure_analysis['conditional_complexity']}
- Loop count: {structure_analysis['loop_count']}
- Uses dynamic allocation: {structure_analysis['dynamic_allocation']}
- Uses arrays: {structure_analysis['array_usage']}
- Uses pointers: {structure_analysis['pointer_usage']}
- Has error checking: {structure_analysis['error_checking']}
- Line count: {structure_analysis['line_count']}
- Comment ratio: {structure_analysis['comment_lines']}/{structure_analysis['line_count']}

EVALUATION CONTEXT:
- Program Type: {metrics.program_type}
- Difficulty Level: {metrics.difficulty_level}
- Pass Rate: {metrics.passrate}% ({metrics.tests_passed}/{metrics.total_tests})
- Memory Score: {metrics.memory_score}
- Execution Time: {metrics.execution_time_ms}ms

C SOURCE CODE TO ANALYZE:
```c
{source_code}
```

Perform deep algorithmic and structural analysis following the JSON format specified.
"""

        messages = [
            {"role": "system", "content": self.code_understanding_prompt},
            {"role": "user", "content": context}
        ]
        
        response_text = self.call_llm_with_retry(messages)
        analysis = self.extract_json_from_response(response_text)
        
        # Store conversation for context
        self.analysis_conversations.append({
            "stage": "code_understanding",
            "response": response_text
        })
        
        return analysis

    def stage_2_failure_analysis(self, source_code: str, metrics: CodeMetrics, code_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Deep analysis of test failures"""
        print("🔍 Stage 2: Test Failure Analysis...")
        
        if not metrics.failed_tests:
            return {
                "failure_pattern_analysis": {
                    "common_root_causes": [],
                    "severity_assessment": "No test failures - excellent!",
                    "fix_priority": []
                },
                "detailed_failure_analysis": [],
                "cascading_effects": {
                    "issues_that_could_cause_multiple_failures": [],
                    "hidden_dependencies": []
                }
            }
        
        context = f"""
PREVIOUS CODE ANALYSIS INSIGHTS:
{json.dumps(code_analysis, indent=2)}

FAILED TEST DETAILS:
{chr(10).join(metrics.failed_tests)}

PERFORMANCE METRICS:
- Memory Score: {metrics.memory_score} (100 = no leaks, 0 = has leaks)
- Robustness Score: {metrics.robustness_score} (how well handles edge cases)
- Execution Time: {metrics.execution_time_ms}ms

C SOURCE CODE:
```c
{source_code}
```

Analyze these test failures in depth to understand root causes and patterns.
"""

        messages = [
            {"role": "system", "content": self.failure_analysis_prompt},
            {"role": "user", "content": context}
        ]
        
        response_text = self.call_llm_with_retry(messages)
        analysis = self.extract_json_from_response(response_text)
        
        self.analysis_conversations.append({
            "stage": "failure_analysis", 
            "response": response_text
        })
        
        return analysis

    def stage_3_edge_case_discovery(self, source_code: str, metrics: CodeMetrics, 
                                    code_analysis: Dict[str, Any], failure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Advanced edge case and vulnerability discovery"""
        print("🔍 Stage 3: Edge Case & Vulnerability Discovery...")
        
        context = f"""
COMPREHENSIVE ANALYSIS CONTEXT:

CODE ANALYSIS INSIGHTS:
{json.dumps(code_analysis, indent=2)}

FAILURE ANALYSIS INSIGHTS:  
{json.dumps(failure_analysis, indent=2)}

KNOWN EDGE CASES IDENTIFIED BY STAGE 1:
{chr(10).join(metrics.potential_edge_cases)}

PROGRAM CHARACTERISTICS:
- Type: {metrics.program_type}
- Difficulty: {metrics.difficulty_level}
- Current Pass Rate: {metrics.passrate}%
- Memory Issues: {"Yes" if metrics.memory_score < 100 else "No"}
- Robustness Issues: {"Yes" if metrics.robustness_score < 100 else "No"}

C SOURCE CODE:
```c
{source_code}
```

Based on this comprehensive analysis, identify critical missing edge cases, security vulnerabilities, and production risks that haven't been covered yet.
"""

        messages = [
            {"role": "system", "content": self.edge_case_discovery_prompt},
            {"role": "user", "content": context}
        ]
        
        response_text = self.call_llm_with_retry(messages)
        analysis = self.extract_json_from_response(response_text)
        
        self.analysis_conversations.append({
            "stage": "edge_case_discovery",
            "response": response_text
        })
        
        return analysis

    def stage_4_comprehensive_feedback(self, source_code: str, metrics: CodeMetrics,
                                       code_analysis: Dict[str, Any], failure_analysis: Dict[str, Any],
                                       edge_case_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 4: Synthesize everything into comprehensive educational feedback"""
        print("🔍 Stage 4: Comprehensive Educational Feedback Synthesis...")
        
        context = f"""
COMPLETE ANALYSIS SYNTHESIS:

ALGORITHM & CODE STRUCTURE ANALYSIS:
{json.dumps(code_analysis, indent=2)}

FAILURE ANALYSIS:
{json.dumps(failure_analysis, indent=2)}

EDGE CASE & SECURITY ANALYSIS:
{json.dumps(edge_case_analysis, indent=2)}

QUANTITATIVE METRICS:
- Overall Pass Rate: {metrics.passrate}% ({metrics.tests_passed}/{metrics.total_tests} tests)
- Weighted Score: {metrics.weighted_score}%
- Memory Management: {metrics.memory_score}/100
- Robustness: {metrics.robustness_score}/100
- Execution Time: {metrics.execution_time_ms}ms
- Failed Tests: {len(metrics.failed_tests)}

ORIGINAL SOURCE CODE:
```c
{source_code}
```

Synthesize all this analysis into comprehensive, educational feedback that helps the programmer understand not just what's wrong, but WHY it's wrong and HOW to improve. Focus on teaching underlying principles and providing a clear learning path.
"""

        messages = [
            {"role": "system", "content": self.comprehensive_feedback_prompt},
            {"role": "user", "content": context}
        ]
        
        response_text = self.call_llm_with_retry(messages)
        analysis = self.extract_json_from_response(response_text)
        
        self.analysis_conversations.append({
            "stage": "comprehensive_feedback",
            "response": response_text
        })
        
        return analysis

    def calculate_final_score(self, metrics: CodeMetrics, comprehensive_analysis: Dict[str, Any]) -> Tuple[str, float]:
        """Calculate final grade and score based on comprehensive analysis"""
        
        # Base scores from metrics
        base_score = (metrics.weighted_score + metrics.memory_score + metrics.robustness_score) / 3
        
        # Adjust based on comprehensive analysis
        adjustments = 0
        
        # Positive adjustments
        exec_summary = comprehensive_analysis.get('executive_summary', {})
        if 'advanced' in exec_summary.get('learning_level', '').lower():
            adjustments += 5
        elif 'intermediate' in exec_summary.get('learning_level', '').lower():
            adjustments += 2
            
        # Negative adjustments for critical issues
        critical_issues = len(exec_summary.get('critical_issues', []))
        if critical_issues > 3:
            adjustments -= 15
        elif critical_issues > 1:
            adjustments -= 8
            
        # Security vulnerability penalty
        edge_analysis = comprehensive_analysis.get('stage_3_edge_case_analysis', {})
        security_vulns = edge_analysis.get('security_vulnerabilities', [])
        critical_vulns = [v for v in security_vulns if v.get('severity') == 'critical']
        adjustments -= len(critical_vulns) * 15
        adjustments -= len(security_vulns) * 3
        
        final_score = max(0, min(100, base_score + adjustments))
        
        # Determine grade
        if final_score >= 95: grade = "A+"
        elif final_score >= 90: grade = "A"
        elif final_score >= 85: grade = "A-"
        elif final_score >= 80: grade = "B+"
        elif final_score >= 75: grade = "B"
        elif final_score >= 70: grade = "B-"
        elif final_score >= 65: grade = "C+"
        elif final_score >= 60: grade = "C"
        elif final_score >= 55: grade = "C-"
        elif final_score >= 50: grade = "D"
        else: grade = "F"
            
        return grade, final_score

    def perform_comprehensive_analysis(self, source_file: str, results_file: str) -> Dict[str, Any]:
        """Main analysis orchestrator - runs all 4 stages"""
        
        print("🚀 Starting Comprehensive Multi-Stage Analysis...")
        print("="*80)
        
        # Load data
        try:
            with open(source_file, 'r') as f:
                source_code = f.read()
            with open(results_file, 'r') as f:
                eval_results = json.load(f)
        except Exception as e:
            return {"error": f"Failed to load files: {e}"}
        
        metrics = self.extract_metrics(eval_results)
        
        # Stage 1: Deep Code Understanding
        code_analysis = self.stage_1_code_understanding(source_code, metrics)
        
        # Stage 2: Test Failure Analysis  
        failure_analysis = self.stage_2_failure_analysis(source_code, metrics, code_analysis)
        
        # Stage 3: Edge Case Discovery
        edge_case_analysis = self.stage_3_edge_case_discovery(source_code, metrics, code_analysis, failure_analysis)
        
        # Stage 4: Comprehensive Feedback
        comprehensive_feedback = self.stage_4_comprehensive_feedback(source_code, metrics, code_analysis, 
                                                                     failure_analysis, edge_case_analysis)
        
        # Calculate final grade
        final_grade, final_score = self.calculate_final_score(metrics, comprehensive_feedback)
        
        # Combine all analyses
        complete_analysis = {
            "meta_information": {
                "analysis_timestamp": time.time(),
                "source_file": source_file,
                "model_used": self.model_name,
                "analysis_stages_completed": 4
            },
            "quantitative_metrics": {
                "passrate": metrics.passrate,
                "weighted_score": metrics.weighted_score,
                "memory_score": metrics.memory_score,
                "robustness_score": metrics.robustness_score,
                "tests_passed": metrics.tests_passed,
                "tests_failed": metrics.tests_failed,
                "total_tests": metrics.total_tests,
                "execution_time_ms": metrics.execution_time_ms
            },
            "final_assessment": {
                "grade": final_grade,
                "score": final_score,
                "assessment_confidence": "high"  # Based on multi-stage analysis
            },
            "stage_1_code_analysis": code_analysis,
            "stage_2_failure_analysis": failure_analysis, 
            "stage_3_edge_case_analysis": edge_case_analysis,
            "stage_4_comprehensive_feedback": comprehensive_feedback,
            "conversation_history": self.analysis_conversations
        }
        
        print("="*80)
        print("🎉 Multi-Stage Analysis Complete!")
        
        return complete_analysis

    def generate_executive_report(self, analysis: Dict[str, Any], output_file: str):
        """Generate a comprehensive executive report"""
        
        with open(output_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("                                COMPREHENSIVE CODE ANALYSIS REPORT\n")
            f.write("                                     Advanced Multi-Stage Analysis\n")
            f.write("="*100 + "\n\n")
            
            # Executive Summary
            final_assessment = analysis.get('final_assessment', {})
            f.write(f"🎓 FINAL GRADE: {final_assessment.get('grade', 'N/A')}\n")
            f.write(f"📊 COMPREHENSIVE SCORE: {final_assessment.get('score', 0):.1f}/100\n")
            f.write(f"🔍 ANALYSIS CONFIDENCE: {final_assessment.get('assessment_confidence', 'unknown').upper()}\n\n")
            
            # Quantitative Summary
            metrics = analysis.get('quantitative_metrics', {})
            f.write("📈 QUANTITATIVE METRICS SUMMARY:\n")
            f.write("-" * 50 + "\n")
            f.write(f"    Test Success Rate: {metrics.get('passrate', 0):.1f}% ({metrics.get('tests_passed', 0)}/{metrics.get('total_tests', 0)} tests)\n")
            f.write(f"    Weighted Test Score: {metrics.get('weighted_score', 0):.1f}%\n")
            f.write(f"    Memory Management: {metrics.get('memory_score', 0)}/100\n")
            f.write(f"    Robustness Score: {metrics.get('robustness_score', 0)}/100\n")
            f.write(f"    Execution Time: {metrics.get('execution_time_ms', 0)}ms\n\n")
            
            # Write each stage analysis
            self._write_stage_analysis(f, "STAGE 1: ALGORITHM & CODE STRUCTURE ANALYSIS", 
                                       analysis.get('stage_1_code_analysis', {}))
            
            self._write_stage_analysis(f, "STAGE 2: TEST FAILURE ANALYSIS", 
                                       analysis.get('stage_2_failure_analysis', {}))
            
            self._write_stage_analysis(f, "STAGE 3: EDGE CASE & SECURITY ANALYSIS", 
                                       analysis.get('stage_3_edge_case_analysis', {}))
            
            self._write_stage_analysis(f, "STAGE 4: COMPREHENSIVE EDUCATIONAL FEEDBACK", 
                                       analysis.get('stage_4_comprehensive_feedback', {}))
            
            f.write("="*100 + "\n")
            f.write("                                        END OF COMPREHENSIVE ANALYSIS\n")
            f.write("="*100 + "\n")

    def _write_stage_analysis(self, file, stage_title: str, stage_data: Dict[str, Any]):
        """Write individual stage analysis to report"""
        file.write(f"\n🔍 {stage_title}\n")
        file.write("="*100 + "\n")
        
        if not stage_data or "error" in stage_data:
            file.write("❌ Analysis stage failed or returned incomplete data.\n\n")
            return
            
        # Write stage-specific content based on structure
        if "algorithm_analysis" in stage_data:
            self._write_algorithm_analysis(file, stage_data)
        elif "failure_pattern_analysis" in stage_data:
            self._write_failure_analysis(file, stage_data)
        elif "critical_missing_edge_cases" in stage_data:
            self._write_edge_case_analysis(file, stage_data)
        elif "executive_summary" in stage_data:
            self._write_comprehensive_feedback(file, stage_data)
        else:
            # Generic fallback
            file.write(json.dumps(stage_data, indent=2) + "\n")
            
        file.write("\n")

    def _write_algorithm_analysis(self, file, data: Dict[str, Any]):
        """Write algorithm analysis section"""
        for key, value in data.items():
            title = ' '.join(word.capitalize() for word in key.split('_'))
            file.write(f"\n--- {title} ---\n")
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    sub_title = ' '.join(word.capitalize() for word in sub_key.split('_'))
                    file.write(f"  {sub_title}: {sub_val}\n")
            elif isinstance(value, list):
                for item in value:
                    file.write(f"- {item}\n")
            else:
                 file.write(f"{value}\n")


    def _write_failure_analysis(self, file, data: Dict[str, Any]):
        """Write failure analysis section"""
        patterns = data.get('failure_pattern_analysis', {})
        if patterns:
            file.write("\n--- Failure Pattern Analysis ---\n")
            for key, val in patterns.items():
                title = ' '.join(word.capitalize() for word in key.split('_'))
                file.write(f"  {title}: {val}\n")

        details = data.get('detailed_failure_analysis', [])
        if details:
            file.write("\n--- Detailed Failure Analysis ---\n")
            for i, failure in enumerate(details):
                file.write(f"\n  Failure #{i+1}:\n")
                for key, val in failure.items():
                    title = ' '.join(word.capitalize() for word in key.split('_'))
                    file.write(f"    {title}: {val}\n")

    def _write_edge_case_analysis(self, file, data: Dict[str, Any]):
        """Write edge case analysis section"""
        for key, value in data.items():
             title = ' '.join(word.capitalize() for word in key.split('_'))
             file.write(f"\n--- {title} ---\n")
             if isinstance(value, list) and value:
                 for i, item in enumerate(value):
                     file.write(f"\n  Item #{i+1}:\n")
                     for sub_key, sub_val in item.items():
                         sub_title = ' '.join(word.capitalize() for word in sub_key.split('_'))
                         file.write(f"    {sub_title}: {sub_val}\n")
             else:
                 file.write("  None identified.\n")


    def _write_comprehensive_feedback(self, file, data: Dict[str, Any]):
        """Write comprehensive feedback section"""
        summary = data.get('executive_summary', {})
        if summary:
            file.write("\n--- Executive Summary ---\n")
            for key, val in summary.items():
                title = ' '.join(word.capitalize() for word in key.split('_'))
                file.write(f"  {title}: {val}\n")

        improvements = data.get('prioritized_improvements', [])
        if improvements:
            file.write("\n--- Prioritized Improvements ---\n")
            for item in sorted(improvements, key=lambda x: x.get('priority', 99)):
                file.write(f"\n  Priority {item.get('priority')}: {item.get('improvement_area')}\n")
                file.write(f"    Why Important: {item.get('why_important')}\n")
                file.write(f"    Learning Concepts: {', '.join(item.get('learning_concepts', []))}\n")
                file.write(f"    Example: \n```c\n{item.get('code_example', 'N/A')}\n```\n")

        insights = data.get('educational_insights', {})
        if insights:
            file.write("\n--- Educational Insights ---\n")
            for key, val in insights.items():
                title = ' '.join(word.capitalize() for word in key.split('_'))
                file.write(f"  {title}: {val}\n")

        guidance = data.get('mentorship_guidance', {})
        if guidance:
            file.write("\n--- Mentorship Guidance ---\n")
            for key, val in guidance.items():
                title = ' '.join(word.capitalize() for word in key.split('_'))
                file.write(f"  {title}: {val}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 advanced_code_analyzer.py <source_file.c> <results_file.json>")
        sys.exit(1)

    source_file = sys.argv[1]
    results_file = sys.argv[2]
    
    analyzer = AdvancedCodeAnalyzer()
    
    # Perform the full multi-stage analysis
    complete_analysis = analyzer.perform_comprehensive_analysis(source_file, results_file)
    
    if "error" in complete_analysis:
        print(f"An error occurred: {complete_analysis['error']}")
        sys.exit(1)

    # Define output file names
    json_output_file = "comprehensive_analysis.json"
    report_output_file = "feedback_report.txt"

    # Save the raw JSON output
    with open(json_output_file, 'w') as f:
        json.dump(complete_analysis, f, indent=2)
    print(f"\n✅ Raw analysis data saved to {json_output_file}")

    # Generate and save the human-readable report
    analyzer.generate_executive_report(complete_analysis, report_output_file)
    print(f"✅ Human-readable feedback report saved to {report_output_file}")
