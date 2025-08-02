#!/usr/bin/env python3
"""
Stage 1: Intelligent Test Case Generator using CodeLlama
Analyzes C code and generates appropriate test cases dynamically
"""

import json
import subprocess
import sys
import re
from typing import List, Dict, Any
import ollama  # For CodeLlama integration

class TestCaseGenerator:
    def __init__(self, model_name="codellama:7b"):
        self.model_name = model_name
        self.system_prompt = """You are an expert C code analyzer and test case generator.

Your task is to:
1. Analyze the given C code to understand its purpose and functionality
2. Generate comprehensive test cases that cover:
   - Normal/expected inputs
   - Edge cases (empty input, boundary values, large inputs)
   - Error conditions (invalid input, overflow scenarios)
   - Corner cases specific to the algorithm

Return ONLY a valid JSON object in this EXACT format:
{
  "program_description": "Brief description of what the program does",
  "program_type": "calculator|string_processor|mathematical|io_handler|data_structure|other",
  "difficulty_level": "basic|intermediate|advanced",
  "test_cases": [
    {
      "input": "exact input string to send to program",
      "expected_output": "exact expected output",
      "description": "what this test checks",
      "category": "normal|edge|error|corner",
      "weight": 1.0
    }
  ],
  "potential_edge_cases": [
    "Description of edge cases to watch for"
  ]
}

IMPORTANT:
- Generate 5-8 test cases minimum
- Include at least 2 edge cases
- For mathematical programs, test boundary values (0, negative, large numbers)
- For string programs, test empty strings, whitespace, special characters
- For interactive programs, test invalid input scenarios
- Make inputs realistic and outputs precise"""

    def analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Pre-analyze code to give LLM more context"""
        analysis = {
            "has_scanf": "scanf" in code,
            "has_printf": "printf" in code,
            "has_loops": any(keyword in code for keyword in ["for", "while", "do"]),
            "has_arrays": "[" in code and "]" in code,
            "has_math": any(op in code for op in ["+", "-", "*", "/", "%"]),
            "has_strings": "char" in code and any(func in code for func in ["strlen", "strcpy", "strcmp", "fgets"]),
            "main_function": "int main" in code,
            "includes": re.findall(r'#include\s*<([^>]+)>', code)
        }
        return analysis

    def generate_test_cases(self, source_file: str) -> Dict[str, Any]:
        """Generate test cases using CodeLlama"""
        try:
            # Read the source code
            with open(source_file, 'r') as f:
                code = f.read()
            
            # Pre-analyze the code
            code_analysis = self.analyze_code_structure(code)
            
            # Create enhanced prompt with code analysis
            enhanced_prompt = f"""
CODE ANALYSIS CONTEXT:
- Uses scanf: {code_analysis['has_scanf']}
- Uses printf: {code_analysis['has_printf']}
- Has loops: {code_analysis['has_loops']}
- Has arrays: {code_analysis['has_arrays']}
- Has math operations: {code_analysis['has_math']}
- Has string operations: {code_analysis['has_strings']}
- Includes: {', '.join(code_analysis['includes'])}

C CODE TO ANALYZE:
```c
{code}
```

Generate comprehensive test cases for this C program following the JSON format specified in the system prompt.
"""

            # Call CodeLlama via Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": enhanced_prompt}
                ],
                options={
                    "temperature": 0.3,  # Lower temperature for more consistent output
                    "top_p": 0.9,
                    "num_predict": 2048
                }
            )
            
            # Extract JSON from response
            response_text = response['message']['content'].strip()
            
            # Try to extract JSON if it's wrapped in markdown or other text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                json_text = response_text
            
            # Parse and validate JSON
            test_data = json.loads(json_text)
            
            # Validate required fields
            required_fields = ["program_description", "test_cases"]
            for field in required_fields:
                if field not in test_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure we have test cases
            if len(test_data["test_cases"]) == 0:
                raise ValueError("No test cases generated")
            
            # Add metadata
            test_data["source_file"] = source_file
            test_data["generation_method"] = "codellama_analysis"
            test_data["code_analysis"] = code_analysis
            
            return test_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response_text}")
            return self._generate_fallback_tests(source_file)
        except Exception as e:
            print(f"Error generating test cases: {e}")
            return self._generate_fallback_tests(source_file)

    def _generate_fallback_tests(self, source_file: str) -> Dict[str, Any]:
        """Generate basic fallback tests if LLM fails"""
        return {
            "program_description": "Unknown program (LLM analysis failed)",
            "program_type": "other",
            "difficulty_level": "basic",
            "test_cases": [
                {
                    "input": "5\n",
                    "expected_output": "5",
                    "description": "Basic numeric input test",
                    "category": "normal",
                    "weight": 1.0
                },
                {
                    "input": "\n",
                    "expected_output": "",
                    "description": "Empty input test",
                    "category": "edge",
                    "weight": 1.0
                }
            ],
            "potential_edge_cases": ["Input validation", "Boundary conditions"],
            "source_file": source_file,
            "generation_method": "fallback"
        }

    def save_test_cases(self, test_data: Dict[str, Any], output_file: str):
        """Save generated test cases to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"✅ Generated {len(test_data['test_cases'])} test cases")
        print(f"📄 Program type: {test_data.get('program_type', 'unknown')}")
        print(f"💾 Test cases saved to: {output_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python test_generator.py <source.c> <output_tests.json>")
        sys.exit(1)
    
    source_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print("🔍 Analyzing C code with CodeLlama...")
    generator = TestCaseGenerator()
    test_data = generator.generate_test_cases(source_file)
    generator.save_test_cases(test_data, output_file)

if __name__ == "__main__":
    main()
