import json
import ollama
from typing import List, Dict, Tuple, Optional

class AutoPrompt:
    def __init__(self, model: str = "llama3.1"):
        """Initialize the AutoPrompt system.
        
        Args:
            model: The Ollama model to use
        """
        self.model = model
        
    def generate_completion(self, prompt: str) -> str:
        """Generate completion using Ollama API.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The generated response text
        """
        response = ollama.generate(
            model=self.model,
            prompt=prompt
        )
        return response['response']
    
    def task_analysis(self, task_description: str) -> Dict:
        """Analyze the task and generate structured analysis.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Dict containing structured task analysis
        """
        analysis_prompt = f"""You are a task analysis expert. I will provide you with basic task information. Please help me analyze the task following this template:
{task_description}
1. First, summarize the basic task information I provided.

2. Then, analyze the task using this structure:
   - Task Description: Provide a clear and detailed explanation of the task
   - Task Goal: Define specific, measurable objectives
   - Input Data Features: Analyze what information/data is needed
   - Output Data Structure: Describe how the results should be organized
   - Technical Requirements: List necessary technical conditions
   - Challenges & Limitations: Identify potential difficulties
   - Evaluation Metrics: Suggest how to measure success

3. Present your analysis in a well-organized format with clear headings.

Remember to:
- Be specific and practical
- Focus on key elements
- Highlight critical requirements"""

        response = self.generate_completion(analysis_prompt)
        try:
            return json.loads(response)
        except:
            # Fallback if response isn't valid JSON
            return {"error": "Failed to parse analysis"}

    def generate_prompt(self, task_analysis: Dict) -> str:
        """Generate prompts.
        
        Args:
            task_analysis: The structured task analysis
            
        Returns:
            prompts
        """
        template = f"""As an expert prompt engineer, generate 5 prompt variations for the given task. Each should employ a different approach.
{json.dumps(task_analysis)}
For each variation:
1. Include essential components:
   - Clear task description
   - Key constraints
   - Relevant examples
   - Expected output format

2. Optimize for:
   - Brevity: Remove redundant words
   - Clarity: Use clear structure
   - Completeness: Cover all requirements
   - Consistency: Maintain uniform style

Structure your response as:
[Variation 1:]
[Prompt content]
...

Note:
- Ensure each variation takes a distinct approach
- Balance detail with conciseness"""

        return self.generate_completion(template)

    def generate_challenge_analysis(self, task_analysis: Dict) -> str:
        """Generate challenge analysis for specific task.
        
        Args:
            task_analysis: The structured task analysis
            
        Returns:
            Analysis of challenges and limitations
        """
        template = f"""Given the following task description:
        {json.dumps(task_analysis)}

        Please analyze all potential technical challenges and risks in implementing this functionality. Focus on:
        1. Core technical complexities
        2. External dependencies
        3. Performance bottlenecks
        4. Security concerns
        5. Data handling risks

        Present your analysis in a structured format.
        """
        return self.generate_completion(template)

    def generate_error_scenarios(self, challanges: str) -> str:
        """Generate error scenarios for specific task.
        
        Args:
            challanges: The challanges of the task
            
        Returns:
            Error scenarios
        """
        template = f"""Based on the previous challenge analysis, identify specific error-prone scenarios that require testing.
        {challanges}
        For each scenario, please provide:
        1. Scenario description
        2. Why it's risky
        3. What could go wrong
        4. Potential impact
        """
        return self.generate_completion(template)
    
    def generate_test_cases(self, error_scenario: str) -> List[Dict]:
        """Generate comprehensive test cases for specific task.
        
        Args:
            error_scenario: The structured task analysis
            
        Returns:
            List of test cases with inputs and expected outputs
        """
        test_prompt = f"""
        For the identified error-prone scenarios: {error_scenario}
        Generate detailed test case that verifies the handling of this scenario. The output should be in the following JSON format:

        {{
        "testCase": {{
            "id": "TC_[number]",
            "description": "[One sentence about what this test verifies]",
            "input": {{
            "field1": "value1",
            "field2": "value2"
            }},
            "expectedOutput": {{
            "result": "value",
            "status": "success/failure"
            }}
        }}
        }}
        """


        response = self.generate_completion(test_prompt)
        try:
            return json.loads(response)
        except:
            return [{"error": "Failed to parse test cases"}]

    def prompt_analysis(self, prompt: str) -> Dict:
        """Analyze the prompt and generate structured analysis.
        
        Args:
            prompt: Description of the task

        Returns:
            prompt analysis
        """
        analysis_prompt = f"""Analyze the following prompt:
            {prompt}

            Your analysis should:
            1. Evaluate prompt clarity and structure
            2. Identify potential ambiguities or gaps
            3. Assess constraints and limitations
            4. Review expected outputs and edge cases

            Please provide your analysis in the following format:
            - Strengths:
            - [List key strengths]
            - Weaknesses:
            - [List potential issues]
            - Improvement suggestions:
            - [List specific recommendations]
            - Edge cases to consider:
            - [List potential edge cases]
            """
        return self.generate_completion(analysis_prompt)

    def prompt_combination(self, prompts: List[str]) -> str:
        """Combine multiple prompts into a single prompt.
        
        Args:
            prompts: List of prompts to combine
            
        Returns:
            Combined prompt
        """
        template = f"""
        Combine the following prompts while maintaining their core functionalities:

        {'\n'.join(f'Prompt {i+1}:\n{prompt}' for i, prompt in enumerate(prompts))}

        Requirements:
        1. Maintain all critical functionality from original prompts
        2. Eliminate redundancies
        3. Ensure clarity and conciseness
        4. Preserve any essential constraints

        Please provide:
        1. Combined prompt
        2. Explanation of integration choices
        3. List of preserved functionalities
        4. Any potential trade-offs made
        """
        return self.generate_completion(template)
    
    def optimize_prompt(self, prompt: str, evaluation_results: str) -> str:
        """Optimize prompt based on evaluation results.
        
        Args:
            prompt: Original prompt
            evaluation_results: Results from evaluation
            
        Returns:
            Optimized prompt
        """
        template = f"""Based on the following evaluation results:
        {evaluation_results}

        prompt: {prompt}
        
        Please optimize the prompt by:
        1. Addressing identified issues
        2. Incorporating successful patterns
        3. Removing ineffective elements

        Please provide:
        1. Optimized prompt version
        2. Changelog detailing modifications
        3. Rationale for each major change
        4. Expected impact on performance
        5. Suggested verification tests"""

        return self.generate_completion(template)

    
    def evaluate_prompt(self, prompt: str, test_cases: List[Dict]) -> Dict:
        """Evaluate a prompt's performance on test cases.
        
        Args:
            prompt: The prompt to evaluate
            test_cases: List of test cases
            
        Returns:
            Evaluation metrics
        """
        results = []
        for test in test_cases:
            full_prompt = f"{prompt}\n\nInput: {test['input']}"
            response = self.generate_completion(full_prompt)
            results.append({
                "test_case": test,
                "response": response,
                "matches_expected": response.strip() == test["expected"].strip()
            })
            
        return {
            "total_cases": len(test_cases),
            "successful_cases": sum(1 for r in results if r["matches_expected"]),
            "detailed_results": results
        }

