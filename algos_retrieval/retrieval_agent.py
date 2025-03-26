import os
import json
import re
import csv
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from anthropic import Anthropic
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prover.lean.verifier import Lean4ServerScheduler

load_dotenv()

@dataclass
class TestCase:
    name: str
    header: str
    informal_prefix: str
    formal_statement: str
    goal: str

@dataclass
class ProofAttempt:
    informal_proof: str
    formal_proof: str
    success: bool
    error_message: Optional[str] = None

class ProofGenerator:
    """Responsible for generating informal mathematical proofs"""
    def __init__(self, model_type="o1", temperature=0):
        self.model_name = model_type
        if "gpt" in model_type.lower():
            self.llm = ChatOpenAI(
                model_name=model_type,
                temperature=temperature,
                openai_api_key=os.environ.get('OPENAI_API_KEY')
            )
            self.use_anthropic = False
        elif "o1" in model_type.lower():
            self.llm = ChatOpenAI(
                model_name=model_type,
                openai_api_key=os.environ.get('OPENAI_API_KEY')
            )
            self.use_anthropic = False
        else:
            self.client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
            self.model_name = model_type
            self.temperature = temperature
            self.use_anthropic = True
            
    def generate_proof(self, context: str, problem: str) -> str:
        if not self.model_name.lower().startswith("o1"):
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a mathematics expert focused on generating clear informal proofs."),
                ("user", """Given the following mathematical problem, generate a clear and detailed informal proof in natural language.
Do not attempt to formalize the proof - focus only on explaining the mathematical reasoning clearly.

Problem:
{problem}

Provide your proof in the following format:

# Informal Proof:
[Your natural language proof here]
""")
        ])
        else:
            prompt_template = ChatPromptTemplate.from_messages(
                ("user", """You are a mathematics expert focused on generating clear informal proofs.
Given the following mathematical problem, generate a clear and detailed informal proof in natural language.
Do not attempt to formalize the proof - focus only on explaining the mathematical reasoning clearly.

Problem:
{problem}

Provide your proof in the following format:

# Informal Proof:
[Your natural language proof here]
""")
            )
        if self.use_anthropic:
            response = self.client.messages.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt_template}]
            )
            return response.content[0].text.strip()
        else:
            chain = prompt_template | self.llm | StrOutputParser()
            return chain.invoke({
                "context": context,
                "problem": problem
            }).strip()

class AutoFormalizer:
    """Responsible for converting informal proofs to Lean 4 formal proofs"""
    def __init__(self, model_name="deepseek-ai/DeepSeek-Prover-V1.5-RL", temperature=0.0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = LLM(model=model_name, max_num_batched_tokens=8192, seed=1, trust_remote_code=True)
        self.temperature = temperature
    
    def formalize_proof(self, header: str, informal_proof: str, informal_prefix: str, formal_statement: str, goal: str) -> str:
        prompt = f"""You are a Lean 4 code generator. 
We have:
  HEADER:
{header}

  INFORMAL PROOF:
{informal_proof}

  PREFIX:
{informal_prefix}

  STATEMENT:
{formal_statement}

GOAL (optional):
{goal}

INSTRUCTIONS:
1. Output exactly one triple-backtick code block containing valid Lean 4 code.
2. Do not include any text or explanations outside the code block.
3. Make sure it compiles in Lean 4.

Required Format:
# Start
```lean4
<Lean code here>
```  # End
"""

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=2048,
            top_p=0.1,
            n=1,
        )

        # Generate the formal proof using the DeepSeek model
        model_outputs = self.model.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
        )

        generated_text = model_outputs[0].outputs[0].text

        formal_proof = informal_prefix + "\ngenerated text\n" + generated_text
        print(formal_proof)

        # Return the formal proof
        return generated_text.strip()

class TwoAgentProver:
    def __init__(
        self,
        lean4_scheduler,
        proof_generator: ProofGenerator,
        auto_formalizer: AutoFormalizer,
        max_depth=3,
        top_k=5,
        max_attempts=3,
        log_file=None
    ):
        self.lean4_scheduler = lean4_scheduler
        self.proof_generator = proof_generator
        self.auto_formalizer = auto_formalizer
        self.max_depth = max_depth
        self.max_attempts = max_attempts
        
        # Set up logging
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'two_agent_prover_log_{timestamp}.csv'
        self.log_file = log_file
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prompt', 'depth', 'attempt', 'informal_proof', 'formal_proof', 'passed'])
    
    def __del__(self):
        pass

    def _verify_lean_proof(self, formal_proof: str) -> Tuple[bool, Optional[Dict]]:
        match = re.search(r'```lean4?\n(.*?)\n```', formal_proof, re.DOTALL)
        if not match:
            return False, {"errors": ["No code block found"]}

        request_id_list = self.lean4_scheduler.submit_all_request([match.group(1)])
        outputs_list = self.lean4_scheduler.get_all_request_outputs(request_id_list)
        result = outputs_list[0]
        print(result)
        return (result['pass'] == True and result['complete'] == True), result

    def _log_attempt(self, problem: str, prompt: str, depth: int, attempt: int, 
                    informal_proof: str, formal_proof: str, passed: bool):
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            informal_escaped = informal_proof.replace('\n', '\\n')
            formal_escaped = formal_proof.replace('\n', '\\n')
            writer.writerow([problem, prompt, depth, attempt, informal_escaped, formal_escaped, passed])

    def process_test_case(self, test_case: TestCase) -> Dict:
        # Without RAG, we do not vary by depth based on any retrieved context. 
        # We can still try multiple depths if desired, but here the depth won't matter.
        for depth in range(self.max_depth + 1):
            print(f"\nTrying at depth {depth}")
            
            # In a no-RAG scenario, we do not retrieve any context.
            # We pass an empty context to the proof generator.
            rag_context = ""

            # Generate informal proof with empty context
            informal_proof = self.proof_generator.generate_proof(
                rag_context, 
                test_case.informal_prefix
            )
            
            # Try multiple formalization attempts
            for attempt in range(self.max_attempts):
                try:
                    print(f"Processing {test_case.name}, Depth {depth}, Attempt {attempt + 1}")
                    
                    # Formalize the proof
                    formal_proof = self.auto_formalizer.formalize_proof(
                        test_case.header,
                        informal_proof,
                        test_case.informal_prefix,
                        test_case.formal_statement,
                        test_case.goal
                    )
                    
                    # Verify with Lean
                    passes, output = self._verify_lean_proof(formal_proof)
                    
                    # Log the attempt
                    self._log_attempt(
                        problem=test_case.name,
                        prompt=test_case.informal_prefix,
                        depth=depth,
                        attempt=attempt + 1,
                        informal_proof=informal_proof,
                        formal_proof=formal_proof,
                        passed=passes
                    )
                    
                    if passes:
                        print(f"Depth {depth}, Attempt {attempt + 1} succeeded")
                        return {
                            'name': test_case.name,
                            'passed': True,
                            'informal_proof': informal_proof,
                            'lean_code': output.get("verified_code"),
                            'depth': depth,
                            'attempts': attempt + 1
                        }
                    
                    # If verification failed, add error context for next attempt
                    if isinstance(output, dict) and 'errors' in output:
                        error_context = "\n".join([error.get('data', '') for error in output['errors']])
                        informal_proof += f"\nPrevious attempt failed with: {error_context}\nPlease revise the proof."
                
                except Exception as e:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    self._log_attempt(
                        problem=test_case.name,
                        prompt=test_case.informal_prefix,
                        depth=depth,
                        attempt=attempt + 1,
                        informal_proof=str(e),
                        formal_proof="",
                        passed=False
                    )

        # If we get here, all depths and attempts failed
        return {
            'name': test_case.name,
            'passed': False,
            'informal_proof': informal_proof,
            'lean_code': None,
            'depth': self.max_depth,
            'attempts': self.max_attempts
        }

def load_test_data(file_path: str) -> List[TestCase]:
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    test_data = [
        TestCase(
            name=entry['name'],
            header=entry.get('header', ''),
            informal_prefix=entry.get('informal_prefix', ''),
            formal_statement=entry.get('formal_statement', ''),
            goal=entry.get('goal', '')
        )
        for entry in data if entry.get('split') == 'test'
    ]
    return test_data

def run_evaluation(prover: TwoAgentProver, test_cases: List[TestCase], output_file: str):
    results = []
    for test_case in test_cases:
        result = prover.process_test_case(test_case)
        results.append(result)
        print(f"Processed {test_case.name}: {'Passed' if result['passed'] else 'Failed'}")

    num_passed = sum(1 for result in results if result['passed'])
    print(f"Total test cases: {len(results)}")
    print(f"Passed: {num_passed}")
    print(f"Failed: {len(results) - num_passed}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

if __name__ == "__main__":
    load_dotenv()
    
    # Initialize components
    lean4_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=3,
        timeout=300,
        memory_limit=10,
        name='verifier'
    )
    
    proof_generator = ProofGenerator(model_type="gpt-4o-mini")
    auto_formalizer = AutoFormalizer()
    
    try:
        # Load test data
        test_cases = load_test_data('datasets/minif2f.jsonl')
        print(f"Total test cases: {len(test_cases)}")
        
        # Create two-agent prover (no RAG)
        prover = TwoAgentProver(
            lean4_scheduler=lean4_scheduler,
            proof_generator=proof_generator,
            auto_formalizer=auto_formalizer,
            max_depth=0,
            max_attempts=3,
            log_file='two_agent_prover_results.csv'
        )
        
        # Run evaluation
        results = run_evaluation(
            prover,
            test_cases,
            'two_agent_prover_results.json'
        )
        
    finally:
        lean4_scheduler.close()
