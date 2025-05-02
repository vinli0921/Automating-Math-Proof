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
from openai import OpenAI
from together import Together
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Note: Removed imports for Neo4j and the vector retriever

load_dotenv()

class TogetherLLM:
    def __init__(self, model: str, temperature: float = 0.7, n: int = 1):
        self.client = Together()
        self.model = model
        self.temperature = temperature
        self.n = n

    def create(self, messages: List[Dict]) -> List[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            n=self.n,
            max_tokens=3072
        )
        # Return a list of response texts, one per candidate.
        return [choice.message.content for choice in response.choices]

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

@dataclass
class ProofCandidate:
    proof_text: str
    score: float = 0.0
    metadata: Dict = None

class VotingProofGenerator:
    """Enhanced proof generator that uses a Best-of-N voting system"""
    
    def __init__(self, 
                 model_type: str = "gpt-4",
                 temperature: float = 0.7,
                 n_candidates: int = 5,
                 ranker_model: str = "gpt-4",  
                 ranker_temperature: float = 0.0):
        self.client = OpenAI()
        self.together = Together()
        self.model_type = model_type
        self.temperature = temperature
        self.n_candidates = n_candidates
        self.ranker_model = ranker_model
        self.ranker_temperature = ranker_temperature
        """
        Initialize the voting-based proof generator.
        """
        self.generation_llm = TogetherLLM(model=model_type, temperature=temperature, n=n_candidates)
        
        self.ranker_llm = ChatOpenAI(
            model_name=ranker_model,
            temperature=ranker_temperature
        )
        
        self.n_candidates = n_candidates

    def generate_proof_candidates(self, context: str, problem: str) -> List[ProofCandidate]:
        """Generate multiple candidate proofs using direct Together API call."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a mathematics expert focused on generating clear informal proofs."
                },
                {
                    "role": "user",
                    "content": f"""Generate a clear and detailed informal proof in natural language.
Pay special attention to:
- Similar theorem statements
- Related proof techniques
- Mathematical patterns
- Definitions and axioms used

Context:
{context}

Problem to Prove:
{problem}

Provide your proof in this format:

# Informal Proof:
[Your natural language proof here]"""
                }
            ]
            responses = self.generation_llm.create(messages)
            return [ProofCandidate(proof_text=resp.strip()) for resp in responses]
        except Exception as e:
            print(f"Error generating proofs: {e}")
            return [ProofCandidate(proof_text="Error generating proof.")]

    def rank_candidates(self, candidates: List[ProofCandidate], context: str, problem: str) -> List[ProofCandidate]:
        """Rank candidates using OpenAI's JSON response format."""
        formatted_candidates = "\n\n".join([
            f"Candidate {i}:\n{c.proof_text}" for i, c in enumerate(candidates)
        ])
        
        try:
            response = self.client.chat.completions.create(
                model=self.ranker_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior mathematician evaluating mathematical proofs."
                    },
                    {
                        "role": "user",
                        "content": f"""Evaluate these candidate proofs:

Problem: {problem}

Context: {context}

{formatted_candidates}"""
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "proof_evaluation_schema",
                        "schema": {      
                            "type": "object",
                            "properties": {
                                "evaluations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "candidate_index": {"type": "integer"},
                                            "score": {"type": "number", "minimum": 0, "maximum": 10},
                                            "justification": {"type": "string"}
                                        },
                                        "required": ["candidate_index", "score", "justification"]
                                    }
                                }
                            },
                            "required": ["evaluations"]
                        }
                    }
                },
                temperature=self.ranker_temperature
            )
            
            evaluations = json.loads(response.choices[0].message.content)["evaluations"]
            
            # Update candidates with scores and metadata
            for eval_data in evaluations:
                idx = eval_data["candidate_index"]
                if idx < len(candidates):
                    candidates[idx].score = eval_data["score"]
                    candidates[idx].metadata = {"justification": eval_data["justification"]}
            
            # Sort candidates by score
            candidates.sort(key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            print(f"Error in ranking: {e}")
            for i, candidate in enumerate(candidates):
                candidate.score = float(self.n_candidates - i)
        
        return candidates

    def generate_best_proof(self, context: str, problem: str) -> Tuple[ProofCandidate, List[ProofCandidate]]:
        """Generate and rank multiple proofs, returning the best one and all candidates."""
        candidates = self.generate_proof_candidates(context, problem)
        ranked_candidates = self.rank_candidates(candidates, context, problem)
        return ranked_candidates[0], ranked_candidates

class TreeSearchProofGenerator(VotingProofGenerator):
    """Extends VotingProofGenerator with tree search capabilities"""
    
    def __init__(self,
                 max_depth: int = 3,
                 beam_width: int = 3,
                 **kwargs):
        """
        Args:
            max_depth: Maximum search depth
            beam_width: Number of candidates to keep at each level
            **kwargs: Arguments passed to VotingProofGenerator
        """
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.beam_width = beam_width
        
        self.refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a mathematics expert refining proofs."),
            ("user", """Given a mathematical proof attempt, generate a refined version that addresses any gaps or weaknesses.

Original Proof:
{proof}

Problem:
{problem}

Context:
{context}

Previous Feedback (if any):
{feedback}

Generate a refined proof that improves upon the original.""")
        ])

    def _refine_proof(self, 
                     proof: str, 
                     context: str, 
                     problem: str,
                     feedback: str = "") -> List[ProofCandidate]:
        """Generate refined versions of a proof."""
        prompt = self.refinement_prompt.format(proof=proof, context=context, problem=problem, feedback=feedback)
        messages = [{"role": "user", "content": prompt}]
        try:
            responses = self.generation_llm.create(messages)
            return [ProofCandidate(proof_text=resp.strip()) for resp in responses]
        except Exception as e:
            print(f"Error in refinement: {e}")
            return [ProofCandidate(proof_text="Error in refinement.")]

    def tree_search(self, 
                   context: str, 
                   problem: str,
                   formal_verifier=None) -> Tuple[ProofCandidate, Dict]:
        """
        Perform tree search to find the best proof.
        
        Args:
            context: Mathematical context (taken directly from the test case)
            problem: Problem to prove
            formal_verifier: Optional function to verify proofs formally
            
        Returns:
            Tuple of (best candidate, search statistics)
        """
        stats = {
            "nodes_explored": 0,
            "formal_checks": 0,
            "successful_proofs": []
        }
        feedback = ""
        # Initial beam using provided context
        current_beam = self.generate_proof_candidates(context, problem)
        current_beam = self.rank_candidates(current_beam, context, problem)
        current_beam = current_beam[:self.beam_width]
        
        for depth in range(self.max_depth):
            for candidate in current_beam:
                stats["nodes_explored"] += 1
                
                if formal_verifier:
                    stats["formal_checks"] += 1
                    try:
                        if formal_verifier(candidate.proof_text):
                            stats["final_depth"] = depth
                            return candidate, stats
                    except Exception as e:
                        feedback = str(e)
                else:
                    feedback = ""
                
                refinements = self._refine_proof(candidate.proof_text, context, problem, feedback)
                current_beam.extend(refinements)
            
            if current_beam:
                current_beam = self.rank_candidates(current_beam, context, problem)
                current_beam = current_beam[:self.beam_width]
            else:
                break
        
        stats["final_depth"] = self.max_depth
        best_candidate = max(current_beam, key=lambda x: x.score)
        return best_candidate, stats

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
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=2048,
            top_p=0.1,
            n=1,
        )
        model_outputs = self.model.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
        )
        generated_text = model_outputs[0].outputs[0].text
        return generated_text.strip()

class TwoAgentProver:
    def __init__(
        self,
        lean4_scheduler,
        auto_formalizer: AutoFormalizer,
        max_depth=3,
        max_attempts=3,
        log_file=None,
        # Voting system parameters
        n_candidates=5,
        beam_width=3,
        search_depth=2,
        model_type="gpt-4o",
        ranker_model="gpt-4o"
    ):
        self.lean4_scheduler = lean4_scheduler
        self.proof_generator = TreeSearchProofGenerator(
            model_type=model_type,
            n_candidates=n_candidates,
            beam_width=beam_width,
            max_depth=search_depth,
            ranker_model=ranker_model
        )
        self.auto_formalizer = auto_formalizer
        self.max_attempts = max_attempts
        
        # Simplified logging: only problem, search_depth, attempt, best_score, informal_proof, formal_proof, passed.
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'two_agent_prover_log_{timestamp}.csv'
        self.log_file = log_file 
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'problem', 'search_depth', 'attempt', 'best_score', 'informal_proof', 'formal_proof', 'passed'
            ])

    def __del__(self):
        # Nothing to close since no persistent connections are used.
        pass

    def _verify_lean_proof(self, formal_proof: str) -> Tuple[bool, Optional[Dict]]:
        match = re.search(r'```lean4?\n(.*?)\n```', formal_proof, re.DOTALL)
        if not match:
            return False, {"error": "No lean code block found"}
        code = match.group(1)
        request_id_list = self.lean4_scheduler.submit_all_request([code])
        outputs_list = self.lean4_scheduler.get_all_request_outputs(request_id_list)
        result = outputs_list[0]
        print(result)
        return (result['pass'] == True and result['complete'] == True), result

    def _log_attempt(self, problem: str, search_depth: int, attempt: int, 
                     best_score: float, informal_proof: str, 
                     formal_proof: str, passed: bool):
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Escape newlines in the proof texts.
            informal_escaped = informal_proof.replace('\n', '\\n')
            formal_escaped = formal_proof.replace('\n', '\\n')
            writer.writerow([
                problem, search_depth, attempt, best_score, informal_escaped, formal_escaped, passed
            ])

    def process_test_case(self, test_case: TestCase) -> Dict:
        # With no graph retrieval, we use the test case's informal_prefix as the context.
        depth = 0
        context = test_case.informal_prefix
        # No visited nodes in this simplified version.
        
        # Formal verification wrapper for tree search.
        def verify_proof(proof_text: str) -> bool:
            try:
                formal_proof = self.auto_formalizer.formalize_proof(
                    test_case.header,
                    proof_text,
                    test_case.informal_prefix,
                    test_case.formal_statement,
                    test_case.goal
                )
                return self._verify_lean_proof(formal_proof)[0]
            except Exception as e:
                return False
            
        # Use tree search to generate the best candidate proof.
        best_candidate, search_stats = self.proof_generator.tree_search(
            context=context,
            problem=test_case.informal_prefix,
            formal_verifier=verify_proof
        )
        
        # Use the final search depth from stats if available.
        final_depth = search_stats.get("final_depth", depth)
        
        for attempt in range(self.max_attempts):
            try:
                formal_proof = self.auto_formalizer.formalize_proof(
                    test_case.header,
                    best_candidate.proof_text,
                    test_case.informal_prefix,
                    test_case.formal_statement,
                    test_case.goal
                )
                passes, output = self._verify_lean_proof(formal_proof)
                self._log_attempt(
                    problem=test_case.name,
                    search_depth=final_depth,
                    attempt=attempt + 1,
                    best_score=best_candidate.score,
                    informal_proof=best_candidate.proof_text,
                    formal_proof=formal_proof,
                    passed=passes
                )
                
                if passes:
                    print(f"Attempt {attempt + 1} succeeded")
                    return {
                        'name': test_case.name,
                        'passed': True,
                        'informal_proof': best_candidate.proof_text,
                        'lean_code': output.get("verified_code"),
                        'search_depth': final_depth,
                        'attempts': attempt + 1,
                        'search_stats': search_stats,
                        'best_score': best_candidate.score
                    }
                    
                # Fallback: if verification fails, generate new candidates with error feedback.
                if isinstance(output, dict) and 'errors' in output:
                    error_context = "\n".join([error.get('data', '') for error in output['errors']])
                    best_candidate, new_stats = self.proof_generator.generate_best_proof(
                        context,
                        test_case.informal_prefix + f"\nPrevious errors: {error_context}"
                    )
                    search_stats["nodes_explored"] += new_stats.get("nodes_explored", 0)
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                self._log_attempt(
                    problem=test_case.name,
                    search_depth=final_depth,
                    attempt=attempt + 1,
                    best_score=0.0,
                    informal_proof=str(e),
                    formal_proof="",
                    passed=False
                )
        return {
            'name': test_case.name,
            'passed': False,
            'informal_proof': best_candidate.proof_text,
            'lean_code': None,
            'search_depth': final_depth,
            'attempts': self.max_attempts,
            'search_stats': search_stats,
            'best_score': best_candidate.score
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
    from prover.lean.verifier import Lean4ServerScheduler  # Assuming this remains unchanged.
    # Initialize Lean 4 verifier scheduler.
    lean4_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=3,
        timeout=300,
        memory_limit=10,
        name='verifier'
    )
    
    auto_formalizer = AutoFormalizer()
    
    try:
        test_cases = load_test_data('datasets/mustard_short.jsonl')
        print(f"Total test cases: {len(test_cases)}")
        
        prover = TwoAgentProver(
            lean4_scheduler=lean4_scheduler,
            auto_formalizer=auto_formalizer,
            model_type="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",              
            ranker_model="gpt-4o",   
            n_candidates=5,
            beam_width=3,
            search_depth=6,
            max_depth=2,
            max_attempts=1,
            log_file='ms_8b_tree36_nograph.csv'
        )
        
        results = run_evaluation(
            prover,
            test_cases,
            'ms_8b_tree36_nograph.json'
        )
        
    finally:
        lean4_scheduler.close()
