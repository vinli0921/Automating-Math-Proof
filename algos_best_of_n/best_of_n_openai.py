import os
import json
import re
import csv
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from RAG.Neo4jVectorRetriever import Neo4jVectorRetriever
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

@dataclass
class ProofCandidate:
    proof_text: str
    score: float = 0.0
    metadata: Dict = None

class VotingProofGenerator:
    """Enhanced proof generator that uses a Best-of-N voting system."""
    
    def __init__(self, 
                 model_type: str = "gpt-4",
                 temperature: float = 0.7,
                 n_candidates: int = 5,
                 ranker_model: str = "gpt-4",  # Using GPT-4 for more reliable ranking
                 ranker_temperature: float = 0.0):
        self.client = OpenAI()
        self.model_type = model_type
        self.temperature = temperature
        self.n_candidates = n_candidates
        self.ranker_model = ranker_model
        self.ranker_temperature = ranker_temperature
        
        # Use ChatOpenAI for generation
        self.generation_llm = ChatOpenAI(
            model_name=model_type,
            temperature=temperature,
            n=n_candidates,
        )
        self.ranker_llm = ChatOpenAI(
            model_name=ranker_model,
            temperature=ranker_temperature
        )
        self.n_candidates = n_candidates

    def generate_proof_candidates(self, context: str, problem: str) -> List[ProofCandidate]:
        """Generate multiple candidate proofs using direct OpenAI API call."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a mathematics expert focused on generating clear informal proofs.

Generate a clear and detailed informal proof in natural language.
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
                ],
                temperature=self.temperature,
                n=self.n_candidates,
            )
            return [
                ProofCandidate(proof_text=choice.message.content.strip())
                for choice in response.choices
            ]
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
            for eval_data in evaluations:
                idx = eval_data["candidate_index"]
                if idx < len(candidates):
                    candidates[idx].score = eval_data["score"]
                    candidates[idx].metadata = {"justification": eval_data["justification"]}
            candidates.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            print(f"Error in ranking: {e}")
            for i, candidate in enumerate(candidates):
                candidate.score = float(self.n_candidates - i)
        return candidates

    def generate_best_proof(self, context: str, problem: str) -> Tuple[ProofCandidate, List[ProofCandidate]]:
        """Generate and rank multiple proofs, returning the best candidate and all candidates."""
        candidates = self.generate_proof_candidates(context, problem)
        ranked_candidates = self.rank_candidates(candidates, context, problem)
        return ranked_candidates[0], ranked_candidates

# ---------------------------------------------------------------------------
# AutoFormalizer (local autoformalization using DeepSeek)
# ---------------------------------------------------------------------------
class AutoFormalizer:
    """Responsible for converting informal proofs to Lean 4 formal proofs."""
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
        model_outputs = self.model.generate([prompt], sampling_params, use_tqdm=False)
        generated_text = model_outputs[0].outputs[0].text
        return generated_text.strip()

# ---------------------------------------------------------------------------
# TwoAgentProver (using best-of-N approach)
# ---------------------------------------------------------------------------
class TwoAgentProver:
    def __init__(
        self,
        lean4_scheduler,
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        auto_formalizer: AutoFormalizer,
        max_depth=3,
        top_k=5,
        max_attempts=1,
        log_file=None,
        n_candidates=5,
        model_type="gpt-4o",
        ranker_model="gpt-4o"
    ):
        self.lean4_scheduler = lean4_scheduler
        # Use the best-of-N generation (without tree search)
        self.proof_generator = VotingProofGenerator(
            model_type=model_type,
            n_candidates=n_candidates,
            temperature=0.7,
            ranker_model=ranker_model,
            ranker_temperature=0.0
        )
        self.auto_formalizer = auto_formalizer
        self.max_depth = max_depth
        self.max_attempts = max_attempts

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.retriever = Neo4jVectorRetriever(driver=self.driver, top_k=top_k)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'two_agent_prover_log_{timestamp}.csv'
        self.log_file = log_file
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'prompt', 'depth', 'attempt', 'visited_node_ids',
                'candidate_count', 'best_score', 'proof_scores',
                'informal_proof', 'formal_proof', 'passed'
            ])
    
    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.close()

    def _verify_lean_proof(self, formal_proof: str) -> Tuple[bool, Optional[Dict]]:
        request_id_list = self.lean4_scheduler.submit_all_request([
            re.search(r'```lean4?\n(.*?)\n```', formal_proof, re.DOTALL).group(1)
        ])
        outputs_list = self.lean4_scheduler.get_all_request_outputs(request_id_list)
        result = outputs_list[0]
        print(result)
        return (result['pass'] == True and result['complete'] == True), result

    def _log_attempt(self, problem: str, prompt: str, depth: int, attempt: int,
                    visited_node_ids: List[str], informal_proof: str,
                    formal_proof: str, passed: bool,
                    candidate_count: int = 1,
                    best_score: float = 0.0,
                    proof_scores: List[float] = None):
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            informal_escaped = informal_proof.replace('\n', '\\n')
            formal_escaped = formal_proof.replace('\n', '\\n')
            visited_str = ",".join(visited_node_ids)
            scores_str = ",".join(map(str, proof_scores)) if proof_scores else ""
            writer.writerow([
                problem, prompt, depth, attempt, visited_str,
                candidate_count, best_score, scores_str,
                informal_escaped, formal_escaped, passed
            ])

    def _get_rag_context(self, query: str, depth: int) -> Tuple[str, List[str]]:
        query_embedding = self.retriever.get_query_embedding(query)
        context = ""
        visited_ids = []
        current_node_id = None

        for d in range(depth + 1):
            if d == 0:
                current_node_id, content = self.retriever.get_top_node(query_embedding)
                if current_node_id is None:
                    return "", []
                visited_ids.append(str(current_node_id))
                context += f"{current_node_id}:\n{content}\n\n"
            else:
                neighbors = self.retriever.get_neighbors(current_node_id, self.retriever.top_k)
                if not neighbors:
                    break
                neighbor_similarities = [
                    (self._cosine_similarity(query_embedding, n['embedding']), n)
                    for n in neighbors if n['embedding'] is not None
                ]
                if not neighbor_similarities:
                    break
                neighbor_similarities.sort(reverse=True, key=lambda x: x[0])
                top_neighbors = neighbor_similarities[:self.retriever.top_k]
                for _, neighbor in top_neighbors:
                    content = neighbor.get('content', '')
                    node_id = neighbor.get('id')
                    visited_ids.append(str(node_id))
                    context += f"{node_id}:\n{content}\n\n"
                current_node_id = top_neighbors[0][1]['id']
        
        return context.strip(), visited_ids

    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def process_test_case(self, test_case: TestCase) -> Dict:
        for depth in range(self.max_depth + 1):
            rag_context, visited_node_ids = self._get_rag_context(test_case.informal_prefix, depth)
            if not rag_context:
                continue

            # Use best-of-N generation to get the best candidate proof.
            best_candidate, ranked_candidates = self.proof_generator.generate_best_proof(
                rag_context,
                test_case.informal_prefix
            )

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
                    prompt=test_case.informal_prefix,
                    depth=depth,
                    attempt=1,
                    visited_node_ids=visited_node_ids,
                    informal_proof=best_candidate.proof_text,
                    formal_proof=formal_proof,
                    passed=passes,
                    candidate_count=len(ranked_candidates),
                    best_score=best_candidate.score,
                    proof_scores=[c.score for c in ranked_candidates]
                )

                if passes:
                    print(f"Depth {depth}, Best-of-N succeeded")
                    return {
                        'name': test_case.name,
                        'passed': True,
                        'visited_node_ids': visited_node_ids,
                        'informal_proof': best_candidate.proof_text,
                        'lean_code': output.get("verified_code"),
                        'depth': depth,
                        'attempts': 1,
                        'candidates': ranked_candidates,
                        'best_score': best_candidate.score
                    }
            except Exception as e:
                print(f"Error processing test case {test_case.name} at depth {depth}: {e}")

        return {
            'name': test_case.name,
            'passed': False,
            'visited_node_ids': visited_node_ids,
            'informal_proof': best_candidate.proof_text,
            'lean_code': None,
            'depth': self.max_depth,
            'attempts': self.max_attempts,
            'candidates': ranked_candidates,
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
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize Lean4 verifier
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
            neo4j_uri=os.environ.get('NEO4J_URI'),
            neo4j_user=os.environ.get('NEO4J_USERNAME'),
            neo4j_password=os.environ.get('NEO4J_PASSWORD'),
            auto_formalizer=auto_formalizer,
            model_type="o1-mini",              
            ranker_model="gpt-4o",   
            n_candidates=5,
            beam_width=3,
            search_depth=2,
            max_depth=2,
            max_attempts=1,
            log_file='ms_o1_tree.csv'
        )
        
        results = run_evaluation(
            prover,
            test_cases,
            'ms_o1_tree.json'
        )
        
    finally:
        lean4_scheduler.close()
