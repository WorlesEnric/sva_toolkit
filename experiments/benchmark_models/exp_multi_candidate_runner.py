#!/usr/bin/env python3
"""
Multi-candidate SVA generation benchmark runner.

This script benchmarks an LLM SVA generation workflow where:
1. The LLM generates N candidate SVAs for each SVAD
2. Each candidate is checked against the reference SVA
3. The best relationship among all candidates is reported following priority:
   equivalent > generated_implies_reference > reference_implies_generated > no_relationship > error
"""

import json
import os
import sys
import time
import tempfile
import argparse
import hashlib
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Pool
from functools import partial

# Add parent directory to path to import sva_toolkit
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sva_toolkit.utils.llm_client import LLMClient, LLMConfig
from sva_toolkit.implication_checker import SVAImplicationChecker
from sva_toolkit.benchmark.runner import (
    RelationshipType,
    _clean_sva_output,
)


# Priority order for relationship types (lower is better)
RELATIONSHIP_PRIORITY: Dict[RelationshipType, int] = {
    RelationshipType.EQUIVALENT: 0,
    RelationshipType.GENERATED_IMPLIES_REFERENCE: 1,
    RelationshipType.REFERENCE_IMPLIES_GENERATED: 2,
    RelationshipType.NO_RELATIONSHIP: 3,
    RelationshipType.ERROR: 4,
}


@dataclass
class CandidateResult:
    """Result for a single candidate SVA."""
    candidate_index: int
    generated_sva: str
    relationship: RelationshipType
    error_message: Optional[str] = None
    generation_time: float = 0.0
    verification_time: float = 0.0


@dataclass
class MultiCandidateSingleResult:
    """Result for a single benchmark item with multiple candidates."""
    svad: str
    reference_sva: str
    candidates: List[CandidateResult]
    best_relationship: RelationshipType
    best_candidate_index: int
    cot: Optional[str] = None
    total_generation_time: float = 0.0
    total_verification_time: float = 0.0


@dataclass
class MultiCandidateBenchmarkResult:
    """Aggregated benchmark results for multi-candidate generation."""
    model_name: str
    num_candidates: int
    total_items: int
    equivalent_count: int = 0
    generated_implies_reference_count: int = 0
    reference_implies_generated_count: int = 0
    no_relationship_count: int = 0
    error_count: int = 0
    avg_generation_time: float = 0.0
    avg_verification_time: float = 0.0
    individual_results: List[MultiCandidateSingleResult] = field(default_factory=list)

    @property
    def equivalent_rate(self) -> float:
        """Rate of equivalent SVAs (any candidate)."""
        return self.equivalent_count / self.total_items if self.total_items > 0 else 0

    @property
    def any_implication_rate(self) -> float:
        """Rate of any implication relationship (including equivalent)."""
        correct = (
            self.equivalent_count
            + self.generated_implies_reference_count
            + self.reference_implies_generated_count
        )
        return correct / self.total_items if self.total_items > 0 else 0

    @property
    def success_rate(self) -> float:
        """Rate of non-error results."""
        return (self.total_items - self.error_count) / self.total_items if self.total_items > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "num_candidates": self.num_candidates,
            "total_items": self.total_items,
            "equivalent_count": self.equivalent_count,
            "generated_implies_reference_count": self.generated_implies_reference_count,
            "reference_implies_generated_count": self.reference_implies_generated_count,
            "no_relationship_count": self.no_relationship_count,
            "error_count": self.error_count,
            "equivalent_rate": self.equivalent_rate,
            "any_implication_rate": self.any_implication_rate,
            "success_rate": self.success_rate,
            "avg_generation_time": self.avg_generation_time,
            "avg_verification_time": self.avg_verification_time,
        }


def _compute_item_hash(svad: str, reference_sva: str, model: str, num_candidates: int) -> str:
    """
    Compute a unique hash for a benchmark item.

    Args:
        svad: Natural language description
        reference_sva: Reference SVA code
        model: Model name
        num_candidates: Number of candidates

    Returns:
        Hash string for the item
    """
    content = f"{svad}|{reference_sva}|{model}|{num_candidates}"
    return hashlib.md5(content.encode()).hexdigest()


def _get_best_relationship(candidates: List[CandidateResult]) -> Tuple[RelationshipType, int]:
    """
    Get the best relationship from a list of candidate results.

    Args:
        candidates: List of CandidateResult objects

    Returns:
        Tuple of (best_relationship, best_candidate_index)
    """
    if not candidates:
        return RelationshipType.ERROR, -1
    best_relationship = RelationshipType.ERROR
    best_index = 0
    best_priority = RELATIONSHIP_PRIORITY[RelationshipType.ERROR]
    for candidate in candidates:
        priority = RELATIONSHIP_PRIORITY.get(candidate.relationship, 4)
        if priority < best_priority:
            best_priority = priority
            best_relationship = candidate.relationship
            best_index = candidate.candidate_index
    return best_relationship, best_index


def _generate_candidates(
    llm_client: LLMClient,
    svad: str,
    num_candidates: int,
    system_prompt: str,
    user_prompt_template: str,
    verible_path: str,
    temperature: float = 6,
) -> List[Tuple[str, float]]:
    """
    Generate multiple candidate SVAs for a given SVAD.

    Args:
        llm_client: LLM client to use
        svad: Natural language description
        num_candidates: Number of candidates to generate
        system_prompt: System prompt for generation
        user_prompt_template: User prompt template
        verible_path: Path to verible-verilog-syntax binary
        temperature: Temperature for generation (higher for diversity)

    Returns:
        List of (generated_sva, generation_time) tuples
    """
    candidates = []
    prompt = user_prompt_template.format(svad=svad)
    for _ in range(num_candidates):
        start_time = time.time()
        try:
            response = llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            generated_sva = _clean_sva_output(response, verible_path)
        except Exception:
            generated_sva = ""
        gen_time = time.time() - start_time
        candidates.append((generated_sva, gen_time))
    return candidates


def _evaluate_candidate(
    checker: SVAImplicationChecker,
    generated_sva: str,
    reference_sva: str,
) -> Tuple[RelationshipType, Optional[str], float]:
    """
    Evaluate a single candidate against the reference.

    Args:
        checker: Implication checker
        generated_sva: Generated SVA code
        reference_sva: Reference SVA code

    Returns:
        Tuple of (relationship, error_message, verification_time)
    """
    if not generated_sva:
        return RelationshipType.ERROR, "Empty generated SVA", 0.0
    start_time = time.time()
    try:
        gen_implies_ref, ref_implies_gen = checker.get_implication_relationship(
            generated_sva, reference_sva
        )
        if gen_implies_ref and ref_implies_gen:
            relationship = RelationshipType.EQUIVALENT
        elif gen_implies_ref:
            relationship = RelationshipType.GENERATED_IMPLIES_REFERENCE
        elif ref_implies_gen:
            relationship = RelationshipType.REFERENCE_IMPLIES_GENERATED
        else:
            relationship = RelationshipType.NO_RELATIONSHIP
        error_msg = None
    except Exception as e:
        relationship = RelationshipType.ERROR
        error_msg = f"Verification error: {str(e)}"
    verify_time = time.time() - start_time
    return relationship, error_msg, verify_time


def _worker_process_item(
    item_data: Dict[str, Any],
    llm_config_dict: Dict[str, Any],
    checker_kwargs: Dict[str, Any],
    system_prompt: str,
    user_prompt_template: str,
    num_candidates: int,
    cache_dir: Optional[str],
    verible_path: str = "verible-verilog-syntax",
    temperature: float = 0.6,
) -> Dict[str, Any]:
    """
    Worker function to process a single benchmark item with multiple candidates.

    Args:
        item_data: Dict with 'svad', 'reference_sva', 'cot', 'index'
        llm_config_dict: LLM configuration as dict
        checker_kwargs: Kwargs for implication checker
        system_prompt: System prompt for SVA generation
        user_prompt_template: User prompt template
        num_candidates: Number of candidates to generate
        cache_dir: Directory for caching results
        verible_path: Path to verible-verilog-syntax binary
        temperature: Temperature for generation

    Returns:
        Dict with result data
    """
    svad = item_data["svad"]
    reference_sva = item_data["reference_sva"]
    cot = item_data.get("cot")
    index = item_data["index"]
    model = llm_config_dict["model"]
    # Check cache first
    item_hash = _compute_item_hash(svad, reference_sva, model, num_candidates)
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"{item_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    cached["from_cache"] = True
                    return cached
            except (json.JSONDecodeError, IOError):
                pass  # Cache corrupted, regenerate
    # Create LLM client in this process
    llm_config = LLMConfig(**llm_config_dict)
    llm_client = LLMClient(llm_config)
    # Create implication checker in this process
    checker = SVAImplicationChecker(**checker_kwargs)
    # Generate candidates
    candidate_results = []
    total_gen_time = 0.0
    total_verify_time = 0.0
    generated_candidates = _generate_candidates(
        llm_client=llm_client,
        svad=svad,
        num_candidates=num_candidates,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        verible_path=verible_path,
        temperature=temperature,
    )
    # Evaluate each candidate
    for i, (generated_sva, gen_time) in enumerate(generated_candidates):
        total_gen_time += gen_time
        relationship, error_msg, verify_time = _evaluate_candidate(
            checker=checker,
            generated_sva=generated_sva,
            reference_sva=reference_sva,
        )
        total_verify_time += verify_time
        candidate_results.append({
            "candidate_index": i,
            "generated_sva": generated_sva,
            "relationship": relationship.value,
            "error_message": error_msg,
            "generation_time": gen_time,
            "verification_time": verify_time,
        })
    # Determine best relationship
    candidate_objs = [
        CandidateResult(
            candidate_index=c["candidate_index"],
            generated_sva=c["generated_sva"],
            relationship=RelationshipType(c["relationship"]),
            error_message=c["error_message"],
            generation_time=c["generation_time"],
            verification_time=c["verification_time"],
        )
        for c in candidate_results
    ]
    best_relationship, best_index = _get_best_relationship(candidate_objs)
    result = {
        "index": index,
        "svad": svad,
        "reference_sva": reference_sva,
        "candidates": candidate_results,
        "best_relationship": best_relationship.value,
        "best_candidate_index": best_index,
        "cot": cot,
        "total_generation_time": total_gen_time,
        "total_verification_time": total_verify_time,
        "from_cache": False,
    }
    # Save to cache
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"{item_hash}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except IOError:
            pass  # Ignore cache write errors
    return result


class MultiCandidateBenchmarkRunner:
    """
    Runner for benchmarking LLM SVA generation with multiple candidates.

    Takes a dataset with SVAD (natural language descriptions) and reference SVA,
    generates N candidate SVAs using an LLM, and evaluates the best relationship
    among all candidates.
    """

    DEFAULT_SYSTEM_PROMPT = """You are an expert SystemVerilog Assertion engineer. Your task is to translate natural language requirements into syntactically correct and semantically accurate SVA properties.

Critical requirements:
1. Property structure: Use 'property' keyword with a descriptive name, then 'endproperty'
2. Clocking: Always specify @(posedge clk) or @(negedge clk) - infer from context if not specified
3. Reset handling: Use 'disable iff (condition)' when reset is mentioned - match the reset polarity correctly
4. Implication: Use |-> for overlapping (same cycle) or |=> for non-overlapping (next cycle) based on timing
5. Delays: Use ##N for N clock cycles, ##0 for same cycle
6. Assertion: Create an assert statement with a descriptive name and error message

Output format: Pure SVA code only, no markdown, no explanations, no comments."""

    DEFAULT_USER_PROMPT_TEMPLATE = """Translate this requirement into a SystemVerilog Assertion:

{svad}

Generate the complete SVA code:"""

    def __init__(
        self,
        llm_clients: List[LLMClient],
        num_candidates: int = 3,
        implication_checker: Optional[SVAImplicationChecker] = None,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        verible_path: str = "verible-verilog-syntax",
        temperature: float = 0.6,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ):
        """
        Initialize the multi-candidate benchmark runner.

        Args:
            llm_clients: List of LLM clients to benchmark
            num_candidates: Number of candidate SVAs to generate per item
            implication_checker: Optional implication checker
            num_workers: Number of worker processes
            cache_dir: Directory for caching progress
            verible_path: Path to verible-verilog-syntax binary
            temperature: Temperature for diverse generation
            system_prompt: System prompt for SVA generation
            user_prompt_template: User prompt template
        """
        self.llm_clients = llm_clients
        self.num_candidates = num_candidates
        self.implication_checker = implication_checker or SVAImplicationChecker()
        self.num_workers = num_workers
        self.verible_path = verible_path
        self.temperature = temperature
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or self.DEFAULT_USER_PROMPT_TEMPLATE
        # Set up cache directory
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        else:
            self.cache_dir = tempfile.mkdtemp(prefix="sva_multi_candidate_cache_")
        # Store checker kwargs for worker processes
        self._checker_kwargs = {
            "ebmc_path": getattr(self.implication_checker, 'ebmc_path', None),
            "depth": getattr(self.implication_checker, 'depth', 20),
            "timeout": getattr(self.implication_checker, 'timeout', 300),
        }

    def run_benchmark(
        self,
        dataset: List[Dict[str, Any]],
        llm_client: LLMClient,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_multiprocessing: bool = True,
    ) -> MultiCandidateBenchmarkResult:
        """
        Run benchmark on a dataset with multiple candidates.

        Args:
            dataset: List of dicts with 'SVAD' and 'SVA' keys
            llm_client: LLM client to benchmark
            progress_callback: Optional callback(current, total)
            use_multiprocessing: Whether to use multiprocessing

        Returns:
            MultiCandidateBenchmarkResult with aggregated statistics
        """
        # Prepare items for processing
        items_to_process = []
        for i, item in enumerate(dataset):
            svad = item.get("SVAD", "")
            reference_sva = item.get("SVA", "")
            cot = item.get("CoT")
            if not svad or not reference_sva:
                continue
            items_to_process.append({
                "index": i,
                "svad": svad,
                "reference_sva": reference_sva,
                "cot": cot,
            })
        total = len(items_to_process)
        if total == 0:
            return MultiCandidateBenchmarkResult(
                model_name=llm_client.config.model,
                num_candidates=self.num_candidates,
                total_items=0,
            )
        # Prepare LLM config dict for worker processes
        llm_config_dict = {
            "base_url": llm_client.config.base_url,
            "model": llm_client.config.model,
            "api_key": llm_client.config.api_key,
            "temperature": llm_client.config.temperature,
            "max_tokens": llm_client.config.max_tokens,
        }
        if use_multiprocessing and self.num_workers > 1:
            results = self._run_multiprocess(
                items_to_process,
                llm_config_dict,
                progress_callback,
                total,
            )
        else:
            results = self._run_single_process(
                items_to_process,
                llm_config_dict,
                progress_callback,
                total,
            )
        # Aggregate results
        benchmark_result = MultiCandidateBenchmarkResult(
            model_name=llm_client.config.model,
            num_candidates=self.num_candidates,
            total_items=len(results),
        )
        total_gen_time = 0.0
        total_verify_time = 0.0
        for r in results:
            best_rel = RelationshipType(r["best_relationship"])
            if best_rel == RelationshipType.EQUIVALENT:
                benchmark_result.equivalent_count += 1
            elif best_rel == RelationshipType.GENERATED_IMPLIES_REFERENCE:
                benchmark_result.generated_implies_reference_count += 1
            elif best_rel == RelationshipType.REFERENCE_IMPLIES_GENERATED:
                benchmark_result.reference_implies_generated_count += 1
            elif best_rel == RelationshipType.NO_RELATIONSHIP:
                benchmark_result.no_relationship_count += 1
            else:
                benchmark_result.error_count += 1
            total_gen_time += r["total_generation_time"]
            total_verify_time += r["total_verification_time"]
            # Convert to MultiCandidateSingleResult
            candidates = [
                CandidateResult(
                    candidate_index=c["candidate_index"],
                    generated_sva=c["generated_sva"],
                    relationship=RelationshipType(c["relationship"]),
                    error_message=c["error_message"],
                    generation_time=c["generation_time"],
                    verification_time=c["verification_time"],
                )
                for c in r["candidates"]
            ]
            single_result = MultiCandidateSingleResult(
                svad=r["svad"],
                reference_sva=r["reference_sva"],
                candidates=candidates,
                best_relationship=best_rel,
                best_candidate_index=r["best_candidate_index"],
                cot=r.get("cot"),
                total_generation_time=r["total_generation_time"],
                total_verification_time=r["total_verification_time"],
            )
            benchmark_result.individual_results.append(single_result)
        if results:
            benchmark_result.avg_generation_time = total_gen_time / len(results)
            benchmark_result.avg_verification_time = total_verify_time / len(results)
        return benchmark_result

    def _run_multiprocess(
        self,
        items: List[Dict[str, Any]],
        llm_config_dict: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]],
        total: int,
    ) -> List[Dict[str, Any]]:
        """Run benchmark using multiprocessing."""
        worker_fn = partial(
            _worker_process_item,
            llm_config_dict=llm_config_dict,
            checker_kwargs=self._checker_kwargs,
            system_prompt=self.system_prompt,
            user_prompt_template=self.user_prompt_template,
            num_candidates=self.num_candidates,
            cache_dir=self.cache_dir,
            verible_path=self.verible_path,
            temperature=self.temperature,
        )
        results = []
        completed = 0
        with Pool(processes=self.num_workers) as pool:
            for result in pool.imap_unordered(worker_fn, items):
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        results.sort(key=lambda x: x["index"])
        return results

    def _run_single_process(
        self,
        items: List[Dict[str, Any]],
        llm_config_dict: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]],
        total: int,
        rate_limit_delay: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Run benchmark in single process mode."""
        results = []
        for i, item in enumerate(items):
            result = _worker_process_item(
                item_data=item,
                llm_config_dict=llm_config_dict,
                checker_kwargs=self._checker_kwargs,
                system_prompt=self.system_prompt,
                user_prompt_template=self.user_prompt_template,
                num_candidates=self.num_candidates,
                cache_dir=self.cache_dir,
                verible_path=self.verible_path,
                temperature=self.temperature,
            )
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, total)
            if not result.get("from_cache", False):
                time.sleep(rate_limit_delay)
        return results

    def clear_cache(self) -> int:
        """Clear all cached results."""
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return 0
        count = 0
        for f in os.listdir(self.cache_dir):
            if f.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, f))
                count += 1
        return count


def run_multi_candidate_benchmark(
    input_file: str,
    output_file: str,
    llm_configs: List[Dict[str, Any]],
    prompt_config: Dict[str, str],
    num_candidates: int = 3,
    workers: int = 4,
    ebmc_path: Optional[str] = None,
    depth: int = 20,
    num_samples: Optional[int] = None,
    verible_path: str = "verible-verilog-syntax",
    temperature: float = 0.6,
) -> Dict[str, Any]:
    """
    Run multi-candidate benchmark with custom prompts.

    Args:
        input_file: Path to dataset file (with SVAD and SVA)
        output_file: Path to output results file
        llm_configs: List of LLM configurations
        prompt_config: Dictionary with SVA_GENERATION_SYSTEM_PROMPT and SVA_GENERATION_USER_PROMPT_TEMPLATE
        num_candidates: Number of candidates to generate per item
        workers: Number of worker processes
        ebmc_path: Path to ebmc binary
        depth: Proof depth for verification
        num_samples: Number of samples to process (None for all)
        verible_path: Path to verible-verilog-syntax binary
        temperature: Temperature for generation

    Returns:
        Dictionary with benchmark results
    """
    print(f"Loading dataset: {input_file}")
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    valid_items = [item for item in dataset if item.get("SVAD") and item.get("SVA")]
    if num_samples:
        valid_items = valid_items[:num_samples]
        print(f"Limiting to {num_samples} samples for testing")
    print(f"Found {len(valid_items)} valid items with SVAD and SVA")
    print(f"Generating {num_candidates} candidates per item (temperature={temperature})")
    if not valid_items:
        raise ValueError("No valid items found in dataset")
    llm_clients = []
    for config in llm_configs:
        client = LLMClient.from_params(
            base_url=config["base_url"],
            model=config["model"],
            api_key=config["api_key"],
        )
        llm_clients.append(client)
    checker = SVAImplicationChecker(ebmc_path=ebmc_path, depth=depth)
    cache_dir = tempfile.mkdtemp(prefix="sva_multi_candidate_cache_")
    all_results = []
    for llm_client in llm_clients:
        print(f"\nRunning benchmark for model: {llm_client.config.model}")
        runner = MultiCandidateBenchmarkRunner(
            llm_clients=[llm_client],
            num_candidates=num_candidates,
            implication_checker=checker,
            num_workers=workers,
            cache_dir=cache_dir,
            verible_path=verible_path,
            temperature=temperature,
            system_prompt=prompt_config.get("SVA_GENERATION_SYSTEM_PROMPT"),
            user_prompt_template=prompt_config.get("SVA_GENERATION_USER_PROMPT_TEMPLATE"),
        )

        def progress_callback(current: int, total: int) -> None:
            if current % 10 == 0 or current == total:
                print(f"  Progress: {current}/{total} ({current * 100 // total}%)")

        result = runner.run_benchmark(
            valid_items,
            llm_client,
            progress_callback=progress_callback,
            use_multiprocessing=workers > 1,
        )
        result_dict = result.to_dict()
        result_dict["individual_results"] = []
        for i, r in enumerate(result.individual_results):
            candidates_dict = [
                {
                    "candidate_index": c.candidate_index,
                    "generated_sva": c.generated_sva,
                    "relationship": c.relationship.value,
                    "error_message": c.error_message,
                    "generation_time": c.generation_time,
                    "verification_time": c.verification_time,
                }
                for c in r.candidates
            ]
            result_dict["individual_results"].append({
                "index": i,
                "svad": r.svad,
                "reference_sva": r.reference_sva,
                "candidates": candidates_dict,
                "best_relationship": r.best_relationship.value,
                "best_candidate_index": r.best_candidate_index,
                "cot": r.cot,
                "total_generation_time": r.total_generation_time,
                "total_verification_time": r.total_verification_time,
            })
        all_results.append(result_dict)
        print(f"  Equivalent rate: {result.equivalent_rate:.2%}")
        print(f"  Any implication rate: {result.any_implication_rate:.2%}")
        print(f"  Success rate: {result.success_rate:.2%}")
    output_data = {
        "prompt_name": prompt_config.get("name", "unknown"),
        "prompt_config": prompt_config,
        "num_candidates": num_candidates,
        "temperature": temperature,
        "results": all_results,
    }
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    return output_data


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run multi-candidate SVA generation benchmarks"
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default="hard_dataset.json",
        help="Input test dataset file (default: hard_dataset.json)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="prompts.json",
        help="Prompts configuration file (default: prompts.json)",
    )
    parser.add_argument(
        "--llm-configs-file",
        type=str,
        default="llm_configs.json",
        help="LLM configurations file (default: llm_configs.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_multi_candidate",
        help="Output directory for results (default: results_multi_candidate)",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Number of candidate SVAs to generate per item (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for diverse generation (default: 0.6)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)",
    )
    parser.add_argument(
        "--ebmc-path",
        type=str,
        default=None,
        help="Path to ebmc binary",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Proof depth for verification (default: 20)",
    )
    parser.add_argument(
        "--verible-path",
        type=str,
        default="verible-verilog-syntax",
        help="Path to verible-verilog-syntax binary (default: verible-verilog-syntax)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples for testing (default: None, use all samples)",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=None,
        help="Use only a specific prompt by index (default: None, use all prompts)",
    )
    parser.add_argument(
        "--llm-index",
        type=int,
        default=None,
        help="Use only a specific LLM by index (default: None, use all LLMs)",
    )
    args = parser.parse_args()
    script_dir = Path(__file__).parent
    input_dataset = script_dir / args.input_dataset
    prompts_file = script_dir / args.prompts_file
    llm_configs_file = script_dir / args.llm_configs_file
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    print(f"Loading prompts from: {prompts_file}")
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    print(f"Loading LLM configs from: {llm_configs_file}")
    with open(llm_configs_file, 'r') as f:
        llm_configs = json.load(f)
    # Filter prompts and LLMs if specified
    if args.prompt_index is not None:
        if 0 <= args.prompt_index < len(prompts):
            prompts = [prompts[args.prompt_index]]
            print(f"Using only prompt at index {args.prompt_index}")
        else:
            raise ValueError(f"Invalid prompt index: {args.prompt_index}")
    if args.llm_index is not None:
        if 0 <= args.llm_index < len(llm_configs):
            llm_configs = [llm_configs[args.llm_index]]
            print(f"Using only LLM at index {args.llm_index}")
        else:
            raise ValueError(f"Invalid LLM index: {args.llm_index}")
    print(f"Found {len(prompts)} prompt configurations")
    print(f"Found {len(llm_configs)} LLM configurations")
    print(f"Generating {args.num_candidates} candidates per item")
    print(f"\nRunning {len(prompts)} × {len(llm_configs)} = {len(prompts) * len(llm_configs)} experiments\n")
    all_experiment_results = []
    for prompt_idx, prompt_config in enumerate(prompts):
        prompt_name = prompt_config.get("name", f"prompt_{prompt_idx}")
        print(f"\n{'=' * 80}")
        print(f"PROMPT {prompt_idx + 1}/{len(prompts)}: {prompt_name}")
        print(f"{'=' * 80}")
        for llm_idx, llm_config in enumerate(llm_configs):
            model_name = llm_config["model"]
            exp_name = f"{prompt_name}_{model_name}_nc{args.num_candidates}"
            print(f"\nExperiment {prompt_idx * len(llm_configs) + llm_idx + 1}/{len(prompts) * len(llm_configs)}: {exp_name}")
            print(f"  Prompt: {prompt_name}")
            print(f"  Model: {model_name}")
            print(f"  Candidates: {args.num_candidates}")
            result_file = output_dir / f"result_{exp_name}.json"
            print(f"  Running benchmark...")
            try:
                result = run_multi_candidate_benchmark(
                    input_file=str(input_dataset),
                    output_file=str(result_file),
                    llm_configs=[llm_config],
                    prompt_config=prompt_config,
                    num_candidates=args.num_candidates,
                    workers=args.workers,
                    ebmc_path=args.ebmc_path,
                    depth=args.depth,
                    num_samples=args.num_samples,
                    verible_path=args.verible_path,
                    temperature=args.temperature,
                )
                all_experiment_results.append(result)
            except Exception as e:
                print(f"  ERROR running benchmark: {e}")
                import traceback
                traceback.print_exc()
                continue
    summary_file = output_dir / "summary.json"
    print(f"\n{'=' * 80}")
    print(f"Saving summary to: {summary_file}")
    with open(summary_file, 'w') as f:
        json.dump(all_experiment_results, f, indent=2)
    print(f"\nCompleted {len(all_experiment_results)} experiments")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
