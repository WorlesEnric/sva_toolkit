"""
SVA Benchmark Runner - Evaluate LLM performance on SVA generation.

Tests LLM's ability to generate SVAs from natural language descriptions,
then uses implication checking to evaluate correctness.
"""

import json
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from sva_toolkit.utils.llm_client import LLMClient, LLMConfig
from sva_toolkit.implication_checker import SVAImplicationChecker
from sva_toolkit.implication_checker.checker import ImplicationResult


class RelationshipType(Enum):
    """Type of relationship between generated and reference SVA."""
    EQUIVALENT = "equivalent"  # Bidirectional implication
    GENERATED_IMPLIES_REFERENCE = "generated_implies_reference"  # Generated is stronger
    REFERENCE_IMPLIES_GENERATED = "reference_implies_generated"  # Reference is stronger
    NO_RELATIONSHIP = "no_relationship"  # No implication either way
    ERROR = "error"  # Verification error


@dataclass
class SingleResult:
    """Result for a single benchmark item."""
    svad: str
    reference_sva: str
    generated_sva: str
    relationship: RelationshipType
    error_message: Optional[str] = None
    generation_time: float = 0.0
    verification_time: float = 0.0


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    model_name: str
    total_items: int
    equivalent_count: int = 0
    generated_implies_reference_count: int = 0
    reference_implies_generated_count: int = 0
    no_relationship_count: int = 0
    error_count: int = 0
    avg_generation_time: float = 0.0
    avg_verification_time: float = 0.0
    individual_results: List[SingleResult] = field(default_factory=list)
    
    @property
    def equivalent_rate(self) -> float:
        """Rate of equivalent SVAs."""
        return self.equivalent_count / self.total_items if self.total_items > 0 else 0
    
    @property
    def any_implication_rate(self) -> float:
        """Rate of any implication relationship (including equivalent)."""
        correct = self.equivalent_count + self.generated_implies_reference_count + self.reference_implies_generated_count
        return correct / self.total_items if self.total_items > 0 else 0
    
    @property
    def success_rate(self) -> float:
        """Rate of non-error results."""
        return (self.total_items - self.error_count) / self.total_items if self.total_items > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
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


class BenchmarkRunner:
    """
    Runner for benchmarking LLM SVA generation.
    
    Takes a dataset with SVAD (natural language descriptions) and reference SVA,
    generates SVAs using one or more LLMs, and evaluates correctness using
    formal implication checking.
    """

    SVA_GENERATION_SYSTEM_PROMPT = """You are an expert in SystemVerilog Assertions (SVA). Your task is to generate precise SVA properties from natural language descriptions.

When generating SVA:
1. Use proper SVA syntax with property and assert statements
2. Include appropriate clock specifications (@(posedge clk) or similar)
3. Include disable iff for reset conditions when mentioned
4. Use correct implication operators (|-> or |=>)
5. Handle timing constraints with ## delays

Output ONLY the SVA code, no explanations or additional text."""

    SVA_GENERATION_USER_PROMPT_TEMPLATE = """Generate a SystemVerilog Assertion for the following requirement:

{svad}

Output only the SVA code:"""

    def __init__(
        self,
        llm_clients: List[LLMClient],
        implication_checker: Optional[SVAImplicationChecker] = None,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            llm_clients: List of LLM clients to benchmark
            implication_checker: Optional implication checker (creates default if not provided)
        """
        self.llm_clients = llm_clients
        self.implication_checker = implication_checker or SVAImplicationChecker()

    @classmethod
    def from_configs(
        cls,
        llm_configs: List[Dict[str, Any]],
        checker_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkRunner":
        """
        Create a BenchmarkRunner from LLM configurations.
        
        Args:
            llm_configs: List of dicts with base_url, model, api_key
            checker_kwargs: Optional kwargs for implication checker
            
        Returns:
            Configured BenchmarkRunner
        """
        llm_clients = []
        for config in llm_configs:
            client = LLMClient.from_params(
                base_url=config["base_url"],
                model=config["model"],
                api_key=config["api_key"],
            )
            llm_clients.append(client)
        
        checker = SVAImplicationChecker(**(checker_kwargs or {}))
        return cls(llm_clients=llm_clients, implication_checker=checker)

    def generate_sva(self, llm_client: LLMClient, svad: str) -> str:
        """
        Generate SVA from natural language description.
        
        Args:
            llm_client: LLM client to use
            svad: Natural language description
            
        Returns:
            Generated SVA code
        """
        prompt = self.SVA_GENERATION_USER_PROMPT_TEMPLATE.format(svad=svad)
        response = llm_client.generate(
            prompt=prompt,
            system_prompt=self.SVA_GENERATION_SYSTEM_PROMPT,
            temperature=0.1,  # Low temperature for consistent output
        )
        return self._clean_sva_output(response)

    def _clean_sva_output(self, response: str) -> str:
        """
        Clean LLM output to extract SVA code.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned SVA code
        """
        # Remove markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines if they're code block markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)
        
        return response.strip()

    def evaluate_relationship(
        self,
        generated_sva: str,
        reference_sva: str,
    ) -> RelationshipType:
        """
        Evaluate the relationship between generated and reference SVA.
        
        Args:
            generated_sva: Generated SVA code
            reference_sva: Reference SVA code
            
        Returns:
            RelationshipType indicating the implication relationship
        """
        try:
            gen_implies_ref, ref_implies_gen = self.implication_checker.get_implication_relationship(
                generated_sva, reference_sva
            )
            
            if gen_implies_ref and ref_implies_gen:
                return RelationshipType.EQUIVALENT
            elif gen_implies_ref:
                return RelationshipType.GENERATED_IMPLIES_REFERENCE
            elif ref_implies_gen:
                return RelationshipType.REFERENCE_IMPLIES_GENERATED
            else:
                return RelationshipType.NO_RELATIONSHIP
                
        except Exception:
            return RelationshipType.ERROR

    def run_single(
        self,
        llm_client: LLMClient,
        svad: str,
        reference_sva: str,
    ) -> SingleResult:
        """
        Run benchmark on a single item.
        
        Args:
            llm_client: LLM client to use
            svad: Natural language description
            reference_sva: Reference SVA code
            
        Returns:
            SingleResult with evaluation
        """
        # Generate SVA
        start_gen = time.time()
        try:
            generated_sva = self.generate_sva(llm_client, svad)
        except Exception as e:
            return SingleResult(
                svad=svad,
                reference_sva=reference_sva,
                generated_sva="",
                relationship=RelationshipType.ERROR,
                error_message=f"Generation error: {str(e)}",
            )
        gen_time = time.time() - start_gen
        
        # Evaluate relationship
        start_verify = time.time()
        try:
            relationship = self.evaluate_relationship(generated_sva, reference_sva)
            error_msg = None
        except Exception as e:
            relationship = RelationshipType.ERROR
            error_msg = f"Verification error: {str(e)}"
        verify_time = time.time() - start_verify
        
        return SingleResult(
            svad=svad,
            reference_sva=reference_sva,
            generated_sva=generated_sva,
            relationship=relationship,
            error_message=error_msg,
            generation_time=gen_time,
            verification_time=verify_time,
        )

    def run_benchmark(
        self,
        dataset: List[Dict[str, Any]],
        llm_client: LLMClient,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        rate_limit_delay: float = 0.5,
    ) -> BenchmarkResult:
        """
        Run benchmark on a dataset.
        
        Args:
            dataset: List of dicts with 'SVAD' and 'SVA' keys
            llm_client: LLM client to benchmark
            progress_callback: Optional callback(current, total) for progress
            rate_limit_delay: Delay between LLM calls
            
        Returns:
            BenchmarkResult with aggregated statistics
        """
        results = []
        total = len(dataset)
        
        total_gen_time = 0.0
        total_verify_time = 0.0
        
        for i, item in enumerate(dataset):
            svad = item.get("SVAD", "")
            reference_sva = item.get("SVA", "")
            
            if not svad or not reference_sva:
                continue
            
            result = self.run_single(llm_client, svad, reference_sva)
            results.append(result)
            
            total_gen_time += result.generation_time
            total_verify_time += result.verification_time
            
            if progress_callback:
                progress_callback(i + 1, total)
            
            time.sleep(rate_limit_delay)
        
        # Aggregate results
        benchmark_result = BenchmarkResult(
            model_name=llm_client.config.model,
            total_items=len(results),
            individual_results=results,
        )
        
        for result in results:
            if result.relationship == RelationshipType.EQUIVALENT:
                benchmark_result.equivalent_count += 1
            elif result.relationship == RelationshipType.GENERATED_IMPLIES_REFERENCE:
                benchmark_result.generated_implies_reference_count += 1
            elif result.relationship == RelationshipType.REFERENCE_IMPLIES_GENERATED:
                benchmark_result.reference_implies_generated_count += 1
            elif result.relationship == RelationshipType.NO_RELATIONSHIP:
                benchmark_result.no_relationship_count += 1
            else:
                benchmark_result.error_count += 1
        
        if results:
            benchmark_result.avg_generation_time = total_gen_time / len(results)
            benchmark_result.avg_verification_time = total_verify_time / len(results)
        
        return benchmark_result

    def run_all_benchmarks(
        self,
        dataset: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        rate_limit_delay: float = 0.5,
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks for all configured LLM clients.
        
        Args:
            dataset: Dataset to benchmark against
            progress_callback: Optional callback(model_name, current, total)
            rate_limit_delay: Delay between LLM calls
            
        Returns:
            List of BenchmarkResult, one per LLM client
        """
        all_results = []
        
        for llm_client in self.llm_clients:
            def wrapped_callback(current, total):
                if progress_callback:
                    progress_callback(llm_client.config.model, current, total)
            
            result = self.run_benchmark(
                dataset,
                llm_client,
                progress_callback=wrapped_callback,
                rate_limit_delay=rate_limit_delay,
            )
            all_results.append(result)
        
        return all_results

    @staticmethod
    def compare_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        Compare benchmark results across models.
        
        Args:
            results: List of BenchmarkResult objects
            
        Returns:
            Comparison dictionary
        """
        if not results:
            return {}
        
        comparison = {
            "models": [r.model_name for r in results],
            "equivalent_rates": [r.equivalent_rate for r in results],
            "any_implication_rates": [r.any_implication_rate for r in results],
            "success_rates": [r.success_rate for r in results],
            "avg_generation_times": [r.avg_generation_time for r in results],
            "best_equivalent_rate": max(r.equivalent_rate for r in results),
            "best_any_implication_rate": max(r.any_implication_rate for r in results),
        }
        
        # Find best model for each metric
        comparison["best_equivalent_model"] = max(results, key=lambda r: r.equivalent_rate).model_name
        comparison["best_implication_model"] = max(results, key=lambda r: r.any_implication_rate).model_name
        
        return comparison
