"""
SVA Benchmark Runner - Evaluate LLM performance on SVA generation.

Tests LLM's ability to generate SVAs from natural language descriptions,
then uses implication checking to evaluate correctness.

Supports multiprocessing for faster evaluation and progress caching
to resume interrupted runs.
"""

import json
import time
import hashlib
import tempfile
import os
import re
import subprocess
import shutil
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

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


def _compute_item_hash(svad: str, reference_sva: str, model: str) -> str:
    """
    Compute a unique hash for a benchmark item.
    
    Args:
        svad: Natural language description
        reference_sva: Reference SVA code
        model: Model name
        
    Returns:
        Hash string for the item
    """
    content = f"{svad}|{reference_sva}|{model}"
    return hashlib.md5(content.encode()).hexdigest()


def _worker_process_item(
    item_data: Dict[str, Any],
    llm_config_dict: Dict[str, Any],
    checker_kwargs: Dict[str, Any],
    system_prompt: str,
    user_prompt_template: str,
    cache_dir: Optional[str],
    verible_path: str = "verible-verilog-syntax",
) -> Dict[str, Any]:
    """
    Worker function to process a single benchmark item.
    
    This function runs in a separate process and handles:
    1. SVA generation using LLM
    2. Relationship evaluation using implication checker
    3. Caching of results
    
    Args:
        item_data: Dict with 'svad', 'reference_sva', 'cot', 'index'
        llm_config_dict: LLM configuration as dict
        checker_kwargs: Kwargs for implication checker
        system_prompt: System prompt for SVA generation
        user_prompt_template: User prompt template
        cache_dir: Directory for caching results
        verible_path: Path to verible-verilog-syntax binary
        
    Returns:
        Dict with result data
    """
    svad = item_data["svad"]
    reference_sva = item_data["reference_sva"]
    cot = item_data.get("cot")
    index = item_data["index"]
    model = llm_config_dict["model"]
    # Check cache first
    item_hash = _compute_item_hash(svad, reference_sva, model)
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
    # Generate SVA
    start_gen = time.time()
    try:
        prompt = user_prompt_template.format(svad=svad)
        response = llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
        )
        generated_sva = _clean_sva_output(response, verible_path)
        gen_error = None
    except Exception as e:
        generated_sva = ""
        gen_error = f"Generation error: {str(e)}"
    gen_time = time.time() - start_gen
    # Evaluate relationship if generation succeeded
    start_verify = time.time()
    if gen_error:
        relationship = RelationshipType.ERROR.value
        verify_error = gen_error
    else:
        try:
            gen_implies_ref, ref_implies_gen = checker.get_implication_relationship(
                generated_sva, reference_sva
            )
            if gen_implies_ref and ref_implies_gen:
                relationship = RelationshipType.EQUIVALENT.value
            elif gen_implies_ref:
                relationship = RelationshipType.GENERATED_IMPLIES_REFERENCE.value
            elif ref_implies_gen:
                relationship = RelationshipType.REFERENCE_IMPLIES_GENERATED.value
            else:
                relationship = RelationshipType.NO_RELATIONSHIP.value
            verify_error = None
        except Exception as e:
            relationship = RelationshipType.ERROR.value
            verify_error = f"Verification error: {str(e)}"
    verify_time = time.time() - start_verify
    result = {
        "index": index,
        "svad": svad,
        "reference_sva": reference_sva,
        "generated_sva": generated_sva,
        "relationship": relationship,
        "cot": cot,
        "error_message": gen_error or verify_error,
        "generation_time": gen_time,
        "verification_time": verify_time,
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


def _check_verible_syntax(code: str, verible_path: str = "verible-verilog-syntax") -> bool:
    """
    Validates SystemVerilog syntax using Verible.
    
    Handles both standalone modules and code fragments by attempting a wrapped check.
    
    Args:
        code: SystemVerilog code to validate
        verible_path: Path to verible-verilog-syntax binary
        
    Returns:
        True if syntax is valid, False otherwise
    """
    # Check if verible exists
    if not shutil.which(verible_path):
        raise FileNotFoundError(f"Verible binary not found at '{verible_path}'. Please install it or add to PATH.")
    def run_check(content: str) -> bool:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            # Run verible-verilog-syntax
            result = subprocess.run(
                [verible_path, tmp_path],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    # Strategy A: Check code exactly as provided (e.g., if LLM generated a full module)
    if run_check(code):
        return True
    # Strategy B: Wrap in a dummy module (e.g., if LLM generated just 'assert property...')
    # We add common SVA imports or contexts if necessary, though usually a raw module wrapper suffices.
    wrapped_code = f"module syntax_check_wrapper;\n{code}\nendmodule"
    if run_check(wrapped_code):
        return True
    return False


def _strip_comments_and_extract_sva(code: str) -> str:
    """
    Strip comments from SystemVerilog code.
    
    Removes:
    - Single-line comments (// ...) but preserves // inside string literals
    - Multi-line comments (/* ... */) but preserves /* */ inside string literals
    - Empty lines after comment removal
    - Lines that are only whitespace and comments
    
    Args:
        code: Code string potentially containing comments
        
    Returns:
        Cleaned code with comments removed, preserving all SVA code
    """
    if not code:
        return ""
    lines = code.split('\n')
    cleaned_lines = []
    in_multiline_comment = False
    for line in lines:
        # Handle multi-line comments (check if not inside string)
        if '/*' in line and not in_multiline_comment:
            # Simple check: if /* appears before any quote, it's likely a comment
            # More sophisticated: track if we're in a string
            quote_chars = ['"', "'"]
            in_string = False
            string_char = None
            comment_start = -1
            for i, char in enumerate(line):
                if char in quote_chars and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                elif not in_string and char == '/' and i + 1 < len(line) and line[i+1] == '*':
                    comment_start = i
                    break
            if comment_start >= 0:
                if '*/' in line[comment_start:]:
                    # Comment starts and ends on same line
                    line = line[:comment_start] + line[line.index('*/', comment_start) + 2:]
                else:
                    # Comment starts on this line
                    line = line[:comment_start]
                    in_multiline_comment = True
        elif in_multiline_comment:
            if '*/' in line:
                # Comment ends on this line
                line = line[line.index('*/') + 2:]
                in_multiline_comment = False
            else:
                # Still in comment, skip this line
                continue
        # Remove single-line comments (but preserve // inside strings)
        if '//' in line and not in_multiline_comment:
            # Check if // is inside a string literal
            quote_chars = ['"', "'"]
            in_string = False
            string_char = None
            comment_pos = -1
            for i, char in enumerate(line):
                if char in quote_chars and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                elif not in_string and char == '/' and i + 1 < len(line) and line[i+1] == '/':
                    comment_pos = i
                    break
            if comment_pos >= 0:
                line = line[:comment_pos]
        # Strip whitespace
        line = line.strip()
        # Keep non-empty lines
        if line:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def _extract_valid_sva(response: str, verible_path: str = "verible-verilog-syntax") -> Optional[str]:
    """
    Extracts code blocks from LLM response and returns the first one that passes SVA syntax checks.
    
    Args:
        response: Raw LLM output
        verible_path: Path to the verible-verilog-syntax binary
        
    Returns:
        The valid SVA code string, or None if no valid code is found
    """
    # 1. Regex to find markdown code blocks (handles systemverilog, verilog, sv, or no language tag)
    # This regex is non-greedy (*?) to capture multiple separate blocks if present.
    code_block_pattern = r"```(?:systemverilog|verilog|sv)?\s*\n?(.*?)```"
    matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
    # 2. If no code blocks found, treat the whole string as potential code (fallback)
    candidates = matches if matches else [response.strip()]
    # 3. Iterate through candidates and validate
    for candidate in candidates:
        cleaned_candidate = candidate.strip()
        if not cleaned_candidate:
            continue
        # Strip comments before validation
        cleaned_candidate = _strip_comments_and_extract_sva(cleaned_candidate)
        if not cleaned_candidate:
            continue
        try:
            if _check_verible_syntax(cleaned_candidate, verible_path):
                return cleaned_candidate
        except (FileNotFoundError, Exception):
            # If Verible is not available or fails, fall through to next candidate
            # or return the cleaned candidate as fallback
            pass
    # Fallback: if Verible validation fails, return the first non-empty candidate (with comments stripped)
    # This maintains backward compatibility
    for candidate in candidates:
        cleaned_candidate = candidate.strip()
        if cleaned_candidate:
            cleaned_candidate = _strip_comments_and_extract_sva(cleaned_candidate)
            if cleaned_candidate:
                return cleaned_candidate
    return None


def _clean_sva_output(response: str, verible_path: str = "verible-verilog-syntax") -> str:
    """
    Clean LLM output to extract SVA code using Verible for semantic validation.
    
    Extracts code blocks from markdown and validates syntax using Verible.
    Falls back to naive extraction if Verible is unavailable.
    
    Args:
        response: Raw LLM response
        verible_path: Path to verible-verilog-syntax binary
        
    Returns:
        Cleaned SVA code (empty string if extraction fails)
    """
    # Try Verible-based extraction first
    try:
        valid_sva = _extract_valid_sva(response, verible_path)
        if valid_sva:
            return valid_sva
    except (FileNotFoundError, Exception):
        # Fall back to naive extraction if Verible is not available
        pass
    # Fallback: naive extraction (original implementation)
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)
    # Strip comments from fallback extraction as well
    response = _strip_comments_and_extract_sva(response.strip())
    return response.strip()


@dataclass
class SingleResult:
    """Result for a single benchmark item."""
    svad: str
    reference_sva: str
    generated_sva: str
    relationship: RelationshipType
    cot: Optional[str] = None
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
    
    Supports multiprocessing for parallel evaluation and progress caching
    to resume interrupted benchmark runs.
    """

    SVA_GENERATION_SYSTEM_PROMPT = "You are an expert SystemVerilog Assertion engineer. Your task is to translate natural language requirements into syntactically correct and semantically accurate SVA properties.\n\nCritical requirements:\n1. Property structure: Use 'property' keyword with a descriptive name, then 'endproperty'\n2. Clocking: Always specify @(posedge clk) or @(negedge clk) - infer from context if not specified\n3. Reset handling: Use 'disable iff (condition)' when reset is mentioned - match the reset polarity correctly\n4. Implication: Use |-> for overlapping (same cycle) or |=> for non-overlapping (next cycle) based on timing\n5. Delays: Use ##N for N clock cycles, ##0 for same cycle\n6. Assertion: Create an assert statement with a descriptive name and error message\n\nOutput format: Pure SVA code only, no markdown, no explanations, no comments."

    SVA_GENERATION_USER_PROMPT_TEMPLATE = "Translate this requirement into a SystemVerilog Assertion:\n\n{svad}\n\nGenerate the complete SVA code:"

    def __init__(
        self,
        llm_clients: List[LLMClient],
        implication_checker: Optional[SVAImplicationChecker] = None,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        verible_path: str = "verible-verilog-syntax",
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            llm_clients: List of LLM clients to benchmark
            implication_checker: Optional implication checker (creates default if not provided)
            num_workers: Number of worker processes for parallel evaluation (default: 4)
            cache_dir: Directory for caching progress (default: creates temp dir)
            verible_path: Path to verible-verilog-syntax binary (default: "verible-verilog-syntax")
        """
        self.llm_clients = llm_clients
        self.implication_checker = implication_checker or SVAImplicationChecker()
        self.num_workers = num_workers
        self.verible_path = verible_path
        # Set up cache directory
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        else:
            self.cache_dir = tempfile.mkdtemp(prefix="sva_benchmark_cache_")
        # Store checker kwargs for worker processes
        self._checker_kwargs = {
            "ebmc_path": getattr(self.implication_checker, 'ebmc_path', None),
            "depth": getattr(self.implication_checker, 'depth', 20),
            "timeout": getattr(self.implication_checker, 'timeout', 300),
        }

    @classmethod
    def from_configs(
        cls,
        llm_configs: List[Dict[str, Any]],
        checker_kwargs: Optional[Dict[str, Any]] = None,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        verible_path: str = "verible-verilog-syntax",
    ) -> "BenchmarkRunner":
        """
        Create a BenchmarkRunner from LLM configurations.
        
        Args:
            llm_configs: List of dicts with base_url, model, api_key
            checker_kwargs: Optional kwargs for implication checker
            num_workers: Number of worker processes for parallel evaluation
            cache_dir: Directory for caching progress
            verible_path: Path to verible-verilog-syntax binary (default: "verible-verilog-syntax")
            
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
        return cls(
            llm_clients=llm_clients,
            implication_checker=checker,
            num_workers=num_workers,
            cache_dir=cache_dir,
            verible_path=verible_path,
        )

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
        return _clean_sva_output(response, self.verible_path)

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
            
        Raises:
            Exception: Re-raises any exception from implication checking
                       so the caller can capture the error message
        """
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

    def run_single(
        self,
        llm_client: LLMClient,
        svad: str,
        reference_sva: str,
        cot: Optional[str] = None,
    ) -> SingleResult:
        """
        Run benchmark on a single item.
        
        Args:
            llm_client: LLM client to use
            svad: Natural language description
            reference_sva: Reference SVA code
            cot: Optional Chain of Thought from dataset
            
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
                cot=cot,
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
            cot=cot,
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
        use_multiprocessing: bool = True,
    ) -> BenchmarkResult:
        """
        Run benchmark on a dataset.
        
        Args:
            dataset: List of dicts with 'SVAD' and 'SVA' keys
            llm_client: LLM client to benchmark
            progress_callback: Optional callback(current, total) for progress
            rate_limit_delay: Delay between LLM calls (only used in single-process mode)
            use_multiprocessing: Whether to use multiprocessing (default: True)
            
        Returns:
            BenchmarkResult with aggregated statistics
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
            return BenchmarkResult(
                model_name=llm_client.config.model,
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
                rate_limit_delay,
            )
        # Convert results to SingleResult objects
        single_results = []
        total_gen_time = 0.0
        total_verify_time = 0.0
        for r in results:
            single_result = SingleResult(
                svad=r["svad"],
                reference_sva=r["reference_sva"],
                generated_sva=r["generated_sva"],
                relationship=RelationshipType(r["relationship"]),
                cot=r.get("cot"),
                error_message=r.get("error_message"),
                generation_time=r["generation_time"],
                verification_time=r["verification_time"],
            )
            single_results.append(single_result)
            total_gen_time += r["generation_time"]
            total_verify_time += r["verification_time"]
        # Aggregate results
        benchmark_result = BenchmarkResult(
            model_name=llm_client.config.model,
            total_items=len(single_results),
            individual_results=single_results,
        )
        for result in single_results:
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
        if single_results:
            benchmark_result.avg_generation_time = total_gen_time / len(single_results)
            benchmark_result.avg_verification_time = total_verify_time / len(single_results)
        return benchmark_result

    def _run_multiprocess(
        self,
        items: List[Dict[str, Any]],
        llm_config_dict: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]],
        total: int,
    ) -> List[Dict[str, Any]]:
        """
        Run benchmark using multiprocessing.
        
        Args:
            items: List of items to process
            llm_config_dict: LLM config as dict
            progress_callback: Progress callback
            total: Total number of items
            
        Returns:
            List of result dicts
        """
        # Create partial function with fixed arguments
        worker_fn = partial(
            _worker_process_item,
            llm_config_dict=llm_config_dict,
            checker_kwargs=self._checker_kwargs,
            system_prompt=self.SVA_GENERATION_SYSTEM_PROMPT,
            user_prompt_template=self.SVA_GENERATION_USER_PROMPT_TEMPLATE,
            cache_dir=self.cache_dir,
            verible_path=self.verible_path,
        )
        results = []
        completed = 0
        with Pool(processes=self.num_workers) as pool:
            for result in pool.imap_unordered(worker_fn, items):
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        # Sort results by original index
        results.sort(key=lambda x: x["index"])
        return results

    def _run_single_process(
        self,
        items: List[Dict[str, Any]],
        llm_config_dict: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]],
        total: int,
        rate_limit_delay: float,
    ) -> List[Dict[str, Any]]:
        """
        Run benchmark in single process mode.
        
        Args:
            items: List of items to process
            llm_config_dict: LLM config as dict
            progress_callback: Progress callback
            total: Total number of items
            rate_limit_delay: Delay between calls
            
        Returns:
            List of result dicts
        """
        results = []
        for i, item in enumerate(items):
            result = _worker_process_item(
                item_data=item,
                llm_config_dict=llm_config_dict,
                checker_kwargs=self._checker_kwargs,
                system_prompt=self.SVA_GENERATION_SYSTEM_PROMPT,
                user_prompt_template=self.SVA_GENERATION_USER_PROMPT_TEMPLATE,
                cache_dir=self.cache_dir,
                verible_path=self.verible_path,
            )
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, total)
            if not result.get("from_cache", False):
                time.sleep(rate_limit_delay)
        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dict with cache statistics
        """
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return {"cached_items": 0, "cache_dir": self.cache_dir}
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        return {
            "cached_items": len(cache_files),
            "cache_dir": self.cache_dir,
        }

    def clear_cache(self) -> int:
        """
        Clear all cached results.
        
        Returns:
            Number of cache files deleted
        """
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return 0
        count = 0
        for f in os.listdir(self.cache_dir):
            if f.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, f))
                count += 1
        return count

    def run_all_benchmarks(
        self,
        dataset: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        rate_limit_delay: float = 0.5,
        use_multiprocessing: bool = True,
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks for all configured LLM clients.
        
        Args:
            dataset: Dataset to benchmark against
            progress_callback: Optional callback(model_name, current, total)
            rate_limit_delay: Delay between LLM calls (single-process mode only)
            use_multiprocessing: Whether to use multiprocessing (default: True)
            
        Returns:
            List of BenchmarkResult, one per LLM client
        """
        all_results = []
        for llm_client in self.llm_clients:
            def wrapped_callback(current: int, total: int) -> None:
                if progress_callback:
                    progress_callback(llm_client.config.model, current, total)
            result = self.run_benchmark(
                dataset,
                llm_client,
                progress_callback=wrapped_callback,
                rate_limit_delay=rate_limit_delay,
                use_multiprocessing=use_multiprocessing,
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
