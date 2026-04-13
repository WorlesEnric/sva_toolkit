"""
SVA Dataset Builder - Build training datasets with SVAD and CoT annotations.

Takes a JSON file with SVA code and generates SVAD (natural language descriptions)
using an LLM, and CoT (Chain-of-Thought) using the CoT builder.

Supports multiprocessing for faster dataset building and progress caching
to resume interrupted builds.
"""

import json
import time
import hashlib
import tempfile
import os
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from multiprocessing import Pool
from functools import partial

from sva_toolkit.utils.llm_client import LLMClient, LLMConfig
from sva_toolkit.cot_builder import SVACoTBuilder
from sva_toolkit.ast_parser import SVAASTParser


def _compute_entry_hash(sva_code: str, model: str) -> str:
    """
    Compute a unique hash for a dataset entry.
    
    Args:
        sva_code: SVA code
        model: Model name used for SVAD generation
        
    Returns:
        Hash string for the entry
    """
    content = f"{sva_code}|{model}"
    return hashlib.md5(content.encode()).hexdigest()


def _worker_process_entry(
    item_data: Dict[str, Any],
    llm_config_dict: Optional[Dict[str, Any]],
    generate_svad: bool,
    generate_cot: bool,
    system_prompt: str,
    user_prompt_template: str,
    cache_dir: Optional[str],
) -> Dict[str, Any]:
    """
    Worker function to process a single dataset entry.
    
    This function runs in a separate process and handles:
    1. SVAD generation using LLM (if enabled)
    2. CoT generation using template matching
    3. Caching of results
    
    Args:
        item_data: Dict with 'sva_code' and 'index'
        llm_config_dict: LLM configuration as dict (None if no SVAD)
        generate_svad: Whether to generate SVAD
        generate_cot: Whether to generate CoT
        system_prompt: System prompt for SVAD generation
        user_prompt_template: User prompt template
        cache_dir: Directory for caching results
        
    Returns:
        Dict with result data
    """
    sva_code = item_data["sva_code"]
    index = item_data["index"]
    model = llm_config_dict["model"] if llm_config_dict else "no_llm"
    # Check cache first
    entry_hash = _compute_entry_hash(sva_code, model)
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"{entry_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    cached["from_cache"] = True
                    return cached
            except (json.JSONDecodeError, IOError):
                pass  # Cache corrupted, regenerate
    result: Dict[str, Any] = {
        "index": index,
        "SVA": sva_code,
        "SVAD": None,
        "CoT": None,
        "metadata": {},
        "from_cache": False,
    }
    # Create AST parser and CoT builder in this process
    ast_parser = SVAASTParser()
    cot_builder = SVACoTBuilder()
    # Generate SVAD if enabled and LLM configured
    if generate_svad and llm_config_dict:
        try:
            llm_config = LLMConfig(**llm_config_dict)
            llm_client = LLMClient(llm_config)
            # Extract signal info
            signal_info = _format_signal_info(sva_code, ast_parser)
            prompt = user_prompt_template.format(
                sva_code=sva_code,
                signal_info=signal_info,
            )
            svad = llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
            )
            result["SVAD"] = svad
        except Exception as e:
            result["metadata"]["svad_error"] = str(e)
    # Generate CoT if enabled
    if generate_cot:
        try:
            cot = cot_builder.build(sva_code)
            result["CoT"] = cot
        except Exception as e:
            result["metadata"]["cot_error"] = str(e)
    # Save to cache
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"{entry_hash}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except IOError:
            pass  # Ignore cache write errors
    return result


def _format_signal_info(sva_code: str, ast_parser: SVAASTParser) -> str:
    """
    Extract and format signal information from SVA code.
    
    Args:
        sva_code: SVA code
        ast_parser: AST parser instance
        
    Returns:
        Formatted string describing all signals in the SVA
    """
    structure = ast_parser.parse(sva_code)
    lines: List[str] = ["The following signals are used in this assertion:"]
    if structure.clock_signal:
        lines.append(f"- Clock signal: {structure.clock_signal} ({structure.clock_edge})")
    if structure.reset_signal:
        polarity = "active-low" if structure.reset_active_low else "active-high"
        lines.append(f"- Reset signal: {structure.reset_signal} ({polarity})")
    other_signals = [
        s.name for s in structure.signals
        if s.name != structure.clock_signal and s.name != structure.reset_signal
    ]
    if other_signals:
        lines.append(f"- Other signals: {', '.join(sorted(other_signals))}")
    if structure.builtin_functions:
        func_names = list(set(f.name for f in structure.builtin_functions))
        lines.append(f"- Built-in functions used: {', '.join(func_names)}")
    return "\n".join(lines)


@dataclass
class DatasetEntry:
    """A single entry in the dataset."""
    SVA: str
    SVAD: Optional[str] = None
    CoT: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"SVA": self.SVA}
        if self.SVAD is not None:
            result["SVAD"] = self.SVAD
        if self.CoT is not None:
            result["CoT"] = self.CoT
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class DatasetBuilder:
    """
    Builder for SVA training datasets.
    
    Takes raw SVA code and generates:
    - SVAD: Natural Language Description (via LLM)
    - CoT: Chain-of-Thought reasoning (via template matching)
    
    Supports multiprocessing for parallel processing and progress caching
    to resume interrupted builds.
    """

    SVAD_SYSTEM_PROMPT = "You are an expert in SystemVerilog Assertions (SVA). Your task is to generate clear, precise natural language descriptions of SVA properties.\n Given an SVA property or assertion, describe: \n1. What the property verifies (the intent)\n2. The trigger condition (antecedent)\n3. The expected behavior (consequent)\n4. Any timing relationships\n5. Reset/disable conditions if present\nKeep the description concise but complete. Focus on the functional meaning, not the syntax."

    SVAD_USER_PROMPT_TEMPLATE = "Generate a natural language description for the following SystemVerilog Assertion:\n    \n    ```systemverilog\n    {sva_code}\n    ```\n    \n    {signal_info}\n    \n    Provide a clear, concise description that captures the property's intent and behavior. Your description MUST explicitly mention ALL the signal names listed above. Do not include any code in your response, only the natural language description.\n"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        cot_builder: Optional[SVACoTBuilder] = None,
        ast_parser: Optional[SVAASTParser] = None,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the dataset builder.
        
        Args:
            llm_client: LLM client for generating SVAD
            cot_builder: CoT builder for generating chain-of-thought
            ast_parser: AST parser for extracting signals from SVA
            num_workers: Number of worker processes for parallel processing (default: 4)
            cache_dir: Directory for caching progress (default: creates temp dir)
            system_prompt: Custom system prompt for SVAD generation (default: uses SVAD_SYSTEM_PROMPT)
        """
        self.llm_client = llm_client
        self.cot_builder = cot_builder or SVACoTBuilder()
        self.ast_parser = ast_parser or SVAASTParser()
        self.num_workers = num_workers
        self.system_prompt = system_prompt or self.SVAD_SYSTEM_PROMPT
        # Set up cache directory
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        else:
            self.cache_dir = tempfile.mkdtemp(prefix="sva_dataset_cache_")

    @classmethod
    def from_llm_config(
        cls,
        base_url: str,
        model: str,
        api_key: str,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **llm_kwargs: Any,
    ) -> "DatasetBuilder":
        """
        Create a DatasetBuilder with LLM configuration.
        
        Args:
            base_url: LLM API base URL
            model: Model name
            api_key: API key
            num_workers: Number of worker processes for parallel processing
            cache_dir: Directory for caching progress
            system_prompt: Custom system prompt for SVAD generation
            **llm_kwargs: Additional LLM config options
            
        Returns:
            Configured DatasetBuilder
        """
        llm_client = LLMClient.from_params(base_url, model, api_key, **llm_kwargs)
        return cls(
            llm_client=llm_client,
            num_workers=num_workers,
            cache_dir=cache_dir,
            system_prompt=system_prompt,
        )

    def _format_signal_info_instance(self, sva_code: str) -> str:
        """
        Extract and format signal information from SVA code (instance method).
        
        Args:
            sva_code: SVA code
            
        Returns:
            Formatted string describing all signals in the SVA
        """
        return _format_signal_info(sva_code, self.ast_parser)

    def generate_svad(self, sva_code: str) -> str:
        """
        Generate natural language description for SVA code.
        
        Args:
            sva_code: SVA code
            
        Returns:
            Natural language description
        """
        if not self.llm_client:
            raise RuntimeError("LLM client not configured. Cannot generate SVAD.")
        signal_info = self._format_signal_info_instance(sva_code)
        prompt = self.SVAD_USER_PROMPT_TEMPLATE.format(
            sva_code=sva_code,
            signal_info=signal_info,
        )
        return self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.3,  # Lower temperature for more consistent descriptions
        )

    def generate_cot(self, sva_code: str) -> str:
        """
        Generate Chain-of-Thought for SVA code.
        
        Args:
            sva_code: SVA code
            
        Returns:
            Chain-of-Thought reasoning
        """
        return self.cot_builder.build(sva_code)

    def process_entry(
        self,
        entry: DatasetEntry,
        generate_svad: bool = True,
        generate_cot: bool = True,
    ) -> DatasetEntry:
        """
        Process a single dataset entry.
        
        Args:
            entry: Dataset entry with SVA code
            generate_svad: Whether to generate SVAD
            generate_cot: Whether to generate CoT
            
        Returns:
            Processed entry with SVAD and/or CoT
        """
        if generate_svad and self.llm_client:
            entry.SVAD = self.generate_svad(entry.SVA)
        
        if generate_cot:
            entry.CoT = self.generate_cot(entry.SVA)
        
        return entry

    def build_dataset(
        self,
        input_data: List[Dict[str, Any]],
        generate_svad: bool = True,
        generate_cot: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        rate_limit_delay: float = 0.5,
        use_multiprocessing: bool = True,
    ) -> List[DatasetEntry]:
        """
        Build dataset from input data.
        
        Args:
            input_data: List of dicts with 'SVA' key
            generate_svad: Whether to generate SVAD
            generate_cot: Whether to generate CoT
            progress_callback: Optional callback(current, total) for progress updates
            rate_limit_delay: Delay between LLM calls (single-process mode only)
            use_multiprocessing: Whether to use multiprocessing (default: True)
            
        Returns:
            List of processed DatasetEntry objects
        """
        # Prepare items for processing
        items_to_process = []
        for i, item in enumerate(input_data):
            sva_code = item.get("SVA", item.get("sva", ""))
            if not sva_code:
                continue
            items_to_process.append({
                "index": i,
                "sva_code": sva_code,
            })
        total = len(items_to_process)
        if total == 0:
            return []
        # Prepare LLM config dict for worker processes
        llm_config_dict = None
        if self.llm_client and generate_svad:
            llm_config_dict = {
                "base_url": self.llm_client.config.base_url,
                "model": self.llm_client.config.model,
                "api_key": self.llm_client.config.api_key,
                "temperature": self.llm_client.config.temperature,
                "max_tokens": self.llm_client.config.max_tokens,
            }
        if use_multiprocessing and self.num_workers > 1:
            results = self._run_multiprocess(
                items_to_process,
                llm_config_dict,
                generate_svad,
                generate_cot,
                progress_callback,
                total,
            )
        else:
            results = self._run_single_process(
                items_to_process,
                llm_config_dict,
                generate_svad,
                generate_cot,
                progress_callback,
                total,
                rate_limit_delay,
            )
        # Convert results to DatasetEntry objects
        entries = []
        for r in results:
            entry = DatasetEntry(
                SVA=r["SVA"],
                SVAD=r.get("SVAD"),
                CoT=r.get("CoT"),
                metadata=r.get("metadata", {}),
            )
            entries.append(entry)
        return entries

    def _run_multiprocess(
        self,
        items: List[Dict[str, Any]],
        llm_config_dict: Optional[Dict[str, Any]],
        generate_svad: bool,
        generate_cot: bool,
        progress_callback: Optional[Callable[[int, int], None]],
        total: int,
    ) -> List[Dict[str, Any]]:
        """
        Build dataset using multiprocessing.
        
        Args:
            items: List of items to process
            llm_config_dict: LLM config as dict
            generate_svad: Whether to generate SVAD
            generate_cot: Whether to generate CoT
            progress_callback: Progress callback
            total: Total number of items
            
        Returns:
            List of result dicts
        """
        worker_fn = partial(
            _worker_process_entry,
            llm_config_dict=llm_config_dict,
            generate_svad=generate_svad,
            generate_cot=generate_cot,
            system_prompt=self.system_prompt,
            user_prompt_template=self.SVAD_USER_PROMPT_TEMPLATE,
            cache_dir=self.cache_dir,
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
        llm_config_dict: Optional[Dict[str, Any]],
        generate_svad: bool,
        generate_cot: bool,
        progress_callback: Optional[Callable[[int, int], None]],
        total: int,
        rate_limit_delay: float,
    ) -> List[Dict[str, Any]]:
        """
        Build dataset in single process mode.
        
        Args:
            items: List of items to process
            llm_config_dict: LLM config as dict
            generate_svad: Whether to generate SVAD
            generate_cot: Whether to generate CoT
            progress_callback: Progress callback
            total: Total number of items
            rate_limit_delay: Delay between LLM calls
            
        Returns:
            List of result dicts
        """
        results = []
        for i, item in enumerate(items):
            result = _worker_process_entry(
                item_data=item,
                llm_config_dict=llm_config_dict,
                generate_svad=generate_svad,
                generate_cot=generate_cot,
                system_prompt=self.system_prompt,
                user_prompt_template=self.SVAD_USER_PROMPT_TEMPLATE,
                cache_dir=self.cache_dir,
            )
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, total)
            # Only delay if SVAD was generated (LLM call made) and not from cache
            if generate_svad and llm_config_dict and not result.get("from_cache", False):
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

    def build_from_file(
        self,
        input_path: str,
        output_path: str,
        generate_svad: bool = True,
        generate_cot: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        rate_limit_delay: float = 0.5,
        use_multiprocessing: bool = True,
    ) -> List[DatasetEntry]:
        """
        Build dataset from input file and save to output file.
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file
            generate_svad: Whether to generate SVAD
            generate_cot: Whether to generate CoT
            progress_callback: Optional callback for progress updates
            rate_limit_delay: Delay between LLM calls (single-process mode only)
            use_multiprocessing: Whether to use multiprocessing (default: True)
            
        Returns:
            List of processed DatasetEntry objects
        """
        # Load input data
        with open(input_path, 'r') as f:
            input_data = json.load(f)
        # Process dataset
        entries = self.build_dataset(
            input_data,
            generate_svad=generate_svad,
            generate_cot=generate_cot,
            progress_callback=progress_callback,
            rate_limit_delay=rate_limit_delay,
            use_multiprocessing=use_multiprocessing,
        )
        # Save output
        output_data = [entry.to_dict() for entry in entries]
        with open(output_path, 'w') as f:
            json.dump(output_data, indent=2, fp=f)
        return entries

    def validate_dataset(self, entries: List[DatasetEntry]) -> Dict[str, Any]:
        """
        Validate dataset entries.
        
        Args:
            entries: List of dataset entries
            
        Returns:
            Validation report
        """
        total = len(entries)
        has_svad = sum(1 for e in entries if e.SVAD)
        has_cot = sum(1 for e in entries if e.CoT)
        has_errors = sum(1 for e in entries if e.metadata.get("svad_error") or e.metadata.get("cot_error"))
        
        return {
            "total_entries": total,
            "entries_with_svad": has_svad,
            "entries_with_cot": has_cot,
            "entries_with_errors": has_errors,
            "svad_coverage": has_svad / total if total > 0 else 0,
            "cot_coverage": has_cot / total if total > 0 else 0,
        }
