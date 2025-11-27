"""
SVA Dataset Builder - Build training datasets with SVAD and CoT annotations.

Takes a JSON file with SVA code and generates SVAD (natural language descriptions)
using an LLM, and CoT (Chain-of-Thought) using the CoT builder.
"""

import json
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from sva_toolkit.utils.llm_client import LLMClient, LLMConfig
from sva_toolkit.cot_builder import SVACoTBuilder


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
    """

    SVAD_SYSTEM_PROMPT = """You are an expert in SystemVerilog Assertions (SVA). Your task is to generate clear, precise natural language descriptions of SVA properties.

Given an SVA property or assertion, describe:
1. What the property verifies (the intent)
2. The trigger condition (antecedent)
3. The expected behavior (consequent)
4. Any timing relationships
5. Reset/disable conditions if present

Keep the description concise but complete. Focus on the functional meaning, not the syntax."""

    SVAD_USER_PROMPT_TEMPLATE = """Generate a natural language description for the following SystemVerilog Assertion:

```systemverilog
{sva_code}
```

Provide a clear, concise description that captures the property's intent and behavior. Do not include any code in your response, only the natural language description."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        cot_builder: Optional[SVACoTBuilder] = None,
    ):
        """
        Initialize the dataset builder.
        
        Args:
            llm_client: LLM client for generating SVAD
            cot_builder: CoT builder for generating chain-of-thought
        """
        self.llm_client = llm_client
        self.cot_builder = cot_builder or SVACoTBuilder()

    @classmethod
    def from_llm_config(
        cls,
        base_url: str,
        model: str,
        api_key: str,
        **llm_kwargs: Any,
    ) -> "DatasetBuilder":
        """
        Create a DatasetBuilder with LLM configuration.
        
        Args:
            base_url: LLM API base URL
            model: Model name
            api_key: API key
            **llm_kwargs: Additional LLM config options
            
        Returns:
            Configured DatasetBuilder
        """
        llm_client = LLMClient.from_params(base_url, model, api_key, **llm_kwargs)
        return cls(llm_client=llm_client)

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
        
        prompt = self.SVAD_USER_PROMPT_TEMPLATE.format(sva_code=sva_code)
        return self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.SVAD_SYSTEM_PROMPT,
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
        max_workers: int = 1,
        rate_limit_delay: float = 0.5,
    ) -> List[DatasetEntry]:
        """
        Build dataset from input data.
        
        Args:
            input_data: List of dicts with 'SVA' key
            generate_svad: Whether to generate SVAD
            generate_cot: Whether to generate CoT
            progress_callback: Optional callback(current, total) for progress updates
            max_workers: Number of parallel workers (for CoT only, SVAD is sequential)
            rate_limit_delay: Delay between LLM calls to avoid rate limiting
            
        Returns:
            List of processed DatasetEntry objects
        """
        entries = []
        total = len(input_data)
        
        for i, item in enumerate(input_data):
            sva_code = item.get("SVA", item.get("sva", ""))
            if not sva_code:
                continue
            
            entry = DatasetEntry(SVA=sva_code)
            
            # Generate SVAD (sequential to respect rate limits)
            if generate_svad and self.llm_client:
                try:
                    entry.SVAD = self.generate_svad(sva_code)
                    time.sleep(rate_limit_delay)
                except Exception as e:
                    entry.metadata["svad_error"] = str(e)
            
            # Generate CoT
            if generate_cot:
                try:
                    entry.CoT = self.generate_cot(sva_code)
                except Exception as e:
                    entry.metadata["cot_error"] = str(e)
            
            entries.append(entry)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return entries

    def build_from_file(
        self,
        input_path: str,
        output_path: str,
        generate_svad: bool = True,
        generate_cot: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        rate_limit_delay: float = 0.5,
    ) -> List[DatasetEntry]:
        """
        Build dataset from input file and save to output file.
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file
            generate_svad: Whether to generate SVAD
            generate_cot: Whether to generate CoT
            progress_callback: Optional callback for progress updates
            rate_limit_delay: Delay between LLM calls
            
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
