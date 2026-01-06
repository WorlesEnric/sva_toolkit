#!/usr/bin/env python3
"""
Experiment runner for benchmarking SVA generation with different prompts and LLMs.

This script runs 12 experiments: 3 prompt combinations × 4 LLMs.
"""

import json
import os
import sys
import time
import tempfile
import argparse
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

# Add parent directory to path to import sva_toolkit
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sva_toolkit.utils.llm_client import LLMClient, LLMConfig
from sva_toolkit.benchmark.runner import BenchmarkRunner, BenchmarkResult, RelationshipType
from sva_toolkit.benchmark.runner import _worker_process_item, _clean_sva_output
from sva_toolkit.implication_checker import SVAImplicationChecker
from sva_toolkit.dataset_builder.builder import DatasetBuilder
from multiprocessing import Pool
from functools import partial


class CustomBenchmarkRunner(BenchmarkRunner):
    """
    Custom BenchmarkRunner that accepts custom prompts for SVA generation.
    """
    def __init__(
        self,
        llm_clients: List[LLMClient],
        system_prompt: str,
        user_prompt_template: str,
        implication_checker: Optional[SVAImplicationChecker] = None,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        verible_path: str = "verible-verilog-syntax",
    ):
        """
        Initialize custom benchmark runner with prompts.
        
        Args:
            llm_clients: List of LLM clients to benchmark
            system_prompt: System prompt for SVA generation
            user_prompt_template: User prompt template for SVA generation
            implication_checker: Optional implication checker
            num_workers: Number of worker processes
            cache_dir: Directory for caching progress
            verible_path: Path to verible-verilog-syntax binary
        """
        super().__init__(
            llm_clients=llm_clients,
            implication_checker=implication_checker,
            num_workers=num_workers,
            cache_dir=cache_dir,
            verible_path=verible_path,
        )
        self.SVA_GENERATION_SYSTEM_PROMPT = system_prompt
        self.SVA_GENERATION_USER_PROMPT_TEMPLATE = user_prompt_template

    def _run_multiprocess(
        self,
        items: List[Dict[str, Any]],
        llm_config_dict: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]],
        total: int,
    ) -> List[Dict[str, Any]]:
        """
        Run benchmark using multiprocessing with custom prompts.
        """
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
        Run benchmark in single process mode with custom prompts.
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


def create_dataset(
    input_file: str,
    output_file: str,
    prompt_config: Dict[str, str],
    llm_config: Dict[str, str],
    num_samples: Optional[int] = None,
    workers: int = 4,
) -> None:
    """
    Generate SVAD dataset using specified prompt and LLM.
    
    Args:
        input_file: Path to input dataset file (with SVA code)
        output_file: Path to output dataset file
        prompt_config: Dictionary with SVAD_SYSTEM_PROMPT and SVAD_USER_PROMPT_TEMPLATE
        llm_config: Dictionary with base_url, model, api_key
        num_samples: Number of samples to process (None for all)
        workers: Number of worker processes
    """
    print(f"Loading input dataset: {input_file}")
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    if num_samples:
        input_data = input_data[:num_samples]
        print(f"Processing {num_samples} samples")
    else:
        print(f"Processing all {len(input_data)} samples")
    llm_client_config = LLMConfig(
        base_url=llm_config["base_url"],
        model=llm_config["model"],
        api_key=llm_config["api_key"],
    )
    llm_client = LLMClient(llm_client_config)
    builder = DatasetBuilder(
        llm_client=llm_client,
        system_prompt=prompt_config.get("SVAD_SYSTEM_PROMPT"),
        num_workers=workers,
    )
    builder.SVAD_USER_PROMPT_TEMPLATE = prompt_config.get("SVAD_USER_PROMPT_TEMPLATE", builder.SVAD_USER_PROMPT_TEMPLATE)
    print("Generating SVAD descriptions...")
    entries = builder.build_dataset(
        input_data,
        generate_svad=True,
        generate_cot=True,
        use_multiprocessing=workers > 1,
    )
    output_data = [entry.to_dict() for entry in entries]
    print(f"Saving dataset to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Generated {len(output_data)} entries")


def run_benchmark(
    input_file: str,
    output_file: str,
    llm_configs: List[Dict[str, Any]],
    prompt_config: Dict[str, str],
    workers: int = 4,
    ebmc_path: Optional[str] = None,
    depth: int = 20,
    num_samples: Optional[int] = None,
    verible_path: str = "verible-verilog-syntax",
) -> Dict[str, Any]:
    """
    Run benchmark with custom prompts.
    
    Args:
        input_file: Path to dataset file (with SVAD and SVA)
        output_file: Path to output results file
        llm_configs: List of LLM configurations
        prompt_config: Dictionary with SVA_GENERATION_SYSTEM_PROMPT and SVA_GENERATION_USER_PROMPT_TEMPLATE
        workers: Number of worker processes
        ebmc_path: Path to ebmc binary
        depth: Proof depth for verification
        num_samples: Number of samples to process (None for all)
        verible_path: Path to verible-verilog-syntax binary
        
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
    cache_dir = tempfile.mkdtemp(prefix="sva_benchmark_cache_")
    all_results = []
    for llm_client in llm_clients:
        print(f"\nRunning benchmark for model: {llm_client.config.model}")
        runner = CustomBenchmarkRunner(
            llm_clients=[llm_client],
            system_prompt=prompt_config["SVA_GENERATION_SYSTEM_PROMPT"],
            user_prompt_template=prompt_config["SVA_GENERATION_USER_PROMPT_TEMPLATE"],
            implication_checker=checker,
            num_workers=workers,
            cache_dir=cache_dir,
            verible_path=verible_path,
        )
        def progress_callback(current: int, total: int) -> None:
            if current % 10 == 0 or current == total:
                print(f"  Progress: {current}/{total} ({current*100//total}%)")
        result = runner.run_benchmark(
            valid_items,
            llm_client,
            progress_callback=progress_callback,
            use_multiprocessing=workers > 1,
        )
        result_dict = result.to_dict()
        result_dict["individual_results"] = []
        for i, r in enumerate(result.individual_results):
            result_dict["individual_results"].append({
                "index": i,
                "svad": r.svad,
                "reference_sva": r.reference_sva,
                "generated_sva": r.generated_sva,
                "relationship": r.relationship.value,
                "cot": r.cot,
                "error_message": r.error_message,
                "generation_time": r.generation_time,
                "verification_time": r.verification_time,
            })
        all_results.append(result_dict)
        print(f"  Equivalent rate: {result.equivalent_rate:.2%}")
        print(f"  Any implication rate: {result.any_implication_rate:.2%}")
        print(f"  Success rate: {result.success_rate:.2%}")
    output_data = {
        "prompt_name": prompt_config.get("name", "unknown"),
        "prompt_config": prompt_config,
        "results": all_results,
    }
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    return output_data


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run SVA generation benchmarks with different prompts and LLMs"
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
        default="results",
        help="Output directory for results (default: results)",
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
        "--skip-dataset-generation",
        action="store_true",
        help="Skip dataset generation (use existing datasets or original dataset)",
    )
    parser.add_argument(
        "--use-original-dataset",
        action="store_true",
        help="Use original dataset directly without regenerating SVAD",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples for testing (default: None, use all samples)",
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
    print(f"Found {len(prompts)} prompt configurations")
    print(f"Found {len(llm_configs)} LLM configurations")
    print(f"\nRunning {len(prompts)} × {len(llm_configs)} = {len(prompts) * len(llm_configs)} experiments\n")
    all_experiment_results = []
    for prompt_idx, prompt_config in enumerate(prompts):
        prompt_name = prompt_config.get("name", f"prompt_{prompt_idx}")
        print(f"\n{'='*80}")
        print(f"PROMPT {prompt_idx + 1}/{len(prompts)}: {prompt_name}")
        print(f"{'='*80}")
        if args.use_original_dataset:
            dataset_file = input_dataset
            print(f"Using original dataset directly: {dataset_file}")
        else:
            dataset_file = output_dir / f"dataset_{prompt_name}.json"
            if not args.skip_dataset_generation:
                print(f"Generating dataset for prompt {prompt_name}...")
                try:
                    create_dataset(
                        input_file=str(input_dataset),
                        output_file=str(dataset_file),
                        prompt_config=prompt_config,
                        llm_config=llm_configs[0],
                        num_samples=args.num_samples,
                        workers=args.workers,
                    )
                except Exception as e:
                    print(f"ERROR generating dataset: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                if not dataset_file.exists():
                    print(f"WARNING: Dataset file not found: {dataset_file}")
                    print(f"Using original dataset: {input_dataset}")
                    dataset_file = input_dataset
                else:
                    print(f"Using existing dataset: {dataset_file}")
        for llm_idx, llm_config in enumerate(llm_configs):
            model_name = llm_config["model"]
            exp_name = f"{prompt_name}_{model_name}"
            print(f"\nExperiment {prompt_idx * len(llm_configs) + llm_idx + 1}/{len(prompts) * len(llm_configs)}: {exp_name}")
            print(f"  Prompt: {prompt_name}")
            print(f"  Model: {model_name}")
            result_file = output_dir / f"result_{exp_name}.json"
            print(f"  Running benchmark...")
            try:
                result = run_benchmark(
                    input_file=str(dataset_file),
                    output_file=str(result_file),
                    llm_configs=[llm_config],
                    prompt_config=prompt_config,
                    workers=args.workers,
                    ebmc_path=args.ebmc_path,
                    depth=args.depth,
                    num_samples=args.num_samples,
                    verible_path=args.verible_path,
                )
                all_experiment_results.append(result)
            except Exception as e:
                print(f"  ERROR running benchmark: {e}")
                import traceback
                traceback.print_exc()
                continue
    summary_file = output_dir / "summary.json"
    print(f"\n{'='*80}")
    print(f"Saving summary to: {summary_file}")
    with open(summary_file, 'w') as f:
        json.dump(all_experiment_results, f, indent=2)
    print(f"\nCompleted {len(all_experiment_results)} experiments")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
