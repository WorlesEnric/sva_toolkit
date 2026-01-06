"""
Implication check test for SVA equivalence verification.

This script checks equivalence between SVA pairs from syntax_check.json
using the SVAImplicationChecker.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from sva_toolkit.implication_checker import SVAImplicationChecker
from sva_toolkit.implication_checker.checker import (
    ImplicationResult,
    CheckResult,
)


@dataclass
class EquivalenceResult:
    """Result of an equivalence check for an SVA pair."""
    id: str
    sva1: str
    sva2: str
    is_equivalent: bool
    result: ImplicationResult
    message: str
    sva1_implies_sva2: bool
    sva2_implies_sva1: bool


def load_sva_pairs(json_path: str) -> List[Dict[str, Any]]:
    """
    Load SVA pairs from JSON file.
    
    Args:
        json_path: Path to the JSON file containing SVA pairs.
        
    Returns:
        List of dictionaries with 'id', 'sva1', and 'sva2' keys.
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def check_sva_pair_equivalence(
    checker: SVAImplicationChecker,
    pair: Dict[str, Any],
    verbose: bool = False
) -> EquivalenceResult:
    """
    Check equivalence for a single SVA pair.
    
    Args:
        checker: The SVAImplicationChecker instance.
        pair: Dictionary containing 'id', 'sva1', and 'sva2'.
        verbose: Whether to print verbose output.
        
    Returns:
        EquivalenceResult with the verification outcome.
    """
    pair_id = pair.get("id", "unknown")
    sva1 = pair.get("sva1", "")
    sva2 = pair.get("sva2", "")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Checking pair: {pair_id}")
        print(f"SVA1: {sva1[:100]}..." if len(sva1) > 100 else f"SVA1: {sva1}")
        print(f"SVA2: {sva2[:100]}..." if len(sva2) > 100 else f"SVA2: {sva2}")
    
    # Check equivalence (bidirectional implication)
    result = checker.check_equivalence(sva1, sva2)
    
    # Get individual implication directions
    sva1_implies_sva2, sva2_implies_sva1 = checker.get_implication_relationship(sva1, sva2)
    
    return EquivalenceResult(
        id=pair_id,
        sva1=sva1,
        sva2=sva2,
        is_equivalent=(result.result == ImplicationResult.EQUIVALENT),
        result=result.result,
        message=result.message,
        sva1_implies_sva2=sva1_implies_sva2,
        sva2_implies_sva1=sva2_implies_sva1,
    )


def run_equivalence_checks(
    json_path: str,
    depth: int = 20,
    keep_files: bool = False,
    verbose: bool = False,
    limit: int = None,
    timeout: int = 300
) -> List[EquivalenceResult]:
    """
    Run equivalence checks for all SVA pairs in the JSON file.
    
    Args:
        json_path: Path to the JSON file containing SVA pairs.
        depth: Proof depth for bounded model checking.
        keep_files: Keep generated verification files.
        verbose: Print verbose output.
        limit: Maximum number of pairs to check (None for all).
        timeout: Timeout in seconds for each verification (default: 300).
        
    Returns:
        List of EquivalenceResult objects.
    """
    # Check if EBMC is available
    if not shutil.which("ebmc"):
        print("WARNING: EBMC not found in PATH. Results may be limited.")
    # Initialize checker
    checker = SVAImplicationChecker(
        depth=depth, keep_files=keep_files, verbose=verbose, timeout=timeout
    )
    
    # Load SVA pairs
    pairs = load_sva_pairs(json_path)
    
    if limit is not None:
        pairs = pairs[:limit]
    
    print(f"Checking equivalence for {len(pairs)} SVA pairs...")
    
    results = []
    for i, pair in enumerate(pairs):
        if verbose:
            print(f"\nProgress: {i+1}/{len(pairs)}")
        
        result = check_sva_pair_equivalence(checker, pair, verbose)
        results.append(result)
        
        # Print brief status
        if result.result == ImplicationResult.TIMEOUT:
            status_icon = "⏱"
        elif result.is_equivalent:
            status_icon = "✓"
        else:
            status_icon = "✗"
        print(f"[{status_icon}] {result.id}: {result.result.value}")
    
    return results


def summarize_results(results: List[EquivalenceResult]) -> Dict[str, Any]:
    """
    Generate a summary of equivalence check results.
    
    Args:
        results: List of EquivalenceResult objects.
        
    Returns:
        Summary dictionary with statistics.
    """
    total = len(results)
    equivalent = sum(1 for r in results if r.result == ImplicationResult.EQUIVALENT)
    not_implies = sum(1 for r in results if r.result == ImplicationResult.NOT_IMPLIES)
    errors = sum(1 for r in results if r.result == ImplicationResult.ERROR)
    timeouts = sum(1 for r in results if r.result == ImplicationResult.TIMEOUT)
    # Count one-way implications
    sva1_implies_only = sum(
        1 for r in results 
        if r.sva1_implies_sva2 and not r.sva2_implies_sva1
    )
    sva2_implies_only = sum(
        1 for r in results 
        if r.sva2_implies_sva1 and not r.sva1_implies_sva2
    )
    return {
        "total": total,
        "equivalent": equivalent,
        "not_equivalent": not_implies,
        "errors": errors,
        "timeouts": timeouts,
        "sva1_implies_sva2_only": sva1_implies_only,
        "sva2_implies_sva1_only": sva2_implies_only,
        "equivalent_percentage": (equivalent / total * 100) if total > 0 else 0,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    """Print formatted summary of results."""
    print("\n" + "=" * 60)
    print("EQUIVALENCE CHECK SUMMARY")
    print("=" * 60)
    print(f"Total pairs checked:        {summary['total']}")
    print(f"Equivalent:                 {summary['equivalent']} ({summary['equivalent_percentage']:.1f}%)")
    print(f"Not equivalent:             {summary['not_equivalent']}")
    print(f"Errors:                     {summary['errors']}")
    print(f"Timeouts:                   {summary['timeouts']}")
    print(f"SVA1 implies SVA2 only:     {summary['sva1_implies_sva2_only']}")
    print(f"SVA2 implies SVA1 only:     {summary['sva2_implies_sva1_only']}")
    print("=" * 60)


def save_results_to_json(
    results: List[EquivalenceResult],
    output_path: str
) -> None:
    """
    Save results to a JSON file.
    
    Args:
        results: List of EquivalenceResult objects.
        output_path: Path to save the JSON output.
    """
    output_data = []
    for r in results:
        output_data.append({
            "id": r.id,
            "is_equivalent": r.is_equivalent,
            "result": r.result.value,
            "message": r.message,
            "sva1_implies_sva2": r.sva1_implies_sva2,
            "sva2_implies_sva1": r.sva2_implies_sva1,
            "sva1": r.sva1,
            "sva2": r.sva2,
        })
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point for the equivalence check script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check equivalence between SVA pairs"
    )
    parser.add_argument(
        "--input", "-i",
        default=str(Path(__file__).parent / "syntax_check_2.json"),
        help="Path to input JSON file with SVA pairs"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(Path(__file__).parent / "equivalence_results.json"),
        help="Path to output JSON file for results"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=20,
        help="Proof depth for bounded model checking (default: 20)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Maximum number of pairs to check (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--keep-files", "-k",
        action="store_true",
        help="Keep generated verification files"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=10,
        help="Timeout in seconds for each verification (default: 300)"
    )
    args = parser.parse_args()
    
    # Run equivalence checks
    results = run_equivalence_checks(
        json_path=args.input,
        depth=args.depth,
        keep_files=args.keep_files,
        verbose=args.verbose,
        limit=args.limit,
        timeout=args.timeout,
    )
    
    # Generate and print summary
    summary = summarize_results(results)
    print_summary(summary)
    
    # Save results to JSON
    save_results_to_json(results, args.output)
    
    return results


if __name__ == "__main__":
    main()
