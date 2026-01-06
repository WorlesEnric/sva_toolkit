#!/usr/bin/env python3
"""
Visualization script for benchmark model results.

This script creates visualizations for:
1. Comparison of rates across all models
2. Individual model detailed breakdowns
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_results(json_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_comparison_summary(comparison: Dict[str, Any], output_path: str = None) -> None:
    """Plot comparison summary showing rates across all models."""
    models = comparison['models']
    equivalent_rates = comparison['equivalent_rates']
    any_implication_rates = comparison['any_implication_rates']
    success_rates = comparison['success_rates']
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - width, equivalent_rates, width, label='Equivalent Rate', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, any_implication_rates, width, label='Any Implication Rate', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, success_rates, width, label='Success Rate', color='#9b59b6', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Performance Rates', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_individual_model(result: Dict[str, Any], output_dir: str = None) -> None:
    """Plot detailed breakdown for a single model."""
    model_name = result['model_name']
    total = result['total_items']
    
    # Prepare data for pie chart
    labels = [
        'Equivalent',
        'Generated → Reference',
        'Reference → Generated',
        'No Relationship',
        'Errors'
    ]
    sizes = [
        result['equivalent_count'],
        result['generated_implies_reference_count'],
        result['reference_implies_generated_count'],
        result['no_relationship_count'],
        result['error_count']
    ]
    colors = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c', '#95a5a6']
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Pie chart
    ax1 = plt.subplot(1, 2, 1)
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 10})
    ax1.set_title(f'{model_name}\nRelationship Breakdown', fontsize=13, fontweight='bold', pad=15)
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Bar chart with metrics
    ax2 = plt.subplot(1, 2, 2)
    metrics = ['Equivalent\nRate', 'Any Implication\nRate', 'Success\nRate']
    values = [
        result['equivalent_rate'],
        result['any_implication_rate'],
        result['success_rate']
    ]
    bars = ax2.bar(metrics, values, color=['#2ecc71', '#3498db', '#9b59b6'], alpha=0.8)
    ax2.set_ylabel('Rate', fontsize=11, fontweight='bold')
    ax2.set_title('Performance Metrics', fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylim([0, 1.1])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add text box with detailed counts
    textstr = f"""Total Items: {total}
Equivalent: {result['equivalent_count']}
Gen→Ref: {result['generated_implies_reference_count']}
Ref→Gen: {result['reference_implies_generated_count']}
No Relationship: {result['no_relationship_count']}
Errors: {result['error_count']}

Avg Generation Time: {result['avg_generation_time']:.2f}s
Avg Verification Time: {result['avg_verification_time']:.4f}s"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    if output_dir:
        filename = f"{model_name.replace('/', '_')}_breakdown.png"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Model breakdown saved to: {filepath}")
    else:
        plt.show()
    plt.close()


def plot_all_models_comparison(results: List[Dict[str, Any]], output_path: str = None) -> None:
    """Plot comparison of relationship breakdowns across all models."""
    models = [r['model_name'] for r in results]
    n_models = len(models)
    
    # Prepare data
    equivalent = [r['equivalent_count'] for r in results]
    gen_to_ref = [r['generated_implies_reference_count'] for r in results]
    ref_to_gen = [r['reference_implies_generated_count'] for r in results]
    no_rel = [r['no_relationship_count'] for r in results]
    errors = [r['error_count'] for r in results]
    
    x = np.arange(n_models)
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Stacked bar chart
    p1 = ax.bar(x, equivalent, width, label='Equivalent', color='#2ecc71', alpha=0.8)
    p2 = ax.bar(x, gen_to_ref, width, bottom=equivalent, label='Generated → Reference', color='#f39c12', alpha=0.8)
    p3 = ax.bar(x, ref_to_gen, width, bottom=np.array(equivalent) + np.array(gen_to_ref),
                label='Reference → Generated', color='#3498db', alpha=0.8)
    p4 = ax.bar(x, no_rel, width,
                bottom=np.array(equivalent) + np.array(gen_to_ref) + np.array(ref_to_gen),
                label='No Relationship', color='#e74c3c', alpha=0.8)
    p5 = ax.bar(x, errors, width,
                bottom=np.array(equivalent) + np.array(gen_to_ref) + np.array(ref_to_gen) + np.array(no_rel),
                label='Errors', color='#95a5a6', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Relationship Breakdown Comparison Across Models', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison breakdown plot saved to: {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    """Main function to generate all visualizations."""
    result_json_path = "/wkspace/sva_toolkit/experiments/benchmark_models/result.json"
    output_dir = Path("/wkspace/sva_toolkit/experiments/benchmark_models/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    data = load_results(result_json_path)
    comparison = data['comparison']
    results = data['results']
    
    print("Generating visualizations...")
    
    # 1. Comparison summary
    print("\n1. Creating comparison summary plot...")
    plot_comparison_summary(comparison, str(output_dir / "comparison_summary.png"))
    
    # 2. Individual model breakdowns
    print("\n2. Creating individual model breakdowns...")
    for result in results:
        plot_individual_model(result, str(output_dir))
    
    # 3. All models comparison breakdown
    print("\n3. Creating relationship breakdown comparison...")
    plot_all_models_comparison(results, str(output_dir / "all_models_breakdown.png"))
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()





