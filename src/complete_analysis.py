"""
Complete analysis and visualization script for memory gap detection.

Analyzes existing LoCoMo results and runs streamlined SelfAware experiments.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append(str(Path(__file__).parent))
from memory_gap_detector import MemoryGapDetector, set_seed


def run_selfaware_small(max_samples=30):
    """Run small-scale SelfAware experiments."""
    print("\n" + "="*80)
    print("Running SelfAware Experiments (Small Scale)")
    print("="*80)

    # Load data
    with open('datasets/SelfAware/data/SelfAware.json', 'r') as f:
        data_dict = json.load(f)

    data = data_dict.get('example', [])

    # Get balanced sample
    answerable = [q for q in data if q.get('answerable', True)]
    unanswerable = [q for q in data if not q.get('answerable', True)]

    print(f"Total: {len(answerable)} answerable, {len(unanswerable)} unanswerable")

    # Sample
    n_each = max_samples // 2
    sampled_answerable = np.random.choice(answerable, min(n_each, len(answerable)), replace=False)
    sampled_unanswerable = np.random.choice(unanswerable, min(n_each, len(unanswerable)), replace=False)

    all_questions = list(sampled_answerable) + list(sampled_unanswerable)
    np.random.shuffle(all_questions)

    # Initialize detector
    detector = MemoryGapDetector()

    results = []
    for question in tqdm(all_questions, desc="Processing questions"):
        query = question['question']
        is_answerable = question.get('answerable', True)

        try:
            result = detector.detect_gap([], query)

            results.append({
                'question': query,
                'answerable': is_answerable,
                'divergence_score': result['divergence_score'],
                'gap_detected': result['gap_detected'],
                'recommended_action': result['recommended_action']
            })
        except Exception as e:
            print(f"Error: {e}")
            continue

    # Save results
    with open('results/selfaware_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results")

    return results


def analyze_results():
    """Analyze both LoCoMo and SelfAware results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS")
    print("="*80)

    # Load results
    with open('results/locomo_results.json', 'r') as f:
        locomo_results = json.load(f)

    with open('results/selfaware_results.json', 'r') as f:
        selfaware_results = json.load(f)

    print(f"\nLoCoMo: {len(locomo_results)} queries")
    print(f"SelfAware: {len(selfaware_results)} questions")

    # LoCoMo analysis
    df_locomo = pd.DataFrame(locomo_results)
    print("\n" + "-"*80)
    print("LoCoMo Analysis")
    print("-"*80)
    print(f"Mean divergence: {df_locomo['divergence_score'].mean():.3f} ± {df_locomo['divergence_score'].std():.3f}")
    print(f"Gap detection rate: {df_locomo['gap_detected'].mean()*100:.1f}%")
    print("\nActions recommended:")
    print(df_locomo['recommended_action'].value_counts())

    # SelfAware analysis
    df_selfaware = pd.DataFrame(selfaware_results)
    print("\n" + "-"*80)
    print("SelfAware Analysis")
    print("-"*80)

    answerable_div = df_selfaware[df_selfaware['answerable']]['divergence_score']
    unanswerable_div = df_selfaware[~df_selfaware['answerable']]['divergence_score']

    print(f"\nAnswerable questions:")
    print(f"  Mean divergence: {answerable_div.mean():.3f} ± {answerable_div.std():.3f}")
    print(f"  Gap detection: {df_selfaware[df_selfaware['answerable']]['gap_detected'].mean()*100:.1f}%")

    print(f"\nUnanswerable questions:")
    print(f"  Mean divergence: {unanswerable_div.mean():.3f} ± {unanswerable_div.std():.3f}")
    print(f"  Gap detection: {df_selfaware[~df_selfaware['answerable']]['gap_detected'].mean()*100:.1f}%")

    # AUC
    y_true = (~df_selfaware['answerable']).astype(int)
    y_scores = df_selfaware['divergence_score'].values

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_scores)
        print(f"\nAUC (unanswerable detection): {auc:.3f}")

    return df_locomo, df_selfaware


def create_visualizations(df_locomo, df_selfaware):
    """Create visualizations."""
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)

    sns.set_style("whitegrid")

    # Figure 1: LoCoMo Analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Histogram of divergence scores
    axes[0].hist(df_locomo['divergence_score'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(0.2, color='orange', linestyle='--', label='Medium Threshold', linewidth=2)
    axes[0].axvline(0.4, color='red', linestyle='--', label='High Threshold', linewidth=2)
    axes[0].set_xlabel('Divergence Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('LoCoMo: Divergence Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Action distribution
    action_counts = df_locomo['recommended_action'].value_counts()
    axes[1].bar(range(len(action_counts)), action_counts.values, edgecolor='black')
    axes[1].set_xticks(range(len(action_counts)))
    axes[1].set_xticklabels([a.replace('_', '\n') for a in action_counts.index], fontsize=10)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('LoCoMo: Recommended Actions', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Box plot by gap detected
    df_locomo['gap_status'] = df_locomo['gap_detected'].map({True: 'Gap Detected', False: 'No Gap'})
    df_locomo.boxplot(column='divergence_score', by='gap_status', ax=axes[2])
    axes[2].set_xlabel('Status', fontsize=12)
    axes[2].set_ylabel('Divergence Score', fontsize=12)
    axes[2].set_title('LoCoMo: Divergence by Detection Status', fontsize=14, fontweight='bold')
    axes[2].get_figure().suptitle('')

    plt.tight_layout()
    plt.savefig('figures/locomo_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/locomo_analysis.png")
    plt.close()

    # Figure 2: SelfAware Analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Box plot
    df_selfaware['category'] = df_selfaware['answerable'].map({True: 'Answerable', False: 'Unanswerable'})
    df_selfaware.boxplot(column='divergence_score', by='category', ax=axes[0])
    axes[0].set_xlabel('Question Type', fontsize=12)
    axes[0].set_ylabel('Divergence Score', fontsize=12)
    axes[0].set_title('SelfAware: Divergence by Type', fontsize=14, fontweight='bold')
    axes[0].get_figure().suptitle('')

    # Histogram comparison
    answerable_div = df_selfaware[df_selfaware['answerable']]['divergence_score']
    unanswerable_div = df_selfaware[~df_selfaware['answerable']]['divergence_score']
    axes[1].hist(answerable_div, bins=15, alpha=0.6, label='Answerable', edgecolor='black', color='green')
    axes[1].hist(unanswerable_div, bins=15, alpha=0.6, label='Unanswerable', edgecolor='black', color='red')
    axes[1].set_xlabel('Divergence Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('SelfAware: Divergence Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # ROC curve
    y_true = (~df_selfaware['answerable']).astype(int)
    y_scores = df_selfaware['divergence_score'].values

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        axes[2].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auc:.3f})')
        axes[2].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
        axes[2].set_xlabel('False Positive Rate', fontsize=12)
        axes[2].set_ylabel('True Positive Rate', fontsize=12)
        axes[2].set_title('SelfAware: ROC Curve', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/selfaware_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/selfaware_analysis.png")
    plt.close()


def save_summary_stats(df_locomo, df_selfaware):
    """Save summary statistics."""
    print("\nSaving summary statistics...")

    answerable_div = df_selfaware[df_selfaware['answerable']]['divergence_score']
    unanswerable_div = df_selfaware[~df_selfaware['answerable']]['divergence_score']

    y_true = (~df_selfaware['answerable']).astype(int)
    y_scores = df_selfaware['divergence_score'].values
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0

    stats = {
        'locomo': {
            'n_samples': len(df_locomo),
            'mean_divergence': float(df_locomo['divergence_score'].mean()),
            'std_divergence': float(df_locomo['divergence_score'].std()),
            'min_divergence': float(df_locomo['divergence_score'].min()),
            'max_divergence': float(df_locomo['divergence_score'].max()),
            'gap_detection_rate': float(df_locomo['gap_detected'].mean()),
            'action_distribution': df_locomo['recommended_action'].value_counts().to_dict()
        },
        'selfaware': {
            'n_samples': len(df_selfaware),
            'n_answerable': int(df_selfaware['answerable'].sum()),
            'n_unanswerable': int((~df_selfaware['answerable']).sum()),
            'mean_divergence_answerable': float(answerable_div.mean()),
            'std_divergence_answerable': float(answerable_div.std()),
            'mean_divergence_unanswerable': float(unanswerable_div.mean()),
            'std_divergence_unanswerable': float(unanswerable_div.std()),
            'auc_unanswerable_detection': float(auc),
            'gap_detection_rate_answerable': float(df_selfaware[df_selfaware['answerable']]['gap_detected'].mean()),
            'gap_detection_rate_unanswerable': float(df_selfaware[~df_selfaware['answerable']]['gap_detected'].mean())
        }
    }

    with open('results/summary_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("Saved: results/summary_statistics.json")
    print("\nSummary Statistics:")
    print(json.dumps(stats, indent=2))

    return stats


def main():
    """Main pipeline."""
    set_seed(42)

    # Run SelfAware experiments (small scale)
    selfaware_results = run_selfaware_small(max_samples=30)

    # Analyze results
    df_locomo, df_selfaware = analyze_results()

    # Create visualizations
    create_visualizations(df_locomo, df_selfaware)

    # Save summary stats
    stats = save_summary_stats(df_locomo, df_selfaware)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
