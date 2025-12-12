"""
Experimental evaluation script for memory gap detection.

This script runs comprehensive experiments on LoCoMo and SelfAware datasets
to evaluate the hypothesis that policy simulation can detect memory gaps.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from memory_gap_detector import (
    MemoryGapDetector,
    ResponseGenerator,
    DivergenceMeasurer,
    FullContextPolicy,
    RecentOnlyPolicy,
    SemanticRetrievalPolicy,
    NoMemoryPolicy,
    set_seed
)


class ExperimentRunner:
    """Runs experiments and collects results."""

    def __init__(self, output_dir: str = "results"):
        """Initialize experiment runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize detector
        self.detector = MemoryGapDetector()

        # Results storage
        self.results = []

    def run_locomo_experiments(self, data_path: str, max_samples: int = 50):
        """
        Run experiments on LoCoMo dataset.

        Tests if divergence correlates with:
        1. Distance to relevant information
        2. Answer correctness
        """
        print("\n" + "="*80)
        print("EXPERIMENT 1: LoCoMo Long-Term Memory Evaluation")
        print("="*80)

        # Load data
        print(f"\nLoading LoCoMo data from {data_path}...")
        with open(data_path, 'r') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} conversations")

        # Process conversations
        results = []

        for conv_idx, conversation in enumerate(tqdm(data[:max_samples], desc="Processing conversations")):
            # Extract conversation history from LoCoMo format
            history = []

            # LoCoMo has conversation dict with session_1, session_2, etc.
            conv_data = conversation.get('conversation', {})

            # Extract sessions in order
            for session_num in range(1, 36):
                session_key = f'session_{session_num}'
                if session_key in conv_data and conv_data[session_key]:
                    for turn in conv_data[session_key]:
                        history.append({
                            'speaker': turn.get('speaker', 'unknown'),
                            'text': turn.get('text', '')
                        })

            # Use QA pairs as test queries
            qa_pairs = conversation.get('qa', [])

            for qa in qa_pairs[:3]:  # Use first 3 QA pairs per conversation
                query = qa['question']
                ground_truth_answer = qa['answer']

                if len(history) < 10:
                    continue

                # Detect gap
                try:
                    result = self.detector.detect_gap(history, query)

                    results.append({
                        'conversation_id': conv_idx,
                        'query': query,
                        'ground_truth_answer': str(ground_truth_answer),
                        'history_length': len(history),
                        'divergence_score': result['divergence_score'],
                        'gap_detected': result['gap_detected'],
                        'recommended_action': result['recommended_action'],
                        'responses': result['responses']
                    })
                except Exception as e:
                    print(f"\nError processing query '{query}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Save results
        results_file = self.output_dir / "locomo_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved {len(results)} results to {results_file}")

        # Analyze results
        self._analyze_locomo_results(results)

        return results

    def _analyze_locomo_results(self, results: List[Dict]):
        """Analyze LoCoMo results."""
        print("\n" + "-"*80)
        print("ANALYSIS: LoCoMo Results")
        print("-"*80)

        if len(results) == 0:
            print("No results to analyze.")
            return

        df = pd.DataFrame(results)

        # Summary statistics
        print(f"\nTotal queries processed: {len(df)}")
        print(f"Average divergence score: {df['divergence_score'].mean():.3f} ± {df['divergence_score'].std():.3f}")
        print(f"Gap detection rate: {df['gap_detected'].mean()*100:.1f}%")

        # Divergence distribution
        print("\nDivergence score distribution:")
        print(df['divergence_score'].describe())

        # Recommended actions
        print("\nRecommended actions:")
        print(df['recommended_action'].value_counts())

    def run_selfaware_experiments(self, data_path: str, max_samples: int = 100):
        """
        Run experiments on SelfAware dataset.

        Tests if divergence distinguishes answerable from unanswerable questions.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 2: SelfAware Self-Knowledge Evaluation")
        print("="*80)

        # Load data
        print(f"\nLoading SelfAware data from {data_path}...")
        with open(data_path, 'r') as f:
            data_dict = json.load(f)

        # Extract questions from 'example' key
        data = data_dict.get('example', [])

        # Sample balanced data
        answerable = [q for q in data if q.get('answerable', True)]
        unanswerable = [q for q in data if not q.get('answerable', True)]

        print(f"Loaded {len(answerable)} answerable and {len(unanswerable)} unanswerable questions")

        # Sample equally
        n_samples = min(max_samples // 2, len(answerable), len(unanswerable))
        sampled_answerable = np.random.choice(answerable, n_samples, replace=False)
        sampled_unanswerable = np.random.choice(unanswerable, n_samples, replace=False)

        all_questions = list(sampled_answerable) + list(sampled_unanswerable)
        np.random.shuffle(all_questions)

        results = []

        for question in tqdm(all_questions, desc="Processing questions"):
            query = question['question']
            is_answerable = question.get('answerable', True)

            # No conversation history for SelfAware
            try:
                result = self.detector.detect_gap([], query)

                results.append({
                    'question': query,
                    'answerable': is_answerable,
                    'divergence_score': result['divergence_score'],
                    'gap_detected': result['gap_detected'],
                    'recommended_action': result['recommended_action'],
                    'responses': result['responses']
                })
            except Exception as e:
                print(f"\nError processing question: {e}")
                continue

        # Save results
        results_file = self.output_dir / "selfaware_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved {len(results)} results to {results_file}")

        # Analyze results
        self._analyze_selfaware_results(results)

        return results

    def _analyze_selfaware_results(self, results: List[Dict]):
        """Analyze SelfAware results."""
        print("\n" + "-"*80)
        print("ANALYSIS: Answerable vs. Unanswerable Questions")
        print("-"*80)

        df = pd.DataFrame(results)

        if len(df) > 0:
            # Divergence by answerability
            print("\nDivergence scores:")
            print(df.groupby('answerable')['divergence_score'].agg(['mean', 'std', 'count']))

            # Gap detection by answerability
            print("\nGap detection rates:")
            print(df.groupby('answerable')['gap_detected'].agg(['mean', 'count']))

            # Classification performance
            # Treat unanswerable as positive class (should have high divergence)
            y_true = (~df['answerable']).astype(int)  # 1 if unanswerable
            y_scores = df['divergence_score'].values

            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_scores)
                print(f"\nAUC (divergence for detecting unanswerable): {auc:.3f}")

    def create_visualizations(self):
        """Create visualization plots from results."""
        print("\n" + "="*80)
        print("Creating Visualizations")
        print("="*80)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # Load results
        locomo_file = self.output_dir / "locomo_results.json"
        selfaware_file = self.output_dir / "selfaware_results.json"

        # LoCoMo visualizations
        if locomo_file.exists():
            print("\nCreating LoCoMo visualizations...")
            with open(locomo_file, 'r') as f:
                locomo_results = json.load(f)

            df_locomo = pd.DataFrame(locomo_results)

            # Plot 1: Divergence vs. Distance
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Scatter plot
            axes[0, 0].scatter(df_locomo['distance_from_end'], df_locomo['divergence_score'], alpha=0.5)
            axes[0, 0].set_xlabel('Distance from End (turns)')
            axes[0, 0].set_ylabel('Divergence Score')
            axes[0, 0].set_title('Divergence vs. Distance to Information')

            # Box plot by category
            df_locomo['distance_category'] = pd.cut(
                df_locomo['distance_from_end'],
                bins=[0, 10, 30, 100, 1000],
                labels=['Very Recent\n(0-10)', 'Recent\n(10-30)', 'Distant\n(30-100)', 'Very Distant\n(100+)']
            )
            df_locomo.boxplot(column='divergence_score', by='distance_category', ax=axes[0, 1])
            axes[0, 1].set_xlabel('Distance Category')
            axes[0, 1].set_ylabel('Divergence Score')
            axes[0, 1].set_title('Divergence Distribution by Distance')
            plt.sca(axes[0, 1])
            plt.xticks(rotation=0, ha='center')

            # Gap detection rates
            gap_rates = df_locomo.groupby('distance_category')['gap_detected'].mean()
            axes[1, 0].bar(range(len(gap_rates)), gap_rates.values)
            axes[1, 0].set_xticks(range(len(gap_rates)))
            axes[1, 0].set_xticklabels(gap_rates.index, rotation=0, ha='center')
            axes[1, 0].set_ylabel('Gap Detection Rate')
            axes[1, 0].set_title('Gap Detection Rate by Distance')
            axes[1, 0].set_ylim([0, 1])

            # Histogram of divergence scores
            axes[1, 1].hist(df_locomo['divergence_score'], bins=30, edgecolor='black', alpha=0.7)
            axes[1, 1].set_xlabel('Divergence Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Divergence Scores')
            axes[1, 1].axvline(0.2, color='orange', linestyle='--', label='Medium Threshold')
            axes[1, 1].axvline(0.4, color='red', linestyle='--', label='High Threshold')
            axes[1, 1].legend()

            plt.suptitle('LoCoMo Experimental Results', fontsize=16, y=1.00)
            plt.tight_layout()
            plt.savefig(self.output_dir.parent / 'figures' / 'locomo_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Saved: figures/locomo_analysis.png")
            plt.close()

        # SelfAware visualizations
        if selfaware_file.exists():
            print("Creating SelfAware visualizations...")
            with open(selfaware_file, 'r') as f:
                selfaware_results = json.load(f)

            df_selfaware = pd.DataFrame(selfaware_results)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Box plot: answerable vs unanswerable
            df_selfaware['category'] = df_selfaware['answerable'].map({True: 'Answerable', False: 'Unanswerable'})
            df_selfaware.boxplot(column='divergence_score', by='category', ax=axes[0])
            axes[0].set_xlabel('Question Type')
            axes[0].set_ylabel('Divergence Score')
            axes[0].set_title('Divergence: Answerable vs. Unanswerable')

            # Histogram comparison
            answerable_div = df_selfaware[df_selfaware['answerable']]['divergence_score']
            unanswerable_div = df_selfaware[~df_selfaware['answerable']]['divergence_score']
            axes[1].hist(answerable_div, bins=20, alpha=0.5, label='Answerable', edgecolor='black')
            axes[1].hist(unanswerable_div, bins=20, alpha=0.5, label='Unanswerable', edgecolor='black')
            axes[1].set_xlabel('Divergence Score')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Divergence Distribution Comparison')
            axes[1].legend()

            # ROC curve
            y_true = (~df_selfaware['answerable']).astype(int)
            y_scores = df_selfaware['divergence_score'].values

            if len(np.unique(y_true)) > 1:
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                auc = roc_auc_score(y_true, y_scores)
                axes[2].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
                axes[2].plot([0, 1], [0, 1], 'k--', label='Random')
                axes[2].set_xlabel('False Positive Rate')
                axes[2].set_ylabel('True Positive Rate')
                axes[2].set_title('ROC: Detecting Unanswerable Questions')
                axes[2].legend()
                axes[2].grid(True)

            plt.suptitle('SelfAware Experimental Results', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir.parent / 'figures' / 'selfaware_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Saved: figures/selfaware_analysis.png")
            plt.close()

    def compute_summary_statistics(self):
        """Compute and save summary statistics."""
        print("\n" + "="*80)
        print("Computing Summary Statistics")
        print("="*80)

        stats = {}

        # LoCoMo stats
        locomo_file = self.output_dir / "locomo_results.json"
        if locomo_file.exists():
            with open(locomo_file, 'r') as f:
                locomo_results = json.load(f)

            df_locomo = pd.DataFrame(locomo_results)

            stats['locomo'] = {
                'n_samples': len(df_locomo),
                'mean_divergence': float(df_locomo['divergence_score'].mean()),
                'std_divergence': float(df_locomo['divergence_score'].std()),
                'gap_detection_rate': float(df_locomo['gap_detected'].mean()),
                'distance_divergence_correlation': float(df_locomo['divergence_score'].corr(df_locomo['distance_from_end']))
            }

        # SelfAware stats
        selfaware_file = self.output_dir / "selfaware_results.json"
        if selfaware_file.exists():
            with open(selfaware_file, 'r') as f:
                selfaware_results = json.load(f)

            df_selfaware = pd.DataFrame(selfaware_results)

            answerable_div = df_selfaware[df_selfaware['answerable']]['divergence_score']
            unanswerable_div = df_selfaware[~df_selfaware['answerable']]['divergence_score']

            y_true = (~df_selfaware['answerable']).astype(int)
            y_scores = df_selfaware['divergence_score'].values

            auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0

            stats['selfaware'] = {
                'n_samples': len(df_selfaware),
                'mean_divergence_answerable': float(answerable_div.mean()),
                'mean_divergence_unanswerable': float(unanswerable_div.mean()),
                'std_divergence_answerable': float(answerable_div.std()),
                'std_divergence_unanswerable': float(unanswerable_div.std()),
                'auc_unanswerable_detection': float(auc)
            }

        # Save stats
        stats_file = self.output_dir / "summary_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nSaved summary statistics to {stats_file}")
        print("\nSummary Statistics:")
        print(json.dumps(stats, indent=2))

        return stats


def main():
    """Main experimental pipeline."""
    print("\n" + "="*80)
    print("MEMORY GAP DETECTION: EXPERIMENTAL EVALUATION")
    print("="*80)

    # Set seed for reproducibility
    set_seed(42)

    # Initialize runner
    runner = ExperimentRunner(output_dir="results")

    # Paths to datasets
    locomo_path = "datasets/locomo/data/locomo10.json"
    selfaware_path = "datasets/SelfAware/data/SelfAware.json"

    # Run experiments
    print("\n>>> Starting LoCoMo experiments...")
    if os.path.exists(locomo_path):
        locomo_results = runner.run_locomo_experiments(locomo_path, max_samples=10)
    else:
        print(f"WARNING: LoCoMo data not found at {locomo_path}")

    print("\n>>> Starting SelfAware experiments...")
    if os.path.exists(selfaware_path):
        selfaware_results = runner.run_selfaware_experiments(selfaware_path, max_samples=100)
    else:
        print(f"WARNING: SelfAware data not found at {selfaware_path}")

    # Create visualizations
    print("\n>>> Creating visualizations...")
    runner.create_visualizations()

    # Compute summary statistics
    print("\n>>> Computing summary statistics...")
    stats = runner.compute_summary_statistics()

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: results/")
    print(f"Figures saved to: figures/")


if __name__ == "__main__":
    main()
