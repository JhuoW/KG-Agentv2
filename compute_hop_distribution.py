"""
Compute hop distribution statistics for KGQA datasets.

This script analyzes the ground-truth reasoning paths to determine
the distribution of reasoning hops (1-hop, 2-hop, ≥3-hop) for each dataset.

Usage:
    # Option 1: Load from HuggingFace (requires internet)
    python compute_hop_distribution.py
    
    # Option 2: Use preprocessed local data (if available)
    python compute_hop_distribution.py --local_data_path data/shortest_gt_paths
    
    # Option 3: Use pre-computed ground_truth_paths field
    python compute_hop_distribution.py --use_gt_paths
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
from functools import partial

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

try:
    from datasets import load_dataset, load_from_disk
except ImportError:
    print("Please install datasets: pip install datasets")
    sys.exit(1)

try:
    import networkx as nx
except ImportError:
    print("Please install networkx: pip install networkx")
    sys.exit(1)


def build_graph(triples, undirected=False):
    """Build a NetworkX graph from triples."""
    if undirected:
        g = nx.Graph()
    else:
        g = nx.DiGraph()
    
    for triple in triples:
        if len(triple) >= 3:
            h, r, t = triple[0], triple[1], triple[2]
            g.add_edge(h, t, relation=r)
    
    return g


def get_shortest_paths(graph, start_nodes, end_nodes):
    """
    Get all shortest paths from start_nodes to end_nodes.
    Returns list of paths, where each path is a list of (head, relation, tail) tuples.
    """
    paths = []
    for start in start_nodes:
        if start not in graph:
            continue
        for end in end_nodes:
            if end not in graph:
                continue
            if start == end:
                continue
            try:
                for path in nx.all_shortest_paths(graph, start, end):
                    # Convert node path to edge path with relations
                    edge_path = []
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        rel = graph[u][v].get('relation', 'unknown')
                        edge_path.append((u, rel, v))
                    if edge_path:
                        paths.append(edge_path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
    return paths


def process_sample(sample, undirected=False):
    """Process a single sample to get the minimum hop count."""
    graph_triples = sample.get('graph', [])
    q_entities = sample.get('q_entity', [])
    a_entities = sample.get('a_entity', [])
    
    if not graph_triples or not q_entities or not a_entities:
        return None
    
    # Build graph
    graph = build_graph(graph_triples, undirected=undirected)
    
    # Get shortest paths
    paths = get_shortest_paths(graph, q_entities, a_entities)
    
    if not paths:
        return None
    
    # Get minimum hop count (length of shortest path)
    min_hops = min(len(p) for p in paths)
    return min_hops


def compute_hop_distribution(dataset_name, split='test', num_workers=8, undirected=False, 
                              local_path=None, use_gt_paths=False):
    """
    Compute hop distribution for a dataset.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'rmanluo/RoG-webqsp')
        split: Dataset split to analyze
        num_workers: Number of parallel workers
        undirected: Whether to treat graph as undirected
        local_path: Path to local preprocessed data (optional)
        use_gt_paths: If True, use pre-computed 'ground_truth_paths' field
    
    Returns:
        Dictionary with hop distribution statistics
    """
    # Load dataset
    if local_path and os.path.exists(local_path):
        # Load from local preprocessed data
        dataset_local_path = os.path.join(local_path, dataset_name.split('/')[-1], split)
        if os.path.exists(dataset_local_path):
            print(f"\nLoading from local path: {dataset_local_path}")
            dataset = load_from_disk(dataset_local_path)
        else:
            print(f"Local path not found: {dataset_local_path}")
            print(f"Falling back to HuggingFace: {dataset_name}")
            dataset = load_dataset(dataset_name, split=split)
    else:
        print(f"\nLoading {dataset_name} ({split} split) from HuggingFace...")
        dataset = load_dataset(dataset_name, split=split)
    
    print(f"Processing {len(dataset)} samples...")
    
    # Check if ground_truth_paths field exists
    sample = dataset[0]
    has_gt_paths = 'ground_truth_paths' in sample and use_gt_paths
    
    if has_gt_paths:
        print("Using pre-computed 'ground_truth_paths' field...")
        hop_counts = []
        for sample in dataset:
            gt_paths = sample.get('ground_truth_paths', [])
            if gt_paths:
                # Each path is a list of triples or a string representation
                # Determine the minimum hop count
                min_hops = float('inf')
                for path in gt_paths:
                    if isinstance(path, list):
                        hops = len(path)
                    elif isinstance(path, str):
                        # Count arrows: "A -> r1 -> B -> r2 -> C" has 2 hops
                        hops = path.count(' -> ') // 2
                        if hops == 0:
                            hops = 1  # At least 1 hop if path exists
                    else:
                        continue
                    min_hops = min(min_hops, hops)
                if min_hops < float('inf'):
                    hop_counts.append(min_hops)
    else:
        # Process samples in parallel using graph analysis
        print(f"Computing paths from graph with {num_workers} workers...")
        process_fn = partial(process_sample, undirected=undirected)
        
        with Pool(processes=num_workers) as pool:
            from tqdm import tqdm
            results = list(tqdm(
                pool.imap(process_fn, dataset),
                total=len(dataset),
                desc=f"Analyzing {dataset_name.split('/')[-1]}"
            ))
        
        # Filter out None results
        hop_counts = [r for r in results if r is not None]
    
    # Compute distribution
    total = len(hop_counts)
    hop_dist = defaultdict(int)
    for hops in hop_counts:
        hop_dist[hops] += 1
    
    # Calculate percentages
    one_hop = hop_dist.get(1, 0)
    two_hop = hop_dist.get(2, 0)
    three_plus_hop = sum(count for hops, count in hop_dist.items() if hops >= 3)
    
    stats = {
        'dataset': dataset_name,
        'split': split,
        'total_samples': len(dataset),
        'samples_with_paths': total,
        'hop_distribution': dict(hop_dist),
        '1-hop_count': one_hop,
        '2-hop_count': two_hop,
        '>=3-hop_count': three_plus_hop,
        '1-hop_pct': (one_hop / total * 100) if total > 0 else 0,
        '2-hop_pct': (two_hop / total * 100) if total > 0 else 0,
        '>=3-hop_pct': (three_plus_hop / total * 100) if total > 0 else 0,
    }
    
    return stats


def print_stats(stats):
    """Print statistics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Dataset: {stats['dataset']} ({stats['split']})")
    print(f"{'='*60}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples with valid paths: {stats['samples_with_paths']}")
    print(f"\nHop Distribution (raw counts):")
    for hops in sorted(stats['hop_distribution'].keys()):
        count = stats['hop_distribution'][hops]
        pct = count / stats['samples_with_paths'] * 100
        print(f"  {hops}-hop: {count} ({pct:.1f}%)")
    
    print(f"\nFor LaTeX table:")
    print(f"  1-hop: {stats['1-hop_pct']:.1f}%")
    print(f"  2-hop: {stats['2-hop_pct']:.1f}%")
    print(f"  >=3-hop: {stats['>=3-hop_pct']:.1f}%" if stats['>=3-hop_count'] > 0 else "  >=3-hop: --")


def print_latex_table_row(name, train_stats, test_stats, num_paths):
    """Print a LaTeX table row for the dataset."""
    # Use test split for hop distribution (as it reflects evaluation complexity)
    stats = test_stats
    
    three_hop_str = f"{stats['>=3-hop_pct']:.1f}\\%" if stats['>=3-hop_count'] > 0 else "--"
    
    print(f"{name} & {train_stats['total_samples']:,} & {test_stats['total_samples']:,} & "
          f"{num_paths:,} & {stats['1-hop_pct']:.1f}\\% & {stats['2-hop_pct']:.1f}\\% & "
          f"{three_hop_str} \\\\")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute hop distribution statistics')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--undirected', action='store_true', help='Treat graph as undirected')
    parser.add_argument('--local_data_path', type=str, default=None, 
                        help='Path to local preprocessed data (e.g., data/shortest_gt_paths)')
    parser.add_argument('--use_gt_paths', action='store_true',
                        help='Use pre-computed ground_truth_paths field if available')
    args = parser.parse_args()
    
    datasets_info = [
        ('rmanluo/RoG-webqsp', 'WebQSP', 28307),   # #Path from GCR paper
        ('rmanluo/RoG-cwq', 'CWQ', 181602),         # #Path from GCR paper
    ]
    
    all_stats = {}
    
    for dataset_path, dataset_name, num_paths in datasets_info:
        print(f"\n{'#'*60}")
        print(f"# Processing {dataset_name}")
        print(f"{'#'*60}")
        
        # Compute stats for both train and test splits
        train_stats = compute_hop_distribution(
            dataset_path, 
            split='train', 
            num_workers=args.num_workers,
            undirected=args.undirected,
            local_path=args.local_data_path,
            use_gt_paths=args.use_gt_paths
        )
        test_stats = compute_hop_distribution(
            dataset_path, 
            split='test', 
            num_workers=args.num_workers,
            undirected=args.undirected,
            local_path=args.local_data_path,
            use_gt_paths=args.use_gt_paths
        )
        
        print_stats(train_stats)
        print_stats(test_stats)
        
        all_stats[dataset_name] = {
            'train': train_stats,
            'test': test_stats,
            'num_paths': num_paths
        }
    
    # Print LaTeX table format
    print(f"\n{'='*60}")
    print("LaTeX Table Rows (copy-paste ready):")
    print(f"{'='*60}")
    print("% Dataset & #Train & #Test & #Path & 1-hop & 2-hop & >=3-hop")
    for dataset_path, dataset_name, num_paths in datasets_info:
        stats = all_stats[dataset_name]
        print_latex_table_row(
            dataset_name, 
            stats['train'], 
            stats['test'], 
            stats['num_paths']
        )
    
    # Also print summary for easy reference
    print(f"\n{'='*60}")
    print("Summary (for filling the table):")
    print(f"{'='*60}")
    for dataset_path, dataset_name, num_paths in datasets_info:
        stats = all_stats[dataset_name]['test']
        print(f"\n{dataset_name}:")
        print(f"  #Train: {all_stats[dataset_name]['train']['total_samples']:,}")
        print(f"  #Test: {stats['total_samples']:,}")
        print(f"  #Path: {num_paths:,}")
        print(f"  1-hop: {stats['1-hop_pct']:.1f}%")
        print(f"  2-hop: {stats['2-hop_pct']:.1f}%")
        if stats['>=3-hop_count'] > 0:
            print(f"  >=3-hop: {stats['>=3-hop_pct']:.1f}%")
        else:
            print(f"  >=3-hop: -- (none in dataset)")


if __name__ == "__main__":
    main()