# -*- coding: utf-8 -*-
"""
Benchmark comparison: D* vs D* Lite
Compares the two path planning algorithms for performance
"""

import time
import statistics
import tracemalloc
# import pandas as pd
from d_star_algoirthm import DStarPathPlanner 
from d_star_lite import DStarLite
from typing import List, Tuple


def benchmark_algorithm(planner_class, grid, start, goal, iterations=10):
    """
    Benchmark an algorithm multiple times and return statistics

    Args:
        planner_class: Either DStarLite or DStarPathPlanner
        grid: Test grid
        start: Start position
        goal: Goal position
        iterations: Number of times to run

    Returns:
        Dictionary with timing statistics
    """
    times = []

    for i in range(iterations):
        planner = planner_class(grid, start, goal)
        planner = DStarLite(grid, start, goal)
        path = planner.plan()
        start_time = time.perf_counter()
        path = planner.plan()
        end_time = time.perf_counter()

        times.append(end_time - start_time)

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'all_times': times
    }


def benchmark_memory(planner_class, grid, start, goal):
    """Measure peak memory usage"""
    tracemalloc.start()

    planner = planner_class(grid, start, goal)
    path = planner.plan()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'current_mb': current / 1024 / 1024,
        'peak_mb': peak / 1024 / 1024
    }


def create_test_grid(size):
    """Create a test grid of given size"""
    return [[0 for _ in range(size)] for _ in range(size)]


def compare_single_test(grid_size=10, iterations=10):
    """Compare both algorithms on a single grid size"""

    print(f"\n{'='*70}")
    print(f"SINGLE TEST COMPARISON (Grid Size: {grid_size}x{grid_size})")
    print(f"Iterations: {iterations}")
    print(f"{'='*70}\n")

    grid = create_test_grid(grid_size)
    start = (0, grid_size - 1)
    goal = (grid_size - 1, 0)

    print("Running D* (Original Implementation)...")
    d_star_stats = benchmark_algorithm(
        DStarPathPlanner, grid, start, goal, iterations)

    print("Running D* Lite...")
    d_lite_stats = benchmark_algorithm(
        DStarLite, grid, start, goal, iterations)

    print("\nMeasuring memory usage...")
    d_star_mem = benchmark_memory(DStarPathPlanner, grid, start, goal)
    d_lite_mem = benchmark_memory(DStarLite, grid, start, goal)

    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")

    print(f"\nD* (Original):")
    print(f"  Mean Time:    {d_star_stats['mean']*1000:.4f} ms")
    print(f"  Median Time:  {d_star_stats['median']*1000:.4f} ms")
    print(f"  Std Dev:      {d_star_stats['std_dev']*1000:.4f} ms")
    print(f"  Min Time:     {d_star_stats['min']*1000:.4f} ms")
    print(f"  Max Time:     {d_star_stats['max']*1000:.4f} ms")
    print(f"  Peak Memory:  {d_star_mem['peak_mb']:.4f} MB")

    print(f"\nD* Lite:")
    print(f"  Mean Time:    {d_lite_stats['mean']*1000:.4f} ms")
    print(f"  Median Time:  {d_lite_stats['median']*1000:.4f} ms")
    print(f"  Std Dev:      {d_lite_stats['std_dev']*1000:.4f} ms")
    print(f"  Min Time:     {d_lite_stats['min']*1000:.4f} ms")
    print(f"  Max Time:     {d_lite_stats['max']*1000:.4f} ms")
    print(f"  Peak Memory:  {d_lite_mem['peak_mb']:.4f} MB")

    # Comparison
    speedup = d_star_stats['mean'] / d_lite_stats['mean']
    time_improvement = (d_star_stats['mean'] - d_lite_stats['mean']) / d_star_stats['mean'] * 100
    memory_saved = (d_star_mem['peak_mb'] - d_lite_mem['peak_mb']) / d_star_mem['peak_mb'] * 100

    print(f"\n{'='*70}")
    print("COMPARISON:")
    print(f"{'='*70}")
    print(f"  Speedup:           {speedup:.2f}x")
    print(f"  Time Improvement:  {time_improvement:.2f}%")
    print(f"  Time Saved:        {(d_star_stats['mean'] - d_lite_stats['mean'])*1000:.4f} ms")
    print(f"  Memory Saved:      {memory_saved:.2f}%")

    if speedup > 1:
        print(f"\n✓ D* Lite is FASTER")
    else:
        print(f"\n✗ D* (Original) is FASTER")
    print(f"{'='*70}\n")


def scalability_test(grid_sizes=[5, 10, 15, 20, 25, 30], iterations=5):
    """Test both algorithms across different grid sizes"""

    print(f"\n{'='*70}")
    print("SCALABILITY TEST")
    print(f"Grid Sizes: {grid_sizes}")
    print(f"Iterations per size: {iterations}")
    print(f"{'='*70}\n")

    results = []

    for size in grid_sizes:
        print(f"\nTesting grid size: {size}x{size}")
        grid = create_test_grid(size)
        start = (0, size - 1)
        goal = (size - 1, 0)

        print("  Running D*...")
        d_star_stats = benchmark_algorithm(
            DStarPathPlanner, grid, start, goal, iterations)

        print("  Running D* Lite...")
        d_lite_stats = benchmark_algorithm(
            DStarLite, grid, start, goal, iterations)

        print("  Measuring memory...")
        d_star_mem = benchmark_memory(DStarPathPlanner, grid, start, goal)
        d_lite_mem = benchmark_memory(DStarLite, grid, start, goal)

        speedup = d_star_stats['mean'] / d_lite_stats['mean']
        time_saved_pct = (
            (d_star_stats['mean'] - d_lite_stats['mean']) / d_star_stats['mean']) * 100
        mem_saved_pct = (
            (d_star_mem['peak_mb'] - d_lite_mem['peak_mb']) / d_star_mem['peak_mb']) * 100

        results.append({
            'Grid Size': f"{size}x{size}",
            'Grid Nodes': size * size,
            'D* Time (ms)': d_star_stats['mean'] * 1000,
            'D* Lite Time (ms)': d_lite_stats['mean'] * 1000,
            'D* Std Dev (ms)': d_star_stats['std_dev'] * 1000,
            'D* Lite Std Dev (ms)': d_lite_stats['std_dev'] * 1000,
            'D* Memory (MB)': d_star_mem['peak_mb'],
            'D* Lite Memory (MB)': d_lite_mem['peak_mb'],
            'Speedup': speedup,
            'Time Saved (%)': time_saved_pct,
            'Memory Saved (%)': mem_saved_pct
        })

        print(
            f"  ✓ Speedup: {speedup:.2f}x, Time saved: {time_saved_pct:.1f}%")

    # df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    print("\n" + "="*70)
    print("D* vs D* Lite Performance Comparison")
    print("="*70)

    print("\n[1/2] Running single comparison test...")
    compare_single_test(grid_size=15, iterations=10)

    print("\n[2/2] Running scalability test...")
    df = scalability_test(
       grid_sizes=[5, 10, 15, 20, 25, 30],
            iterations=5
    )

    print(f"\n{'='*70}")
    print("SUMMARY TABLE:")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))

    df.to_csv('benchmark_results.csv', index=False)
    print(f"\n✓ Results saved to 'benchmark_results.csv'")

    print(f"\n{'='*70}")
    print("OVERALL STATISTICS:")
    print(f"{'='*70}")
    print(f"Average Speedup:      {df['Speedup'].mean():.2f}x")
    print(f"Maximum Speedup:      {df['Speedup'].max():.2f}x")
    print(f"Minimum Speedup:      {df['Speedup'].min():.2f}x")
    print(f"Avg Time Saved:       {df['Time Saved (%)'].mean():.1f}%")
    print(f"Avg Memory Saved:     {df['Memory Saved (%)'].mean():.1f}%")
    print(f"{'='*70}\n")

    print("\n✓ Benchmark complete!")
