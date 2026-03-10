#!/usr/bin/env python3
"""
Analyze Nsight Systems profiling results for cuda_native

Extracts key metrics from nsys profiles to understand:
1. GPU time breakdown (compute vs memory)
2. Kernel execution patterns
3. Comparison between c=1 and c=16 workloads
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_nsys_stats(stats_file):
    """Parse nsys stats output file"""
    with open(stats_file) as f:
        content = f.read()

    metrics = {}
    in_cuda_summary = False

    for line in content.split('\n'):
        if 'CUDA API Summary' in line:
            in_cuda_summary = True
            continue

        if in_cuda_summary and line.strip().startswith('*'):
            parts = line.strip().split()
            if len(parts) >= 10:
                try:
                    api_name = parts[0]
                    num_calls = int(parts[1])
                    total_time_us = float(parts[3])
                    avg_time_us = float(parts[4])
                    metrics[api_name] = {
                        'calls': num_calls,
                        'total_time_us': total_time_us,
                        'avg_time_us': avg_time_us,
                    }
                except:
                    pass

    return metrics


def get_nsys_sqlite_export(profile_file):
    """Export nsys profile to SQLite and query for key metrics"""
    import sqlite3

    # Convert Path to string and replace extension
    profile_file_str = str(profile_file)
    sqlite_file = Path(profile_file_str.replace('.nsys-rep', '.sqlite'))

    if not sqlite_file.exists():
        print(f"SQLite file not found: {sqlite_file}")
        return None

    conn = sqlite3.connect(str(sqlite_file))
    cursor = conn.cursor()

    # Query CUDA kernel executions
    try:
        cursor.execute("""
            SELECT
                k.functionName,
                COUNT(*) as calls,
                SUM(k.totalDuration) as total_duration_ns,
                AVG(k.totalDuration) as avg_duration_ns
            FROM CUPTI_ACTIVITY_KIND ck
            JOIN CUPTI_ACTIVITY k ON ck.id = k.activityId
            WHERE ck.stringId = 1 AND ck.value = 'Kernel'
            GROUP BY k.functionName
            ORDER BY total_duration_ns DESC
            LIMIT 20
        """)

        kernels = cursor.fetchall()

        print("\n" + "=" * 80)
        print("Top 20 CUDA Kernels by Total Duration")
        print("=" * 80)
        print(f"{'Kernel':<50} {'Calls':>8} {'Total (ms)':>12} {'Avg (ms)':>10}")
        print("-" * 82)

        total_kernel_time = 0
        for row in kernels:
            func_name, calls, total_ns, avg_ns = row
            total_ms = total_ns / 1_000_000
            avg_ms = avg_ns / 1_000_000
            total_kernel_time += total_ms

            # Truncate function name
            if len(func_name) > 47:
                func_name = func_name[:44] + "..."

            print(f"{func_name:<50} {calls:>8} {total_ms:>12.2f} {avg_ms:>10.4f}")

        print(f"\nTotal kernel time: {total_kernel_time:.2f} ms")

    except Exception as e:
        print(f"Error querying kernels: {e}")

    # Query memory copies
    try:
        cursor.execute("""
            SELECT
                m.functionName,
                COUNT(*) as calls,
                SUM(m.totalDuration) as total_duration_ns,
                SUM(m.bytes) as total_bytes
            FROM CUPTI_ACTIVITY_KIND cm
            JOIN CUPTI_ACTIVITY m ON cm.id = m.activityId
            WHERE cm.stringId = 1 AND cm.value = 'Memory Copy'
            GROUP BY m.functionName
            ORDER BY total_duration_ns DESC
            LIMIT 10
        """)

        memcpy_ops = cursor.fetchall()

        print("\n" + "=" * 80)
        print("Top 10 Memory Copy Operations")
        print("=" * 80)
        print(f"{'Operation':<50} {'Calls':>8} {'Total (ms)':>12} {'MB':>10}")
        print("-" * 82)

        for row in memcpy_ops:
            func_name, calls, total_ns, total_bytes = row
            total_ms = total_ns / 1_000_000
            mb = total_bytes / (1024 * 1024) if total_bytes else 0

            # Truncate function name
            if len(func_name) > 47:
                func_name = func_name[:44] + "..."

            print(f"{func_name:<50} {calls:>8} {total_ms:>12.2f} {mb:>10.2f}")

    except Exception as e:
        print(f"Error querying memory copies: {e}")

    # Query GPU memory usage
    try:
        cursor.execute("""
            SELECT
                SUM(m.size) / (1024.0 * 1024.0 * 1024.0) as total_memory_gb
            FROM CUPTI_ACTIVITY_KIND cm
            JOIN CUPTI_ACTIVITY m ON cm.id = m.activityId
            WHERE cm.stringId = 1 AND cm.value = 'Memory Allocation'
        """)

        result = cursor.fetchone()
        if result and result[0]:
            print(f"\nGPU memory allocated: {result[0]:.2f} GB")

    except Exception as e:
        print(f"Error querying memory: {e}")

    conn.close()

    return kernels


def compare_profiles(c1_stats, c16_stats):
    """Compare c=1 and c=16 profiles"""
    print("\n" + "=" * 80)
    print("COMPARISON: c=1 vs c=16")
    print("=" * 80)

    # Compare kernel counts
    print("\nCUDA API Call Counts:")
    print(f"{'API':<30} {'c=1':>12} {'c=16':>12} {'Ratio':>10}")
    print("-" * 64)

    for api in sorted(set(c1_stats.keys()) | set(c16_stats.keys())):
        c1_calls = c1_stats.get(api, {}).get('calls', 0)
        c16_calls = c16_stats.get(api, {}).get('calls', 0)

        if c1_calls == 0 and c16_calls == 0:
            continue

        ratio = c16_calls / c1_calls if c1_calls > 0 else 0

        print(f"{api:<30} {c1_calls:>12} {c16_calls:>12} {ratio:>10.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cuda_native Nsight Systems profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("profile_dir", help="Directory containing nsys profiles")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)

    if not profile_dir.exists():
        print(f"Error: Directory not found: {profile_dir}")
        return 1

    # Load stats
    c1_stats = parse_nsys_stats(profile_dir / "stats_c1.txt")
    c16_stats = parse_nsys_stats(profile_dir / "stats_c16.txt")

    # Analyze c=1 profile
    print(f"\n{'=' * 80}")
    print(f"ANALYZING: c=1 profile")
    print(f"{'=' * 80}\n")
    get_nsys_sqlite_export(profile_dir / "profile_c1.nsys-rep")

    # Analyze c=16 profile
    print(f"\n{'=' * 80}")
    print(f"ANALYZING: c=16 profile")
    print(f"{'=' * 80}\n")
    get_nsys_sqlite_export(profile_dir / "profile_c16.nsys-rep")

    # Compare
    compare_profiles(c1_stats, c16_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
