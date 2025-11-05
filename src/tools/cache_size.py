"""
Cache-aware tile size calculator for matrix multiplication optimization.

This module provides analytical models to determine optimal tile sizes (BM, BN, BK)
for tiled matrix multiplication based on CPU cache hierarchy (L1, L2, L3).

Based on research:
- Data Footprint Models (DL/ML)
- Cache capacity and associativity constraints
- Multi-level tiling strategies
"""

import math
import subprocess
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CacheInfo:
    """CPU cache hierarchy information."""
    l1_data_size: int  # in bytes
    l2_size: int       # in bytes
    l3_size: int       # in bytes
    cache_line_size: int = 64  # typical cache line size
    l1_associativity: int = 8
    l2_associativity: int = 8
    l3_associativity: int = 16

    def __str__(self):
        return (f"L1 Data: {self.l1_data_size // 1024} KB, "
                f"L2: {self.l2_size // 1024} KB, "
                f"L3: {self.l3_size // (1024*1024)} MB")


@dataclass
class TileRecommendation:
    """Recommended tile sizes for matrix multiplication."""
    # L1-level tiles (innermost, register blocking)
    bk_l1: List[int]  # K dimension for L1

    # L2-level tiles (middle level)
    bm_l2: List[int]  # M dimension for L2
    bk_l2: List[int]  # K dimension for L2

    # L3-level tiles (outermost, if applicable)
    bn_l3: List[int]  # N dimension for L3

    # Combined recommendations (for simpler single-level tiling)
    bm_simple: List[int]
    bn_simple: List[int]
    bk_simple: List[int]

    reasoning: str


def detect_cache_sizes() -> Optional[CacheInfo]:
    """
    Detect cache sizes from the system using multiple methods.

    Returns:
        CacheInfo object with detected cache sizes, or None if detection fails.
    """
    # Method 1: Try lscpu with --caches flag (newer versions)
    # Format: NAME ONE-SIZE ALL-SIZE WAYS TYPE LEVEL ...
    # L1d     32768   262144    8 Data            1 ...
    try:
        result = subprocess.run(['lscpu', '--caches', '--bytes'],
                              capture_output=True, text=True, check=True, timeout=2)
        output = result.stdout

        l1d_size = None
        l1d_ways = None
        l2_size = None
        l2_ways = None
        l3_size = None
        l3_ways = None

        lines = output.strip().split('\n')

        # Skip header line
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 6:
                continue

            name = parts[0]
            one_size = parts[1]
            ways = parts[3]
            cache_type = parts[4]
            level = parts[5]

            try:
                size_bytes = int(one_size)
                associativity = int(ways)

                # L1d (L1 Data cache)
                if name == 'L1d' and cache_type == 'Data' and level == '1':
                    l1d_size = size_bytes
                    l1d_ways = associativity

                # L2 cache
                elif name == 'L2' and level == '2':
                    l2_size = size_bytes
                    l2_ways = associativity

                # L3 cache
                elif name == 'L3' and level == '3':
                    l3_size = size_bytes
                    l3_ways = associativity

            except (ValueError, IndexError):
                continue

        if l1d_size and l2_size and l3_size:
            return CacheInfo(
                l1_data_size=l1d_size,
                l2_size=l2_size,
                l3_size=l3_size,
                l1_associativity=l1d_ways or 8,
                l2_associativity=l2_ways or 8,
                l3_associativity=l3_ways or 16
            )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: Try lscpu without --caches flag (older format)
    try:
        result = subprocess.run(['lscpu'],
                              capture_output=True, text=True, check=True, timeout=2)
        output = result.stdout

        l1d_size = None
        l2_size = None
        l3_size = None

        for line in output.split('\n'):
            if 'L1d cache:' in line:
                match = re.search(r'(\d+)\s*([KMG])i?B?', line)
                if match:
                    size, unit = match.groups()
                    multiplier = {'K': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
                    l1d_size = int(size) * multiplier.get(unit, 1)

            elif 'L2 cache:' in line:
                match = re.search(r'(\d+)\s*([KMG])i?B?', line)
                if match:
                    size, unit = match.groups()
                    multiplier = {'K': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
                    l2_size = int(size) * multiplier.get(unit, 1)

            elif 'L3 cache:' in line:
                match = re.search(r'(\d+)\s*([KMG])i?B?', line)
                if match:
                    size, unit = match.groups()
                    multiplier = {'K': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
                    l3_size = int(size) * multiplier.get(unit, 1)

        if l1d_size and l2_size and l3_size:
            return CacheInfo(
                l1_data_size=l1d_size,
                l2_size=l2_size,
                l3_size=l3_size
            )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 3: Try reading from /sys/devices/system/cpu (Linux-specific)
    try:
        import os
        cpu0_path = "/sys/devices/system/cpu/cpu0/cache"

        if os.path.exists(cpu0_path):
            l1d_size = None
            l2_size = None
            l3_size = None

            for cache_dir in os.listdir(cpu0_path):
                cache_path = os.path.join(cpu0_path, cache_dir)

                # Read cache type
                type_file = os.path.join(cache_path, "type")
                if os.path.exists(type_file):
                    with open(type_file) as f:
                        cache_type = f.read().strip()

                # Read cache level
                level_file = os.path.join(cache_path, "level")
                if os.path.exists(level_file):
                    with open(level_file) as f:
                        level = int(f.read().strip())

                # Read cache size
                size_file = os.path.join(cache_path, "size")
                if os.path.exists(size_file):
                    with open(size_file) as f:
                        size_str = f.read().strip()
                        match = re.search(r'(\d+)\s*([KMG])', size_str)
                        if match:
                            size, unit = match.groups()
                            multiplier = {'K': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
                            size_bytes = int(size) * multiplier.get(unit, 1)

                            if level == 1 and cache_type == "Data":
                                l1d_size = size_bytes
                            elif level == 2:
                                l2_size = size_bytes
                            elif level == 3:
                                l3_size = size_bytes

            if l1d_size and l2_size and l3_size:
                return CacheInfo(
                    l1_data_size=l1d_size,
                    l2_size=l2_size,
                    l3_size=l3_size
                )
    except Exception:
        pass

    return None


def get_default_cache_info(architecture: str) -> CacheInfo:
    """
    Get default cache information for known architectures.

    Args:
        architecture: Architecture identifier (e.g., "AMD-Ryzen-7-6800HS")

    Returns:
        CacheInfo object with default values for the architecture.
    """
    defaults = {
        "AMD-Ryzen-7-6800HS": CacheInfo(
            l1_data_size=32 * 1024,      # 32 KB L1 data cache per core
            l2_size=512 * 1024,           # 512 KB L2 per core
            l3_size=16 * 1024 * 1024,     # 16 MB L3 shared
            l1_associativity=8,
            l2_associativity=8,
            l3_associativity=16
        ),
        "Intel-i7-1195G7": CacheInfo(
            l1_data_size=48 * 1024,      # 48 KB L1 data cache per core
            l2_size=1280 * 1024,          # 1.25 MB L2 per core
            l3_size=12 * 1024 * 1024,     # 12 MB L3 shared
            l1_associativity=12,
            l2_associativity=10,
            l3_associativity=12
        ),
    }

    return defaults.get(architecture, CacheInfo(
        l1_data_size=32 * 1024,
        l2_size=256 * 1024,
        l3_size=8 * 1024 * 1024
    ))


def calculate_tile_sizes(cache_info: CacheInfo,
                        element_size: int = 4,
                        num_cores: int = 1,
                        use_conservative: bool = True) -> TileRecommendation:
    """
    Calculate optimal tile sizes using analytical cache models.

    This implements research-based approaches:
    1. L1 tiling: Based on sqrt(M_fast / 3) heuristic
    2. L2 tiling: Sized to fit working set in L2
    3. L3 tiling: Considers shared cache for multi-core

    Args:
        cache_info: CPU cache hierarchy information
        element_size: Size of matrix element in bytes (4 for float)
        num_cores: Number of cores for L3 tiling
        use_conservative: Use conservative (DL) model vs aggressive (ML) model

    Returns:
        TileRecommendation with suggested tile sizes
    """

    # --- L1 Tiling (Innermost Level) ---
    # Based on heuristic: BLOCK_SIZE <= sqrt(M_fast / 3)
    # For GEMM, we need to fit: BM*BK + BK*BN + BM*BN in L1
    # Conservative approach: focus on BK dimension for K-loop reuse

    l1_capacity_elements = cache_info.l1_data_size // element_size

    # For L1, we want small BK values that allow A panel and B panel to fit
    # Assume we want: BM*BK (A panel) + BK*BN (B panel) <= L1_capacity * utilization
    # Simplified: prioritize BK for inner loop, BM/BN handled by registers

    l1_utilization = 0.5 if use_conservative else 0.75
    available_l1 = int(l1_capacity_elements * l1_utilization)

    # BK for L1: sqrt approximation
    bk_l1_base = int(math.sqrt(available_l1 / 3))
    bk_l1_options = [
        max(8, (bk_l1_base // 8) * 8 - 8),
        (bk_l1_base // 8) * 8,
        (bk_l1_base // 8) * 8 + 8,
    ]
    bk_l1_options = [x for x in bk_l1_options if 8 <= x <= 128]
    if not bk_l1_options:
        bk_l1_options = [16, 24, 32]

    # --- L2 Tiling (Middle Level) ---
    # L2 should fit: (BM_L2 * BK_L2) panel of A
    # This allows reuse of A panel across multiple N iterations

    l2_capacity_elements = cache_info.l2_size // element_size
    l2_utilization = 0.6 if use_conservative else 0.8
    available_l2 = int(l2_capacity_elements * l2_utilization)

    # BK_L2 should be larger than BK_L1 (multiple L1 tiles)
    bk_l2_base = max(bk_l1_options) * 2
    bk_l2_options = [
        max(32, (bk_l2_base // 16) * 16 - 16),
        (bk_l2_base // 16) * 16,
        (bk_l2_base // 16) * 16 + 16,
    ]
    bk_l2_options = [x for x in bk_l2_options if 32 <= x <= 256]
    if not bk_l2_options:
        bk_l2_options = [48, 64, 80]

    # BM_L2: Given BK_L2, how large can BM be?
    bm_l2_options = []
    for bk in bk_l2_options:
        bm_max = available_l2 // bk
        bm_base = min(bm_max, 192)
        bm_l2_options.extend([
            max(32, (bm_base // 16) * 16 - 16),
            (bm_base // 16) * 16,
            min(256, (bm_base // 16) * 16 + 16),
        ])

    bm_l2_options = sorted(set([x for x in bm_l2_options if 32 <= x <= 256]))
    if not bm_l2_options:
        bm_l2_options = [64, 96, 128]

    # --- L3 Tiling (Outermost Level) ---
    # L3 should fit: BN_L3 * BK_L2 panel of B
    # For multi-core, divide L3 capacity by number of cores

    l3_capacity_elements = cache_info.l3_size // element_size
    l3_per_core = l3_capacity_elements // max(1, num_cores)
    l3_utilization = 0.5 if use_conservative else 0.7
    available_l3 = int(l3_per_core * l3_utilization)

    bn_l3_options = []
    for bk in bk_l2_options:
        bn_max = available_l3 // bk
        bn_base = min(bn_max, 384)
        bn_l3_options.extend([
            max(32, (bn_base // 32) * 32 - 32),
            (bn_base // 32) * 32,
            min(512, (bn_base // 32) * 32 + 32),
        ])

    bn_l3_options = sorted(set([x for x in bn_l3_options if 32 <= x <= 512]))
    if not bn_l3_options:
        bn_l3_options = [128, 192, 256]

    # --- Simple Single-Level Recommendations ---
    # For implementations that use single-level tiling (BM, BN, BK)
    # Aim to fit in L2 cache: BM*BK + BK*BN + BM*BN <= L2

    # Solve for balanced BM = BN = BK case
    balanced_tile = int((available_l2 / 3) ** 0.5)

    simple_base_options = [32, 48, 64, 96, 128, 192]
    bm_simple = [x for x in simple_base_options if x <= balanced_tile * 1.5]
    bn_simple = bm_simple.copy()
    bk_simple = [x for x in simple_base_options if x <= balanced_tile]

    # Ensure we have reasonable defaults
    if not bm_simple:
        bm_simple = [64, 96, 128]
    if not bn_simple:
        bn_simple = [64, 96, 128]
    if not bk_simple:
        bk_simple = [32, 48, 64]

    reasoning = f"""
Cache-Aware Tile Size Analysis:
================================
L1 Data: {cache_info.l1_data_size // 1024} KB (per core)
L2: {cache_info.l2_size // 1024} KB (per core)
L3: {cache_info.l3_size // (1024*1024)} MB (shared across {num_cores} cores)

Multi-Level Tiling Strategy:
- L1 tiles (BK): Focus on K-dimension for inner loop reuse in L1
- L2 tiles (BM, BK): A panel (BM×BK) should fit in L2 for reuse across N
- L3 tiles (BN): B panel (BN×BK) should fit in L3 (per-core) for reuse across M

Single-Level Tiling (Simplified):
- Targets L2 cache fit: BM×BK + BK×BN + BM×BN ≤ L2 capacity × utilization
- Balanced approach with BM ≈ BN > BK for typical GEMM patterns
- Balanced tile size estimate: ~{balanced_tile} elements

Recommendations use {'conservative (DL)' if use_conservative else 'aggressive (ML)'} model.
Test multiple values from each range for best performance on your workload.
"""

    return TileRecommendation(
        bk_l1=bk_l1_options[:3],
        bm_l2=bm_l2_options[:5],
        bk_l2=bk_l2_options[:3],
        bn_l3=bn_l3_options[:5],
        bm_simple=bm_simple,
        bn_simple=bn_simple,
        bk_simple=bk_simple,
        reasoning=reasoning.strip()
    )


def format_for_llm_prompt(recommendation: TileRecommendation,
                          cache_info: CacheInfo,
                          num_cores: int = 1) -> str:
    """
    Format tile recommendations for inclusion in LLM prompt.

    Args:
        recommendation: TileRecommendation object
        cache_info: CacheInfo object
        num_cores: Number of CPU cores

    Returns:
        Formatted string to include in LLM prompt
    """

    prompt_section = f"""
**CACHE-AWARE TILING GUIDANCE:**

Target CPU Cache Hierarchy:
- L1 Data Cache: {cache_info.l1_data_size // 1024} KB (per core, {cache_info.l1_associativity}-way)
- L2 Cache: {cache_info.l2_size // 1024} KB (per core, {cache_info.l2_associativity}-way)
- L3 Cache: {cache_info.l3_size // (1024*1024)} MB (shared, {cache_info.l3_associativity}-way)
- Cache Line Size: {cache_info.cache_line_size} bytes

**Recommended Tile Sizes (BM, BN, BK):**

For SINGLE-LEVEL tiling (recommended for initial implementation):
- BM (M-dimension blocking): Test values from {{{', '.join(map(str, recommendation.bm_simple))}}}
- BN (N-dimension blocking): Test values from {{{', '.join(map(str, recommendation.bn_simple))}}}
- BK (K-dimension blocking): Test values from {{{', '.join(map(str, recommendation.bk_simple))}}}

**Rationale:**
These values are analytically derived to:
1. Fit the working set (tiles of A, B, C) primarily in L2 cache ({cache_info.l2_size // 1024} KB)
2. Maximize data reuse while minimizing cache misses
3. Balance between cache capacity and computational intensity
4. Account for {cache_info.l2_associativity}-way set associativity in L2

**Recommended defaults to start with:**
- BM = {recommendation.bm_simple[len(recommendation.bm_simple)//2]} (middle value)
- BN = {recommendation.bn_simple[len(recommendation.bn_simple)//2]} (middle value)
- BK = {recommendation.bk_simple[len(recommendation.bk_simple)//2]} (middle value)

For MULTI-LEVEL tiling (advanced, hierarchical blocking):
- Inner K tiles (L1-focused, {cache_info.l1_data_size // 1024} KB): {{{', '.join(map(str, recommendation.bk_l1))}}}
- Middle M tiles (L2-focused, {cache_info.l2_size // 1024} KB): {{{', '.join(map(str, recommendation.bm_l2[:3]))}}}
- Middle K tiles (L2-focused): {{{', '.join(map(str, recommendation.bk_l2))}}}
- Outer N tiles (L3-focused, ~{cache_info.l3_size // (1024*1024*num_cores)} MB per core): {{{', '.join(map(str, recommendation.bn_l3[:3]))}}}

**Implementation Guidelines:**
1. Start with single-level tiling using the recommended defaults above
2. Make BM, BN, BK compile-time constants (constexpr) or template parameters for easy tuning
3. Ensure tile sizes are multiples of SIMD vector width (8 for AVX2, 16 for AVX-512 with float32)
4. Consider that optimal values may vary ±20% based on:
   - Matrix shapes (square vs. rectangular)
   - Memory access patterns and stride
   - SIMD vector width and micro-kernel register blocking
   - TLB effects (prefer tiles that don't cause excessive TLB misses)
   - Hardware prefetching behavior

5. For multi-threaded code ({num_cores} cores available):
   - Outer tiles should be large enough to distribute work across threads
   - Inner tiles should still respect single-core cache constraints
   - Consider using OpenMP collapse(2) for BM×BN outer loops when tiles are small

6. Memory layout considerations:
   - For row-major C (M×N): parallelize over M dimension (rows)
   - Ensure BN is a multiple of cache line size / element_size = {cache_info.cache_line_size // 4} elements
   - This gives aligned, contiguous memory access patterns
"""

    return prompt_section.strip()


def get_tile_recommendations(architecture: str,
                            num_cores: int = 1,
                            use_conservative: bool = True) -> Tuple[TileRecommendation, CacheInfo, str]:
    """
    Main entry point: Get tile size recommendations for a given architecture.

    Args:
        architecture: Architecture identifier
        num_cores: Number of cores to consider for L3 tiling
        use_conservative: Use conservative cache model

    Returns:
        Tuple of (TileRecommendation, CacheInfo, formatted_prompt_section)
    """

    # Try to detect cache sizes, fall back to defaults
    cache_info = detect_cache_sizes()
    if cache_info is None:
        print(f"[Cache Detection] Could not auto-detect, using defaults for {architecture}")
        cache_info = get_default_cache_info(architecture)
    else:
        print(f"[Cache Detection] Successfully detected: {cache_info}")

    # Calculate recommendations
    recommendation = calculate_tile_sizes(
        cache_info=cache_info,
        element_size=4,  # float32
        num_cores=num_cores,
        use_conservative=use_conservative
    )

    # Format for LLM
    prompt_section = format_for_llm_prompt(recommendation, cache_info, num_cores)

    return recommendation, cache_info, prompt_section


# Example usage and testing
if __name__ == "__main__":
    # Test for both architectures
    for arch in ["AMD-Ryzen-7-6800HS", "Intel-i7-1195G7"]:
        print(f"\n{'='*60}")
        print(f"Architecture: {arch}")
        print('='*60)

        recommendation, cache_info, prompt = get_tile_recommendations(
            architecture=arch,
            num_cores=8 if "Ryzen" in arch else 4,
            use_conservative=True
        )

        print(f"\n{recommendation.reasoning}")
        print(f"\nSimple recommendations:")
        print(f"  BM: {recommendation.bm_simple}")
        print(f"  BN: {recommendation.bn_simple}")
        print(f"  BK: {recommendation.bk_simple}")
        print(f"\n{'-'*60}\n")
        print(prompt)