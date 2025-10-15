#!/bin/bash

echo "=========================================="
echo "AMD Ryzen L3 Traffic Measurement Diagnostic"
echo "=========================================="

echo ""
echo "=== 1. All Available Cache Events ==="
perf list | grep -E "l3|l2_cache|data_cache" | head -30

echo ""
echo "=== 2. Looking for Zen-specific prefetch/cache events ==="
perf list | grep -i "prefetch\|fill\|evict" | head -20

echo ""
echo "=== 3. Core to L3 cache request events ==="
perf list | grep "l2_cache_req_stat" -A 2 | head -40

echo ""
echo "=== 4. Testing specific AMD events (one by one) ==="
# Try some common AMD Zen 3 cache events

echo "Testing: l2_cache_req_stat.all_l2_cache_requests_miss"
perf stat -e l2_cache_req_stat.all_l2_cache_requests_miss -- sleep 0.1 2>&1 | grep -E "l2_cache|event"

echo ""
echo "Testing: l2_cache_req_stat.miss"
perf stat -e l2_cache_req_stat.miss -- sleep 0.1 2>&1 | grep -E "miss|event"

echo ""
echo "Testing: ic_cache_fill_sys (instruction cache fills from system)"
perf stat -e ic_cache_fill_sys -- sleep 0.1 2>&1 | grep -E "ic_cache|event"

echo ""
echo "=== 5. Trying raw PMC events (if available) ==="
echo "Checking if raw events work..."
perf stat -e r076 -- sleep 0.1 2>&1 | tail -5

echo ""
echo "=== 6. All raw event codes available ==="
perf list | grep "0x" | head -20

echo ""
echo "=== 7. Check CPU details ==="
lscpu | grep -E "Model name|CPU family|Model:|Stepping"

echo ""
echo "=== 8. Run a simple test with available metrics ==="
echo "Running: perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses -- sleep 0.5"
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses -- sleep 0.5 2>&1 | grep -v "^$"