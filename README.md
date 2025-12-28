# HPC Agent

-`results/`: Stores successfully generated and optimized code

-`workspace`: Temporary files (source, binaries)

-`main.py`: Main script to run the optimization loop

-`src/`: Source code for the project

CPU Configurations

```markdown
**Target Platform (Host CPU):**
            - Architecture: x86_64 (Intel 11th Gen Core i7-1195G7)
            - SIMD ISA: AVX2, FMA, and AVX-512
            - Threads: 8 logical CPUs (4 cores, SMT/HT=2)
            - OS: Linux (assume recent GCC/Clang toolchain)
```

```markdown
**Target Platform (Host CPU):**
            - Architecture: x86_64 (AMD Ryzen 7 6800HS)
            - SIMD ISA: AVX, AVX2, and FMA
            - Threads: 16 logical CPUs (8 cores, SMT/HT=2)
            - OS: Linux (assume recent GCC/Clang toolchain)
```

```markdown
**Target Platform (Host CPU):**
            - Architecture: x86_64 (Zen 4/5 AMD EPYC 9365)
            - SIMD ISA: AVX2, FMA, and AVX-512 (Full 512-bit registers)
            - Cores: 72 Physical Cores, 144 Logical Threads
            - Topology: 2 Sockets, 2 NUMA Nodes
            - L3 Cache: 384 MiB total (very large, favorable for blocking)
            - OS: Linux (GCC/Clang)
```
