# Intel HEXL Integration Guide

Intel HEXL (Homomorphic Encryption Acceleration Library) provides AVX512-accelerated
NTT operations for SEAL, which TenSEAL wraps. This can yield 2-7× speedup on HE
operations with zero code changes — it's a build-time optimization.

## Prerequisites

- CPU with AVX512 support (Intel Ice Lake+, or AMD Zen 4+)
- CMake 3.13+, GCC 7+ or Clang 5+

### Check AVX512 Support

```bash
grep -c avx512 /proc/cpuinfo
# If 0, this optimization won't help — skip it.
```

## Build Steps

### 1. Install Intel HEXL

```bash
git clone https://github.com/intel/hexl.git
cd hexl
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j$(nproc)
sudo cmake --install build
cd ..
```

### 2. Rebuild Microsoft SEAL with HEXL

```bash
git clone -b v4.1.1 https://github.com/microsoft/SEAL.git
cd SEAL
cmake -S . -B build \
    -DSEAL_USE_INTEL_HEXL=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j$(nproc)
sudo cmake --install build
cd ..
```

### 3. Rebuild TenSEAL Against HEXL-enabled SEAL

```bash
pip uninstall tenseal -y
git clone https://github.com/OpenMined/TenSEAL.git
cd TenSEAL

# Patch CMakeLists.txt to enable HEXL
sed -i 's/SEAL_USE_INTEL_HEXL OFF/SEAL_USE_INTEL_HEXL ON/g' CMakeLists.txt

pip install .
cd ..
```

### 4. Verify

```python
import tenseal as ts
import time

# Quick benchmark
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                 coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx.global_scale = 2**40
vec = ts.ckks_vector(ctx, [1.0] * 4096)
matrix = [[float(i+j) for j in range(100)] for i in range(4096)]

t0 = time.perf_counter()
for _ in range(10):
    vec.mm(matrix)
elapsed = (time.perf_counter() - t0) / 10
print(f"Average mm time: {elapsed*1000:.1f}ms")
# With HEXL this should be noticeably faster than without
```

## Using the `--hexl` Flag

The codebase doesn't require any code changes for HEXL — it's transparent at the
SEAL/TenSEAL level. However, we provide an optional `--hexl` flag on the server
to log whether HEXL acceleration is detected:

```bash
python -m server.server --hexl
```

This will log a warning if HEXL is not available.

## Expected Impact

| Operation | Without HEXL | With HEXL | Speedup |
|-----------|-------------|-----------|---------|
| NTT (4096 slots) | ~2ms | ~0.3ms | ~7× |
| HE matmul (2048×2048) | ~800ms | ~200ms | ~4× |
| Full layer (4 rounds) | ~10s | ~3s | ~3× |

*Benchmarks are approximate and vary by CPU model.*

## Troubleshooting

- **"HEXL not found"**: Ensure `/usr/local/lib/libhexl.so` exists and is in `LD_LIBRARY_PATH`
- **No speedup observed**: Verify AVX512 is actually being used: `lscpu | grep avx512`
- **TenSEAL build fails**: Try building SEAL as a shared library (`-DBUILD_SHARED_LIBS=ON`)
