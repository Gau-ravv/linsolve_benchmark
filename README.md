# Linear solver benchmark

Solves **Ax = b** three ways, times each, and checks every answer — to measure
how much low-level optimization of a matrix multiply is actually worth to a
real numerical task.

| Method | What it is |
|---|---|
| `inversion` | A⁻¹ by Gauss-Jordan, then x = A⁻¹b. The bad way. |
| `unblocked-lu` | LU with partial pivoting, plain triple loop. Correct, unoptimized. |
| `blocked-lu/*` | Right-looking blocked LU whose Schur update is a matrix multiply. Six rows — **one algorithm, six kernels**. |

The six multiply kernels each add one optimization on top of the last:
`naive`, `reordered` (i-k-j loop order), `blocked` (64×64 cache tiles), `simd`
(AVX2 FMA intrinsics), `parallel` (OpenMP), `full` (all of it).

The kernel goes in as a plain function pointer:

```cpp
Matrix solve_blocked_lu(const Matrix& A, const Matrix& b,
                        Matrix (*matmul)(const Matrix&, const Matrix&));
```

No templates anywhere — a function pointer means all six rows run the *same*
compiled factorization, so the kernel is the only variable in the comparison.

## Run it

```bat
run.bat
```

Builds `linsolve.exe` if missing, then runs it. Results also land in
`results_solve.csv`. Sizes are one constant at the top of `linsolve.cpp`:

```cpp
const int SOLVE_SIZES[NUM_SOLVE_SIZES] = {128, 256, 512, 1024, 2048};
```

Most of the wall time is `inversion` and `blocked-lu/naive` at 2048 — drop
2048 from that list for a fast run.

## The blocked LU

Block size 64. For each panel of 64 columns starting at column *k*:

1. **Factor the panel** `A[k:n, k:k+64]` with unblocked LU + partial pivoting,
   recording each pivot row.
2. **Replay the row swaps** across the full row width.
3. **Triangular solve** `A[k:k+64, k+64:n] = L11⁻¹ · A[k:k+64, k+64:n]`.
4. **Schur update** — this is the matmul:
   `A[k+64:n, k+64:n] -= A[k+64:n, k:k+64] · A[k:k+64, k+64:n]`

Steps 1–3 cost O(n·nb²) and O(n²·nb); step 4 is O(n³). **At n = 2048 about 95%
of the arithmetic is in that one update**, which is why swapping the kernel
moves the whole solve time — and why Amdahl's law caps the achievable gain
near 20×.

Blocking does not change the answer: same pivots, same L and U, same x. All six
kernels produce byte-identical residuals, which is the check that the
regrouping is correct.

**Implementation note.** `Matrix` is a flat row-major buffer with no stride
support, so a submatrix view can't be handed to the kernels. Step 4 copies the
two operand blocks into fresh matrices, calls the kernel unchanged, and
subtracts the result back. The copies move O(m²) elements against O(m²·64)
flops of real work — a 1/64 sliver — so they don't distort the timings. That
was the deliberate tradeoff: a little copying to keep all six kernels
completely untouched.

## Checking the answers

Built backwards on purpose: pick `x_true`, compute `b = A · x_true`, then have
each solver recover x from A and b alone. A is random in [−1, 1] plus n on
every diagonal entry — strictly diagonally dominant, hence non-singular and
well conditioned, so the benchmark measures speed rather than numerical
blowup.

Two columns follow: **residual** = max |A·x − b|, and **error** =
max |x − x_true| against a solution known exactly rather than against another
solver's output.

The pass threshold scales with n, since elimination's residual grows like
eps·n·‖b‖:

```cpp
tolerance = 256 · eps · n · max|b|
```

A fixed 1e-9 sits below the noise floor at n = 2048 and would fail perfectly
correct answers.

## Measured

| | |
|---|---|
| CPU | Intel Core i7-9750H, 6 cores / 12 threads, AVX2 + FMA |
| Compiler | MSYS2 mingw64 g++, `-O2 -std=c++17 -march=native -fopenmp -static` |

Every row passes the scaled residual check at every size.

```
Size   Method                Time(ms)      Residual     Error       Speedup
2048   inversion             23605.013     1.82e-11     8.99e-15      1.00
2048   unblocked-lu           5031.202     1.64e-11     7.88e-15      4.69
2048   blocked-lu/naive       7111.770     1.32e-11     6.44e-15      3.32
2048   blocked-lu/reordered   4875.220     1.32e-11     6.44e-15      4.84
2048   blocked-lu/blocked     4979.141     1.32e-11     6.44e-15      4.74
2048   blocked-lu/simd        3446.753     1.32e-11     6.44e-15      6.85
2048   blocked-lu/parallel    1417.510     1.32e-11     6.44e-15     16.65
2048   blocked-lu/full        1148.549     1.32e-11     6.44e-15     20.55
```

`Speedup` is against the slowest method at that size, which is `inversion`
everywhere.

### Swapping only the kernel

Same factorization, same pivots, same answer — only the Schur-update kernel
changes:

| n | naive | full | Speedup |
|---:|---:|---:|---:|
| 128 | 1.915 ms | 1.736 ms | **1.10×** |
| 256 | 11.365 ms | 6.793 ms | **1.67×** |
| 512 | 111.486 ms | 33.012 ms | **3.38×** |
| 1024 | 848.490 ms | 206.645 ms | **4.11×** |
| 2048 | 7111.770 ms | 1148.549 ms | **6.19×** |

**6.2× with the solver untouched** — that is the number this project exists to
produce.

The trend matters as much as the number. At n = 128 the whole matrix fits in
L2, the Schur updates are tiny, and `blocked-lu/parallel` (3.33 ms) is actually
*slower* than serial `reordered` (1.91 ms) — OpenMP thread startup costs more
than the work it distributes. The advantage appears only once the multiply is
genuinely the bottleneck, and grows monotonically with n.

Two things bound the 2048 result: the ~5% serial panel work no kernel touches
(Amdahl, ~20× ceiling), and the shape of the update. It is (m × 64) · (64 × m),
so the inner dimension is only 64 — even the naive kernel's strided reads stay
short and mostly in cache, and it isn't penalized nearly as hard as it would be
on a square multiply.

### Inversion loses twice

At n = 2048, `inversion` takes 23.6 s against `unblocked-lu`'s 5.0 s — roughly
the 3× arithmetic ratio, plus the cost of touching 2n² elements per elimination
step instead of n². Against `blocked-lu/full` it is **20.6× slower**.

And it is less accurate at every size:

| n | Error (inversion) | Error (LU) |
|---:|---:|---:|
| 128 | 2.44e-15 | 1.11e-15 |
| 512 | 4.00e-15 | 3.00e-15 |
| 2048 | 8.99e-15 | 6.44e-15 |

Forming A⁻¹ explicitly commits rounding that solving in place never commits.
Slower *and* wronger, consistently — which is the whole case against it, made
concrete instead of folkloric.
