// linsolve.cpp -- what is a fast matrix multiply actually worth?
//
// Timing a multiply on its own tells you how fast the kernel is. It does not
// tell you whether the kernel makes a real program faster. Solving Ax = b is
// a real task with an answer that can be checked, so this benchmark solves
// the same system three different ways and times each:
//
//   1. solve_by_inversion   compute A^-1 by Gauss-Jordan, then x = A^-1 b.
//                           The deliberately bad baseline -- about 3x the
//                           arithmetic of elimination, and less accurate.
//   2. solve_unblocked_lu   LU with partial pivoting, plain triple loop.
//                           Correct, standard, and completely unoptimized.
//   3. solve_blocked_lu     right-looking blocked LU whose Schur update is a
//                           matrix multiply -- run once per kernel, so the
//                           same algorithm appears six times in the table and
//                           differs only in which multiply it calls.
//
// The six multiply kernels each add one optimization to the previous:
//   1. multiply_naive      textbook i-j-k loop order
//   2. multiply_reordered  i-k-j loop order (stride-1 memory access)
//   3. multiply_blocked    cache tiling on top of the reordered loop
//   4. multiply_simd       AVX2 intrinsics on top of the reordered loop
//   5. multiply_parallel   tiling + OpenMP threads
//   6. multiply_full       tiling + AVX2 + OpenMP threads
//
// The kernel goes into solve_blocked_lu as a plain function pointer, so all
// six rows run the same compiled factorization and the kernel is the only
// variable in the comparison.
//
// Tuned for this machine: Intel Core i7-9750H (6 cores / 12 threads, AVX2 +
// FMA, 32 KB L1d per core, 256 KB L2 per core, 12 MB shared L3), built with
// MSYS2 mingw64 g++ 14.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <immintrin.h>   // AVX2 / FMA intrinsics (x86 only)

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Cache tile: 64 x 64 doubles = 32 KB. Three of those (A, B, C tiles) is
// 96 KB, which sits comfortably inside this CPU's 256 KB per-core L2.
// The blocked LU reuses this as its panel width, so the Schur-update multiply
// lands on the tile boundaries the kernels already like.
const int BLOCK_SIZE = 64;

// One AVX2 register is 256 bits and a double is 64 bits, so 4 doubles fit
// side by side and one FMA instruction does 4 multiply-adds at once.
// BLOCK_SIZE (64) is a multiple of 4, so tiles never split a vector.
const int SIMD_WIDTH = 4;

// Edit this list to change the benchmark. The naive-kernel and inversion rows
// at 2048 are the slow ones -- drop 2048 for a quick run.
const int NUM_SOLVE_SIZES = 5;
const int SOLVE_SIZES[NUM_SOLVE_SIZES] = {128, 256, 512, 1024, 2048};

const unsigned RANDOM_SEED_MATRIX = 42;
const unsigned RANDOM_SEED_SOLUTION = 1337;
const double RANDOM_MIN = -1.0;
const double RANDOM_MAX = 1.0;

// A purely random matrix can sit arbitrarily close to singular. Adding n to
// every diagonal entry makes it strictly diagonally dominant, which
// guarantees it is non-singular and keeps its condition number near 1 -- so
// the benchmark measures speed instead of numerical blowup.
const double DIAGONAL_BOOST_PER_N = 1.0;

// Gaussian elimination is backward stable, so an honest solver's residual
// lands around eps * n * ||b||. This is that bound with slack. A fixed
// absolute tolerance will not do: at n = 2048 an absolute 1e-9 sits below the
// noise floor and would fail perfectly good answers.
const double RESIDUAL_TOLERANCE_UNITS = 256.0;

// A pivot smaller than this means the matrix is singular to working
// precision. The diagonal boost should make this unreachable; it is here so a
// bad test problem reports itself instead of dividing by zero.
const double SINGULAR_PIVOT_THRESHOLD = 1.0e-300;

const int NUM_METHODS = 8;   // inversion + unblocked LU + six blocked-LU rows

const std::string RESULTS_CSV_PATH = "results_solve.csv";

// ---------------------------------------------------------------------------
// Matrix: one flat row-major buffer
// ---------------------------------------------------------------------------

// A vector<vector<double>> would heap-allocate every row separately and
// scatter them across memory, which defeats every cache trick below. One
// contiguous buffer keeps a row -- and a whole tile -- physically close.
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> data;

    double& at(int row, int col) {
        return data[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    }

    const double& at(int row, int col) const {
        return data[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    }
};

Matrix create_matrix(int rows, int cols) {
    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.data.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0);
    return matrix;
}

void fill_random(Matrix& matrix, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(RANDOM_MIN, RANDOM_MAX);
    for (size_t index = 0; index < matrix.data.size(); ++index) {
        matrix.data[index] = distribution(generator);
    }
}

// ---------------------------------------------------------------------------
// 1. Naive: textbook i-j-k order
// ---------------------------------------------------------------------------

// The inner loop over k reads A(i,k) contiguously, but reads B(k,j) with
// stride B.cols -- a new row, and a new cache line, on every step. For any
// matrix wider than a cache line that means a miss almost every iteration.
// This is the slow baseline everything else is measured against.
Matrix multiply_naive(const Matrix& A, const Matrix& B) {
    Matrix C = create_matrix(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < A.cols; ++k) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
    return C;
}

// ---------------------------------------------------------------------------
// 2. Reordered: i-k-j order
// ---------------------------------------------------------------------------

// Swapping j and k makes A(i,k) a loop-invariant scalar and leaves the
// innermost loop walking both B(k,j) and C(i,j) with stride 1 -- whole rows,
// read and written in the order they sit in memory. Every cache line that
// gets loaded is now used for several consecutive iterations.
Matrix multiply_reordered(const Matrix& A, const Matrix& B) {
    Matrix C = create_matrix(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            double a_ik = A.at(i, k);
            for (int j = 0; j < B.cols; ++j) {
                C.at(i, j) += a_ik * B.at(k, j);
            }
        }
    }
    return C;
}

// ---------------------------------------------------------------------------
// 3. Blocked: cache tiling on top of the reordered loop
// ---------------------------------------------------------------------------

// At large N a full row of B or C is far bigger than L1/L2, so by the time
// the i loop comes back to reuse a row it has already been evicted. Tiling
// all three loops into 64x64 chunks keeps each pass working on a tile of A,
// B and C small enough to sit in cache together for the whole k range.
Matrix multiply_blocked(const Matrix& A, const Matrix& B) {
    Matrix C = create_matrix(A.rows, B.cols);
    int n_rows = A.rows;
    int n_cols = B.cols;
    int n_inner = A.cols;

    for (int block_row = 0; block_row < n_rows; block_row += BLOCK_SIZE) {
        int row_end = std::min(block_row + BLOCK_SIZE, n_rows);
        for (int block_col = 0; block_col < n_cols; block_col += BLOCK_SIZE) {
            int col_end = std::min(block_col + BLOCK_SIZE, n_cols);
            for (int block_inner = 0; block_inner < n_inner; block_inner += BLOCK_SIZE) {
                int inner_end = std::min(block_inner + BLOCK_SIZE, n_inner);

                // Inside a tile this is exactly multiply_reordered's loop.
                for (int i = block_row; i < row_end; ++i) {
                    for (int k = block_inner; k < inner_end; ++k) {
                        double a_ik = A.at(i, k);
                        for (int j = block_col; j < col_end; ++j) {
                            C.at(i, j) += a_ik * B.at(k, j);
                        }
                    }
                }
            }
        }
    }
    return C;
}

// ---------------------------------------------------------------------------
// 4. SIMD: AVX2 on top of the reordered loop
// ---------------------------------------------------------------------------

// Same i-k-j access pattern, but the innermost loop handles 4 doubles per
// instruction:
//   _mm256_set1_pd(a_ik)          copy the scalar A(i,k) into all 4 lanes
//   _mm256_loadu_pd(&B(k,j))      read 4 consecutive B values
//   _mm256_loadu_pd(&C(i,j))      read 4 running sums
//   _mm256_fmadd_pd(a, b, c)      c = a*b + c, one instruction, one rounding
//   _mm256_storeu_pd(&C(i,j), c)  write the 4 sums back
// which is "C(i,j) += a_ik * B(k,j)" done four times at once. The loads are
// the unaligned ("u") variants because row starts are not guaranteed to land
// on 32-byte boundaries.
Matrix multiply_simd(const Matrix& A, const Matrix& B) {
    Matrix C = create_matrix(A.rows, B.cols);
    int n_rows = A.rows;
    int n_cols = B.cols;
    int n_inner = A.cols;

    // Where the vector loop must stop and the scalar tail takes over.
    int simd_cols_end = n_cols - (n_cols % SIMD_WIDTH);

    for (int i = 0; i < n_rows; ++i) {
        for (int k = 0; k < n_inner; ++k) {
            double a_ik = A.at(i, k);
            __m256d a_ik_vec = _mm256_set1_pd(a_ik);

            int j = 0;
            for (; j < simd_cols_end; j += SIMD_WIDTH) {
                __m256d b_vec = _mm256_loadu_pd(&B.at(k, j));
                __m256d c_vec = _mm256_loadu_pd(&C.at(i, j));
                c_vec = _mm256_fmadd_pd(a_ik_vec, b_vec, c_vec);
                _mm256_storeu_pd(&C.at(i, j), c_vec);
            }

            // Scalar tail: the last (n_cols % 4) columns, too few for a
            // full-width load/store.
            for (; j < n_cols; ++j) {
                C.at(i, j) += a_ik * B.at(k, j);
            }
        }
    }
    return C;
}

// ---------------------------------------------------------------------------
// 5. Parallel: tiling + OpenMP threads
// ---------------------------------------------------------------------------

// multiply_blocked with the outermost loop (over row blocks) split across
// threads. Each thread owns a disjoint set of row blocks, so two threads
// never write the same element of C -- no races, no synchronization.
Matrix multiply_parallel(const Matrix& A, const Matrix& B) {
    Matrix C = create_matrix(A.rows, B.cols);
    int n_rows = A.rows;
    int n_cols = B.cols;
    int n_inner = A.cols;
    int num_row_blocks = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #pragma omp parallel for
    for (int block_index = 0; block_index < num_row_blocks; ++block_index) {
        int block_row = block_index * BLOCK_SIZE;
        int row_end = std::min(block_row + BLOCK_SIZE, n_rows);

        for (int block_col = 0; block_col < n_cols; block_col += BLOCK_SIZE) {
            int col_end = std::min(block_col + BLOCK_SIZE, n_cols);
            for (int block_inner = 0; block_inner < n_inner; block_inner += BLOCK_SIZE) {
                int inner_end = std::min(block_inner + BLOCK_SIZE, n_inner);

                for (int i = block_row; i < row_end; ++i) {
                    for (int k = block_inner; k < inner_end; ++k) {
                        double a_ik = A.at(i, k);
                        for (int j = block_col; j < col_end; ++j) {
                            C.at(i, j) += a_ik * B.at(k, j);
                        }
                    }
                }
            }
        }
    }
    return C;
}

// ---------------------------------------------------------------------------
// 6. Full: tiling + AVX2 + OpenMP
// ---------------------------------------------------------------------------

// Everything at once: row blocks spread over the 12 hardware threads, each
// thread's working set kept tile-sized so it stays in its own L1/L2, and the
// innermost loop running 4 doubles per FMA.
Matrix multiply_full(const Matrix& A, const Matrix& B) {
    Matrix C = create_matrix(A.rows, B.cols);
    int n_rows = A.rows;
    int n_cols = B.cols;
    int n_inner = A.cols;
    int num_row_blocks = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #pragma omp parallel for
    for (int block_index = 0; block_index < num_row_blocks; ++block_index) {
        int block_row = block_index * BLOCK_SIZE;
        int row_end = std::min(block_row + BLOCK_SIZE, n_rows);

        for (int block_col = 0; block_col < n_cols; block_col += BLOCK_SIZE) {
            int col_end = std::min(block_col + BLOCK_SIZE, n_cols);
            // BLOCK_SIZE is a multiple of SIMD_WIDTH, so only a final partial
            // column block can need a scalar tail.
            int simd_col_end = col_end - ((col_end - block_col) % SIMD_WIDTH);

            for (int block_inner = 0; block_inner < n_inner; block_inner += BLOCK_SIZE) {
                int inner_end = std::min(block_inner + BLOCK_SIZE, n_inner);

                for (int i = block_row; i < row_end; ++i) {
                    for (int k = block_inner; k < inner_end; ++k) {
                        double a_ik = A.at(i, k);
                        __m256d a_ik_vec = _mm256_set1_pd(a_ik);

                        int j = block_col;
                        for (; j < simd_col_end; j += SIMD_WIDTH) {
                            __m256d b_vec = _mm256_loadu_pd(&B.at(k, j));
                            __m256d c_vec = _mm256_loadu_pd(&C.at(i, j));
                            c_vec = _mm256_fmadd_pd(a_ik_vec, b_vec, c_vec);
                            _mm256_storeu_pd(&C.at(i, j), c_vec);
                        }

                        for (; j < col_end; ++j) {
                            C.at(i, j) += a_ik * B.at(k, j);
                        }
                    }
                }
            }
        }
    }
    return C;
}

// ---------------------------------------------------------------------------
// Small dense helpers
// ---------------------------------------------------------------------------

// The right-hand side and the solution are n x 1, so every operation on them
// is O(n^2) at worst -- nothing here is ever a meaningful share of the
// runtime, and none of it is what the benchmark is measuring. A plain loop
// keeps these kernel-neutral: no solver gets an accidental head start from
// which multiply happens to be used on the vector work.
Matrix matvec(const Matrix& A, const Matrix& x) {
    Matrix y = create_matrix(A.rows, 1);
    for (int i = 0; i < A.rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < A.cols; ++j) {
            sum += A.at(i, j) * x.at(j, 0);
        }
        y.at(i, 0) = sum;
    }
    return y;
}

double max_abs_difference(const Matrix& left, const Matrix& right) {
    double worst = 0.0;
    for (size_t index = 0; index < left.data.size(); ++index) {
        double difference = std::fabs(left.data[index] - right.data[index]);
        if (difference > worst) {
            worst = difference;
        }
    }
    return worst;
}

double max_abs_value(const Matrix& matrix) {
    double worst = 0.0;
    for (size_t index = 0; index < matrix.data.size(); ++index) {
        double magnitude = std::fabs(matrix.data[index]);
        if (magnitude > worst) {
            worst = magnitude;
        }
    }
    return worst;
}

// Exchange two full rows of a matrix over the column range [col_begin, col_end).
// Partial pivoting only ever swaps whole rows, which is why row-major storage
// makes it cheap: both rows are contiguous runs.
void swap_rows(Matrix& matrix, int row_a, int row_b, int col_begin, int col_end) {
    if (row_a == row_b) {
        return;
    }
    for (int col = col_begin; col < col_end; ++col) {
        std::swap(matrix.at(row_a, col), matrix.at(row_b, col));
    }
}

// Index of the largest-magnitude entry in column `col`, searching rows
// [row_begin, row_end). Choosing the biggest pivot is what keeps the
// multipliers below 1 and elimination numerically stable.
int find_pivot_row(const Matrix& matrix, int col, int row_begin, int row_end) {
    int best_row = row_begin;
    double best_magnitude = std::fabs(matrix.at(row_begin, col));
    for (int row = row_begin + 1; row < row_end; ++row) {
        double magnitude = std::fabs(matrix.at(row, col));
        if (magnitude > best_magnitude) {
            best_magnitude = magnitude;
            best_row = row;
        }
    }
    return best_row;
}

// Ly = Pb, where L is unit lower triangular and sits in the strict lower
// triangle of the factored matrix. O(n^2) -- free next to the factorization.
Matrix forward_substitution(const Matrix& LU, const std::vector<int>& permutation, const Matrix& b) {
    int n = LU.rows;
    Matrix y = create_matrix(n, 1);
    for (int i = 0; i < n; ++i) {
        double sum = b.at(permutation[i], 0);
        for (int j = 0; j < i; ++j) {
            sum -= LU.at(i, j) * y.at(j, 0);
        }
        y.at(i, 0) = sum;   // L(i,i) is 1 by construction, so no divide
    }
    return y;
}

// Ux = y, where U is the upper triangle (diagonal included) of the factored
// matrix. Also O(n^2).
Matrix back_substitution(const Matrix& LU, const Matrix& y) {
    int n = LU.rows;
    Matrix x = create_matrix(n, 1);
    for (int i = n - 1; i >= 0; --i) {
        double sum = y.at(i, 0);
        for (int j = i + 1; j < n; ++j) {
            sum -= LU.at(i, j) * x.at(j, 0);
        }
        x.at(i, 0) = sum / LU.at(i, i);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Method 1: solve by explicit inversion  (the one nobody should use)
// ---------------------------------------------------------------------------

// Gauss-Jordan on the augmented matrix [A | I]: drive the left half to the
// identity and whatever the right half becomes is A^-1. Then x = A^-1 b.
//
// This is in the benchmark to be beaten. It costs about 2n^3 flops against
// elimination's 2/3 n^3 -- three times the work -- and then throws another
// O(n^2) matvec on top. It is also less accurate: forming A^-1 explicitly and
// multiplying introduces rounding that solving in place never commits, which
// shows up in the error column at every size.
Matrix solve_by_inversion(const Matrix& A, const Matrix& b) {
    int n = A.rows;
    const int augmented_cols = 2 * n;

    Matrix augmented = create_matrix(n, augmented_cols);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented.at(i, j) = A.at(i, j);
        }
        augmented.at(i, n + i) = 1.0;
    }

    for (int col = 0; col < n; ++col) {
        int pivot_row = find_pivot_row(augmented, col, col, n);
        swap_rows(augmented, col, pivot_row, 0, augmented_cols);

        double pivot = augmented.at(col, col);
        if (std::fabs(pivot) < SINGULAR_PIVOT_THRESHOLD) {
            std::cerr << "solve_by_inversion: matrix is singular at column " << col << "\n";
            return create_matrix(n, 1);
        }

        // Normalize the pivot row, then clear the pivot column in every other
        // row. Clearing rows above as well as below is exactly the extra work
        // that makes this 3x elimination -- LU stops at the rows below.
        double inverse_pivot = 1.0 / pivot;
        for (int j = col; j < augmented_cols; ++j) {
            augmented.at(col, j) *= inverse_pivot;
        }

        for (int row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            double factor = augmented.at(row, col);
            if (factor == 0.0) {
                continue;
            }
            for (int j = col; j < augmented_cols; ++j) {
                augmented.at(row, j) -= factor * augmented.at(col, j);
            }
        }
    }

    Matrix inverse = create_matrix(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inverse.at(i, j) = augmented.at(i, n + j);
        }
    }

    return matvec(inverse, b);
}

// ---------------------------------------------------------------------------
// Method 2: unblocked LU with partial pivoting  (the honest baseline)
// ---------------------------------------------------------------------------

// Textbook right-looking LU: for each column pick the largest pivot, swap it
// up, scale the column below it into the multipliers of L, then rank-1 update
// the whole trailing submatrix. Correct and standard, but the update is a
// rank-1 operation -- two memory streams for every single multiply-add, no
// data reuse to speak of, and nothing for the cache or the vector units to
// get hold of. This is what the blocked version fixes.
Matrix solve_unblocked_lu(const Matrix& A, const Matrix& b) {
    int n = A.rows;
    Matrix LU = A;

    std::vector<int> permutation(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        permutation[static_cast<size_t>(i)] = i;
    }

    for (int k = 0; k < n; ++k) {
        int pivot_row = find_pivot_row(LU, k, k, n);
        if (pivot_row != k) {
            swap_rows(LU, k, pivot_row, 0, n);
            std::swap(permutation[static_cast<size_t>(k)], permutation[static_cast<size_t>(pivot_row)]);
        }

        double pivot = LU.at(k, k);
        if (std::fabs(pivot) < SINGULAR_PIVOT_THRESHOLD) {
            std::cerr << "solve_unblocked_lu: matrix is singular at column " << k << "\n";
            return create_matrix(n, 1);
        }

        double inverse_pivot = 1.0 / pivot;
        for (int row = k + 1; row < n; ++row) {
            LU.at(row, k) *= inverse_pivot;
        }

        // Rank-1 update of the trailing submatrix.
        for (int row = k + 1; row < n; ++row) {
            double multiplier = LU.at(row, k);
            if (multiplier == 0.0) {
                continue;
            }
            for (int col = k + 1; col < n; ++col) {
                LU.at(row, col) -= multiplier * LU.at(k, col);
            }
        }
    }

    Matrix y = forward_substitution(LU, permutation, b);
    return back_substitution(LU, y);
}

// ---------------------------------------------------------------------------
// Method 3: blocked LU -- the Schur update is a matrix multiply
// ---------------------------------------------------------------------------

typedef Matrix (*MultiplyFunction)(const Matrix&, const Matrix&);

// Same factorization as solve_unblocked_lu, reorganized so the bulk of the
// arithmetic lands in one place: a matrix-matrix multiply, supplied by the
// caller as a plain function pointer.
//
// Per panel of BLOCK_SIZE columns starting at k:
//   1. factor the panel A[k:n, k:k+nb] with unblocked LU + partial pivoting
//   2. apply the panel's row swaps across the full row width
//   3. triangular solve  A[k:k+nb, k+nb:n] = L11^-1 * A[k:k+nb, k+nb:n]
//   4. Schur update      A[k+nb:n, k+nb:n] -= A[k+nb:n, k:k+nb] * A[k:k+nb, k+nb:n]
//
// Steps 1-3 are O(n * nb^2) and O(n^2 * nb). Step 4 is O(n^2 * nb) per panel
// and O(n^3) in total -- at n = 2048 it is roughly 95% of the flops, which is
// why swapping the kernel moves the whole solve time.
Matrix solve_blocked_lu(const Matrix& A, const Matrix& b,
                        Matrix (*matmul)(const Matrix&, const Matrix&)) {
    int n = A.rows;
    Matrix LU = A;

    std::vector<int> permutation(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        permutation[static_cast<size_t>(i)] = i;
    }

    // Pivot row chosen for each column of the current panel, so the swaps can
    // be replayed across the columns outside the panel in step 2.
    std::vector<int> panel_pivots(static_cast<size_t>(BLOCK_SIZE));

    for (int k = 0; k < n; k += BLOCK_SIZE) {
        int panel_width = std::min(BLOCK_SIZE, n - k);
        int panel_end = k + panel_width;

        // --- 1. factor the panel, swapping only inside the panel columns ---
        for (int j = k; j < panel_end; ++j) {
            int pivot_row = find_pivot_row(LU, j, j, n);
            panel_pivots[static_cast<size_t>(j - k)] = pivot_row;

            if (pivot_row != j) {
                swap_rows(LU, j, pivot_row, k, panel_end);
                std::swap(permutation[static_cast<size_t>(j)], permutation[static_cast<size_t>(pivot_row)]);
            }

            double pivot = LU.at(j, j);
            if (std::fabs(pivot) < SINGULAR_PIVOT_THRESHOLD) {
                std::cerr << "solve_blocked_lu: matrix is singular at column " << j << "\n";
                return create_matrix(n, 1);
            }

            double inverse_pivot = 1.0 / pivot;
            for (int row = j + 1; row < n; ++row) {
                LU.at(row, j) *= inverse_pivot;
            }

            // Rank-1 update, but only within the panel's own columns.
            for (int row = j + 1; row < n; ++row) {
                double multiplier = LU.at(row, j);
                if (multiplier == 0.0) {
                    continue;
                }
                for (int col = j + 1; col < panel_end; ++col) {
                    LU.at(row, col) -= multiplier * LU.at(j, col);
                }
            }
        }

        // --- 2. replay those swaps across everything outside the panel ---
        // Order matters: the swaps must be applied in the same sequence the
        // panel factorization performed them.
        for (int j = k; j < panel_end; ++j) {
            int pivot_row = panel_pivots[static_cast<size_t>(j - k)];
            swap_rows(LU, j, pivot_row, 0, k);
            swap_rows(LU, j, pivot_row, panel_end, n);
        }

        int trailing_size = n - panel_end;
        if (trailing_size <= 0) {
            continue;
        }

        // --- 3. triangular solve against the unit lower-triangular L11 ---
        // A12 <- L11^-1 * A12, done in place by forward substitution across
        // all trailing columns at once.
        for (int i = k + 1; i < panel_end; ++i) {
            for (int t = k; t < i; ++t) {
                double multiplier = LU.at(i, t);
                if (multiplier == 0.0) {
                    continue;
                }
                for (int col = panel_end; col < n; ++col) {
                    LU.at(i, col) -= multiplier * LU.at(t, col);
                }
            }
        }

        // --- 4. Schur update: this is the matmul, and it dominates ---
        //
        // `Matrix` is a flat buffer with no stride / leading-dimension
        // support, so there is no way to hand a submatrix view to the six
        // kernels. Rather than modify them -- which would wreck the point of
        // the comparison -- the two operand blocks are copied into fresh
        // Matrix objects, the unchanged kernel is called, and the result is
        // subtracted back in a simple loop. The copies move O(m^2) elements
        // against O(m^2 * nb) flops of real work, i.e. a 1/64 sliver at full
        // panel width, so they do not distort the timings. That was the
        // deliberate tradeoff: pay a little copying to keep all six kernels
        // completely untouched.
        Matrix A21 = create_matrix(trailing_size, panel_width);
        for (int i = 0; i < trailing_size; ++i) {
            for (int j = 0; j < panel_width; ++j) {
                A21.at(i, j) = LU.at(panel_end + i, k + j);
            }
        }

        Matrix A12 = create_matrix(panel_width, trailing_size);
        for (int i = 0; i < panel_width; ++i) {
            for (int j = 0; j < trailing_size; ++j) {
                A12.at(i, j) = LU.at(k + i, panel_end + j);
            }
        }

        // The kernels already read A.rows, A.cols and B.cols separately, so
        // this skinny (m x nb) * (nb x m) product works as-is -- no shape
        // special-casing was needed.
        Matrix update = matmul(A21, A12);

        for (int i = 0; i < trailing_size; ++i) {
            for (int j = 0; j < trailing_size; ++j) {
                LU.at(panel_end + i, panel_end + j) -= update.at(i, j);
            }
        }
    }

    Matrix y = forward_substitution(LU, permutation, b);
    return back_substitution(LU, y);
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

struct SolveResult {
    std::string name;
    double elapsed_ms = 0.0;
    double residual = 0.0;
    double error = 0.0;
    bool passed = false;
};

// A = random in [-1, 1], plus n on every diagonal entry. Strictly diagonally
// dominant, so non-singular and well conditioned by construction.
Matrix create_test_matrix(int n) {
    Matrix A = create_matrix(n, n);
    fill_random(A, RANDOM_SEED_MATRIX);
    double boost = DIAGONAL_BOOST_PER_N * static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
        A.at(i, i) += boost;
    }
    return A;
}

// The tolerance has to grow with the problem: elimination accumulates roughly
// n rounding errors per output component, each about eps * ||b||. A fixed
// 1e-9 would pass at n = 128 and fail at n = 2048 for reasons that have
// nothing to do with a bug.
double residual_tolerance(int n, double b_magnitude) {
    double epsilon = std::numeric_limits<double>::epsilon();
    return RESIDUAL_TOLERANCE_UNITS * epsilon * static_cast<double>(n) * b_magnitude;
}

void print_table_header() {
    std::cout << std::left
              << std::setw(7) << "Size"
              << std::setw(22) << "Method"
              << std::setw(14) << "Time(ms)"
              << std::setw(13) << "Residual"
              << std::setw(13) << "Error"
              << std::setw(11) << "Speedup"
              << std::setw(7) << "Check"
              << "\n";
    std::cout << std::string(87, '-') << "\n";
}

void print_table_row(int n, const SolveResult& result, double speedup) {
    std::cout << std::left << std::setw(7) << n
              << std::setw(22) << result.name
              << std::fixed << std::setprecision(3) << std::setw(14) << result.elapsed_ms
              << std::scientific << std::setprecision(2)
              << std::setw(13) << result.residual
              << std::setw(13) << result.error
              << std::fixed << std::setprecision(2) << std::setw(11) << speedup
              << std::setw(7) << (result.passed ? "PASS" : "FAIL")
              << "\n";
}

int main() {
    MultiplyFunction kernels[6] = {
        multiply_naive,
        multiply_reordered,
        multiply_blocked,
        multiply_simd,
        multiply_parallel,
        multiply_full
    };
    std::string kernel_names[6] = {
        "naive",
        "reordered",
        "blocked",
        "simd",
        "parallel",
        "full"
    };

    std::cout << "Linear solver benchmark: Ax = b (Windows / x86-64, AVX2 + FMA)\n";
    std::cout << "Blocked-LU rows run the identical algorithm and differ only in\n";
    std::cout << "which matmul kernel performs the Schur update.\n";
#ifdef _OPENMP
    std::cout << "OpenMP enabled, max threads = " << omp_get_max_threads() << "\n\n";
#else
    std::cout << "OpenMP NOT enabled (rebuild with -fopenmp)\n\n";
#endif

    std::ofstream csv_file(RESULTS_CSV_PATH);
    csv_file << "size,method,elapsed_ms,residual,error,speedup,check\n";

    print_table_header();

    for (int size_index = 0; size_index < NUM_SOLVE_SIZES; ++size_index) {
        int n = SOLVE_SIZES[size_index];

        Matrix A = create_test_matrix(n);

        // Pick the answer first, then manufacture the question. That way the
        // error column compares against a solution that is known exactly,
        // rather than against another solver's output.
        Matrix x_true = create_matrix(n, 1);
        fill_random(x_true, RANDOM_SEED_SOLUTION);
        Matrix b = matvec(A, x_true);

        double tolerance = residual_tolerance(n, max_abs_value(b));

        SolveResult results[NUM_METHODS];

        for (int method_index = 0; method_index < NUM_METHODS; ++method_index) {
            Matrix x;

            std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
            if (method_index == 0) {
                x = solve_by_inversion(A, b);
            } else if (method_index == 1) {
                x = solve_unblocked_lu(A, b);
            } else {
                x = solve_blocked_lu(A, b, kernels[method_index - 2]);
            }
            std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

            SolveResult& result = results[method_index];
            if (method_index == 0) {
                result.name = "inversion";
            } else if (method_index == 1) {
                result.name = "unblocked-lu";
            } else {
                result.name = "blocked-lu/" + kernel_names[method_index - 2];
            }
            result.elapsed_ms = elapsed.count();
            result.residual = max_abs_difference(matvec(A, x), b);
            result.error = max_abs_difference(x, x_true);
            result.passed = (result.residual <= tolerance);
        }

        // Speedup is measured against the slowest method at this size, which
        // is whichever of inversion / blocked-lu-naive lost the race.
        double slowest_ms = results[0].elapsed_ms;
        for (int method_index = 1; method_index < NUM_METHODS; ++method_index) {
            slowest_ms = std::max(slowest_ms, results[method_index].elapsed_ms);
        }

        for (int method_index = 0; method_index < NUM_METHODS; ++method_index) {
            const SolveResult& result = results[method_index];
            double speedup = slowest_ms / result.elapsed_ms;

            print_table_row(n, result, speedup);

            csv_file << n << ","
                     << result.name << ","
                     << std::fixed << std::setprecision(6) << result.elapsed_ms << ","
                     << std::scientific << std::setprecision(6) << result.residual << ","
                     << result.error << ","
                     << std::fixed << std::setprecision(6) << speedup << ","
                     << (result.passed ? "PASS" : "FAIL") << "\n";
        }

        std::cout << std::string(87, '-') << "\n";
        std::cout.flush();
    }

    csv_file.close();
    std::cout << "\nResults written to " << RESULTS_CSV_PATH << "\n";
    return 0;
}
