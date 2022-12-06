#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
typedef ssize_t ptrdiff_t;
struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }

  scalar_t *ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t> &x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE)
    throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t *out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = val;
}

void Fill(CudaArray *out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from
// strides
__device__ size_t IndexToOffset(size_t flatten_idx, CudaVec shape,
                                CudaVec strides, size_t offset) {
  for (int i = shape.size - 1; i >= 0; --i) {
    size_t idx = flatten_idx % shape.data[i];
    offset += idx * strides.data[i];
    flatten_idx /= shape.data[i];
  }
  return offset;
}

__global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size,
                              CudaVec shape, CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a
   * single entry in the non-compact input a, to the corresponding item (at
   * location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past
   * passing to CUDA kernel) strides: vector of strides of a array offset:
   * offset of a array
   */
  ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if (gid < size)
    out[gid] = a[IndexToOffset(gid, shape, strides, offset)];
  /// END YOUR SOLUTION
}

void Compact(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will
   * primarily call the relevant CUDA kernel.  In this case, we illustrate how
   * you should set this up (i.e., we give you the code for this fuction, and
   * also the prototype for the CompactKernel() function).  For the functions
   * after this, however, you'll need to define these kernels as you see fit to
   * execute the underlying function.
   *
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being
   * compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(
      a.ptr, out->ptr, out->size, VecToCuda(shape), VecToCuda(strides), offset);
}

__global__ void EwiseSetKernel(const scalar_t *a, scalar_t *out, size_t size,
                               CudaVec shape, CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the EwiseSet opeation. This should effectively map a
   * single entry in the compact input a (at location gid), to the corresponding
   * item in the non-compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past
   * passing to CUDA kernel) strides: vector of strides of out array offset:
   * offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if (gid < size)
    out[IndexToOffset(gid, shape, strides, offset)] = a[gid];
  /// END YOUR SOLUTION
}

void EwiseSetitem(const CudaArray &a, CudaArray *out,
                  std::vector<int32_t> shape, std::vector<int32_t> strides,
                  size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want
   * to implement a EwiseSetitemKernel() function, similar to those above, that
   * will do the actual work.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being
   * compact)
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetKernel<<<dim.grid, dim.block>>>(
      a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

__global__ void ScalarSetKernel(const scalar_t val, scalar_t *out, size_t size,
                                CudaVec shape, CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the EwiseSet opeation. This should effectively map
   * scalar to the corresponding item in the non-compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past
   * passing to CUDA kernel) strides: vector of strides of out array offset:
   * offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size)
    out[IndexToOffset(gid, shape, strides, offset)] = val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray *out,
                   std::vector<int32_t> shape, std::vector<int32_t> strides,
                   size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note
   * be the same as out.size, because out is a non-compact subset array);  it
   * _will_ be the same as the product of items in shape, but covenient to just
   * pass it here. val: scalar value to write to out: non-compact array whose
   * items are to be written shape: shapes of each dimension of out strides:
   * strides of the out array offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetKernel<<<dim.grid, dim.block>>>(
      val, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous
 * elementise and and scalar operators for the following functions.  See the
 * numpy backend for examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define
 * these functions (however you want to do so, as long as the functions match
 * the proper) signatures above.
 */

/// BEGIN YOUR SOLUTION
// Mul
__global__ void EwiseMulKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  /**
   * Mul two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray &a, scalar_t val, CudaArray *out) {
  /**
   * Mul a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
// Div
__global__ void EwiseDivKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  /**
   * Div two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray &a, scalar_t val, CudaArray *out) {
  /**
   * Div a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
// Pow
__global__ void ScalarPowKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = pow(a[gid], val);
}

void ScalarPower(const CudaArray &a, scalar_t val, CudaArray *out) {
  /**
   * Div a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
// Max
__global__ void EwiseMaximumKernel(const scalar_t *a, const scalar_t *b,
                                   scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = max(a[gid], b[gid]);
}

void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  /**
   * Maximum two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr,
                                              out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t *a, scalar_t val,
                                    scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = max(a[gid], val);
}

void ScalarMaximum(const CudaArray &a, scalar_t val, CudaArray *out) {
  /**
   * Maximum a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
// Eq
__global__ void EwiseEqKernel(const scalar_t *a, const scalar_t *b,
                              scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] == b[gid];
}

void EwiseEq(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  /**
   * Eq two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                               size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] == val;
}

void ScalarEq(const CudaArray &a, scalar_t val, CudaArray *out) {
  /**
   * Eq a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
// Ge
__global__ void EwiseGeKernel(const scalar_t *a, const scalar_t *b,
                              scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] >= b[gid];
}

void EwiseGe(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  /**
   * Ge two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                               size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] >= val;
}

void ScalarGe(const CudaArray &a, scalar_t val, CudaArray *out) {
  /**
   * Ge a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
// Log
__global__ void EwiseLogKernel(const scalar_t *a, scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = log(a[gid]);
}

void EwiseLog(const CudaArray &a, CudaArray *out) {
  /**
   * Log a CUDA array.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}
// Exp
__global__ void EwiseExpKernel(const scalar_t *a, scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = exp(a[gid]);
}

void EwiseExp(const CudaArray &a, CudaArray *out) {
  /**
   * Exp a CUDA array.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}
// Tanh
__global__ void EwiseTanhKernel(const scalar_t *a, scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = tanh(a[gid]);
}

void EwiseTanh(const CudaArray &a, CudaArray *out) {
  /**
   * Tanh a CUDA array.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}
/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

using func_t = scalar_t (*)(scalar_t, scalar_t);
__device__ scalar_t Add(scalar_t x, scalar_t y) { return x + y; }
__device__ scalar_t Mul(scalar_t x, scalar_t y) { return x * y; }
__device__ scalar_t Div(scalar_t x, scalar_t y) { return x / y; }
__device__ scalar_t Power(scalar_t x, scalar_t y) { return pow(x, y); }
__device__ scalar_t Negate(scalar_t x, scalar_t y) { return -x; }
__device__ scalar_t Tanh(scalar_t x, scalar_t y) { return tanh(x); }
__device__ scalar_t Exp(scalar_t x, scalar_t y) { return exp(x); }
__device__ scalar_t ReLU(scalar_t x, scalar_t y) { return x > 0 ? x : 0; }
__device__ scalar_t Log(scalar_t x, scalar_t y) { return log(x); }
__global__ void MatmulKernelFused(scalar_t **tensor_input,
                                  scalar_t *scalar_input, scalar_t *out,
                                  int32_t M, int32_t N, int32_t P,
                                  func_t *ewise_ops, int ewise_ops_num,
                                  int *is_scalar_op) {
  const int S = 32, L = 32, V = 2;
  __shared__ float sA[S][L], sB[S][L];
  float C[V][V]{0};
  float A[V], B[V];
  const scalar_t *a = tensor_input[0];
  const scalar_t *b = tensor_input[1];
  for (int ko = 0; ko < N; ko += S) {
    __syncthreads();
    for (int i = 0; i < S; i += blockDim.x) {
      for (int j = 0; j < L; j += blockDim.y) {
        if ((blockIdx.x * L + j + threadIdx.y) < M &&
            (ko + i + threadIdx.x) < N) {
          sA[i + threadIdx.x][j + threadIdx.y] =
              a[(blockIdx.x * L + j + threadIdx.y) * N +
                (ko + i + threadIdx.x)];
        } else {
          sA[i + threadIdx.x][j + threadIdx.y] = 0;
        }

        if ((ko + i + threadIdx.x) < N &&
            (blockIdx.y * L + j + threadIdx.y) < P) {
          sB[i + threadIdx.x][j + threadIdx.y] =
              b[(ko + i + threadIdx.x) * P +
                (blockIdx.y * L + j + threadIdx.y)];
        } else {
          sB[i + threadIdx.x][j + threadIdx.y] = 0;
        }
      }
    }
    __syncthreads();
    for (int ki = 0; ki < S; ki++) {
      for (int v = 0; v < V; v++) {
        A[v] = sA[ki][threadIdx.x * V + v];
      }
      for (int v = 0; v < V; v++) {
        B[v] = sB[ki][threadIdx.y * V + v];
      }
      for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
          C[i][j] += A[i] * B[j];
        }
      }
    }
  }
  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++) {
      if (((blockIdx.x * blockDim.x + threadIdx.x) * V + i) < M &&
          ((blockIdx.y * blockDim.y + threadIdx.y) * V + j) < P) {
        scalar_t tmp = C[i][j];
        for (int k = 0; k < ewise_ops_num; k++) {
          if (is_scalar_op[k]) {
            tmp = ewise_ops[k](tmp, scalar_input[k + 2]);
          } else {
            tmp = ewise_ops[k](
                tmp,
                tensor_input[k + 2]
                            [((blockIdx.x * blockDim.x + threadIdx.x) * V + i) *
                                 P +
                             (blockIdx.y * blockDim.y + threadIdx.y) * V + j]);
          }
        }
        out[((blockIdx.x * blockDim.x + threadIdx.x) * V + i) * P +
            (blockIdx.y * blockDim.y + threadIdx.y) * V + j] = tmp;
      }
    }
  }
}
void MatmulFused(const CudaArray &a, const CudaArray &b, CudaArray *out,
                 int32_t M, int32_t N, int32_t P,
                 std::vector<std::string> ewise_ops,
                 std::vector<CudaArray> ewise_tensor_input,
                 std::vector<scalar_t> ewise_scalar_input,
                 std::vector<int> is_scalar_op) {
  size_t grid_x = (M + 32 - 1) / 32;
  size_t grid_y = (N + 32 - 1) / 32;
  dim3 gridDim(grid_x, grid_y, 1);
  dim3 blockDim(16, 16, 1);
  // handle fused ewise
  int ewise_ops_num = ewise_ops.size();
  const std::unordered_map<std::string, func_t> ewise_func_map = {
      {"EwiseAdd", Add},      {"EwiseMul", Mul},  {"EwiseDiv", Div},
      {"AddScalar", Add},     {"MulScalar", Mul}, {"DivScalar", Div},
      {"PowerScalar", Power}, {"Negate", Negate}, {"Tanh", Tanh},
      {"Exp", Exp},           {"ReLU", ReLU},     {"Log", Log}};
  std::vector<func_t> ewise_ops_func;
  for (const auto &op : ewise_ops) {
    ewise_ops_func.push_back(ewise_func_map.at(op));
  }

  scalar_t** tensor_input_cuda;
  cudaMalloc(&tensor_input_cuda, (ewise_ops_num + 2) * sizeof(scalar_t*));
  CudaArray scalar_input_cuda(ewise_ops_num + 2);
  scalar_t **host_tensor_input_ptr = (scalar_t **)std::malloc((ewise_ops_num+2) * sizeof(scalar_t *));
  scalar_t *host_scalar_input_ptr = (scalar_t *)std::malloc((ewise_ops_num+2) * sizeof(scalar_t));
  host_tensor_input_ptr[0] = a.ptr;
  host_tensor_input_ptr[1] = b.ptr;
  for (int i = 0; i < ewise_ops_num; i++) {
    if (is_scalar_op[i]) {
      host_scalar_input_ptr[i + 2] = ewise_scalar_input[i];
    } else {
      host_tensor_input_ptr[i + 2] = ewise_tensor_input[i].ptr;
    }
  }
  cudaError_t err =
      cudaMemcpy(host_scalar_input_ptr, scalar_input_cuda.ptr, (ewise_ops_num + 2) * sizeof(scalar_t), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(err));
  err =
      cudaMemcpy(host_tensor_input_ptr, tensor_input_cuda, (ewise_ops_num + 2) * sizeof(scalar_t*), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(err));
  MatmulKernelFused<<<gridDim, blockDim>>>(tensor_input_cuda, scalar_input_cuda.ptr, out->ptr, M, N, P, ewise_ops_func.data(), ewise_ops_num, is_scalar_op.data());
  /// END YOUR SOLUTION
}

__global__ void MatmulKernel(const scalar_t *a, const scalar_t *b,
                             scalar_t *out, int32_t M, int32_t N, int32_t P) {
  const int S = 32, L = 32, V = 2;
  __shared__ float sA[S][L], sB[S][L];
  float C[V][V]{0};
  float A[V], B[V];
  for (int ko = 0; ko < N; ko += S) {
    __syncthreads();
    for (int i = 0; i < S; i += blockDim.x) {
      for (int j = 0; j < L; j += blockDim.y) {
        if ((blockIdx.x * L + j + threadIdx.y) < M &&
            (ko + i + threadIdx.x) < N) {
          sA[i + threadIdx.x][j + threadIdx.y] =
              a[(blockIdx.x * L + j + threadIdx.y) * N +
                (ko + i + threadIdx.x)];
        } else {
          sA[i + threadIdx.x][j + threadIdx.y] = 0;
        }

        if ((ko + i + threadIdx.x) < N &&
            (blockIdx.y * L + j + threadIdx.y) < P) {
          sB[i + threadIdx.x][j + threadIdx.y] =
              b[(ko + i + threadIdx.x) * P +
                (blockIdx.y * L + j + threadIdx.y)];
        } else {
          sB[i + threadIdx.x][j + threadIdx.y] = 0;
        }
      }
    }
    __syncthreads();
    for (int ki = 0; ki < S; ki++) {
      for (int v = 0; v < V; v++) {
        A[v] = sA[ki][threadIdx.x * V + v];
      }
      for (int v = 0; v < V; v++) {
        B[v] = sB[ki][threadIdx.y * V + v];
      }
      for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
          C[i][j] += A[i] * B[j];
        }
      }
    }
  }
  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++) {
      if (((blockIdx.x * blockDim.x + threadIdx.x) * V + i) < M &&
          ((blockIdx.y * blockDim.y + threadIdx.y) * V + j) < P) {
        out[((blockIdx.x * blockDim.x + threadIdx.x) * V + i) * P +
            (blockIdx.y * blockDim.y + threadIdx.y) * V + j] = C[i][j];
      }
    }
  }
}

void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, int32_t M,
            int32_t N, int32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You
   * will want to look at the lecture and notes on GPU-based linear algebra to
   * see how to do this.  Since ultimately mugrade is just evaluating
   * correctness, you _can_ implement a version that simply parallelizes over
   * (i,j) entries in the output array.  However, to really get the full benefit
   * of this problem, we would encourage you to use cooperative fetching, shared
   * memory register tiling, and other ideas covered in the class notes.  Note
   * that unlike the tiled matmul function in the CPU backend, here you should
   * implement a single function that works across all size matrices, whether or
   * not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel
   * call, and you should implement the logic in a separate MatmulKernel() call.
   *
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  size_t grid_x = (M + 32 - 1) / 32;
  size_t grid_y = (N + 32 - 1) / 32;
  dim3 gridDim(grid_x, grid_y, 1);
  dim3 blockDim(16, 16, 1);
  MatmulKernel<<<gridDim, blockDim>>>(a.ptr, b.ptr, out->ptr, M, N, P);

  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out,
                                size_t reduce_size, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t val = a[gid * reduce_size];
    for (size_t i = 0; i < reduce_size; ++i) {
      val = max(val, a[gid * reduce_size + i]);
    }
    out[gid] = val;
  }
}

void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though
   * it is inefficient, for simplicity you can perform each reduction in a
   * single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size,
                                           out->size);
  /// END YOUR SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out,
                                size_t reduce_size, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t val = 0;
    for (size_t i = 0; i < reduce_size; ++i) {
      val += a[gid * reduce_size + i];
    }
    out[gid] = val;
  }
}

void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again,
   * for simplicity you can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size,
                                           out->size);
  /// END YOUR SOLUTION
}

} // namespace cuda
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(),
                   numpy_strides.begin(),
                   [](size_t &c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t *host_ptr = (scalar_t *)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0)
      throw std::bad_alloc();
    cudaError_t err =
        cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void *p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset,
                                 deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out) {
    cudaError_t err = cudaMemcpy(out->ptr, a.request().ptr,
                                 out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_fused", MatmulFused);
  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}