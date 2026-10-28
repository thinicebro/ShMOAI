#include "ShRotate.cuh"

#include <chrono>

#include "evaluate.cuh"
#include "ntt.cuh"
#include "polymath.cuh"
#include "prng.cuh"
#include "rns.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using phantom::arith::uint128_t;

namespace moai {

// ---------------------------------------------------------------------------
// device kernel
// ---------------------------------------------------------------------------

/**
 * Shared-a key-switch inner product: out = sum_{g=0}^{beta-1} d_g * key_g.
 *   t_mod_up : gadget decomposition of the (rotated) shared a-part, produced by
 *              DRNSTool::modup; digit g is at t_mod_up + g * (size_QlP * n), in the
 *              QlP basis (local component order: Ql primes, then special primes).
 *   key_polys: gadget polys in the full QP basis (global component order); only the
 *              components belonging to QlP are read (global index twr).
 *   key_g    : key_polys + g * key_stride.
 * All values are < q (< 2^60), so beta << 2^60 products fit the 128-bit accumulator.
 */
__global__ void sh_ks_inner_prod_kernel(uint64_t *out, const uint64_t *t_mod_up, const uint64_t *key_polys,
                                        const DModulus *modulus_QP, size_t beta, size_t key_stride, size_t n,
                                        size_t size_Ql, size_t size_QlP, size_t size_Q) {
    const size_t poly_comp = size_QlP * n;
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_comp; tid += blockDim.x * gridDim.x) {
        const size_t nid = tid / n; // local QlP component
        const size_t twr = (nid >= size_Ql) ? size_Q + (nid - size_Ql) : nid; // global component
        const DModulus mod = modulus_QP[twr];
        uint128_t acc = {0, 0};
        for (size_t g = 0; g < beta; g++) {
            uint128_t prod = phantom::arith::multiply_uint64_uint64(
                    t_mod_up[g * poly_comp + tid], key_polys[g * key_stride + twr * n + (tid % n)]);
            phantom::arith::add_uint128_uint128(acc, prod, acc);
        }
        out[tid] = phantom::arith::barrett_reduce_uint128_uint64(acc, mod.value(), mod.const_ratio());
    }
}

// ---------------------------------------------------------------------------
// shared-a Galois key generation (Sh-SWKGen, paper Sec. 6.1)
// ---------------------------------------------------------------------------

SharedAGaloisKey SharedAKeyGen::gen_shared_galois_key(int step, const cudaStream_t &stream) {
    const auto keygen_t0 = std::chrono::system_clock::now();
    struct KeygenTimer {
        double &acc;
        const std::chrono::system_clock::time_point t0;
        const cudaStream_t &s;
        ~KeygenTimer() {
            cudaStreamSynchronize(s);
            acc += std::chrono::duration<double>(std::chrono::system_clock::now() - t0).count();
        }
    } keygen_timer{galois_keygen_sec_, keygen_t0, stream};

    auto &key_parms = context_.get_context_data(0).parms();
    const size_t N = key_parms.poly_modulus_degree();
    const size_t size_P = key_parms.special_modulus_size();
    const size_t size_QP = key_parms.coeff_modulus().size();
    const size_t size_Q = size_QP - size_P;
    const size_t dnum = size_Q / size_P;
    const size_t poly_comp = size_QP * N;

    // memory preflight: dnum * (n+1) polys in the QP basis.
    // NOTE: phantom sets the mempool release threshold to UINT64_MAX, so cudaMemGetInfo
    // reports retained (but reusable) pool memory as used. A trial allocation is the only
    // reliable check.
    {
        const size_t key_bytes = dnum * (n_ + 1) * poly_comp * sizeof(uint64_t);
        void *probe = nullptr;
        const cudaError_t err = cudaMallocAsync(&probe, key_bytes, stream);
        if (err != cudaSuccess) {
            cudaGetLastError(); // clear the error state
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            throw runtime_error("SharedAKeyGen::gen_shared_galois_key: insufficient GPU memory for batch n=" +
                                to_string(n_) + " (need ~" + to_string(key_bytes >> 20) + " MiB, free " +
                                to_string(free_b >> 20) + " MiB)");
        }
        cudaFreeAsync(probe, stream);
        cudaStreamSynchronize(stream);
    }

    SharedAGaloisKey key;
    key.step_ = step;
    key.n_ = n_;
    key.dnum_ = dnum;

    // Galois permutation tables for this step's automorphism (owned by the key)
    auto elts = get_elts_from_steps(vector<int>{step}, N);
    key.galois_elt_ = elts[0];
    const bool is_bfv = key_parms.scheme() == scheme_type::bfv;
    key.tool_ = make_unique<PhantomGaloisTool>(elts, static_cast<int>(phantom::arith::get_power_of_two(N)), stream,
                                               is_bfv);

    key.A_ = make_cuda_auto_ptr<uint64_t>(dnum * poly_comp, stream);
    key.B_ = make_cuda_auto_ptr<uint64_t>(dnum * n_ * poly_comp, stream);

    const DModulus *base_rns = context_.gpu_rns_tables().modulus();
    auto &ntt_tables = context_.gpu_rns_tables();
    auto &rns_tool_top = context_.get_context_data(0).gpu_rns_tool();

    // sigma(sk'_i), one per member
    vector<cuda_auto_ptr<uint64_t>> sigma_sk;
    sigma_sk.reserve(n_);
    for (size_t i = 0; i < n_; i++) {
        sigma_sk.push_back(make_cuda_auto_ptr<uint64_t>(poly_comp, stream));
        key.tool_->apply_galois_ntt(const_cast<uint64_t *>(aux_sks_[i]->secret_key_poly()), size_QP, 0,
                                    sigma_sk[i].get(), stream);
    }

    const size_t grid = poly_comp / blockDimGlb.x;
    auto e = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
    auto seed = make_cuda_auto_ptr<uint8_t>(global_variables::prng_seed_byte_count, stream);

    for (size_t g = 0; g < dnum; g++) {
        // A_g : uniform poly in R_QP, values used directly as NTT form
        random_bytes(seed.get(), global_variables::prng_seed_byte_count, stream);
        sample_uniform_poly<<<grid, blockDimGlb, 0, stream>>>(key.A_.get() + g * poly_comp, seed.get(), base_rns, N,
                                                              size_QP);
        for (size_t i = 0; i < n_; i++) {
            // B_{g,i} = -(A_g * sk'_i + e)
            random_bytes(seed.get(), global_variables::prng_seed_byte_count, stream);
            sample_error_poly<<<grid, blockDimGlb, 0, stream>>>(e.get(), seed.get(), base_rns, N, size_QP);
            nwt_2d_radix8_forward_inplace(e.get(), ntt_tables, size_QP, 0, stream);
            multiply_and_add_negate_rns_poly<<<grid, blockDimGlb, 0, stream>>>(
                    key.A_.get() + g * poly_comp, aux_sks_[i]->secret_key_poly(), e.get(), base_rns,
                    key.B_.get() + (g * n_ + i) * poly_comp, N, size_QP);
        }
    }

    // B_{g,i} += P * (CRT unit of prime g) * sigma(sk'_i), one launch per member
    auto B_ptrs_dev = make_cuda_auto_ptr<uint64_t *>(dnum, stream);
    vector<uint64_t *> B_ptrs_host(dnum);
    const size_t grid_tm = (N * dnum * size_P) / blockDimGlb.x;
    for (size_t i = 0; i < n_; i++) {
        for (size_t g = 0; g < dnum; g++) {
            B_ptrs_host[g] = key.B_.get() + (g * n_ + i) * poly_comp;
        }
        // H2D copy of the pointer table must go through stream s: the phantom default
        // stream is cudaStreamNonBlocking, so a legacy-stream cudaMemcpy is not ordered
        // against subsequent kernels on s and can race the mempool reuse of B_ptrs_dev.
        cudaMemcpyAsync(B_ptrs_dev.get(), B_ptrs_host.data(), dnum * sizeof(uint64_t *), cudaMemcpyHostToDevice,
                        stream);
        cudaStreamSynchronize(stream); // keep B_ptrs_host alive until the copy landed
        multiply_temp_mod_and_add_rns_poly<<<grid_tm, blockDimGlb, 0, stream>>>(
                sigma_sk[i].get(), const_cast<const uint64_t *const *>(B_ptrs_dev.get()), base_rns, B_ptrs_dev.get(),
                N, dnum, size_P, rns_tool_top.bigP_mod_q(), rns_tool_top.bigP_mod_q_shoup());
    }

    return key;
}

// ---------------------------------------------------------------------------
// shared-a Galois key cache access (摊销模式) / scratch access (不摊销模式)
// ---------------------------------------------------------------------------

SharedAKeyGen::~SharedAKeyGen() = default;

const SharedAGaloisKey &SharedAKeyGen::acquire_shared_galois_key(int step, const cudaStream_t &stream) {
    // 不摊销 (default): regenerate every time into the scratch slot; the previous
    // content is released on reassignment, so peak memory stays at one key.
    if (galois_cache_budget_ == 0) {
        scratch_key_ = std::make_unique<SharedAGaloisKey>(gen_shared_galois_key(step, stream));
        return *scratch_key_;
    }
    // 摊销: reuse the cached key when this step was already generated (every chunk
    // of a batch bootstrap uses the same deterministic step sequence).
    auto it = galois_cache_.find(step);
    if (it != galois_cache_.end()) {
        return *it->second;
    }
    SharedAGaloisKey key = gen_shared_galois_key(step, stream);
    const size_t N = context_.get_context_data(0).parms().poly_modulus_degree();
    const size_t size_QP = context_.get_context_data(0).parms().coeff_modulus().size();
    const size_t key_bytes = key.dnum_ * (key.n_ + 1) * size_QP * N * sizeof(uint64_t);
    if (galois_cache_bytes_ + key_bytes > galois_cache_budget_) {
        // over budget: fall back to the scratch path for this step
        scratch_key_ = std::make_unique<SharedAGaloisKey>(std::move(key));
        return *scratch_key_;
    }
    galois_cache_bytes_ += key_bytes;
    auto inserted = galois_cache_.emplace(step, std::make_unique<SharedAGaloisKey>(std::move(key)));
    return *inserted.first->second;
}

// ---------------------------------------------------------------------------
// shared-a rotation (Sh-Rot, paper Sec. 6.1)
// ---------------------------------------------------------------------------

void shared_a_rotate(const PhantomContext &context, const SharedACiphertextBatch &in,
                     const SharedAGaloisKey &key, SharedACiphertextBatch &out, const cudaStream_t &stream) {
    if (!in.is_valid()) {
        throw invalid_argument("shared_a_rotate: input batch is not allocated");
    }
    if (!in.is_ntt_form_) {
        throw invalid_argument("shared_a_rotate: only NTT form is supported");
    }
    if (in.n_ != key.n_ || key.tool_ == nullptr) {
        throw invalid_argument("shared_a_rotate: batch size does not match the Galois key");
    }

    const size_t n = in.n_;
    const size_t chain_index = in.chain_index_;
    auto &key_parms = context.get_context_data(0).parms();
    const size_t N = key_parms.poly_modulus_degree();
    const size_t size_P = key_parms.special_modulus_size();
    const size_t size_QP = key_parms.coeff_modulus().size();
    const size_t size_Q = size_QP - size_P;
    const size_t size_Ql = in.size_Ql_;
    const size_t size_QlP = size_Ql + size_P;
    const size_t poly_comp = size_QlP * N;
    const size_t size_Ql_n = size_Ql * N;

    auto &rns_tool = context.get_context_data(chain_index).gpu_rns_tool();
    auto &ntt_tables = context.gpu_rns_tables();
    const auto scheme = key_parms.scheme();
    const size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    // 1. apply the automorphism to the shared a-part (once) and raise it to QlP (once)
    auto sigma_a = make_cuda_auto_ptr<uint64_t>(size_Ql_n, stream);
    key.tool_->apply_galois_ntt(in.a_.get(), size_Ql, 0, sigma_a.get(), stream);
    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * poly_comp, stream);
    rns_tool.modup(t_mod_up.get(), sigma_a.get(), ntt_tables, scheme, stream);

    out.allocate(context, chain_index, n, in.scale_, stream);
    const DModulus *mod_QP = ntt_tables.modulus();
    const size_t grid = poly_comp / blockDimGlb.x;

    // 2. the new shared a-part: a' = moddown( sum_g d_g * A_g ), identical for all members
    auto scratch_qp = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
    auto scratch_q = make_cuda_auto_ptr<uint64_t>(size_Ql_n, stream);
    sh_ks_inner_prod_kernel<<<grid, blockDimGlb, 0, stream>>>(scratch_qp.get(), t_mod_up.get(), key.A_.get(), mod_QP,
                                                              beta, size_QP * N, N, size_Ql, size_QlP, size_Q);
    rns_tool.moddown_from_NTT(out.a_.get(), scratch_qp.get(), ntt_tables, scheme, stream);

    // 3. per member: b'_i = sigma(b_i) + moddown( sum_g d_g * B_{g,i} )
    for (size_t i = 0; i < n; i++) {
        key.tool_->apply_galois_ntt(in.b_[i].get(), size_Ql, 0, out.b_[i].get(), stream);
        sh_ks_inner_prod_kernel<<<grid, blockDimGlb, 0, stream>>>(
                scratch_qp.get(), t_mod_up.get(), key.B_.get() + i * size_QP * N, mod_QP, beta, n * size_QP * N, N,
                size_Ql, size_QlP, size_Q);
        rns_tool.moddown_from_NTT(scratch_q.get(), scratch_qp.get(), ntt_tables, scheme, stream);
        add_rns_poly<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(out.b_[i].get(), scratch_q.get(), mod_QP,
                                                                            out.b_[i].get(), N, size_Ql);
    }
}

} // namespace moai
