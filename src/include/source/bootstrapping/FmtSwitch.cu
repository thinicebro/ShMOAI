#include "FmtSwitch.cuh"

#include "ShRotate.cuh" // complete SharedAGaloisKey for the key-cache unique_ptr members

#include "evaluate.cuh"
#include "ntt.cuh"
#include "polymath.cuh"
#include "prng.cuh"
#include "rns.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using phantom::arith::uint128_t;
// note: DModulus is declared in the global namespace (gputype.h)

namespace moai {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

namespace {

// modular inverse of a mod m (m < 2^63), extended Euclid with 128-bit intermediates
uint64_t modinv_u64(uint64_t a, uint64_t m) {
    __int128 t = 0, newt = 1;
    __int128 r = m, newr = a % m;
    while (newr != 0) {
        __int128 q = r / newr;
        __int128 tmp = t - q * newt;
        t = newt;
        newt = tmp;
        tmp = r - q * newr;
        r = newr;
        newr = tmp;
    }
    if (t < 0) t += m;
    return static_cast<uint64_t>(t);
}

// Gather the QlP modulus table (regular primes of the level, then special primes)
// into a contiguous device array matching the layout of the QlP poly buffers.
cuda_auto_ptr<DModulus> gather_QlP_modulus(const PhantomContext &context, size_t chain_index, size_t size_Ql,
                                           size_t size_P, const cudaStream_t &stream) {
    const size_t size_QP = context.get_context_data(0).parms().coeff_modulus().size();
    const DModulus *global_mod = context.gpu_rns_tables().modulus();

    auto mod_QlP = make_cuda_auto_ptr<DModulus>(size_Ql + size_P, stream);
    cudaMemcpyAsync(mod_QlP.get(), global_mod, size_Ql * sizeof(DModulus), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(mod_QlP.get() + size_Ql, global_mod + (size_QP - size_P), size_P * sizeof(DModulus),
                    cudaMemcpyDeviceToDevice, stream);
    return mod_QlP;
}

// Gather a poly from the full QP basis (secret key layout) into the QlP basis.
void gather_QlP_poly(uint64_t *dst, const uint64_t *src_QP, size_t N, size_t size_Ql, size_t size_P, size_t size_QP,
                     const cudaStream_t &stream) {
    cudaMemcpyAsync(dst, src_QP, size_Ql * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(dst + size_Ql * N, src_QP + (size_QP - size_P) * N, size_P * N * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);
}

} // namespace

// ---------------------------------------------------------------------------
// device kernels
// ---------------------------------------------------------------------------

/**
 * Exact RNS base extension from base Ql to the special prime P (BEHZ).
 * coeff_poly holds size_Ql RNS components (coefficient domain) of a poly whose
 * integer coefficients are in [0, Ql). p_comp receives the same coefficients mod P.
 *
 *   x mod P = ( sum_g y_g * QHat_g  -  floor(sum_g y_g / q_g) * Ql ) mod P
 *
 * with y_g = (x_g * inv_g) mod q_g, QHat_g = Ql/q_g, inv_g = QHat_g^{-1} mod q_g,
 * and c_g = QHat_g mod P. The correction factor must be the FLOOR of the sum:
 * x = sum_g y_g*QHat_g - floor(v)*Ql holds exactly for x in [0, Ql), whereas the
 * round(v) variant is off by Ql whenever frac(v) > 1/2 (verified empirically).
 */
__global__ void extend_Ql_to_P_kernel(uint64_t *p_comp, const uint64_t *coeff_poly, const uint64_t *inv_g,
                                      const uint64_t *c_g, uint64_t Ql_mod_P, const DModulus *modulus_QlP,
                                      size_t size_Ql, size_t n, size_t p_idx) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
        const DModulus modP = modulus_QlP[size_Ql + p_idx];
        double v = 0.0;
        uint128_t acc = {0, 0};
        for (size_t g = 0; g < size_Ql; g++) {
            const DModulus q = modulus_QlP[g];
            const uint64_t x = coeff_poly[g * n + tid];
            const uint64_t y = phantom::arith::multiply_and_barrett_reduce_uint64(x, inv_g[g], q.value(),
                                                                                  q.const_ratio());
            v += static_cast<double>(y) / static_cast<double>(q.value());
            uint128_t prod = phantom::arith::multiply_uint64_uint64(y, c_g[g]);
            phantom::arith::add_uint128_uint128(acc, prod, acc);
        }
        uint64_t acc_mod = phantom::arith::barrett_reduce_uint128_uint64(acc, modP.value(), modP.const_ratio());
        // floor(v) with a guard: double rounding may land an almost-integer v just
        // below the true integer; the fractional error of the double sum is ~1e-15.
        uint64_t alpha = static_cast<uint64_t>(v);
        if (v - static_cast<double>(alpha) > 1.0 - 1e-9)
            alpha += 1;
        uint64_t v_corr = phantom::arith::multiply_and_barrett_reduce_uint64(alpha, Ql_mod_P, modP.value(),
                                                                             modP.const_ratio());
        p_comp[tid] = phantom::arith::sub_uint64_uint64_mod(acc_mod, v_corr, modP.value());
    }
}

/**
 * out = sum_{j=0}^{num_in-1} ext_j * key_j  (pointwise, NTT domain, QlP basis).
 * ext_j  is at ext + j * (size_QlP * n)
 * key_j  is at key + j * key_stride
 * All values are < q (< 2^60), so num_in << 2^60 products fit into the 128-bit
 * accumulator without intermediate reduction.
 */
__global__ void fmt_inner_prod_kernel(uint64_t *out, const uint64_t *ext, const uint64_t *key,
                                      const DModulus *modulus_QlP, size_t num_in, size_t key_stride, size_t n,
                                      size_t size_QlP) {
    const size_t poly_comp = size_QlP * n;
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_comp; tid += blockDim.x * gridDim.x) {
        const DModulus mod = modulus_QlP[tid / n];
        uint128_t acc = {0, 0};
        for (size_t j = 0; j < num_in; j++) {
            uint128_t prod = phantom::arith::multiply_uint64_uint64(ext[j * poly_comp + tid],
                                                                    key[j * key_stride + tid]);
            phantom::arith::add_uint128_uint128(acc, prod, acc);
        }
        out[tid] = phantom::arith::barrett_reduce_uint128_uint64(acc, mod.value(), mod.const_ratio());
    }
}

/**
 * Adds P * sk to the diagonal entries of B_fmt (only the regular-prime components;
 * P * sk == 0 mod P by construction):
 *   B_diag[comp][k] += (bigP mod q_comp) * sk[comp][k],  comp < size_Ql
 */
__global__ void add_bigP_sk_kernel(uint64_t *B_diag, const uint64_t *sk_QlP, const uint64_t *bigP_mod_q,
                                   const uint64_t *bigP_mod_q_shoup, const DModulus *modulus_QlP, size_t n,
                                   size_t size_Ql) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size_Ql * n; tid += blockDim.x * gridDim.x) {
        const size_t comp = tid / n;
        const DModulus mod = modulus_QlP[comp];
        uint64_t tmp = phantom::arith::multiply_and_reduce_shoup(sk_QlP[tid], bigP_mod_q[comp],
                                                                 bigP_mod_q_shoup[comp], mod.value());
        B_diag[tid] = phantom::arith::add_uint64_uint64_mod(B_diag[tid], tmp, mod.value());
    }
}

// ---------------------------------------------------------------------------
// SharedAKeyGen
// ---------------------------------------------------------------------------

SharedAKeyGen::SharedAKeyGen(const PhantomContext &context, PhantomSecretKey &secret_key, size_t n)
        : context_(context), secret_key_(secret_key), n_(n) {
    if (n == 0) {
        throw invalid_argument("SharedAKeyGen: batch size must be positive");
    }
    aux_sks_.reserve(n);
    for (size_t i = 0; i < n; i++) {
        aux_sks_.push_back(make_unique<PhantomSecretKey>(context));
    }
}

SharedAFormatKey SharedAKeyGen::gen_format_key(size_t chain_index, const cudaStream_t &stream) {
    auto &key_parms = context_.get_context_data(0).parms();
    const size_t N = key_parms.poly_modulus_degree();
    const size_t size_P = key_parms.special_modulus_size();
    const size_t size_QP = key_parms.coeff_modulus().size();
    if (chain_index == 0 || chain_index > size_QP - size_P) {
        throw invalid_argument("SharedAKeyGen::gen_format_key: invalid chain index");
    }
    const size_t size_Ql = context_.get_context_data(chain_index).parms().coeff_modulus().size();
    const size_t size_QlP = size_Ql + size_P;
    const size_t poly_comp = size_QlP * N;

    auto &rns_tool = context_.get_context_data(chain_index).gpu_rns_tool();
    auto &ntt_tables = context_.gpu_rns_tables();

    // memory preflight: the format key grows quadratically with n and every backward
    // key is a full hybrid key-switching key; fail with a readable error instead of OOM
    {
        const size_t size_Q = size_QP - size_P;
        const size_t dnum = size_Q / size_P;
        const size_t fmt_bytes = (n_ + n_ * n_) * poly_comp * sizeof(uint64_t);
        const size_t backward_bytes = n_ * dnum * 2 * size_QP * N * sizeof(uint64_t);
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        // phantom allocates via cudaMallocAsync: freed blocks stay in the default mempool
        // and are reusable, but cudaMemGetInfo reports them as used. Count the pool's
        // cached (reserved but not live) memory as available to avoid false OOMs.
        size_t avail = free_b;
        {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaMemPool_t pool = nullptr;
            if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess && pool != nullptr) {
                uint64_t reserved = 0, used = 0;
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
                avail += static_cast<size_t>(reserved > used ? reserved - used : 0);
            }
        }
        if (fmt_bytes + backward_bytes > static_cast<size_t>(avail * 0.9)) {
            throw runtime_error("SharedAKeyGen::gen_format_key: insufficient GPU memory for batch n=" +
                                to_string(n_) + " (need ~" + to_string((fmt_bytes + backward_bytes) >> 20) +
                                " MiB, available " + to_string(avail >> 20) + " MiB)");
        }
    }

    SharedAFormatKey key;
    key.n_ = n_;
    key.chain_index_ = chain_index;
    key.size_Ql_ = size_Ql;
    key.size_P_ = size_P;
    key.a_fmt_ = make_cuda_auto_ptr<uint64_t>(n_ * poly_comp, stream);
    key.B_fmt_ = make_cuda_auto_ptr<uint64_t>(n_ * n_ * poly_comp, stream);

    auto mod_QlP = gather_QlP_modulus(context_, chain_index, size_Ql, size_P, stream);

    // gather the main secret and the auxiliary secrets into the QlP basis (NTT form)
    auto sk_main = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
    gather_QlP_poly(sk_main.get(), secret_key_.secret_key_poly(), N, size_Ql, size_P, size_QP, stream);
    vector<cuda_auto_ptr<uint64_t>> sk_aux;
    sk_aux.reserve(n_);
    for (size_t i = 0; i < n_; i++) {
        sk_aux.push_back(make_cuda_auto_ptr<uint64_t>(poly_comp, stream));
        gather_QlP_poly(sk_aux[i].get(), aux_sks_[i]->secret_key_poly(), N, size_Ql, size_P, size_QP, stream);
    }

    const size_t grid = poly_comp / blockDimGlb.x;
    auto e = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
    // seed buffers are allocated once and rewritten each iteration (fresh randomness
    // per call to random_bytes), avoiding O(n^2) cudaMalloc/cudaFree pairs
    auto seed_a = make_cuda_auto_ptr<uint8_t>(global_variables::prng_seed_byte_count, stream);
    auto seed_e = make_cuda_auto_ptr<uint8_t>(global_variables::prng_seed_byte_count, stream);

    for (size_t j = 0; j < n_; j++) {
        // a_fmt[j] : uniform poly in R_{QlP}; stored values are used directly as NTT form
        // (same convention as PhantomSecretKey::encrypt_zero_symmetric)
        random_bytes(seed_a.get(), global_variables::prng_seed_byte_count, stream);
        sample_uniform_poly<<<grid, blockDimGlb, 0, stream>>>(key.a_fmt_.get() + j * poly_comp, seed_a.get(),
                                                              mod_QlP.get(), N, size_QlP);

        for (size_t i = 0; i < n_; i++) {
            // B[j][i] = -(a_fmt[j] * sk'_i + e)
            random_bytes(seed_e.get(), global_variables::prng_seed_byte_count, stream);
            sample_error_poly<<<grid, blockDimGlb, 0, stream>>>(e.get(), seed_e.get(), mod_QlP.get(), N, size_QlP);
            nwt_2d_radix8_forward_inplace(e.get(), ntt_tables, size_Ql, 0, stream);
            nwt_2d_radix8_forward_inplace_include_special_mod(e.get(), ntt_tables, size_P, size_Ql, size_QP, size_P,
                                                              stream);
            multiply_and_add_negate_rns_poly<<<grid, blockDimGlb, 0, stream>>>(
                    key.a_fmt_.get() + j * poly_comp, sk_aux[i].get(), e.get(), mod_QlP.get(),
                    key.B_fmt_.get() + (j * n_ + i) * poly_comp, N, size_QlP);
        }

        // diagonal: B[j][j] += P * sk
        add_bigP_sk_kernel<<<(size_Ql * N) / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                key.B_fmt_.get() + (j * n_ + j) * poly_comp, sk_main.get(), rns_tool.bigP_mod_q(),
                rns_tool.bigP_mod_q_shoup(), mod_QlP.get(), N, size_Ql);
    }

    // backward key-switching keys sk'_i -> sk (hybrid, top level)
    key.backward_keys_.reserve(n_);
    for (size_t i = 0; i < n_; i++) {
        key.backward_keys_.emplace_back(secret_key_.gen_kswitch_key(context_, aux_sks_[i]->secret_key_poly()));
    }

    return key;
}

// ---------------------------------------------------------------------------
// FmtSwitch / Backward-FmtSwitch
// ---------------------------------------------------------------------------

void fmt_switch_forward(const PhantomContext &context, const vector<PhantomCiphertext> &cts,
                        const SharedAFormatKey &key, SharedACiphertextBatch &out, const cudaStream_t &stream) {
    const size_t n = key.n_;
    if (cts.size() != n) {
        throw invalid_argument("fmt_switch_forward: number of ciphertexts does not match the format key");
    }

    const size_t chain_index = key.chain_index_;
    auto &key_parms = context.get_context_data(0).parms();
    const size_t N = key_parms.poly_modulus_degree();
    const size_t size_P = key_parms.special_modulus_size();
    const size_t size_QP = key_parms.coeff_modulus().size();
    const size_t size_Ql = key.size_Ql_;
    const size_t size_QlP = size_Ql + size_P;
    const size_t poly_comp = size_QlP * N;
    const size_t size_Ql_n = size_Ql * N;

    for (size_t j = 0; j < n; j++) {
        if (cts[j].chain_index() != chain_index || cts[j].coeff_modulus_size() != size_Ql || cts[j].size() != 2) {
            throw invalid_argument("fmt_switch_forward: ciphertext level/size mismatch with the format key");
        }
        if (cts[j].poly_modulus_degree() != N) {
            throw invalid_argument("fmt_switch_forward: ciphertext polynomial degree mismatch with the context");
        }
        if (!cts[j].is_ntt_form()) {
            throw invalid_argument("fmt_switch_forward: ciphertexts must be in NTT form");
        }
        if (fabs(cts[j].scale() - cts[0].scale()) > 1e-6 * cts[0].scale()) {
            throw invalid_argument("fmt_switch_forward: all ciphertexts must share the same scale");
        }
    }

    auto &rns_tool = context.get_context_data(chain_index).gpu_rns_tool();
    auto &ntt_tables = context.gpu_rns_tables();
    const auto scheme = context.get_context_data(0).parms().scheme();

    auto mod_QlP = gather_QlP_modulus(context, chain_index, size_Ql, size_P, stream);

    // BEHZ constants for the Ql -> P extension (host side), per special prime
    vector<uint64_t> h_inv_g(size_Ql), h_c_g(size_P * size_Ql), h_Ql_mod_P(size_P);
    {
        auto &host_coeff = context.get_context_data(chain_index).parms().coeff_modulus();
        auto &host_key_coeff = key_parms.coeff_modulus();
        vector<uint64_t> q(size_Ql);
        for (size_t g = 0; g < size_Ql; g++) q[g] = host_coeff[g].value();
        for (size_t k = 0; k < size_P; k++) {
            const uint64_t P_value = host_key_coeff[size_QP - size_P + k].value();
            uint64_t Ql_mod_P = 1;
            for (size_t g = 0; g < size_Ql; g++) {
                Ql_mod_P = static_cast<uint64_t>((static_cast<unsigned __int128>(Ql_mod_P) * q[g]) % P_value);
            }
            h_Ql_mod_P[k] = Ql_mod_P;
            for (size_t g = 0; g < size_Ql; g++) {
                uint64_t qhat_mod_P = 1, qhat_mod_qg = 1;
                for (size_t h = 0; h < size_Ql; h++) {
                    if (h == g) continue;
                    qhat_mod_P = static_cast<uint64_t>((static_cast<unsigned __int128>(qhat_mod_P) * q[h]) % P_value);
                    qhat_mod_qg = static_cast<uint64_t>((static_cast<unsigned __int128>(qhat_mod_qg) * q[h]) % q[g]);
                }
                if (k == 0)
                    h_inv_g[g] = modinv_u64(qhat_mod_qg, q[g]);
                h_c_g[k * size_Ql + g] = qhat_mod_P; // c_g = QHat_g mod p_k (kernel multiplies by y_g)
            }
        }
    }
    auto d_inv_g = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
    auto d_c_g = make_cuda_auto_ptr<uint64_t>(size_P * size_Ql, stream);
    // H2D copies must go through stream s: the phantom default stream is
    // cudaStreamNonBlocking, so legacy-stream cudaMemcpy is not ordered against
    // subsequent kernels on s and can race the mempool reuse of these buffers.
    cudaMemcpyAsync(d_inv_g.get(), h_inv_g.data(), size_Ql * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_c_g.get(), h_c_g.data(), size_P * size_Ql * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream); // keep the pageable host vectors alive until the copies land

    // 1. extend every a-part to the QlP basis
    auto ext = make_cuda_auto_ptr<uint64_t>(n * poly_comp, stream);
    auto scratch = make_cuda_auto_ptr<uint64_t>(size_Ql_n, stream);
    for (size_t j = 0; j < n; j++) {
        const uint64_t *a_j = cts[j].data() + cts[j].coeff_modulus_size() * N; // c1
        uint64_t *ext_j = ext.get() + j * poly_comp;
        cudaMemcpyAsync(ext_j, a_j, size_Ql_n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
        nwt_2d_radix8_backward(scratch.get(), a_j, ntt_tables, size_Ql, 0, stream);
        for (size_t k = 0; k < size_P; k++) {
            extend_Ql_to_P_kernel<<<(N + blockDimGlb.x - 1) / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                    ext_j + (size_Ql + k) * N, scratch.get(), d_inv_g.get(), d_c_g.get() + k * size_Ql, h_Ql_mod_P[k],
                    mod_QlP.get(), size_Ql, N, k);
        }
        nwt_2d_radix8_forward_inplace_include_special_mod(ext_j, ntt_tables, size_P, size_Ql, size_QP, size_P, stream);
    }

    // 2. inner products with the format key
    auto tmp_a = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
    auto tmp_b = make_cuda_auto_ptr<uint64_t>(n * poly_comp, stream);
    const size_t grid = poly_comp / blockDimGlb.x;
    fmt_inner_prod_kernel<<<grid, blockDimGlb, 0, stream>>>(tmp_a.get(), ext.get(), key.a_fmt_.get(), mod_QlP.get(),
                                                            n, poly_comp, N, size_QlP);
    for (size_t i = 0; i < n; i++) {
        fmt_inner_prod_kernel<<<grid, blockDimGlb, 0, stream>>>(tmp_b.get() + i * poly_comp, ext.get(),
                                                                key.B_fmt_.get() + i * poly_comp, mod_QlP.get(), n,
                                                                n * poly_comp, N, size_QlP);
    }

    // 3. moddown (divide by P with rounding) and accumulate
    out.allocate(context, chain_index, n, cts[0].scale(), stream);
    auto scratch_qp = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
    auto scratch_q = make_cuda_auto_ptr<uint64_t>(size_Ql_n, stream);

    cudaMemcpyAsync(scratch_qp.get(), tmp_a.get(), poly_comp * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    rns_tool.moddown_from_NTT(out.a_.get(), scratch_qp.get(), ntt_tables, scheme, stream);

    for (size_t i = 0; i < n; i++) {
        cudaMemcpyAsync(scratch_qp.get(), tmp_b.get() + i * poly_comp, poly_comp * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
        rns_tool.moddown_from_NTT(scratch_q.get(), scratch_qp.get(), ntt_tables, scheme, stream);
        add_rns_poly<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(cts[i].data(), scratch_q.get(),
                                                                            mod_QlP.get(), out.b_[i].get(), N,
                                                                            size_Ql);
    }
}

void fmt_switch_backward(const PhantomContext &context, const SharedACiphertextBatch &batch,
                         const SharedAFormatKey &key, vector<PhantomCiphertext> &out, const cudaStream_t &stream) {
    const size_t n = batch.n_;
    if (n != key.n_) {
        throw invalid_argument("fmt_switch_backward: batch size does not match the format key");
    }
    if (!batch.is_valid()) {
        throw invalid_argument("fmt_switch_backward: batch is not allocated");
    }
    if (!batch.is_ntt_form_) {
        throw invalid_argument("fmt_switch_backward: only NTT form is supported");
    }
    if (batch.chain_index_ != key.chain_index_) {
        throw invalid_argument("fmt_switch_backward: batch level does not match the format key");
    }
    const size_t size_Ql_n = batch.size_Ql_ * context.get_context_data(batch.chain_index_)
                                                       .parms()
                                                       .poly_modulus_degree();

    out.clear();
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        PhantomCiphertext ct;
        ct.resize(context, batch.chain_index_, 2, stream);
        cudaMemcpyAsync(ct.data(), batch.b_[i].get(), size_Ql_n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(ct.data() + size_Ql_n, 0, size_Ql_n * sizeof(uint64_t), stream);
        ct.set_scale(batch.scale_);
        ct.set_ntt_form(true);
        ct.set_correction_factor(1);
        // ct = (b_i + KS_0(a), KS_1(a)) : a standard ciphertext under sk
        keyswitch_inplace(context, ct, batch.a_.get(), key.backward_keys_[i], false, stream);
        out.push_back(std::move(ct));
    }
}

} // namespace moai
