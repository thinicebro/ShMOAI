#include "SharedABatchOps.cuh"

#include "evaluate.cuh"
#include "ntt.cuh"
#include "polymath.cuh"
#include "rns.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;

namespace moai {

// same numerics as Bootstrapper's kernel_modraise_inplace: lift the centered
// residue mod q0 into each target prime q
__global__ void shared_a_modraise_kernel(uint64_t *poly_dest, const uint64_t *poly_src_zero, const uint64_t modulus,
                                         const uint64_t minus_q0_modulus, size_t N, uint64_t q0) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        auto q = modulus;
        poly_dest[i] = poly_src_zero[i] % q;
        if (poly_src_zero[i] > (q0 >> 1)) {
            poly_dest[i] += minus_q0_modulus;
            poly_dest[i] -= (poly_dest[i] >= q) ? q : 0;
        }
    }
}

void shared_a_copy(const PhantomContext &context, const SharedACiphertextBatch &src, SharedACiphertextBatch &dst,
                   const cudaStream_t &stream) {
    if (!src.is_valid()) {
        throw invalid_argument("shared_a_copy: source batch is not allocated");
    }
    const size_t comp = src.size_Ql_ * context.get_context_data(src.chain_index_).parms().poly_modulus_degree();
    dst.allocate(context, src.chain_index_, src.n_, src.scale_, stream);
    cudaMemcpyAsync(dst.a_.get(), src.a_.get(), comp * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    for (size_t i = 0; i < src.n_; i++) {
        cudaMemcpyAsync(dst.b_[i].get(), src.b_[i].get(), comp * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    }
}

void shared_a_add_inplace(const PhantomContext &context, SharedACiphertextBatch &a, const SharedACiphertextBatch &b,
                          const cudaStream_t &stream) {
    if (!a.is_valid() || !b.is_valid()) {
        throw invalid_argument("shared_a_add_inplace: batch is not allocated");
    }
    if (a.n_ != b.n_ || a.chain_index_ != b.chain_index_ || a.size_Ql_ != b.size_Ql_) {
        throw invalid_argument("shared_a_add_inplace: batch level/size mismatch");
    }
    if (!a.is_ntt_form_ || !b.is_ntt_form_) {
        throw invalid_argument("shared_a_add_inplace: only NTT form is supported");
    }
    if (fabs(a.scale_ - b.scale_) > 1e-6 * a.scale_) {
        throw invalid_argument("shared_a_add_inplace: scale mismatch");
    }
    const size_t N = context.get_context_data(a.chain_index_).parms().poly_modulus_degree();
    const DModulus *base_rns = context.gpu_rns_tables().modulus();
    const size_t grid = a.size_Ql_ * N / blockDimGlb.x;
    add_rns_poly<<<grid, blockDimGlb, 0, stream>>>(a.a_.get(), b.a_.get(), base_rns, a.a_.get(), N, a.size_Ql_);
    for (size_t i = 0; i < a.n_; i++) {
        add_rns_poly<<<grid, blockDimGlb, 0, stream>>>(a.b_[i].get(), b.b_[i].get(), base_rns, a.b_[i].get(), N,
                                                       a.size_Ql_);
    }
}

void shared_a_multiply_plain_inplace(const PhantomContext &context, SharedACiphertextBatch &a,
                                     const PhantomPlaintext &plain, const cudaStream_t &stream) {
    if (!a.is_valid()) {
        throw invalid_argument("shared_a_multiply_plain_inplace: batch is not allocated");
    }
    if (!a.is_ntt_form_) {
        throw invalid_argument("shared_a_multiply_plain_inplace: only NTT form is supported");
    }
    if (plain.chain_index() != a.chain_index_) {
        throw invalid_argument("shared_a_multiply_plain_inplace: plaintext level mismatch");
    }
    const size_t N = context.get_context_data(a.chain_index_).parms().poly_modulus_degree();
    const DModulus *base_rns = context.gpu_rns_tables().modulus();
    const size_t grid = a.size_Ql_ * N / blockDimGlb.x;
    multiply_rns_poly<<<grid, blockDimGlb, 0, stream>>>(a.a_.get(), plain.data(), base_rns, a.a_.get(), N, a.size_Ql_);
    for (size_t i = 0; i < a.n_; i++) {
        multiply_rns_poly<<<grid, blockDimGlb, 0, stream>>>(a.b_[i].get(), plain.data(), base_rns, a.b_[i].get(), N,
                                                            a.size_Ql_);
    }
    a.scale_ *= plain.scale();
}

void shared_a_rescale_to_next_inplace(const PhantomContext &context, SharedACiphertextBatch &a,
                                      const cudaStream_t &stream) {
    if (!a.is_valid()) {
        throw invalid_argument("shared_a_rescale_to_next_inplace: batch is not allocated");
    }
    if (!a.is_ntt_form_) {
        throw invalid_argument("shared_a_rescale_to_next_inplace: only NTT form is supported");
    }
    auto &context_data = context.get_context_data(a.chain_index_);
    auto &rns_tool = context_data.gpu_rns_tool();
    auto &ntt_tables = context.gpu_rns_tables();
    const uint64_t q_last = context_data.parms().coeff_modulus().back().value();
    const size_t N = context_data.parms().poly_modulus_degree();
    const size_t next_size_Ql = a.size_Ql_ - 1;

    // NOT safe in place: the final kernel reads the original ci[j] from src while
    // dst has already overwritten it (dst == src makes every output exactly 0), so
    // each poly is rescaled into a fresh, one-prime-smaller buffer.
    auto rescale_poly = [&](phantom::util::cuda_auto_ptr<uint64_t> &poly) {
        auto shrunk = make_cuda_auto_ptr<uint64_t>(next_size_Ql * N, stream);
        rns_tool.divide_and_round_q_last_ntt(poly.get(), 1, ntt_tables, shrunk.get(), stream);
        poly = std::move(shrunk);
    };
    rescale_poly(a.a_);
    for (size_t i = 0; i < a.n_; i++) {
        rescale_poly(a.b_[i]);
    }

    a.chain_index_ += 1;
    a.size_Ql_ -= 1;
    a.scale_ /= static_cast<double>(q_last);
}

void shared_a_modraise_inplace(const PhantomContext &context, SharedACiphertextBatch &a, const cudaStream_t &stream) {
    if (!a.is_valid()) {
        throw invalid_argument("shared_a_modraise_inplace: batch is not allocated");
    }
    if (a.size_Ql_ != 1) {
        throw invalid_argument("shared_a_modraise_inplace: only bottom-level batches (1 prime) are supported");
    }
    if (!a.is_ntt_form_) {
        throw invalid_argument("shared_a_modraise_inplace: only NTT form is supported");
    }

    const auto &context_data = context.first_context_data(); // top data level
    const size_t top_chain = context_data.chain_index();
    const auto &modulus = context_data.parms().coeff_modulus();
    const size_t mod_count = modulus.size();
    const size_t N = context_data.parms().poly_modulus_degree();
    auto &ntt_tables = context.gpu_rns_tables();
    const uint64_t q0 = modulus[0].value();

    vector<uint64_t> minus_q0(mod_count);
    minus_q0[0] = 0;
    for (size_t l = 1; l < mod_count; l++) {
        minus_q0[l] = modulus[l].value() - q0 % modulus[l].value();
    }

    const uint64_t grid = N / blockDimGlb.x;
    auto raise_poly = [&](cuda_auto_ptr<uint64_t> &poly) {
        // to coefficient domain (single prime, global RNS index 0)
        nwt_2d_radix8_backward_inplace(poly.get(), ntt_tables, 1, 0, stream);
        auto raised = make_cuda_auto_ptr<uint64_t>(mod_count * N, stream);
        for (size_t j = 0; j < mod_count; j++) {
            shared_a_modraise_kernel<<<grid, blockDimGlb, 0, stream>>>(raised.get() + j * N, poly.get(),
                                                                       modulus[j].value(), minus_q0[j], N, q0);
        }
        nwt_2d_radix8_forward_inplace(raised.get(), ntt_tables, mod_count, 0, stream);
        poly = std::move(raised);
    };

    raise_poly(a.a_);
    for (size_t i = 0; i < a.n_; i++) {
        raise_poly(a.b_[i]);
    }

    a.chain_index_ = top_chain;
    a.size_Ql_ = mod_count;
}

} // namespace moai
