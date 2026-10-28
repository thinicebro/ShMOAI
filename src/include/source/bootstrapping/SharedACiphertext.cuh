#pragma once

#include <vector>

#include "phantom.h"

namespace moai {

/**
 * Shared-a (multi-secret RLWE) representation of n ciphertexts, see
 * "Plaintext-Ciphertext Matrix Multiplication and FHE Bootstrapping: Fast and Fused"
 * (Bae, Cheon, Hanrot, Park, Stehle), Sec. 3.1.
 *
 * n plaintexts are held as n + 1 ring elements instead of 2n:
 *   (a, b_i) with  a * sk_i + b_i ~= m_i  (mod Ql),  0 <= i < n,
 * where sk_0..sk_{n-1} are auxiliary secret keys (one per slot of the batch).
 *
 * All polynomials are stored in NTT form with size_Ql RNS components (the regular
 * primes of the ciphertext level). a_ is a single poly shared by the whole batch;
 * b_[i] is the b-part of the i-th member.
 */
struct SharedACiphertextBatch {
    size_t n_ = 0;           // batch size
    size_t chain_index_ = 0; // level of the ciphertexts
    size_t size_Ql_ = 0;     // number of RNS components per poly
    double scale_ = 1.0;     // CKKS scale of the members
    bool is_ntt_form_ = true; // all polys (a_ and every b_[i]) are in NTT form

    phantom::util::cuda_auto_ptr<uint64_t> a_;              // size_Ql * N, NTT form
    std::vector<phantom::util::cuda_auto_ptr<uint64_t>> b_; // n buffers of size_Ql * N, NTT form

    SharedACiphertextBatch() = default;

    SharedACiphertextBatch(const SharedACiphertextBatch &) = delete;

    SharedACiphertextBatch &operator=(const SharedACiphertextBatch &) = delete;

    SharedACiphertextBatch(SharedACiphertextBatch &&) = default;

    SharedACiphertextBatch &operator=(SharedACiphertextBatch &&) = default;

    /** Whether the batch has been allocated and holds n members. */
    [[nodiscard]] bool is_valid() const { return a_.get() != nullptr && b_.size() == n_ && n_ > 0; }

    /** Release all device memory and reset to the empty state. */
    void release() {
        a_.reset();
        b_.clear();
        n_ = 0;
        chain_index_ = 0;
        size_Ql_ = 0;
        scale_ = 1.0;
        is_ntt_form_ = true;
    }

    void allocate(const PhantomContext &context, size_t chain_index, size_t n, double scale,
                  const cudaStream_t &stream) {
        auto &parms = context.get_context_data(chain_index).parms();
        const size_t N = parms.poly_modulus_degree();
        const size_t size_Ql = parms.coeff_modulus().size();

        n_ = n;
        chain_index_ = chain_index;
        size_Ql_ = size_Ql;
        scale_ = scale;
        is_ntt_form_ = true;

        a_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_Ql * N, stream);
        b_.resize(n);
        for (size_t i = 0; i < n; i++) {
            b_[i] = phantom::util::make_cuda_auto_ptr<uint64_t>(size_Ql * N, stream);
        }
    }

    /** Assemble the i-th member (b_i, a) into a standard size-2 ciphertext in NTT form.
     * Useful for decryption (with the i-th auxiliary secret key) and for the
     * backward format conversion. */
    PhantomCiphertext assemble(const PhantomContext &context, size_t i, const cudaStream_t &stream) const {
        if (!is_valid()) {
            throw std::invalid_argument("SharedACiphertextBatch::assemble: batch is not allocated");
        }
        if (i >= n_) {
            throw std::invalid_argument("SharedACiphertextBatch::assemble: index out of range");
        }
        if (!is_ntt_form_) {
            throw std::invalid_argument("SharedACiphertextBatch::assemble: only NTT form is supported");
        }
        const size_t comp = size_Ql_ * context.get_context_data(chain_index_).parms().poly_modulus_degree();

        PhantomCiphertext ct;
        ct.resize(context, chain_index_, 2, stream);
        cudaMemcpyAsync(ct.data(), b_[i].get(), comp * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(ct.data() + comp, a_.get(), comp * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
        ct.set_scale(scale_);
        ct.set_ntt_form(true);
        ct.set_correction_factor(1);
        return ct;
    }
};

} // namespace moai
