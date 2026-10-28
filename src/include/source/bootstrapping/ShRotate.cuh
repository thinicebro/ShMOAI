#pragma once

#include <memory>

#include "FmtSwitch.cuh"
#include "galois.cuh"
#include "phantom.h"

namespace moai {

/**
 * Shared-a rotation (Galois) key for ONE rotation step, paper Sec. 6.1 (Sh-SWKGen).
 *
 * For a batch of n shared-a ciphertexts under secrets sk'_0..sk'_{n-1} and a Galois
 * automorphism sigma, the key consists of gadget-decomposed switching keys
 * sigma(sk'_i) -> sk'_i that all share the same a-part:
 *   A_g     : dnum uniform polys (QP basis, NTT form), shared by the whole batch
 *   B_{g,i} : -A_g * sk'_i + e + P * (CRT unit of prime g) * sigma(sk'_i)
 * Because the a-part is shared, applying the key to the (single) rotated a-part of
 * the batch yields the same output a-part for every member, so the batch stays in
 * shared-a form. Key size is (n+1)/2 times a standard switching key (instead of n).
 *
 * The key owns the Galois permutation tables of its automorphism.
 */
struct SharedAGaloisKey {
    int step_ = 0;
    uint32_t galois_elt_ = 0;
    size_t n_ = 0;    // batch size
    size_t dnum_ = 0; // gadget decomposition count

    std::unique_ptr<PhantomGaloisTool> tool_; // permutation tables for galois_elt_

    phantom::util::cuda_auto_ptr<uint64_t> A_; // dnum polys (QP basis, NTT form)
    phantom::util::cuda_auto_ptr<uint64_t> B_; // dnum * n polys; B_{g,i} at index (g * n + i)
};

/**
 * Shared-a rotation (Sh-Rot, paper Sec. 6.1): rotates every member of the batch by
 * the key's step, preserving the shared-a format.
 *
 * The Galois automorphism and the key-switching ModUp are applied to the shared
 * a-part only once; each member's b-part gets its own MultSwk + ModDown.
 *   in  : (a, b_i) with a * sk'_i + b_i ~= m_i
 *   out : (a', b'_i) with a' * sk'_i + b'_i ~= Rot_step(m_i)
 */
void shared_a_rotate(const PhantomContext &context, const SharedACiphertextBatch &in,
                     const SharedAGaloisKey &key, SharedACiphertextBatch &out, const cudaStream_t &stream);

} // namespace moai
