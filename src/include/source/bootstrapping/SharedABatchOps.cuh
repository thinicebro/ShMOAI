#pragma once

#include "SharedACiphertext.cuh"
#include "phantom.h"

namespace moai {

/**
 * Element-wise operations on a shared-a ciphertext batch (all polys in NTT form at
 * the batch's chain index). These mirror the corresponding phantom ciphertext ops:
 * an operation applied to an ordinary ciphertext (c0, c1) is applied to the shared
 * a-part and to every member's b-part, so the batch result decrypts (under the
 * auxiliary secrets) to the per-member operation results.
 */

/** Deep copy src -> dst (dst is (re)allocated at src's level). */
void shared_a_copy(const PhantomContext &context, const SharedACiphertextBatch &src,
                   SharedACiphertextBatch &dst, const cudaStream_t &stream);

/** a += b, polynomial-wise. Both batches must be at the same level with equal scales. */
void shared_a_add_inplace(const PhantomContext &context, SharedACiphertextBatch &a,
                          const SharedACiphertextBatch &b, const cudaStream_t &stream);

/** a *= plain, pointwise in the NTT domain (CKKS semantics of multiply_plain_ntt):
 * every poly is multiplied by the plaintext poly and the batch scale is multiplied
 * by the plaintext scale. plain must already sit at the batch's chain index. */
void shared_a_multiply_plain_inplace(const PhantomContext &context, SharedACiphertextBatch &a,
                                     const PhantomPlaintext &plain, const cudaStream_t &stream);

/** CKKS rescale (divide-and-round by the last RNS prime) of every poly; drops the
 * batch to the next chain index and divides the scale by the dropped prime. */
void shared_a_rescale_to_next_inplace(const PhantomContext &context, SharedACiphertextBatch &a,
                                      const cudaStream_t &stream);

/** CKKS bootstrapping-style mod raise: takes a batch at the bottom level (a single
 * RNS prime) and raises it to the top data level, with exactly the same numerics as
 * Bootstrapper::modraise_inplace (centered lift mod q0, then per-prime residues and
 * a forward NTT). The caller is responsible for setting the new scale afterwards. */
void shared_a_modraise_inplace(const PhantomContext &context, SharedACiphertextBatch &a,
                               const cudaStream_t &stream);

} // namespace moai
