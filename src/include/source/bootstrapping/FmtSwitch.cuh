#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "SharedACiphertext.cuh"
#include "phantom.h"

namespace moai {

struct SharedAGaloisKey; // defined in ShRotate.cuh

/**
 * Format-switching key material for the shared-a (multi-secret RLWE) format,
 * paper Sec. 3.2 (naive BV-style FmtSwitch, sufficient at the bottom level where
 * the special modulus P is comparable to the ciphertext modulus Ql).
 *
 * Forward key (converts n ciphertexts under the shared secret sk into the
 * shared-a format under auxiliary secrets sk'_0..sk'_{n-1}):
 *   a_fmt[j]    : n uniform polys in R_{QlP}, NTT form
 *   B_fmt[j][i] : -a_fmt[j] * sk'_i + e_{j,i} + (j == i ? P * sk : 0)  in R_{QlP}, NTT form
 * stored contiguously, B row-major with index (j * n + i).
 *
 * Backward keys: n hybrid (gadget-decomposed) key-switching keys sk'_i -> sk,
 * generated at the top level (QP) so they can be used at any level.
 */
struct SharedAFormatKey {
    size_t n_ = 0;           // batch size
    size_t chain_index_ = 0; // level at which the forward conversion applies
    size_t size_Ql_ = 0;     // number of regular RNS components at that level
    size_t size_P_ = 0;      // number of special primes

    phantom::util::cuda_auto_ptr<uint64_t> a_fmt_; // n * size_QlP * N, NTT form
    phantom::util::cuda_auto_ptr<uint64_t> B_fmt_; // n * n * size_QlP * N, NTT form

    std::vector<PhantomRelinKey> backward_keys_; // n keys, sk'_i -> sk
}; //格式转换密钥结构体

/**
 * Generates the auxiliary secret keys and the SharedAFormatKey.
 */
class SharedAKeyGen {
public:
    /** Samples n auxiliary secret keys with the same distribution (and hamming weight)
     * as the main secret key. */
    SharedAKeyGen(const PhantomContext &context, PhantomSecretKey &secret_key, size_t n);
    ~SharedAKeyGen(); // defined in ShRotate.cu (SharedAGaloisKey is incomplete here)

    [[nodiscard]] size_t size() const { return n_; }

    [[nodiscard]] PhantomSecretKey &aux_secret_key(size_t i) { return *aux_sks_[i]; }

    /** Generates the forward format-switching key at the given chain index (this is
     * where fmt_switch_forward must be called, typically the bottom level right before
     * bootstrapping), and the n backward key-switching keys. */
    SharedAFormatKey gen_format_key(size_t chain_index, const cudaStream_t &stream);

    /** Generates a shared-a rotation key for one rotation step (paper Sec. 6.1):
     * gadget-decomposed switching keys sigma(sk'_i) -> sk'_i sharing one a-part,
     * where sigma is the Galois automorphism of the given step.
     * The wall-clock time spent here is accumulated into galois_keygen_sec_. */
    SharedAGaloisKey gen_shared_galois_key(int step, const cudaStream_t &stream);

    /** Returns the shared-a rotation key for step (used by the batch linear
     * transforms). Two modes:
     *   不摊销 (default, cache budget = 0): the key is freshly generated on every
     *     call and the returned reference points to an internal scratch that is
     *     overwritten by the next call. Peak device memory stays at one key.
     *   摊销 (budget > 0, see set_galois_key_cache): the first generation of each
     *     step is cached on the GPU and later calls (e.g. the following chunks of
     *     a batch bootstrap) reuse it for free, until the cache exceeds the byte
     *     budget; steps beyond the budget fall back to the scratch path.
     * In both modes the reference stays valid until the next acquire call. */
    const SharedAGaloisKey &acquire_shared_galois_key(int step, const cudaStream_t &stream);

    /** 摊销模式开关: cache Sh-Galois keys on the GPU up to budget_bytes
     * (0 disables caching = 不摊销). One key costs dnum*(n+1)*size_QP*N*8 bytes,
     * ~0.4GB at batch 4 to ~2.5GB at batch 16 with 5 special primes, so size the
     * budget for your card. */
    void set_galois_key_cache(size_t budget_bytes) { galois_cache_budget_ = budget_bytes; }

    [[nodiscard]] size_t galois_cache_size() const { return galois_cache_.size(); }
    [[nodiscard]] size_t galois_cache_bytes() const { return galois_cache_bytes_; }

    /** Drop all cached Sh-Galois keys, freeing their GPU memory. The cache
     * refills transparently on the next acquire_shared_galois_key calls (each
     * step is regenerated once and re-cached), so this only costs one extra
     * keyset regeneration -- used to free ~26.7 GiB before the 768-wide FFN
     * phase, which does not need the keys but OOMs with them resident. */
    void clear_galois_key_cache()
    {
        galois_cache_.clear();
        galois_cache_bytes_ = 0;
    }

    /** Total seconds spent inside gen_shared_galois_key (for amortized-vs-single-shot
     * timing reports; read/reset freely). */
    double galois_keygen_sec_ = 0.0;

private:
    const PhantomContext &context_;
    PhantomSecretKey &secret_key_;
    size_t n_;
    std::vector<std::unique_ptr<PhantomSecretKey>> aux_sks_;

    // 摊销模式缓存 (see acquire_shared_galois_key); unique_ptr because
    // SharedAGaloisKey is only forward-declared in this header
    std::unordered_map<int, std::unique_ptr<SharedAGaloisKey>> galois_cache_;
    std::unique_ptr<SharedAGaloisKey> scratch_key_; // 不摊销路径的临时存放点
    size_t galois_cache_budget_ = 0;
    size_t galois_cache_bytes_ = 0;
}; //格式转换密钥生成器类

/**
 * Forward format conversion (paper Sec. 3.2, FmtSwitch):
 *   in : n ciphertexts ct_j = (b_j, a_j) under sk, all at the key's chain index
 *   out: shared-a batch (a', b'_i) under sk'_0..sk'_{n-1} with
 *          a'   = round( <(a_0..a_{n-1}), a_fmt> / P )
 *          b'_i = b_i + round( (a_0..a_{n-1}) . B_fmt[:, i] / P )
 */
void fmt_switch_forward(const PhantomContext &context, const std::vector<PhantomCiphertext> &cts,
                        const SharedAFormatKey &key, SharedACiphertextBatch &out,
                        const cudaStream_t &stream); //格式转换

/**
 * Backward format conversion (paper Sec. 3.4): converts each member of the shared-a
 * batch back into an ordinary ciphertext under the shared secret sk, using the
 * backward key-switching keys.
 */
void fmt_switch_backward(const PhantomContext &context, const SharedACiphertextBatch &batch,
                         const SharedAFormatKey &key, std::vector<PhantomCiphertext> &out,
                         const cudaStream_t &stream); //反向格式转换

} // namespace moai
