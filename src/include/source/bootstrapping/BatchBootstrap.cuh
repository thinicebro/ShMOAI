#pragma once

#include <memory>
#include <set>
#include <vector>

#include "Bootstrapper.cuh"
#include "FmtSwitch.cuh"
#include "ShRotate.cuh"
#include "SharedABatchOps.cuh"
#include "common/func.cuh"

/**
 * Batch bootstrap pipeline (stage-3 MOAI integration).
 *
 * BatchBootstrapper wraps the two end-to-end bootstrap paths compared in this
 * project behind a single reusable object:
 *
 *   ordinary   : modraise -> coefftoslot_full_3 -> EvalMod -> slottocoeff_full_3
 *                Galois keys are generated once per layer and reused for every
 *                member of the call (and freed between layers so the full key set
 *                never has to fit in GPU memory at once).
 *
 *   shared-a   : FmtSwitch -> shared-a modraise -> Sh-C2S -> Backward-FmtSwitch
 *                -> ordinary EvalMod -> ordinary S2C
 *                The one-time key material (auxiliary secrets + bottom-level
 *                format key) is generated in prepare_shared_keys() and reused
 *                across batches; the per-batch Sh-Rot Galois keys and the
 *                output-level format key are regenerated inside every call
 *                (they are too large to keep resident, see stage-2 findings).
 *
 * Timing convention: every entry point splits wall-clock time into
 *   keygen_sec    - key material generation (Galois / format keys)
 *   transform_sec - everything else (modraise, linear transforms, EvalMod, ...)
 * so a caller can quote both single-shot and key-amortized figures.
 */

/** Rotation steps used by one rotated_bsgs layer (for the ordinary reference path). */
static vector<int> collect_rotated_layer_steps(int totlen, int basicstep, long Nh)
{
    int gs2 = giantstep(totlen + 1);
    int giantlast2 = floor((totlen + 0.0) / (gs2 + 0.0));
    std::set<int> uniq;
    for (int i = 1; i < gs2; i++)
        uniq.insert((int)((Nh + i * basicstep) % Nh));
    for (int i = 1; i <= giantlast2; i++)
        uniq.insert((int)((Nh + i * gs2 * basicstep) % Nh));
    return vector<int>(uniq.begin(), uniq.end());
}

/** Rotation steps used by one bsgs layer (for the ordinary reference path). */
static vector<int> collect_bsgs_layer_steps(int totlen, int basicstep, long Nh)
{
    int gs1 = giantstep(2 * totlen + 1);
    int basicstart1 = -totlen + gs1 * floor((totlen + 0.0) / (gs1 + 0.0));
    int giantfirst1 = -floor((totlen + 0.0) / (gs1 + 0.0));
    int giantlast1 = floor((2 * totlen + 0.0) / (gs1 + 0.0)) + giantfirst1;
    std::set<int> uniq;
    for (int i = basicstart1; i < basicstart1 + gs1; i++)
        if (i != 0)
            uniq.insert((int)((Nh + i * basicstep) % Nh));
    for (int i = giantfirst1; i <= giantlast1; i++)
        if (i != 0)
            uniq.insert((int)((Nh + i * gs1 * basicstep) % Nh));
    return vector<int>(uniq.begin(), uniq.end());
}

/**
 * Ordinary coefftoslot_full_3 for a whole batch of ciphertexts, with the Galois
 * keys generated once per layer (and freed between layers) and reused for every
 * member. Key generation time is accumulated into keygen_sec, everything else
 * (modraise, transforms, conjugation) into transform_sec.
 */
static void ordinary_coefftoslot_full_3_batch(Bootstrapper &bootstrapper, CKKSEvaluator &ckks_evaluator,
                                              vector<PhantomCiphertext> &rtn1, vector<PhantomCiphertext> &rtn2,
                                              vector<PhantomCiphertext> &ciphers, double &keygen_sec,
                                              double &transform_sec)
{
    const long logn = bootstrapper.logn;
    const long Nh = bootstrapper.Nh;
    const size_t n = ciphers.size();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();

    int div_part1 = floor(logn / 3.0);
    int div_part2 = floor((logn - div_part1) / 2.0);
    int div_part3 = logn - div_part1 - div_part2;
    int totlen1 = (1 << div_part1) - 1, totlen2 = (1 << div_part2) - 1, totlen3 = (1 << div_part3) - 1;
    int basicstep1 = (1 << (logn - div_part1)), basicstep2 = (1 << (logn - div_part1 - div_part2)), basicstep3 = 1;

    PhantomGaloisKey &layer_keys = *(ckks_evaluator.galois_keys);

    auto timed_keys = [&](auto &&fn) {
        auto t0 = system_clock::now();
        fn();
        cudaStreamSynchronize(stream);
        keygen_sec += duration<double>(system_clock::now() - t0).count();
    };
    auto timed_xform = [&](auto &&fn) {
        auto t0 = system_clock::now();
        fn();
        cudaStreamSynchronize(stream);
        transform_sec += duration<double>(system_clock::now() - t0).count();
    };

    // modraise per member (mirrors bootstrap_full_3: scale := q0 afterwards)
    const double q0 = static_cast<double>(ckks_evaluator.context->first_context_data().parms().coeff_modulus()[0].value());
    vector<PhantomCiphertext> stage1(n), stage2(n);
    timed_xform([&] {
        for (size_t i = 0; i < n; i++)
        {
            bootstrapper.modraise_inplace(ciphers[i]);
            ciphers[i].scale() = q0;
        }
    });

    timed_keys([&] {
        vector<int> steps = collect_rotated_layer_steps(totlen1, basicstep1, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
    });
    timed_xform([&] {
        for (size_t i = 0; i < n; i++)
        {
            bootstrapper.rotated_bsgs_linear_transform(stage1[i], ciphers[i], totlen1, basicstep1, logn,
                                                       bootstrapper.invfftcoeff1[bootstrapper.slot_index]);
            ckks_evaluator.evaluator.rescale_to_next_inplace(stage1[i]);
        }
    });
    layer_keys = PhantomGaloisKey();

    timed_keys([&] {
        vector<int> steps = collect_bsgs_layer_steps(totlen2, basicstep2, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
    });
    timed_xform([&] {
        for (size_t i = 0; i < n; i++)
        {
            bootstrapper.bsgs_linear_transform(stage2[i], stage1[i], totlen2, basicstep2, logn,
                                               bootstrapper.invfftcoeff2[bootstrapper.slot_index]);
            ckks_evaluator.evaluator.rescale_to_next_inplace(stage2[i]);
        }
    });
    layer_keys = PhantomGaloisKey();

    timed_keys([&] {
        vector<int> steps = collect_bsgs_layer_steps(totlen3, basicstep3, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
    });
    timed_xform([&] {
        for (size_t i = 0; i < n; i++)
        {
            bootstrapper.bsgs_linear_transform(stage1[i], stage2[i], totlen3, basicstep3, logn,
                                               bootstrapper.invfftcoeff3[bootstrapper.slot_index]);
            ckks_evaluator.evaluator.rescale_to_next_inplace(stage1[i]);
        }
    });
    layer_keys = PhantomGaloisKey();

    // tail of coefftoslot_full_3: -i multiply, conjugation pair, adds
    timed_keys([&] {
        vector<int> steps{0};
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
    });
    timed_xform([&] {
        complex<double> iunit(0.0, 1.0);
        vector<complex<double>> tmpvec(Nh, 0);
        for (long i = 0; i < Nh; i++)
            tmpvec[i] -= iunit;
        rtn1.resize(n);
        rtn2.resize(n);
        for (size_t i = 0; i < n; i++)
        {
            PhantomPlaintext tmpplain;
            ckks_evaluator.encoder.encode(tmpvec, 1.0, tmpplain);
            ckks_evaluator.evaluator.mod_switch_to_inplace(tmpplain, stage1[i].params_id());
            PhantomCiphertext mct, conj1, conj2;
            ckks_evaluator.evaluator.multiply_plain(stage1[i], tmpplain, mct);
            ckks_evaluator.evaluator.complex_conjugate(stage1[i], layer_keys, conj1);
            ckks_evaluator.evaluator.complex_conjugate(mct, layer_keys, conj2);
            ckks_evaluator.evaluator.add_reduced_error(stage1[i], conj1, rtn1[i]);
            ckks_evaluator.evaluator.add_reduced_error(mct, conj2, rtn2[i]);
        }
    });
    layer_keys = PhantomGaloisKey();
}

/**
 * EvalMod + recombine for a whole batch: modular_reduction on each member of both
 * ciphertext pairs, then mod1 + i * mod2 (the slottocoeff preamble). This is the
 * entire post-C2S tail of S2C-first bootstrapping, and the first section of the
 * C2S-first tail (ordinary_evalmod_s2c_batch below). Needs no key material.
 */
static void ordinary_evalmod_combine_batch(Bootstrapper &bootstrapper, CKKSEvaluator &ckks_evaluator,
                                           vector<PhantomCiphertext> &rtns, vector<PhantomCiphertext> &ct1s,
                                           vector<PhantomCiphertext> &ct2s, double &keygen_sec,
                                           double &transform_sec)
{
    const long Nh = bootstrapper.Nh;
    const size_t n = ct1s.size();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();
    (void)keygen_sec; // no key material needed in this section

    auto t0 = system_clock::now();
    rtns.resize(n);
    std::cout << "    [evalmod] combine input: n=" << n << " primes=" << ct1s[0].coeff_modulus_size()
              << " scale=log2 " << log2(ct1s[0].scale()) << std::endl;
    for (size_t i = 0; i < n; i++)
    {
        PhantomCiphertext mod1, mod2;
        bootstrapper.mod_reducer->modular_reduction(mod1, ct1s[i]);
        bootstrapper.mod_reducer->modular_reduction(mod2, ct2s[i]);

        // slottocoeff preamble: mod1 + i * mod2
        complex<double> iunit(0.0, 1.0);
        vector<complex<double>> tmpvec(Nh, 0);
        for (long k = 0; k < Nh; k++)
            tmpvec[k] += iunit;
        PhantomPlaintext tmpplain;
        ckks_evaluator.encoder.encode(tmpvec, 1.0, tmpplain);
        ckks_evaluator.evaluator.mod_switch_to_inplace(tmpplain, mod2.params_id());
        PhantomCiphertext mct;
        ckks_evaluator.evaluator.multiply_plain(mod2, tmpplain, mct);
        ckks_evaluator.evaluator.add_reduced_error(mod1, mct, rtns[i]);
    }
    cudaStreamSynchronize(stream);
    const double dt = duration<double>(system_clock::now() - t0).count();
    std::cout << "    [evalmod] combine xform: " << dt << "s" << std::endl;
    transform_sec += dt;
}

/**
 * Ordinary EvalMod + slottocoeff_full_3 for a whole batch (the bootstrap tail shared
 * by both paths). Galois keys are generated once per S2C layer for all members
 * (keygen_sec); EvalMod, the i-combination and the transforms go to transform_sec.
 */
static void ordinary_evalmod_s2c_batch(Bootstrapper &bootstrapper, CKKSEvaluator &ckks_evaluator,
                                       vector<PhantomCiphertext> &rtns, vector<PhantomCiphertext> &ct1s,
                                       vector<PhantomCiphertext> &ct2s, double &keygen_sec, double &transform_sec)
{
    const long logn = bootstrapper.logn;
    const long Nh = bootstrapper.Nh;
    const size_t n = ct1s.size();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();

    // sfl_full_3 layer split (note the reversed division order)
    int div_part3 = floor(logn / 3.0);
    int div_part2 = floor((logn - div_part3) / 2.0);
    int div_part1 = logn - div_part3 - div_part2;
    int totlen1 = (1 << div_part1) - 1, totlen2 = (1 << div_part2) - 1, totlen3 = (1 << div_part3) - 1;
    int basicstep1 = 1, basicstep2 = (1 << div_part1), basicstep3 = (1 << (div_part1 + div_part2));

    PhantomGaloisKey &layer_keys = *(ckks_evaluator.galois_keys);

    auto timed_keys = [&](const char *name, auto &&fn) {
        auto t0 = system_clock::now();
        fn();
        cudaStreamSynchronize(stream);
        const double dt = duration<double>(system_clock::now() - t0).count();
        std::cout << "    [tail] " << name << " keys: " << dt << "s" << std::endl;
        keygen_sec += dt;
    };
    auto timed_xform = [&](const char *name, auto &&fn) {
        auto t0 = system_clock::now();
        fn();
        cudaStreamSynchronize(stream);
        const double dt = duration<double>(system_clock::now() - t0).count();
        std::cout << "    [tail] " << name << " xform: " << dt << "s" << std::endl;
        transform_sec += dt;
    };

    vector<PhantomCiphertext> stage1, stage2(n);
    ordinary_evalmod_combine_batch(bootstrapper, ckks_evaluator, stage1, ct1s, ct2s, keygen_sec, transform_sec);

    timed_keys("s2c-layer1", [&] {
        vector<int> steps = collect_bsgs_layer_steps(totlen1, basicstep1, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
    });
    timed_xform("s2c-layer1", [&] {
        for (size_t i = 0; i < n; i++)
        {
            bootstrapper.bsgs_linear_transform(stage2[i], stage1[i], totlen1, basicstep1, logn,
                                               bootstrapper.fftcoeff1[bootstrapper.slot_index]);
            ckks_evaluator.evaluator.rescale_to_next_inplace(stage2[i]);
        }
    });
    layer_keys = PhantomGaloisKey();

    timed_keys("s2c-layer2", [&] {
        vector<int> steps = collect_bsgs_layer_steps(totlen2, basicstep2, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
    });
    timed_xform("s2c-layer2", [&] {
        for (size_t i = 0; i < n; i++)
        {
            bootstrapper.bsgs_linear_transform(stage1[i], stage2[i], totlen2, basicstep2, logn,
                                               bootstrapper.fftcoeff2[bootstrapper.slot_index]);
            ckks_evaluator.evaluator.rescale_to_next_inplace(stage1[i]);
        }
    });
    layer_keys = PhantomGaloisKey();

    timed_keys("s2c-layer3", [&] {
        vector<int> steps = collect_rotated_layer_steps(totlen3, basicstep3, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
    });
    timed_xform("s2c-layer3", [&] {
        // third layer: per-member scale-adjusted coefficients (mirrors sfl_full_3)
        const auto modulus = ckks_evaluator.context->first_context_data().parms().coeff_modulus();
        const double mod_zero = static_cast<double>(modulus[0].value());
        rtns.resize(n);
        for (size_t i = 0; i < n; i++)
        {
            auto curr_level = ckks_evaluator.context->get_context_data(stage1[i].params_id()).chain_depth();
            const double curr_mod = static_cast<double>(modulus[curr_level].value());
            const double factor = curr_mod * mod_zero * bootstrapper.final_scale /
                                  (stage1[i].scale() * stage1[i].scale() * bootstrapper.initial_scale);
            vector<vector<complex<double>>> fftcoeff3_scale(2 * totlen3 + 1);
            for (int r = 0; r < totlen2 + 1; r++)
            {
                fftcoeff3_scale[r].resize(bootstrapper.n);
                for (long j = 0; j < bootstrapper.n; j++)
                    fftcoeff3_scale[r][j] = bootstrapper.fftcoeff3[bootstrapper.slot_index][r][j] * factor;
            }
            bootstrapper.rotated_bsgs_linear_transform(rtns[i], stage1[i], totlen3, basicstep3, logn, fftcoeff3_scale);
            ckks_evaluator.evaluator.rescale_to_next_inplace(rtns[i]);
            rtns[i].scale() = bootstrapper.final_scale;
        }
    });
    layer_keys = PhantomGaloisKey();
}

/**
 * Ordinary sfl_full_3 (slot-to-coeff linear transform) for a whole batch, executed
 * at the LOW input level - the first step of S2C-first bootstrapping. Mirrors
 * sfl_full_3: reversed level split, fftcoeff1/2/3, third layer with an explicit
 * coefficient factor (layer3_factor; the pipelines pass kLowS2cLayer3Factor).
 * Galois keys are generated once per layer for all members (keygen_sec).
 */
static void ordinary_sfl_full_3_batch(Bootstrapper &bootstrapper, CKKSEvaluator &ckks_evaluator,
                                      vector<PhantomCiphertext> &rtn, vector<PhantomCiphertext> &ciphers,
                                      double &keygen_sec, double &transform_sec,
                                      double layer3_factor = 1.0)
{
    const long logn = bootstrapper.logn;
    const long Nh = bootstrapper.Nh;
    const size_t n = ciphers.size();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();

    // sfl_full_3 layer split (note the reversed division order)
    int div_part3 = floor(logn / 3.0);
    int div_part2 = floor((logn - div_part3) / 2.0);
    int div_part1 = logn - div_part3 - div_part2;
    int totlen1 = (1 << div_part1) - 1, totlen2 = (1 << div_part2) - 1, totlen3 = (1 << div_part3) - 1;
    int basicstep1 = 1, basicstep2 = (1 << div_part1), basicstep3 = (1 << (div_part1 + div_part2));

    PhantomGaloisKey &layer_keys = *(ckks_evaluator.galois_keys);

    auto timed_keys = [&](const char *name, auto &&fn) {
        auto t0 = system_clock::now();
        fn();
        cudaStreamSynchronize(stream);
        const double dt = duration<double>(system_clock::now() - t0).count();
        std::cout << "    [s2c-low] " << name << " keys: " << dt << "s" << std::endl;
        keygen_sec += dt;
    };
    auto timed_xform = [&](const char *name, auto &&fn) {
        auto t0 = system_clock::now();
        fn();
        cudaStreamSynchronize(stream);
        const double dt = duration<double>(system_clock::now() - t0).count();
        std::cout << "    [s2c-low] " << name << " xform: " << dt << "s" << std::endl;
        transform_sec += dt;
    };

    vector<PhantomCiphertext> stage1(n), stage2(n);
    timed_keys("layer1", [&] {
        vector<int> steps = collect_bsgs_layer_steps(totlen1, basicstep1, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
    });
    timed_xform("layer1", [&] {
        for (size_t i = 0; i < n; i++)
        {
            bootstrapper.bsgs_linear_transform(stage2[i], ciphers[i], totlen1, basicstep1, logn,
                                               bootstrapper.fftcoeff1[bootstrapper.slot_index]);
            ckks_evaluator.evaluator.rescale_to_next_inplace(stage2[i]);
        }
    });
    layer_keys = PhantomGaloisKey();

    timed_keys("layer2", [&] {
        vector<int> steps = collect_bsgs_layer_steps(totlen2, basicstep2, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
    });
    timed_xform("layer2", [&] {
        for (size_t i = 0; i < n; i++)
        {
            bootstrapper.bsgs_linear_transform(stage1[i], stage2[i], totlen2, basicstep2, logn,
                                               bootstrapper.fftcoeff2[bootstrapper.slot_index]);
            ckks_evaluator.evaluator.rescale_to_next_inplace(stage1[i]);
        }
    });
    layer_keys = PhantomGaloisKey();

    timed_keys("layer3", [&] {
        vector<int> steps = collect_rotated_layer_steps(totlen3, basicstep3, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
    });
    timed_xform("layer3", [&] {
        // Third layer with the caller-supplied factor F. The chain's message
        // amplitude is (F/(q0/scale)) * m, restored to m by the post-EvalMod
        // plaintext correction (s2c_first_correct_amplitude). Use
        // kLowS2cLayer3Factor: small F keeps the EvalMod fractional part tiny
        // (extraction error ~F^3) and the post-S2C polynomial coefficients far
        // below q0/2. Do NOT use sfl_full_3's working-chain scale-factor formula
        // here (~32 at this level): it pushes the fractional part far outside
        // the EvalMod approximation domain.
        rtn.resize(n);
        for (size_t i = 0; i < n; i++)
        {
            const double factor = layer3_factor;
            vector<vector<complex<double>>> fftcoeff3_scale(2 * totlen3 + 1);
            for (int r = 0; r < totlen2 + 1; r++)
            {
                fftcoeff3_scale[r].resize(bootstrapper.n);
                for (long j = 0; j < bootstrapper.n; j++)
                    fftcoeff3_scale[r][j] = bootstrapper.fftcoeff3[bootstrapper.slot_index][r][j] * factor;
            }
            bootstrapper.rotated_bsgs_linear_transform(rtn[i], stage1[i], totlen3, basicstep3, logn, fftcoeff3_scale);
            ckks_evaluator.evaluator.rescale_to_next_inplace(rtn[i]);
            // keep the natural (preserved) scale: the output feeds the modraise,
            // whose downstream C2S/EvalMod expect the original message scale
            rtn[i].scale() = stage1[i].scale();
        }
    });
    layer_keys = PhantomGaloisKey();
}

/**
 * Amplitude + scale correction for the S2C-first tails: the chain up to EvalMod
 * yields (F / (q0/scale)) * m at the EvalMod output scale s_em. Multiplying by
 * the plaintext constant C = (q0/scale)/F encoded at scale scale/s_em recovers
 * the message m AND the original ckks scale in one shot (a plaintext multiply
 * consumes no level; its value and its encoding scale are independent knobs).
 */
static void s2c_first_correct_amplitude(CKKSEvaluator &ckks_evaluator, vector<PhantomCiphertext> &cts,
                                        double layer3_factor)
{
    const double q0 =
        static_cast<double>(ckks_evaluator.context->first_context_data().parms().coeff_modulus()[0].value());
    const double corr = (q0 / ckks_evaluator.scale) / layer3_factor;
    vector<double> cvec(ckks_evaluator.encoder.slot_count(), corr);
    for (auto &ct : cts)
    {
        PhantomPlaintext cp;
        ckks_evaluator.encoder.encode(cvec, ckks_evaluator.scale / ct.scale(), cp);
        ckks_evaluator.evaluator.mod_switch_to_inplace(cp, ct.params_id());
        ckks_evaluator.evaluator.multiply_plain(ct, cp, ct);
    }
}

/**
 * Reusable batch-bootstrap pipeline; see the file header for the two paths and the
 * timing convention. Typical use:
 *
 *   BatchBootstrapper pipe(bootstrapper, ckks_evaluator, batch_n);
 *   pipe.prepare();                              // LT coefficients, one-time
 *   pipe.prepare_shared_keys(secret_key, chain); // shared-a key material, one-time
 *   for each chunk of batch_n bottom-level ciphertexts:
 *       pipe.bootstrap_shared_a(out, chunk, keygen_sec, transform_sec);
 */
class BatchBootstrapper
{
public:
    Bootstrapper &bootstrapper;
    CKKSEvaluator *ckks;
    PhantomContext *context;
    size_t batch_n;

    // one-time shared-a key material
    std::unique_ptr<moai::SharedAKeyGen> keygen;
    moai::SharedAFormatKey fmt_key_bottom;
    bool shared_keys_ready = false;
    double shared_prep_sec = 0.0; // wall time of prepare_shared_keys (one-time)

    BatchBootstrapper(Bootstrapper &boot, CKKSEvaluator &ckks_evaluator, size_t n)
        : bootstrapper(boot), ckks(&ckks_evaluator), context(ckks_evaluator.context), batch_n(n)
    {
        if (n == 0)
            throw invalid_argument("BatchBootstrapper: batch size must be positive");
    }

    /** One-time setup: minimax polynomials + linear-transform coefficients. */
    void prepare()
    {
        bootstrapper.slot_vec.push_back(bootstrapper.logn);
        bootstrapper.slot_index = 0;
        bootstrapper.prepare_mod_polynomial();
        bootstrapper.generate_LT_coefficient_3();
        bootstrapper.initial_scale = ckks->scale; // what bootstrap_3 would set from the input ct
    }

    /**
     * One-time shared-a key material: auxiliary secrets sk'_i and the format key
     * at the level where fmt_switch_forward will run. Reused across all
     * subsequent bootstrap_shared_a / bootstrap_shared_a_s2c_first calls.
     *
     * input_chain_index: chain index of the bootstrap INPUT ciphertexts
     * (fmt_switch_forward runs at their level). For the C2S-first path this is
     * the bottom level (1 prime); for the S2C-first path it is the input level
     * (lS2C+1 = 4 primes) — NOT the bottom, despite the historical member name
     * fmt_key_bottom.
     */
    void prepare_shared_keys(PhantomSecretKey &secret_key, size_t input_chain_index)
    {
        const auto &stream = phantom::util::global_variables::default_stream->get_stream();
        auto t0 = system_clock::now();
        keygen = std::make_unique<moai::SharedAKeyGen>(*context, secret_key, batch_n);
        if (amortize_chunk_keys_)
            keygen->set_galois_key_cache(galois_cache_budget_); // 摊销: 开启 Sh-Galois 密钥缓存
        fmt_key_bottom = keygen->gen_format_key(input_chain_index, stream);
        cudaStreamSynchronize(stream);
        shared_prep_sec = duration<double>(system_clock::now() - t0).count();
        shared_keys_ready = true;
    }

    /**
     * 摊销 / 不摊销开关 (per-chunk key material caching).
     *
     * 不摊销 (default): every chunk regenerates its Sh-Galois keys and the
     *   output-level format key fmt_key_out. This is the "total" (conservative)
     *   figure in the benchmark reports.
     * 摊销 (enabled here): the first chunk's key material is cached on the GPU and
     *   reused by every later chunk at the same levels (the keys are identical in
     *   distribution across chunks: same secrets, same rotation steps, same
     *   levels). This realizes the "keygen amortized" figure. GPU memory cost:
     *     - fmt_key_out: (n + n^2) * size_QlP * N * 8  +  n * dnum * 2 * size_QP * N * 8
     *     - each cached Sh-Galois key: dnum * (n+1) * size_QP * N * 8
     *   galois_cache_bytes caps the Sh-Galois cache (steps beyond the budget are
     *   still regenerated on the fly). Cache statistics: galois_cache_stats().
     */
    void enable_chunk_key_amortization(size_t galois_cache_bytes)
    {
        amortize_chunk_keys_ = true;
        galois_cache_budget_ = galois_cache_bytes;
        if (keygen)
            keygen->set_galois_key_cache(galois_cache_bytes);
    }

    /** 摊销模式缓存统计: (cached Sh-Galois keys, their total bytes). */
    std::pair<size_t, size_t> galois_cache_stats() const
    {
        return keygen ? std::make_pair(keygen->galois_cache_size(), keygen->galois_cache_bytes())
                      : std::make_pair<size_t, size_t>(0, 0);
    }

    /** Release the cached Sh-Galois keys (~26.7 GiB at n=4) so non-bootstrap
     * phases can use the memory. The cache regenerates on the next bootstrap
     * segment (one extra keyset keygen, accounted as preprocessing). */
    void release_galois_cache()
    {
        if (keygen)
            keygen->clear_galois_key_cache();
    }

    // 摊销模式状态 (see enable_chunk_key_amortization); 不摊销时全部闲置
    bool amortize_chunk_keys_ = false;   // false = 不摊销 (每组重新生成密钥材料)
    size_t galois_cache_budget_ = 0;     // Sh-Galois 密钥缓存的显存预算
    moai::SharedAFormatKey cached_fmt_key_out_; // 摊销缓存的输出侧格式密钥
    bool fmt_key_out_cached_ = false;

    /**
     * Shared-a C2S-first bootstrap (MaMBo paper Sec 6.1) of exactly batch_n
     * bottom-level ciphertexts:
     *   FmtSwitch -> shared-a modraise -> Sh-C2S -> Backward-FmtSwitch
     *   -> ordinary EvalMod -> ordinary S2C.
     * (Named "hybrid" historically: the linear transforms run in the shared-a
     * format while EvalMod and the final S2C are per-member. The S2C-first
     * variant used by the single-layer test is bootstrap_shared_a_s2c_first.)
     * The inputs are not modified. Per-batch key generation (Sh-Rot Galois keys,
     * output-level format key, ordinary S2C layer keys) goes to keygen_sec.
     */
    void bootstrap_shared_a(vector<PhantomCiphertext> &rtn, vector<PhantomCiphertext> &cts,
                            double &keygen_sec, double &transform_sec)
    {
        if (!shared_keys_ready)
            throw logic_error("BatchBootstrapper: call prepare_shared_keys first");
        if (cts.size() != batch_n)
            throw invalid_argument("BatchBootstrapper: shared-a bootstrap needs exactly batch_n ciphertexts");
        const auto &stream = phantom::util::global_variables::default_stream->get_stream();

        auto t0 = system_clock::now();
        moai::SharedACiphertextBatch batch;
        moai::fmt_switch_forward(*context, cts, fmt_key_bottom, batch, stream);
        moai::shared_a_modraise_inplace(*context, batch, stream);
        batch.scale_ = static_cast<double>(context->first_context_data().parms().coeff_modulus()[0].value());
        cudaStreamSynchronize(stream);
        transform_sec += duration<double>(system_clock::now() - t0).count();

        keygen->galois_keygen_sec_ = 0.0;
        t0 = system_clock::now();
        moai::SharedACiphertextBatch c2s1, c2s2;
        bootstrapper.shared_a_coefftoslot_full_3(c2s1, c2s2, batch, *keygen);
        cudaStreamSynchronize(stream);
        const double c2s_total = duration<double>(system_clock::now() - t0).count();
        keygen_sec += keygen->galois_keygen_sec_;
        transform_sec += c2s_total - keygen->galois_keygen_sec_;

        t0 = system_clock::now();
        moai::SharedAFormatKey fmt_key_out = keygen->gen_format_key(c2s1.chain_index_, stream);
        cudaStreamSynchronize(stream);
        keygen_sec += duration<double>(system_clock::now() - t0).count();

        t0 = system_clock::now();
        vector<PhantomCiphertext> sa1, sa2;
        moai::fmt_switch_backward(*context, c2s1, fmt_key_out, sa1, stream);
        moai::fmt_switch_backward(*context, c2s2, fmt_key_out, sa2, stream);
        cudaStreamSynchronize(stream);
        transform_sec += duration<double>(system_clock::now() - t0).count();
        std::cout << "  [dbg] shared-a tail input: " << sa1[0].coeff_modulus_size() << " primes, chain "
                  << sa1[0].chain_index() << std::endl;
        {
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            std::cout << "  [dbg] GPU mem before tail: used " << (tot - fr) / 1e9 << " / " << tot / 1e9 << " GB"
                      << std::endl;
        }

        // release the shared-a batches before the ordinary tail allocates keys
        batch.release();
        c2s1.release();
        c2s2.release();

        // The shared-a phase churns >17GB per chunk through the default memory pool
        // (per-step Sh-Rot key generation), and with phantom's UINT64_MAX release
        // threshold the retained odd-sized blocks fragment the pool, pushing the
        // ordinary tail's bsgs allocations onto the (WSL2-slow) driver path
        // (measured 15-23x slowdown of bsgs_linear_transform). Trimming here cannot
        // fix that - the next chunk refills the pool anyway - but it measurably
        // reduces the per-chunk key regeneration time (~4s -> ~2.2s per chunk).
        {
            cudaMemPool_t pool;
            cudaDeviceGetDefaultMemPool(&pool, 0);
            cudaMemPoolTrimTo(pool, 0);
        }

        ordinary_evalmod_s2c_batch(bootstrapper, *ckks, rtn, sa1, sa2, keygen_sec, transform_sec);
    }

    /**
     * Ordinary bootstrap of any number of bottom-level ciphertexts:
     *   modraise -> coefftoslot_full_3 -> EvalMod -> slottocoeff_full_3,
     * Galois keys generated once per layer for the whole call. Inputs are not
     * modified (the path works on deep copies).
     */
    void bootstrap_ordinary(vector<PhantomCiphertext> &rtn, vector<PhantomCiphertext> &cts,
                            double &keygen_sec, double &transform_sec)
    {
        vector<PhantomCiphertext> copy;
        copy.reserve(cts.size());
        for (auto &ct : cts)
            copy.push_back(ct); // deep copy; cts stays intact
        vector<PhantomCiphertext> ct1, ct2;
        ordinary_coefftoslot_full_3_batch(bootstrapper, *ckks, ct1, ct2, copy, keygen_sec, transform_sec);
        ordinary_evalmod_s2c_batch(bootstrapper, *ckks, rtn, ct1, ct2, keygen_sec, transform_sec);
    }

    /**
     * Ordinary S2C-first bootstrap (MaMBo paper Sec 2.2) of any number of
     * ciphertexts that still have lS2C+1 = 4 primes left:
     *   sfl_full_3 (S2C at LOW level) -> modraise -> coefftoslot_full_3 (C2S at
     *   high level) -> EvalMod -> recombine (mod1 + i*mod2).
     * Galois keys generated once per layer for the whole call. Inputs are not
     * modified (the path works on deep copies).
     */
    void bootstrap_ordinary_s2c_first(vector<PhantomCiphertext> &rtn, vector<PhantomCiphertext> &cts,
                                      double &keygen_sec, double &transform_sec,
                                      double layer3_factor = kLowS2cLayer3Factor)
    {
        vector<PhantomCiphertext> copy;
        copy.reserve(cts.size());
        for (auto &ct : cts)
            copy.push_back(ct); // deep copy; cts stays intact
        if (copy[0].coeff_modulus_size() != 4)
            throw invalid_argument("bootstrap_ordinary_s2c_first: inputs must have exactly 4 primes (lS2C+1)");

        // S2C at the low input level (consumes 3 levels -> bottom)
        vector<PhantomCiphertext> s2c_out;
        ordinary_sfl_full_3_batch(bootstrapper, *ckks, s2c_out, copy, keygen_sec, transform_sec,
                                  layer3_factor);

        // C2S at the top level (ordinary_coefftoslot_full_3_batch performs the
        // bottom-level modraise + scale := q0 itself), then EvalMod + recombine
        vector<PhantomCiphertext> ct1, ct2;
        ordinary_coefftoslot_full_3_batch(bootstrapper, *ckks, ct1, ct2, s2c_out, keygen_sec, transform_sec);
        ordinary_evalmod_combine_batch(bootstrapper, *ckks, rtn, ct1, ct2, keygen_sec, transform_sec);

        // amplitude + scale correction (see s2c_first_correct_amplitude)
        {
            const auto &stream = phantom::util::global_variables::default_stream->get_stream();
            auto t0 = system_clock::now();
            s2c_first_correct_amplitude(*ckks, rtn, layer3_factor);
            cudaStreamSynchronize(stream);
            transform_sec += duration<double>(system_clock::now() - t0).count();
        }
    }

    /**
     * Shared-a S2C-first bootstrap of exactly batch_n ciphertexts with 4 primes
     * left:
     *   FmtSwitch -> Sh-S2C (low level) -> shared-a modraise -> Sh-C2S (high
     *   level) -> Backward-FmtSwitch -> per-member EvalMod + recombine.
     * All linear transforms run in the shared-a format; only EvalMod is per
     * member. The one-time key material must have been prepared at the INPUT
     * level (prepare_shared_keys with the input chain index, not the bottom).
     * layer3_factor: the S2C-first amplitude convention F (must match the
     * factor baked into shared_a_sfl_full_3's third layer and the post-EvalMod
     * correction s2c_first_correct_amplitude — keep all three in sync by
     * passing it through this parameter instead of editing call sites).
     */
    void bootstrap_shared_a_s2c_first(vector<PhantomCiphertext> &rtn, vector<PhantomCiphertext> &cts,
                                      double &keygen_sec, double &transform_sec,
                                      double layer3_factor = kLowS2cLayer3Factor)
    {
        if (!shared_keys_ready)
            throw logic_error("BatchBootstrapper: call prepare_shared_keys first");
        if (cts.size() != batch_n)
            throw invalid_argument("BatchBootstrapper: shared-a bootstrap needs exactly batch_n ciphertexts");
        if (cts[0].coeff_modulus_size() != 4)
            throw invalid_argument("bootstrap_shared_a_s2c_first: inputs must have exactly 4 primes (lS2C+1)");
        const auto &stream = phantom::util::global_variables::default_stream->get_stream();

        auto t0 = system_clock::now();
        moai::SharedACiphertextBatch batch;
        moai::fmt_switch_forward(*context, cts, fmt_key_bottom, batch, stream);
        cudaStreamSynchronize(stream);
        transform_sec += duration<double>(system_clock::now() - t0).count();

        // Sh-S2C at the low input level (consumes 3 levels -> bottom)
        keygen->galois_keygen_sec_ = 0.0;
        t0 = system_clock::now();
        moai::SharedACiphertextBatch s2c_out;
        bootstrapper.shared_a_sfl_full_3(s2c_out, batch, *keygen, layer3_factor);
        cudaStreamSynchronize(stream);
        double sec = duration<double>(system_clock::now() - t0).count();
        keygen_sec += keygen->galois_keygen_sec_;
        transform_sec += sec - keygen->galois_keygen_sec_;
        batch.release();

        // shared-a modraise at the bottom (scale := q0, mirrors bootstrap_full_3)
        t0 = system_clock::now();
        moai::shared_a_modraise_inplace(*context, s2c_out, stream);
        s2c_out.scale_ = static_cast<double>(context->first_context_data().parms().coeff_modulus()[0].value());
        cudaStreamSynchronize(stream);
        transform_sec += duration<double>(system_clock::now() - t0).count();

        // Sh-C2S at the top level
        keygen->galois_keygen_sec_ = 0.0;
        t0 = system_clock::now();
        moai::SharedACiphertextBatch c2s1, c2s2;
        bootstrapper.shared_a_coefftoslot_full_3(c2s1, c2s2, s2c_out, *keygen);
        cudaStreamSynchronize(stream);
        sec = duration<double>(system_clock::now() - t0).count();
        keygen_sec += keygen->galois_keygen_sec_;
        transform_sec += sec - keygen->galois_keygen_sec_;
        s2c_out.release();

        // output-level format key. 不摊销: regenerated for every chunk; 摊销:
        // generated on the first chunk and reused afterwards (same secrets and the
        // same level across chunks, so one copy is valid for all of them).
        const moai::SharedAFormatKey *fmt_key_out;
        moai::SharedAFormatKey fmt_key_out_fresh;
        if (amortize_chunk_keys_ && fmt_key_out_cached_ && cached_fmt_key_out_.chain_index_ == c2s1.chain_index_)
        {
            fmt_key_out = &cached_fmt_key_out_;
        }
        else
        {
            t0 = system_clock::now();
            fmt_key_out_fresh = keygen->gen_format_key(c2s1.chain_index_, stream);
            cudaStreamSynchronize(stream);
            keygen_sec += duration<double>(system_clock::now() - t0).count();
            if (amortize_chunk_keys_)
            {
                cached_fmt_key_out_ = std::move(fmt_key_out_fresh);
                fmt_key_out_cached_ = true;
                fmt_key_out = &cached_fmt_key_out_;
            }
            else
            {
                fmt_key_out = &fmt_key_out_fresh;
            }
        }

        t0 = system_clock::now();
        vector<PhantomCiphertext> sa1, sa2;
        moai::fmt_switch_backward(*context, c2s1, *fmt_key_out, sa1, stream);
        moai::fmt_switch_backward(*context, c2s2, *fmt_key_out, sa2, stream);
        cudaStreamSynchronize(stream);
        transform_sec += duration<double>(system_clock::now() - t0).count();

        // release the shared-a batches before the ordinary tail allocates keys
        c2s1.release();
        c2s2.release();

        // see bootstrap_shared_a for why we trim the pool here
        {
            cudaMemPool_t pool;
            cudaDeviceGetDefaultMemPool(&pool, 0);
            cudaMemPoolTrimTo(pool, 0);
        }

        // per-member EvalMod + recombine (no further linear transforms)
        ordinary_evalmod_combine_batch(bootstrapper, *ckks, rtn, sa1, sa2, keygen_sec, transform_sec);

        // amplitude + scale correction (see s2c_first_correct_amplitude)
        t0 = system_clock::now();
        s2c_first_correct_amplitude(*ckks, rtn, layer3_factor);
        cudaStreamSynchronize(stream);
        transform_sec += duration<double>(system_clock::now() - t0).count();
    }
};
