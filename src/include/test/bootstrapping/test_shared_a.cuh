#pragma once

#include <random>
#include <set>

#include "include.cuh"
#include "source/bootstrapping/BatchBootstrap.cuh"
#include "polymath.cuh"
#include "ntt.cuh"

using namespace std;
using namespace phantom;
using namespace moai;

/** Report and clear any pending CUDA error, tagged with a location. */
static void report_cuda_error(const char *where)
{
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        std::cout << "  *** CUDA ERROR at " << where << ": " << cudaGetErrorString(e) << " ***" << endl;
}

/**
 * Unit test for the shared-a (multi-secret RLWE) format conversion, paper Sec. 3:
 *   1. encrypt n messages under the shared secret sk, drop them to the bottom level
 *   2. FmtSwitch -> shared-a batch under auxiliary secrets sk'_0..sk'_{n-1}
 *   3. decrypt each member with sk'_i and compare with the original message
 *   4. Backward-FmtSwitch -> ordinary ciphertexts under sk, decrypt and compare
 *
 * keep_primes: how many RNS primes the input ciphertexts are dropped to before
 * the format switch (1 = bottom level, the original test).
 */
void shared_a_fmt_switch_test(size_t keep_primes = 1)
{
    // same CKKS parameters as the bootstrapping unit test
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    // FmtSwitch requires P >= Ql at the S2C input level (MaMBo paper Lemma 1);
    // with 4 regular primes left (~189 bits) we use 5 special primes (~255 bits)
    int num_special_primes = 5;

    int secret_key_hamming_weight = 192;

    int remaining_level = 10;
    int boot_level = 14;
    int total_level = remaining_level + boot_level;

    const size_t batch_n = 4; //批处理大小

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
        coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; i++)
        coeff_bit_vec.push_back(logq);
    for (int i = 0; i < num_special_primes; i++)
        coeff_bit_vec.push_back(log_special_prime);

    std::cout << "Setting Parameters..." << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_special_modulus_size(num_special_primes);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    PhantomContext context(parms);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();

    // auxiliary secrets + format key
    std::cout << "Generating shared-a key material (batch n = " << batch_n << ")..." << endl;
    SharedAKeyGen keygen(context, secret_key, batch_n);

    // random messages, encrypt, drop to the bottom level
    vector<vector<double>> inputs(batch_n, vector<double>(slot_count, 0.0));
    vector<PhantomCiphertext> cts;
    cts.reserve(batch_n);
    for (size_t i = 0; i < batch_n; i++)
    {
        vector<double> sparse(sparse_slots, 0.0);
        random_real(sparse, sparse_slots);
        for (size_t k = 0; k < slot_count; k++)
            inputs[i][k] = sparse[k % sparse_slots];

        PhantomPlaintext plain;
        PhantomCiphertext cipher;
        ckks_evaluator.encoder.encode(inputs[i], scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, cipher);
        const int drops = total_level + 1 - static_cast<int>(keep_primes);
        for (int lvl = 0; lvl < drops; lvl++)
            ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
        cts.push_back(std::move(cipher));
    }
    size_t bottom_chain = cts[0].chain_index();
    std::cout << "Bottom chain index: " << bottom_chain
              << ", coeff_modulus_size: " << cts[0].coeff_modulus_size() << endl;

    auto start = system_clock::now();
    SharedAFormatKey fmt_key = keygen.gen_format_key(bottom_chain, stream);
    cudaStreamSynchronize(stream);
    duration<double> sec = system_clock::now() - start;
    std::cout << "Format key generation took: " << sec.count() << "s" << endl;

    // forward: n ordinary ciphertexts -> shared-a batch
    SharedACiphertextBatch batch;
    start = system_clock::now();
    fmt_switch_forward(context, cts, fmt_key, batch, stream);
    cudaStreamSynchronize(stream);
    sec = system_clock::now() - start;
    std::cout << "FmtSwitch (forward) took: " << sec.count() * 1000 << " ms" << endl;

    // check: decrypt member i with auxiliary secret sk'_i
    double max_err_shared = 0;
    for (size_t i = 0; i < batch_n; i++)
    {
        PhantomCiphertext member = batch.assemble(context, i, stream);
        PhantomPlaintext plain;
        keygen.aux_secret_key(i).decrypt(context, member, plain);
        vector<double> decoded(slot_count, 0.0);
        ckks_evaluator.encoder.decode(plain, decoded);
        double err = 0, mean = 0;
        for (long k = 0; k < sparse_slots; k++)
        {
            err = max(err, abs(inputs[i][k] - decoded[k]));
            mean += abs(inputs[i][k] - decoded[k]);
        }
        mean /= sparse_slots;
        std::cout << "  member " << i << ": max " << err << ", mean " << mean << endl;
        max_err_shared = max(max_err_shared, err);
    }
    std::cout << "Max error after FmtSwitch (decrypt with sk'_i): " << max_err_shared << endl;

    if (max_err_shared > 1e-3)
    {
        std::cout << "=== FAILURE: member 0 first slots ===" << endl;
        PhantomCiphertext member = batch.assemble(context, 0, stream);
        PhantomPlaintext plain;
        keygen.aux_secret_key(0).decrypt(context, member, plain);
        vector<double> decoded(slot_count, 0.0);
        ckks_evaluator.encoder.decode(plain, decoded);
        for (long k = 0; k < 8; k++)
            std::cout << "  slot " << k << ": input " << inputs[0][k] << " decoded " << decoded[k] << endl;
    }

    // sanity: decrypt the inputs under sk
    double max_err_input = 0;
    for (size_t i = 0; i < batch_n; i++)
    {
        PhantomPlaintext plain;
        ckks_evaluator.decryptor.decrypt(cts[i], plain);
        vector<double> decoded(slot_count, 0.0);
        ckks_evaluator.encoder.decode(plain, decoded);
        for (long k = 0; k < sparse_slots; k++)
            max_err_input = max(max_err_input, abs(inputs[i][k] - decoded[k]));
    }
    std::cout << "Max error of input ciphertexts (decrypt with sk): " << max_err_input << endl;

    // backward: shared-a batch -> n ordinary ciphertexts under sk
    vector<PhantomCiphertext> out;
    start = system_clock::now();
    fmt_switch_backward(context, batch, fmt_key, out, stream);
    cudaStreamSynchronize(stream);
    sec = system_clock::now() - start;
    std::cout << "Backward-FmtSwitch took: " << sec.count() * 1000 << " ms" << endl;

    double max_err_back = 0;
    for (size_t i = 0; i < batch_n; i++)
    {
        PhantomPlaintext plain;
        ckks_evaluator.decryptor.decrypt(out[i], plain);
        vector<double> decoded(slot_count, 0.0);
        ckks_evaluator.encoder.decode(plain, decoded);
        double err = 0;
        for (long k = 0; k < sparse_slots; k++)
            err = max(err, abs(inputs[i][k] - decoded[k]));
        max_err_back = max(max_err_back, err);
    }
    std::cout << "Max error after Backward-FmtSwitch (decrypt with sk): " << max_err_back << endl;

    // ---- shared-a rotation (Sh-Rot, paper Sec. 6.1) vs phantom's ordinary rotate ----
    const int rot_step = 5;
    std::cout << "Generating shared-a Galois key (step " << rot_step << ")..." << endl;
    start = system_clock::now();
    SharedAGaloisKey sh_gal_key = keygen.gen_shared_galois_key(rot_step, stream);
    cudaStreamSynchronize(stream);
    sec = system_clock::now() - start;
    std::cout << "Shared-a Galois key generation took: " << sec.count() << "s" << endl;

    SharedACiphertextBatch rot_batch;
    start = system_clock::now();
    shared_a_rotate(context, batch, sh_gal_key, rot_batch, stream);
    cudaStreamSynchronize(stream);
    sec = system_clock::now() - start;
    std::cout << "Sh-Rot took: " << sec.count() * 1000 << " ms" << endl;

    // reference: ordinary rotate of the original ciphertexts under sk
    PhantomGaloisKey ref_gal_keys = secret_key.create_galois_keys_from_steps(context, {rot_step});
    double max_err_rot = 0;
    for (size_t i = 0; i < batch_n; i++)
    {
        PhantomCiphertext ref = cts[i]; // cts is no longer used after this point
        rotate_vector_inplace(context, ref, rot_step, ref_gal_keys,
                              *phantom::util::global_variables::default_stream);
        PhantomPlaintext ref_plain;
        ckks_evaluator.decryptor.decrypt(ref, ref_plain);
        vector<double> ref_decoded(slot_count, 0.0);
        ckks_evaluator.encoder.decode(ref_plain, ref_decoded);

        PhantomCiphertext member = rot_batch.assemble(context, i, stream);
        PhantomPlaintext plain;
        keygen.aux_secret_key(i).decrypt(context, member, plain);
        vector<double> decoded(slot_count, 0.0);
        ckks_evaluator.encoder.decode(plain, decoded);
        double err = 0;
        for (long k = 0; k < sparse_slots; k++)
            err = max(err, abs(ref_decoded[k] - decoded[k]));
        max_err_rot = max(max_err_rot, err);
    }
    std::cout << "Max error of Sh-Rot vs ordinary rotate: " << max_err_rot << endl;

    if (max_err_shared < 1e-3 && max_err_back < 1e-3 && max_err_rot < 1e-3)
        std::cout << "shared-a format switch test PASSED" << endl;
    else
        std::cout << "shared-a format switch test FAILED (error too large)" << endl;
}

/**
 * Repeated randomized test of the shared-a format switch: fresh auxiliary secrets
 * and format key every round, batch of 8. Checks both the decode-path error and a
 * direct coefficient-domain error E_i = (b'_i + a' sk'_i) - (b_i + a_i sk) mod q0.
 */
void shared_a_stress_test(int rounds)
{
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    // FmtSwitch requires P >= Ql at the S2C input level (MaMBo paper Lemma 1);
    // with 4 regular primes left (~189 bits) we use 5 special primes (~255 bits)
    int num_special_primes = 5;

    int secret_key_hamming_weight = 192;

    int remaining_level = 10;
    int boot_level = 14;
    int total_level = remaining_level + boot_level;

    const size_t batch_n = 8;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
        coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; i++)
        coeff_bit_vec.push_back(logq);
    for (int i = 0; i < num_special_primes; i++)
        coeff_bit_vec.push_back(log_special_prime);

    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_special_modulus_size(num_special_primes);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    PhantomContext context(parms);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();

    int failures = 0;
    for (int round = 0; round < rounds; round++)
    {
        SharedAKeyGen keygen(context, secret_key, batch_n);

        vector<vector<double>> inputs(batch_n, vector<double>(slot_count, 0.0));
        vector<PhantomCiphertext> cts;
        cts.reserve(batch_n);
        for (size_t i = 0; i < batch_n; i++)
        {
            vector<double> sparse(sparse_slots, 0.0);
            random_real(sparse, sparse_slots);
            for (size_t k = 0; k < slot_count; k++)
                inputs[i][k] = sparse[k % sparse_slots];

            PhantomPlaintext plain;
            PhantomCiphertext cipher;
            ckks_evaluator.encoder.encode(inputs[i], scale, plain);
            ckks_evaluator.encryptor.encrypt(plain, cipher);
            for (int lvl = 0; lvl < total_level; lvl++)
                ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
            cts.push_back(std::move(cipher));
        }
        const size_t bottom_chain = cts[0].chain_index();

        auto input_err = [&]() {
            double e = 0;
            for (size_t i = 0; i < batch_n; i++)
            {
                PhantomPlaintext plain;
                ckks_evaluator.decryptor.decrypt(cts[i], plain);
                vector<double> decoded(slot_count, 0.0);
                ckks_evaluator.encoder.decode(plain, decoded);
                for (long k = 0; k < sparse_slots; k++)
                    e = max(e, abs(inputs[i][k] - decoded[k]));
            }
            return e;
        };
        const double pre_err = input_err();
        if (pre_err > 1e-3)
            std::cout << "[round " << round << "] WARNING: inputs already bad BEFORE forward: " << pre_err << endl;

        SharedAFormatKey fmt_key = keygen.gen_format_key(bottom_chain, stream);
        SharedACiphertextBatch batch;
        fmt_switch_forward(context, cts, fmt_key, batch, stream);
        cudaStreamSynchronize(stream);
        report_cuda_error("after fmt_switch_forward");

        // direct batch correctness check in the coefficient domain, bypassing the
        // phantom decrypt/encoder path: E_i must be small (~2^20)
        auto direct_err = [&](SharedACiphertextBatch &bch, SharedAKeyGen &kg) {
            const size_t N = poly_modulus_degree;
            const uint64_t q0 = context.get_context_data(bottom_chain).parms().coeff_modulus()[0].value();
            auto &ntt_tables = context.gpu_rns_tables();
            auto mod1 = phantom::util::make_cuda_auto_ptr<DModulus>(1, stream);
            cudaMemcpyAsync(mod1.get(), ntt_tables.modulus(), sizeof(DModulus), cudaMemcpyDeviceToDevice, stream);
            auto E = phantom::util::make_cuda_auto_ptr<uint64_t>(N, stream);
            auto t1 = phantom::util::make_cuda_auto_ptr<uint64_t>(N, stream);
            vector<uint64_t> h_E(N);
            double worst = 0;
            for (size_t i = 0; i < batch_n; i++)
            {
                multiply_rns_poly<<<N / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                        bch.a_.get(), kg.aux_secret_key(i).secret_key_poly(), mod1.get(), t1.get(), N, 1);
                add_rns_poly<<<N / blockDimGlb.x, blockDimGlb, 0, stream>>>(bch.b_[i].get(), t1.get(), mod1.get(),
                                                                            E.get(), N, 1);
                multiply_rns_poly<<<N / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                        cts[i].data() + N, secret_key.secret_key_poly(), mod1.get(), t1.get(), N, 1);
                sub_rns_poly<<<N / blockDimGlb.x, blockDimGlb, 0, stream>>>(E.get(), cts[i].data(), mod1.get(),
                                                                            E.get(), N, 1);
                sub_rns_poly<<<N / blockDimGlb.x, blockDimGlb, 0, stream>>>(E.get(), t1.get(), mod1.get(), E.get(), N,
                                                                            1);
                nwt_2d_radix8_backward_inplace(E.get(), ntt_tables, 1, 0, stream);
                cudaMemcpyAsync(h_E.data(), E.get(), N * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                uint64_t m = 0;
                for (size_t k = 0; k < N; k++)
                {
                    const uint64_t c = h_E[k] < q0 - h_E[k] ? h_E[k] : q0 - h_E[k];
                    if (c > m)
                        m = c;
                }
                if (static_cast<double>(m) > worst)
                    worst = static_cast<double>(m);
            }
            return worst;
        };
        const double d_err = direct_err(batch, keygen);
        report_cuda_error("after direct_err");

        auto member_err = [&](SharedACiphertextBatch &b, SharedAKeyGen &kg, size_t i) {
            PhantomCiphertext member = b.assemble(context, i, stream);
            PhantomPlaintext plain;
            kg.aux_secret_key(i).decrypt(context, member, plain);
            vector<double> decoded(slot_count, 0.0);
            ckks_evaluator.encoder.decode(plain, decoded);
            double err = 0;
            for (long k = 0; k < sparse_slots; k++)
                err = max(err, abs(inputs[i][k] - decoded[k]));
            return err;
        };

        double max_err_shared = 0;
        size_t worst_member = 0;
        for (size_t i = 0; i < batch_n; i++)
        {
            const double e_i = member_err(batch, keygen, i);
            if (e_i > max_err_shared)
            {
                max_err_shared = e_i;
                worst_member = i;
            }
        }

        double max_err_input = input_err();

        std::cout << "[round " << round << "] input err " << max_err_input << ", direct err " << d_err
                  << ", member errs";
        for (size_t i = 0; i < batch_n; i++)
            std::cout << " " << member_err(batch, keygen, i);
        std::cout << " (max " << max_err_shared << " @" << worst_member << ")";
        if (max_err_shared < 1e-3)
        {
            std::cout << " OK" << endl;
            continue;
        }

        failures++;
        std::cout << " FAILURE" << endl;
        std::cout << "  worst member: " << worst_member << endl;
        {
            PhantomCiphertext member = batch.assemble(context, worst_member, stream);
            PhantomPlaintext plain;
            keygen.aux_secret_key(worst_member).decrypt(context, member, plain);
            vector<double> decoded(slot_count, 0.0);
            ckks_evaluator.encoder.decode(plain, decoded);
            for (long k = 0; k < 8; k++)
                std::cout << "  slot " << k << ": input " << inputs[worst_member][k] << " decoded " << decoded[k]
                          << endl;
        }
    }
    std::cout << "stress test done: " << failures << " failures in " << rounds << " rounds" << endl;
}

// ---------------------------------------------------------------------------
// shared-a coefficient-to-slot test
// ---------------------------------------------------------------------------

// collect_rotated_layer_steps / collect_bsgs_layer_steps moved to
// source/bootstrapping/BatchBootstrap.cuh

/**
 * Ordinary coefftoslot_3 with the Galois keys generated layer by layer (and freed
 * between layers) so the whole key set never has to fit in GPU memory at once.
 * Transform time (excluding key generation) is accumulated into transform_sec.
 */
static void ordinary_coefftoslot_full_3_chunked(Bootstrapper &bootstrapper, CKKSEvaluator &ckks_evaluator,
                                                PhantomCiphertext &rtn1, PhantomCiphertext &rtn2,
                                                PhantomCiphertext &cipher, double &transform_sec)
{
    const long logn = bootstrapper.logn;
    const long Nh = bootstrapper.Nh;
    int div_part1 = floor(logn / 3.0);
    int div_part2 = floor((logn - div_part1) / 2.0);
    int div_part3 = logn - div_part1 - div_part2;
    int totlen1 = (1 << div_part1) - 1, totlen2 = (1 << div_part2) - 1, totlen3 = (1 << div_part3) - 1;
    int basicstep1 = (1 << (logn - div_part1)), basicstep2 = (1 << (logn - div_part1 - div_part2)), basicstep3 = 1;

    // NOTE: the Bootstrapper transforms always rotate with *(ckks->galois_keys), so the
    // per-layer keys must be created into that member (and freed between layers).
    PhantomGaloisKey &layer_keys = *(ckks_evaluator.galois_keys);
    PhantomCiphertext tmpct, tmpct2, tmpct3;

    auto timed = [&](auto &&fn) {
        const auto &stream = phantom::util::global_variables::default_stream->get_stream();
        auto t0 = system_clock::now();
        fn();
        cudaStreamSynchronize(stream);
        transform_sec += duration<double>(system_clock::now() - t0).count();
    };

    auto steps1 = collect_rotated_layer_steps(totlen1, basicstep1, Nh);
    ckks_evaluator.decryptor.create_galois_keys_from_steps(steps1, layer_keys);
    timed([&] {
        bootstrapper.rotated_bsgs_linear_transform(tmpct, cipher, totlen1, basicstep1, logn,
                                                   bootstrapper.invfftcoeff1[bootstrapper.slot_index]);
        ckks_evaluator.evaluator.rescale_to_next_inplace(tmpct);
    });
    layer_keys = PhantomGaloisKey();

    auto steps2 = collect_bsgs_layer_steps(totlen2, basicstep2, Nh);
    ckks_evaluator.decryptor.create_galois_keys_from_steps(steps2, layer_keys);
    timed([&] {
        bootstrapper.bsgs_linear_transform(tmpct2, tmpct, totlen2, basicstep2, logn,
                                           bootstrapper.invfftcoeff2[bootstrapper.slot_index]);
        ckks_evaluator.evaluator.rescale_to_next_inplace(tmpct2);
    });
    layer_keys = PhantomGaloisKey();

    // full-slot variant: the third layer uses coeff_logn = logn (slotlen == Nh)
    auto steps3 = collect_bsgs_layer_steps(totlen3, basicstep3, Nh);
    ckks_evaluator.decryptor.create_galois_keys_from_steps(steps3, layer_keys);
    timed([&] {
        bootstrapper.bsgs_linear_transform(tmpct3, tmpct2, totlen3, basicstep3, logn,
                                           bootstrapper.invfftcoeff3[bootstrapper.slot_index]);
        ckks_evaluator.evaluator.rescale_to_next_inplace(tmpct3);
    });
    layer_keys = PhantomGaloisKey();

    vector<int> conj_steps{0};
    ckks_evaluator.decryptor.create_galois_keys_from_steps(conj_steps, layer_keys);
    timed([&] {
        // mirror coefftoslot_full_3: mct = -i * tmpct3, rtn1 = tmpct3 + conj(tmpct3),
        // rtn2 = mct + conj(mct)
        complex<double> iunit(0.0, 1.0);
        vector<complex<double>> tmpvec(Nh, 0);
        for (int i = 0; i < Nh; i++)
            tmpvec[i] -= iunit;
        PhantomPlaintext tmpplain;
        ckks_evaluator.encoder.encode(tmpvec, 1.0, tmpplain);
        ckks_evaluator.evaluator.mod_switch_to_inplace(tmpplain, tmpct3.params_id());

        PhantomCiphertext mct, conj1, conj2;
        ckks_evaluator.evaluator.multiply_plain(tmpct3, tmpplain, mct);
        ckks_evaluator.evaluator.complex_conjugate(tmpct3, layer_keys, conj1);
        ckks_evaluator.evaluator.complex_conjugate(mct, layer_keys, conj2);
        ckks_evaluator.evaluator.add_reduced_error(tmpct3, conj1, rtn1);
        ckks_evaluator.evaluator.add_reduced_error(mct, conj2, rtn2);
    });
}

/**
 * Test for the shared-a (batch) coefficient-to-slot transform:
 *   1. encrypt n messages under sk, drop them to the bottom level
 *   2. FmtSwitch -> shared-a batch, shared-a mod raise to the top level
 *   3. Bootstrapper::shared_a_coefftoslot_3 on the batch (Sh-Rot rotations)
 *   4. reference: ordinary modraise + coefftoslot_3 per ciphertext (keys chunked
 *      per layer to bound GPU memory)
 *   5. decrypt member i with sk'_i and compare with the ordinary reference
 */
void shared_a_coefftoslot_test()
{
    // same CKKS parameters as the bootstrapping unit test
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    // FmtSwitch requires P >= Ql at the S2C input level (MaMBo paper Lemma 1);
    // with 4 regular primes left (~189 bits) we use 5 special primes (~255 bits)
    int num_special_primes = 5;

    int secret_key_hamming_weight = 192;

    int remaining_level = 10;
    int boot_level = 14;
    int total_level = remaining_level + boot_level;

    const size_t batch_n = 4;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
        coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; i++)
        coeff_bit_vec.push_back(logq);
    for (int i = 0; i < num_special_primes; i++)
        coeff_bit_vec.push_back(log_special_prime);

    std::cout << "Setting Parameters..." << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_special_modulus_size(num_special_primes);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    PhantomContext context(parms);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();

    Bootstrapper bootstrapper(10, logn, logN - 1, total_level, scale, 25, 59, 2, 1, &ckks_evaluator);
    bootstrapper.slot_vec.push_back(logn);
    bootstrapper.slot_index = 0;
    std::cout << "Generating Linear Transformation Coefficients..." << endl;
    bootstrapper.generate_LT_coefficient_3();

    // random messages, encrypt under sk, drop to the bottom level
    vector<vector<double>> inputs(batch_n, vector<double>(slot_count, 0.0));
    vector<PhantomCiphertext> cts;
    cts.reserve(batch_n);
    for (size_t i = 0; i < batch_n; i++)
    {
        vector<double> sparse(sparse_slots, 0.0);
        random_real(sparse, sparse_slots);
        for (size_t k = 0; k < slot_count; k++)
            inputs[i][k] = sparse[k % sparse_slots];

        PhantomPlaintext plain;
        PhantomCiphertext cipher;
        ckks_evaluator.encoder.encode(inputs[i], scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, cipher);
        for (int lvl = 0; lvl < total_level; lvl++)
            ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
        cts.push_back(std::move(cipher));
    }
    const size_t bottom_chain = cts[0].chain_index();

    // ---- shared-a path: FmtSwitch -> shared-a mod raise -> shared-a C2S ----
    std::cout << "Generating shared-a key material (batch n = " << batch_n << ")..." << endl;
    SharedAKeyGen keygen(context, secret_key, batch_n);

    auto t0 = system_clock::now();
    SharedAFormatKey fmt_key = keygen.gen_format_key(bottom_chain, stream);
    SharedACiphertextBatch batch;
    fmt_switch_forward(context, cts, fmt_key, batch, stream);
    shared_a_modraise_inplace(context, batch, stream);
    batch.scale_ = static_cast<double>(context.first_context_data().parms().coeff_modulus()[0].value());
    cudaStreamSynchronize(stream);
    const double sec_prep = duration<double>(system_clock::now() - t0).count();
    std::cout << "FmtSwitch + shared-a mod raise took: " << sec_prep << "s" << endl;

    // NOTE: the shared-a transforms generate their Sh-Rot keys on demand (and release
    // them right after use), so this timing includes ~40 shared-a Galois key
    // generations; the ordinary reference below excludes its key generation entirely.
    t0 = system_clock::now();
    SharedACiphertextBatch rtn_batch1, rtn_batch2;
    bootstrapper.shared_a_coefftoslot_full_3(rtn_batch1, rtn_batch2, batch, keygen);
    cudaStreamSynchronize(stream);
    const double sec_shared_c2s = duration<double>(system_clock::now() - t0).count();
    std::cout << "shared-a coefftoslot_full_3 (incl. on-demand Sh-Rot keygen) took: " << sec_shared_c2s << "s" << endl;

    // ---- reference: ordinary modraise + coefftoslot_full_3 per ciphertext ----
    double sec_ref_total = 0;
    vector<vector<double>> ref1_decoded(batch_n, vector<double>(slot_count, 0.0));
    vector<vector<double>> ref2_decoded(batch_n, vector<double>(slot_count, 0.0));
    for (size_t i = 0; i < batch_n; i++)
    {
        PhantomCiphertext ref = cts[i];
        bootstrapper.modraise_inplace(ref);
        ref.scale() = static_cast<double>(context.first_context_data().parms().coeff_modulus()[0].value());

        double transform_sec = 0;
        PhantomCiphertext ref_rtn1, ref_rtn2;
        ordinary_coefftoslot_full_3_chunked(bootstrapper, ckks_evaluator, ref_rtn1, ref_rtn2, ref, transform_sec);
        sec_ref_total += transform_sec;

        PhantomPlaintext plain;
        ckks_evaluator.decryptor.decrypt(ref_rtn1, plain);
        ckks_evaluator.encoder.decode(plain, ref1_decoded[i]);
        ckks_evaluator.decryptor.decrypt(ref_rtn2, plain);
        ckks_evaluator.encoder.decode(plain, ref2_decoded[i]);
    }
    std::cout << "ordinary coefftoslot_full_3 x" << batch_n << " (transforms only, excl. keygen) took: " << sec_ref_total
              << "s" << endl;

    // ---- compare each shared-a member against the ordinary reference ----
    std::cout << "sample values (member 0, first 4 slots): shared-a vs ordinary" << endl;
    double max_err = 0;
    for (size_t i = 0; i < batch_n; i++)
    {
        PhantomCiphertext member1 = rtn_batch1.assemble(context, i, stream);
        PhantomCiphertext member2 = rtn_batch2.assemble(context, i, stream);
        PhantomPlaintext plain;
        vector<double> decoded1(slot_count, 0.0), decoded2(slot_count, 0.0);
        keygen.aux_secret_key(i).decrypt(context, member1, plain);
        ckks_evaluator.encoder.decode(plain, decoded1);
        keygen.aux_secret_key(i).decrypt(context, member2, plain);
        ckks_evaluator.encoder.decode(plain, decoded2);
        if (i == 0)
            for (long k = 0; k < 4; k++)
                std::cout << "  slot " << k << ": " << decoded1[k] << " vs " << ref1_decoded[i][k] << endl;
        double err = 0;
        for (long k = 0; k < sparse_slots; k++)
        {
            err = max(err, abs(ref1_decoded[i][k] - decoded1[k]));
            err = max(err, abs(ref2_decoded[i][k] - decoded2[k]));
        }
        max_err = max(max_err, err);
        std::cout << "member " << i << ": max |shared-a C2S - ordinary C2S| = " << err << endl;
    }

    if (max_err < 1e-3)
        std::cout << "shared-a coefftoslot test PASSED" << endl;
    else
        std::cout << "shared-a coefftoslot test FAILED (error too large)" << endl;
}


// ---------------------------------------------------------------------------
// shared-a end-to-end bootstrap comparison test
// (ordinary_coefftoslot_full_3_batch / ordinary_evalmod_s2c_batch and the
//  BatchBootstrapper pipeline wrapper live in source/bootstrapping/BatchBootstrap.cuh)
// ---------------------------------------------------------------------------

/**
 * End-to-end bootstrap comparison, batch of n ciphertexts:
 *   shared-a hybrid : FmtSwitch -> shared-a modraise -> Sh-C2S -> Backward-FmtSwitch
 *                     -> ordinary EvalMod -> ordinary S2C        (per member)
 *   ordinary        : modraise -> coefftoslot_full_3 -> EvalMod -> S2C (per member)
 * Both paths decrypt to the original message under the main secret key; the report
 * prints per-section timings, single-shot and key-amortized totals, and the final
 * message-recovery error.
 */
void shared_a_bootstrap_test(size_t batch_n = 4)
{
    // same CKKS parameters as the bootstrapping unit test
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    // FmtSwitch requires P >= Ql at the S2C input level (MaMBo paper Lemma 1);
    // with 4 regular primes left (~189 bits) we use 5 special primes (~255 bits)
    int num_special_primes = 5;

    int secret_key_hamming_weight = 192;

    int remaining_level = 10;
    int boot_level = 14;
    int total_level = remaining_level + boot_level;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
        coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; i++)
        coeff_bit_vec.push_back(logq);
    for (int i = 0; i < num_special_primes; i++)
        coeff_bit_vec.push_back(log_special_prime);

    std::cout << "Setting Parameters... (batch n = " << batch_n << ")" << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_special_modulus_size(num_special_primes);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    PhantomContext context(parms);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();

    Bootstrapper bootstrapper(10, logn, logN - 1, total_level, scale, 25, 59, 2, 1, &ckks_evaluator);
    bootstrapper.slot_vec.push_back(logn);
    bootstrapper.slot_index = 0;
    std::cout << "Generating Optimal Minimax Polynomials..." << endl;
    bootstrapper.prepare_mod_polynomial();
    std::cout << "Generating Linear Transformation Coefficients..." << endl;
    bootstrapper.generate_LT_coefficient_3();
    bootstrapper.initial_scale = scale; // what bootstrap_3 would set from the input ct

    // random messages, encrypt under sk, drop to the bottom level
    vector<vector<double>> inputs(batch_n, vector<double>(slot_count, 0.0));
    vector<vector<double>> before(batch_n, vector<double>(slot_count, 0.0));
    vector<PhantomCiphertext> cts;
    cts.reserve(batch_n);
    for (size_t i = 0; i < batch_n; i++)
    {
        vector<double> sparse(sparse_slots, 0.0);
        random_real(sparse, sparse_slots);
        for (size_t k = 0; k < slot_count; k++)
            inputs[i][k] = sparse[k % sparse_slots];

        PhantomPlaintext plain;
        PhantomCiphertext cipher;
        ckks_evaluator.encoder.encode(inputs[i], scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, cipher);
        for (int lvl = 0; lvl < total_level; lvl++)
            ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
        ckks_evaluator.decryptor.decrypt(cipher, plain);
        ckks_evaluator.encoder.decode(plain, before[i]);
        cts.push_back(std::move(cipher));
    }
    const size_t bottom_chain = cts[0].chain_index();

    // ================= shared-a hybrid bootstrap =================
    std::cout << "Generating shared-a key material (batch n = " << batch_n << ")..." << endl;
    SharedAKeyGen keygen(context, secret_key, batch_n);

    auto t0 = system_clock::now();
    SharedAFormatKey fmt_key_bottom = keygen.gen_format_key(bottom_chain, stream);
    SharedACiphertextBatch batch;
    fmt_switch_forward(context, cts, fmt_key_bottom, batch, stream);
    shared_a_modraise_inplace(context, batch, stream);
    batch.scale_ = static_cast<double>(context.first_context_data().parms().coeff_modulus()[0].value());
    cudaStreamSynchronize(stream);
    const double sa_t_prep = duration<double>(system_clock::now() - t0).count();

    keygen.galois_keygen_sec_ = 0.0;
    t0 = system_clock::now();
    SharedACiphertextBatch c2s1, c2s2;
    bootstrapper.shared_a_coefftoslot_full_3(c2s1, c2s2, batch, keygen);
    cudaStreamSynchronize(stream);
    const double sa_t_c2s = duration<double>(system_clock::now() - t0).count();
    const double sa_t_c2s_keys = keygen.galois_keygen_sec_;

    t0 = system_clock::now();
    SharedAFormatKey fmt_key_out = keygen.gen_format_key(c2s1.chain_index_, stream);
    vector<PhantomCiphertext> sa1, sa2;
    fmt_switch_backward(context, c2s1, fmt_key_out, sa1, stream);
    fmt_switch_backward(context, c2s2, fmt_key_out, sa2, stream);
    cudaStreamSynchronize(stream);
    const double sa_t_back = duration<double>(system_clock::now() - t0).count();

    // release the shared-a input/output batches before the ordinary path allocates keys
    batch.release();
    c2s1.release();
    c2s2.release();

    // ================= ordinary C2S =================
    vector<PhantomCiphertext> ord_cts;
    ord_cts.reserve(batch_n);
    for (size_t i = 0; i < batch_n; i++)
        ord_cts.push_back(cts[i]); // deep copy; cts stays intact
    double ord_t_c2s_keys = 0, ord_t_c2s_xform = 0;
    vector<PhantomCiphertext> ord1, ord2;
    ordinary_coefftoslot_full_3_batch(bootstrapper, ckks_evaluator, ord1, ord2, ord_cts, ord_t_c2s_keys,
                                      ord_t_c2s_xform);

    // ================= common tail: EvalMod + S2C =================
    double sa_t_tail_keys = 0, sa_t_tail_xform = 0;
    vector<PhantomCiphertext> sa_final;
    ordinary_evalmod_s2c_batch(bootstrapper, ckks_evaluator, sa_final, sa1, sa2, sa_t_tail_keys, sa_t_tail_xform);

    double ord_t_tail_keys = 0, ord_t_tail_xform = 0;
    vector<PhantomCiphertext> ord_final;
    ordinary_evalmod_s2c_batch(bootstrapper, ckks_evaluator, ord_final, ord1, ord2, ord_t_tail_keys, ord_t_tail_xform);

    // ================= correctness =================
    double sa_mean_err = 0, ord_mean_err = 0, cross_max_err = 0;
    for (size_t i = 0; i < batch_n; i++)
    {
        PhantomPlaintext plain;
        vector<double> dec(slot_count, 0.0), dec_ord(slot_count, 0.0);
        ckks_evaluator.decryptor.decrypt(sa_final[i], plain);
        ckks_evaluator.encoder.decode(plain, dec);
        ckks_evaluator.decryptor.decrypt(ord_final[i], plain);
        ckks_evaluator.encoder.decode(plain, dec_ord);
        for (long k = 0; k < sparse_slots; k++)
        {
            sa_mean_err += abs(before[i][k] - dec[k]);
            ord_mean_err += abs(before[i][k] - dec_ord[k]);
            cross_max_err = max(cross_max_err, abs(dec[k] - dec_ord[k]));
        }
    }
    sa_mean_err /= batch_n * sparse_slots;
    ord_mean_err /= batch_n * sparse_slots;

    // ================= report =================
    const double sa_single = sa_t_prep + sa_t_c2s + sa_t_back + sa_t_tail_keys + sa_t_tail_xform;
    const double ord_single = ord_t_c2s_keys + ord_t_c2s_xform + ord_t_tail_keys + ord_t_tail_xform;
    const double sa_amort = sa_t_prep + (sa_t_c2s - sa_t_c2s_keys) + sa_t_back + sa_t_tail_xform;
    const double ord_amort = ord_t_c2s_xform + ord_t_tail_xform;

    std::cout << "================ shared-a vs ordinary bootstrap (batch n = " << batch_n
              << ") ================" << endl;
    std::cout << "C2S section:" << endl;
    std::cout << "  shared-a : prep " << sa_t_prep << "s (fmt key + FmtSwitch + modraise) + Sh-C2S " << sa_t_c2s
              << "s (Sh-Rot keygen " << sa_t_c2s_keys << "s, transforms " << sa_t_c2s - sa_t_c2s_keys
              << "s) + backward " << sa_t_back << "s" << endl;
    std::cout << "  ordinary : keygen " << ord_t_c2s_keys << "s + transforms " << ord_t_c2s_xform << "s" << endl;
    std::cout << "common tail (EvalMod + S2C, per path): keygen " << sa_t_tail_keys << "s + transforms "
              << sa_t_tail_xform << "s (shared-a path), keygen " << ord_t_tail_keys << "s + transforms "
              << ord_t_tail_xform << "s (ordinary path)" << endl;
    std::cout << "end-to-end single-shot (all keygen included): shared-a " << sa_single << "s vs ordinary "
              << ord_single << "s  (speedup x" << ord_single / sa_single << ")" << endl;
    std::cout << "end-to-end amortized  (one-time keygen excluded): shared-a " << sa_amort << "s vs ordinary "
              << ord_amort << "s  (speedup x" << ord_amort / sa_amort << ")" << endl;
    std::cout << "returned level (shared-a / ordinary): " << sa_final[0].coeff_modulus_size() << " / "
              << ord_final[0].coeff_modulus_size() << " primes" << endl;
    std::cout << "final message recovery: mean |err| shared-a " << sa_mean_err << ", ordinary " << ord_mean_err
              << ", max |shared - ordinary| " << cross_max_err << endl;
    std::cout << "==================================================================" << endl;
}

// ---------------------------------------------------------------------------
// stage-3 MOAI-style pipeline comparison (multi-batch workload)
// ---------------------------------------------------------------------------

/**
 * Simulates a MOAI single-layer-scale bootstrap workload: total_cts bottom-level
 * ciphertexts (a MOAI layer bootstraps 768, one per matrix column group) are run
 * through both BatchBootstrapper paths:
 *   ordinary : all ciphertexts in one call, Galois keys generated once per layer
 *              (best case for the ordinary path)
 *   shared-a : processed in chunks of batch_n; the one-time key material
 *              (aux secrets + bottom format key) is generated once up front,
 *              per-batch Sh-Rot / output format / S2C layer keys are regenerated
 *              inside every chunk (they are too large to keep resident).
 * The report prints per-chunk timings (the first chunk includes memory-pool
 * warmup), totals, per-ciphertext amortized times, an extrapolation to the real
 * MOAI single-layer scale (768 ciphertexts), and the message-recovery error of
 * both paths.
 */
void moai_pipeline_bootstrap_test(size_t total_cts = 16, size_t batch_n = 4, bool run_ordinary = true,
                                  bool run_shared = true)
{
    // same CKKS parameters as the bootstrapping unit test
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    // FmtSwitch requires P >= Ql at the S2C input level (MaMBo paper Lemma 1);
    // with 4 regular primes left (~189 bits) we use 5 special primes (~255 bits)
    int num_special_primes = 5;

    int secret_key_hamming_weight = 192;

    int remaining_level = 10;
    int boot_level = 14;
    int total_level = remaining_level + boot_level;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
        coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; i++)
        coeff_bit_vec.push_back(logq);
    for (int i = 0; i < num_special_primes; i++)
        coeff_bit_vec.push_back(log_special_prime);

    if (total_cts % batch_n != 0)
        throw invalid_argument("moai_pipeline_bootstrap_test: total_cts must be a multiple of batch_n");

    std::cout << "Setting Parameters... (MOAI pipeline, " << total_cts << " cts, batch n = " << batch_n
              << ")" << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_special_modulus_size(num_special_primes);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    PhantomContext context(parms);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();

    std::cout << "Generating Optimal Minimax Polynomials + LT coefficients..." << endl;
    Bootstrapper bootstrapper(10, logn, logN - 1, total_level, scale, 25, 59, 2, 1, &ckks_evaluator);
    BatchBootstrapper pipe(bootstrapper, ckks_evaluator, batch_n);
    pipe.prepare();

    // random messages; ciphertexts are encrypted on demand per chunk. Holding
    // all total_cts inputs on the GPU starves the memory pool and makes every
    // later allocation hit the (WSL2-slow) driver path: measured ~10x slowdown.
    vector<vector<double>> before(total_cts, vector<double>(slot_count, 0.0));
    for (size_t i = 0; i < total_cts; i++)
    {
        vector<double> sparse(sparse_slots, 0.0);
        random_real(sparse, sparse_slots);
        vector<double> full(slot_count, 0.0);
        for (size_t k = 0; k < slot_count; k++)
            full[k] = sparse[k % sparse_slots];

        PhantomPlaintext plain;
        PhantomCiphertext cipher;
        ckks_evaluator.encoder.encode(full, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, cipher);
        for (int lvl = 0; lvl < total_level; lvl++)
            ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
        ckks_evaluator.decryptor.decrypt(cipher, plain);
        ckks_evaluator.encoder.decode(plain, before[i]);
    }
    auto encrypt_input = [&](size_t i) {
        PhantomPlaintext plain;
        PhantomCiphertext cipher;
        ckks_evaluator.encoder.encode(before[i], scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, cipher);
        for (int lvl = 0; lvl < total_level; lvl++)
            ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
        return cipher;
    };
    const size_t bottom_chain = encrypt_input(0).chain_index();

    // one-time shared-a key material (aux secrets + bottom format key)
    if (run_shared)
    {
        std::cout << "Generating shared-a key material (batch n = " << batch_n << ")..." << endl;
        pipe.prepare_shared_keys(secret_key, bottom_chain);
    }

    // ================= ordinary path: chunks of 64 (GPU memory bound) =================
    double ord_keygen = 0, ord_xform = 0;
    double ord_c2s_keys = 0, ord_c2s_xform = 0, ord_tail_keys = 0, ord_tail_xform = 0;
    vector<PhantomCiphertext> ord_final;
    if (run_ordinary)
    {
        const size_t ord_chunk = 64;
        std::cout << "Running ordinary batch bootstrap (" << total_cts << " cts, chunks of " << ord_chunk << ")..."
                  << endl;
        ord_final.reserve(total_cts);
        for (size_t base = 0; base < total_cts; base += ord_chunk)
        {
            const size_t len = std::min<size_t>(ord_chunk, total_cts - base);
            vector<PhantomCiphertext> ord_cts;
            ord_cts.reserve(len);
            for (size_t i = base; i < base + len; i++)
                ord_cts.push_back(encrypt_input(i)); // encrypted on demand (GPU memory)
            vector<PhantomCiphertext> ord1, ord2, out;
            auto tb = system_clock::now();
            ordinary_coefftoslot_full_3_batch(bootstrapper, ckks_evaluator, ord1, ord2, ord_cts, ord_c2s_keys,
                                              ord_c2s_xform);
            ordinary_evalmod_s2c_batch(bootstrapper, ckks_evaluator, out, ord1, ord2, ord_tail_keys, ord_tail_xform);
            std::cout << "  ordinary chunk " << base / ord_chunk << ": "
                      << duration<double>(system_clock::now() - tb).count() << "s" << endl;
            for (auto &ct : out)
                if (ord_final.size() < 32) // keep a sample for verification, drop the rest (GPU memory)
                    ord_final.push_back(std::move(ct));
        }
        std::cout << "  ordinary C2S  : keygen " << ord_c2s_keys << "s, transforms " << ord_c2s_xform << "s" << endl;
        std::cout << "  ordinary tail : keygen " << ord_tail_keys << "s, transforms " << ord_tail_xform << "s" << endl;
        ord_keygen = ord_c2s_keys + ord_tail_keys;
        ord_xform = ord_c2s_xform + ord_tail_xform;
    }
    const double ord_total = ord_keygen + ord_xform;

    // ================= shared-a path: chunks of batch_n =================
    vector<double> sa_chunk_times;
    double sa_keygen = 0, sa_xform = 0;
    vector<PhantomCiphertext> sa_final;
    if (run_shared)
    {
        std::cout << "Running shared-a batch bootstrap (" << total_cts / batch_n << " chunks of " << batch_n
                  << ")..." << endl;
        sa_final.reserve(total_cts);
        for (size_t base = 0; base < total_cts; base += batch_n)
        {
            vector<PhantomCiphertext> chunk;
            chunk.reserve(batch_n);
            for (size_t i = base; i < base + batch_n; i++)
                chunk.push_back(encrypt_input(i)); // encrypted on demand (GPU memory)
            vector<PhantomCiphertext> out;
            double k = 0, x = 0;
            auto tb = system_clock::now();
            pipe.bootstrap_shared_a(out, chunk, k, x);
            const double bt = duration<double>(system_clock::now() - tb).count();
            sa_chunk_times.push_back(bt);
            sa_keygen += k;
            sa_xform += x;
            for (auto &ct : out)
                if (sa_final.size() < 32) // keep a sample for verification, drop the rest (GPU memory)
                    sa_final.push_back(std::move(ct));
            std::cout << "  chunk " << base / batch_n << ": " << bt << "s (keygen " << k << "s, transforms " << x
                      << "s)" << (base == 0 ? "   [includes memory-pool warmup]" : "") << endl;
        }
    }
    const double sa_total = pipe.shared_prep_sec + sa_keygen + sa_xform;

    // ================= correctness =================
    double sa_mean_err = 0, ord_mean_err = 0, cross_max_err = 0;
    const size_t verify_cts = std::min<size_t>(total_cts, 32); // outputs beyond this were dropped
    for (size_t i = 0; i < verify_cts; i++)
    {
        PhantomPlaintext plain;
        vector<double> dec, dec_ord;
        if (run_shared)
        {
            dec.assign(slot_count, 0.0);
            ckks_evaluator.decryptor.decrypt(sa_final[i], plain);
            ckks_evaluator.encoder.decode(plain, dec);
        }
        if (run_ordinary)
        {
            dec_ord.assign(slot_count, 0.0);
            ckks_evaluator.decryptor.decrypt(ord_final[i], plain);
            ckks_evaluator.encoder.decode(plain, dec_ord);
        }
        for (long k = 0; k < sparse_slots; k++)
        {
            if (run_shared)
                sa_mean_err += abs(before[i][k] - dec[k]);
            if (run_ordinary)
                ord_mean_err += abs(before[i][k] - dec_ord[k]);
            if (run_shared && run_ordinary)
                cross_max_err = max(cross_max_err, abs(dec[k] - dec_ord[k]));
        }
    }
    sa_mean_err /= verify_cts * sparse_slots;
    ord_mean_err /= verify_cts * sparse_slots;

    // ================= report =================
    double sa_steady = 0;
    for (size_t i = 1; i < sa_chunk_times.size(); i++)
        sa_steady += sa_chunk_times[i];
    sa_steady /= max<size_t>(1, sa_chunk_times.size() - 1);

    const size_t moai_layer_cts = 768; // MOAI single-layer bootstrap workload
    std::cout << "================ MOAI pipeline: shared-a vs ordinary (" << total_cts << " cts, batch n = "
              << batch_n << ") ================" << endl;
    if (run_ordinary)
    {
        // per-chunk keygen is identical across chunks, so the fair deployment
        // number amortizes it: one keygen + all transforms
        const size_t ord_chunks = (total_cts + 63) / 64;
        const double ord_amortized = ord_keygen / ord_chunks + ord_xform;
        std::cout << "ordinary  : total " << ord_total << "s (keygen " << ord_keygen << "s + transforms "
                  << ord_xform << "s = C2S " << ord_c2s_xform << "s + tail " << ord_tail_xform << "s), per-ct "
                  << ord_total / total_cts << "s" << endl;
        std::cout << "ordinary  : keygen amortized (once per level, not per chunk): total " << ord_amortized
                  << "s, per-ct " << ord_amortized / total_cts << "s" << endl;
        std::cout << "ordinary   error: mean |err| " << ord_mean_err << endl;
        std::cout << "ordinary   extrapolation to a MOAI layer (" << moai_layer_cts << " cts): ~"
                  << ord_amortized / total_cts * moai_layer_cts << "s (keygen amortized)" << endl;
    }
    if (run_shared)
    {
        // per-chunk keygen (Sh-Rot keys, output-level format key) is identical
        // across chunks; amortize it the same way as the ordinary path
        const size_t sa_chunks = total_cts / batch_n;
        const double sa_amortized = pipe.shared_prep_sec + sa_keygen / sa_chunks + sa_xform;
        std::cout << "shared-a  : total " << sa_total << "s = one-time prep " << pipe.shared_prep_sec
                  << "s + keygen " << sa_keygen << "s + transforms " << sa_xform << "s, per-ct "
                  << sa_total / total_cts << "s" << endl;
        std::cout << "shared-a  : keygen amortized (once per level, not per chunk): total " << sa_amortized
                  << "s, per-ct " << sa_amortized / total_cts << "s" << endl;
        std::cout << "shared-a steady-state chunk (" << batch_n << " cts, warmup excluded): " << sa_steady
                  << "s, per-ct " << sa_steady / batch_n << "s" << endl;
        std::cout << "shared-a   error: mean |err| " << sa_mean_err << endl;
        std::cout << "shared-a   extrapolation to a MOAI layer (" << moai_layer_cts << " cts): ~"
                  << (pipe.shared_prep_sec + (sa_keygen / sa_chunks + sa_xform) / total_cts * moai_layer_cts)
                  << "s (keygen amortized)" << endl;
    }
    if (run_ordinary && run_shared)
    {
        const double ord_amortized = ord_keygen / ((total_cts + 63) / 64) + ord_xform;
        const double sa_amortized = pipe.shared_prep_sec + sa_keygen / (total_cts / batch_n) + sa_xform;
        std::cout << "end-to-end speedup (total): x" << ord_total / sa_total << endl;
        std::cout << "end-to-end speedup (keygen amortized): x" << ord_amortized / sa_amortized << endl;
        std::cout << "returned level (shared-a / ordinary): " << sa_final[0].coeff_modulus_size() << " / "
                  << ord_final[0].coeff_modulus_size() << " primes" << endl;
        std::cout << "max |shared - ordinary| " << cross_max_err << endl;
    }
    std::cout << "==================================================================" << endl;

    const bool ok = (!run_shared || sa_mean_err < 1e-3) && (!run_ordinary || ord_mean_err < 1e-3);
    if (ok)
        std::cout << "MOAI pipeline bootstrap test PASSED" << endl;
    else
        std::cout << "MOAI pipeline bootstrap test FAILED (error too large)" << endl;
}

// ---------------------------------------------------------------------------
// S2C-first bootstrap comparison (MaMBo paper Sec 2.2)
// ---------------------------------------------------------------------------

/**
 * S2C-first variant of moai_pipeline_bootstrap_test. The bootstrap is triggered
 * with lS2C+1 = 4 primes left (instead of 1), so the ciphertexts are mod-switched
 * down only total_level - 3 times. Both paths then run:
 *   ordinary : sfl_full_3 (low level) -> modraise -> coefftoslot_full_3 -> EvalMod
 *              -> recombine
 *   shared-a : FmtSwitch -> Sh-S2C (low) -> Sh-ModRaise -> Sh-C2S (high)
 *              -> Backward-FmtSwitch -> per-member EvalMod + recombine
 * All linear transforms of the shared-a path run in the shared-a format; only
 * EvalMod is per member. Run modes match the pipeline test (independent-process
 * comparisons via run_ordinary / run_shared).
 */
void s2c_first_bootstrap_test(size_t total_cts = 16, size_t batch_n = 4, bool run_ordinary = true,
                              bool run_shared = true, double layer3_factor = kLowS2cLayer3Factor,
                              bool amortize_chunk_keys = false, size_t galois_cache_bytes = (size_t)2 << 30)
{
    // same CKKS parameters as the bootstrapping unit test
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    // FmtSwitch requires P >= Ql at the S2C input level (MaMBo paper Lemma 1);
    // with 4 regular primes left (~189 bits) we use 5 special primes (~255 bits)
    int num_special_primes = 5;

    int secret_key_hamming_weight = 192;

    // remaining_level = 20 (was 10): align with the single-layer test
    // (test_single_layer.cuh) -> chain 1x51 + 20x46 + 14x51 + 5x51 = 40 primes,
    // dnum = 35/5 = 7. Pure-bootstrap comparison under single-layer params.
    int remaining_level = 20;
    int boot_level = 14;
    int total_level = remaining_level + boot_level;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
        coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; i++)
        coeff_bit_vec.push_back(logq);
    for (int i = 0; i < num_special_primes; i++)
        coeff_bit_vec.push_back(log_special_prime);

    if (total_cts % batch_n != 0)
        throw invalid_argument("s2c_first_bootstrap_test: total_cts must be a multiple of batch_n");

    std::cout << "Setting Parameters... (S2C-first, " << total_cts << " cts, batch n = " << batch_n << ")"
              << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_special_modulus_size(num_special_primes);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    PhantomContext context(parms);
    // cap async mempool retention: phantom sets the release threshold to
    // UINT64_MAX (context.cu:135), hoarding freed blocks. With the amortized
    // key cache (~57 GiB at dnum=7) live on an 80GB card, hoarding leaves too
    // little headroom for the per-chunk fmt key generation (~9.5 GiB incl.
    // workspace) -> keep the pool at 2 GiB so blocks return to the driver.
    cudaMemPool_t s2c_pool;
    cudaDeviceGetDefaultMemPool(&s2c_pool, 0);
    uint64_t s2c_thr = (uint64_t)60 << 30;
    cudaMemPoolSetAttribute(s2c_pool, cudaMemPoolAttrReleaseThreshold, &s2c_thr);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();

    std::cout << "Generating Optimal Minimax Polynomials + LT coefficients..." << endl;
    Bootstrapper bootstrapper(10, logn, logN - 1, total_level, scale, 25, 59, 2, 1, &ckks_evaluator);
    BatchBootstrapper pipe(bootstrapper, ckks_evaluator, batch_n);
    pipe.prepare();
    if (amortize_chunk_keys)
    {
        // 摊销模式: 第一组生成的 Sh-Galois 密钥(显存预算 galois_cache_bytes)和
        // 输出侧格式密钥 fmt_key_out 缓存驻留, 后续组直接复用; 不摊销(默认)
        // 每组重新生成, 对应报告里的 total(保守口径)。
        pipe.enable_chunk_key_amortization(galois_cache_bytes);
    }

    // random messages; ciphertexts are encrypted on demand per chunk (see the
    // MOAI-pipeline test above for why inputs are not kept on the GPU)
    vector<vector<double>> before(total_cts, vector<double>(slot_count, 0.0));
    for (size_t i = 0; i < total_cts; i++)
    {
        vector<double> sparse(sparse_slots, 0.0);
        random_real(sparse, sparse_slots);
        vector<double> full(slot_count, 0.0);
        for (size_t k = 0; k < slot_count; k++)
            full[k] = sparse[k % sparse_slots];

        PhantomPlaintext plain;
        PhantomCiphertext cipher;
        ckks_evaluator.encoder.encode(full, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, cipher);
        for (int lvl = 0; lvl < total_level - 3; lvl++)
            ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
        ckks_evaluator.decryptor.decrypt(cipher, plain);
        ckks_evaluator.encoder.decode(plain, before[i]);
    }
    auto encrypt_input = [&](size_t i) {
        PhantomPlaintext plain;
        PhantomCiphertext cipher;
        ckks_evaluator.encoder.encode(before[i], scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, cipher);
        for (int lvl = 0; lvl < total_level - 3; lvl++)
            ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
        return cipher;
    };
    const size_t input_chain = encrypt_input(0).chain_index();
    std::cout << "Input level: " << encrypt_input(0).coeff_modulus_size()
              << " primes (bootstrap triggered early for S2C)" << endl;

    // one-time shared-a key material at the INPUT level (not the bottom)
    if (run_shared)
    {
        std::cout << "Generating shared-a key material (batch n = " << batch_n << ")..." << endl;
        // release pool-hoarded blocks: the fmt key for batch n needs one big
        // contiguous-ish allocation, and the async pool may be retaining tens
        // of GB of cached blocks from the setup phase above
        cudaDeviceSynchronize();
        cudaMemPoolTrimTo(s2c_pool, 0);
        pipe.prepare_shared_keys(secret_key, input_chain);
    }

    // ================= ordinary S2C-first: chunks of 64 (GPU memory bound) =================
    double ord_keygen = 0, ord_xform = 0;
    vector<PhantomCiphertext> ord_final;
    if (run_ordinary)
    {
        // chunk sized for GPU memory: 64 fits >=16GB cards; 16 for ~8GB cards
        const size_t ord_chunk = 64;
        std::cout << "Running ordinary S2C-first bootstrap (" << total_cts << " cts, chunks of " << ord_chunk
                  << ")..." << endl;
        ord_final.reserve(total_cts);
        for (size_t base = 0; base < total_cts; base += ord_chunk)
        {
            const size_t len = std::min<size_t>(ord_chunk, total_cts - base);
            vector<PhantomCiphertext> ord_cts;
            ord_cts.reserve(len);
            for (size_t i = base; i < base + len; i++)
                ord_cts.push_back(encrypt_input(i)); // encrypted on demand (GPU memory)
            vector<PhantomCiphertext> out;
            double k = 0, x = 0;
            auto tb = system_clock::now();
            pipe.bootstrap_ordinary_s2c_first(out, ord_cts, k, x, layer3_factor);
            std::cout << "  ordinary chunk " << base / ord_chunk << ": "
                      << duration<double>(system_clock::now() - tb).count() << "s (keygen " << k << "s, transforms "
                      << x << "s)" << endl;
            ord_keygen += k;
            ord_xform += x;
            for (auto &ct : out)
                if (ord_final.size() < 32) // keep a sample for verification, drop the rest (GPU memory)
                    ord_final.push_back(std::move(ct));
        }
    }
    const double ord_total = ord_keygen + ord_xform;

    // ================= shared-a S2C-first: chunks of batch_n =================
    vector<double> sa_chunk_times;
    double sa_keygen = 0, sa_xform = 0;
    vector<PhantomCiphertext> sa_final;
    if (run_shared)
    {
        std::cout << "Running shared-a S2C-first bootstrap (" << total_cts / batch_n << " chunks of " << batch_n
                  << ")..." << endl;
        sa_final.reserve(total_cts);
        for (size_t base = 0; base < total_cts; base += batch_n)
        {
            vector<PhantomCiphertext> chunk;
            chunk.reserve(batch_n);
            for (size_t i = base; i < base + batch_n; i++)
                chunk.push_back(encrypt_input(i)); // encrypted on demand (GPU memory)
            vector<PhantomCiphertext> out;
            double k = 0, x = 0;
            auto tb = system_clock::now();
            pipe.bootstrap_shared_a_s2c_first(out, chunk, k, x);
            const double bt = duration<double>(system_clock::now() - tb).count();
            sa_chunk_times.push_back(bt);
            sa_keygen += k;
            sa_xform += x;
            for (auto &ct : out)
                if (sa_final.size() < 32) // keep a sample for verification, drop the rest (GPU memory)
                    sa_final.push_back(std::move(ct));
            std::cout << "  chunk " << base / batch_n << ": " << bt << "s (keygen " << k << "s, transforms " << x
                      << "s)" << (base == 0 ? "   [includes memory-pool warmup]" : "") << endl;
        }
    }
    const double sa_total = pipe.shared_prep_sec + sa_keygen + sa_xform;

    // ================= correctness =================
    // the pipelines end with the amplitude correction, so the outputs should be
    // m itself and mean |err| is the true final error
    double sa_mean_err = 0, ord_mean_err = 0, cross_max_err = 0;
    bool ratio_printed = false;
    const size_t verify_cts = std::min<size_t>(total_cts, 32); // outputs beyond this were dropped
    for (size_t i = 0; i < verify_cts; i++)
    {
        PhantomPlaintext plain;
        vector<double> dec, dec_ord;
        if (run_shared)
        {
            dec.assign(slot_count, 0.0);
            ckks_evaluator.decryptor.decrypt(sa_final[i], plain);
            ckks_evaluator.encoder.decode(plain, dec);
        }
        if (run_ordinary)
        {
            dec_ord.assign(slot_count, 0.0);
            ckks_evaluator.decryptor.decrypt(ord_final[i], plain);
            ckks_evaluator.encoder.decode(plain, dec_ord);
        }
        if (run_ordinary && !ratio_printed)
        {
            ratio_printed = true;
            std::cout << "  first values (before -> after, ratio):" << endl;
            for (int k = 0; k < 6; k++)
                std::cout << "    " << before[0][k] << " -> " << dec_ord[k]
                          << "  (" << dec_ord[k] / before[0][k] << ")" << endl;
        }
        for (long k = 0; k < sparse_slots; k++)
        {
            if (run_shared)
                sa_mean_err += abs(before[i][k] - dec[k]);
            if (run_ordinary)
                ord_mean_err += abs(before[i][k] - dec_ord[k]);
            if (run_shared && run_ordinary)
                cross_max_err = max(cross_max_err, abs(dec[k] - dec_ord[k]));
        }
    }
    sa_mean_err /= verify_cts * sparse_slots;
    ord_mean_err /= verify_cts * sparse_slots;

    // ================= report =================
    double sa_steady = 0;
    for (size_t i = 1; i < sa_chunk_times.size(); i++)
        sa_steady += sa_chunk_times[i];
    sa_steady /= max<size_t>(1, sa_chunk_times.size() - 1);

    const size_t moai_layer_cts = 768; // MOAI single-layer bootstrap workload
    std::cout << "================ S2C-first: shared-a vs ordinary (" << total_cts << " cts, batch n = "
              << batch_n << ") ================" << endl;
    if (run_ordinary)
    {
        std::cout << "ordinary  : total " << ord_total << "s (keygen " << ord_keygen << "s + transforms "
                  << ord_xform << "s), per-ct " << ord_total / total_cts << "s" << endl;
        std::cout << "ordinary   error: mean |err| " << ord_mean_err << endl;
        std::cout << "ordinary   extrapolation to a MOAI layer (" << moai_layer_cts << " cts): ~"
                  << ord_total / total_cts * moai_layer_cts << "s" << endl;
    }
    if (run_shared)
    {
        // per-chunk keygen (Sh-Rot keys, output-level format key) is identical
        // across chunks; amortize it the same way as the ordinary path
        const size_t sa_chunks = total_cts / batch_n;
        const double sa_amortized = pipe.shared_prep_sec + sa_keygen / sa_chunks + sa_xform;
        if (amortize_chunk_keys)
        {
            // 摊销模式: sa_total 已是实测摊销值(缓存密钥只生成一次)
            auto cache_stats = pipe.galois_cache_stats();
            std::cout << "shared-a  : [摊销模式] cached " << cache_stats.first << " Sh-Galois keys ("
                      << (cache_stats.second >> 20) << " MiB) + fmt_key_out reused across chunks" << endl;
        }
        std::cout << "shared-a  : total " << sa_total << "s = one-time prep " << pipe.shared_prep_sec
                  << "s + keygen " << sa_keygen << "s + transforms " << sa_xform << "s, per-ct "
                  << sa_total / total_cts << "s" << (amortize_chunk_keys ? "  [摊销实测]" : "  [不摊销]") << endl;
        std::cout << "shared-a  : keygen amortized (once per level, not per chunk): total " << sa_amortized
                  << "s, per-ct " << sa_amortized / total_cts << "s" << endl;
        std::cout << "shared-a steady-state chunk (" << batch_n << " cts, warmup excluded): " << sa_steady
                  << "s, per-ct " << sa_steady / batch_n << "s" << endl;
        std::cout << "shared-a   error: mean |err| " << sa_mean_err << endl;
        std::cout << "shared-a   extrapolation to a MOAI layer (" << moai_layer_cts << " cts): ~"
                  << (pipe.shared_prep_sec + (sa_keygen / sa_chunks + sa_xform) / total_cts * moai_layer_cts)
                  << "s (keygen amortized)" << endl;
    }
    if (run_ordinary && run_shared)
    {
        const double ord_amortized = ord_keygen / ((total_cts + 63) / 64) + ord_xform;
        const double sa_amortized = pipe.shared_prep_sec + sa_keygen / (total_cts / batch_n) + sa_xform;
        std::cout << "end-to-end speedup (total): x" << ord_total / sa_total << endl;
        std::cout << "end-to-end speedup (keygen amortized): x" << ord_amortized / sa_amortized << endl;
        std::cout << "returned level (shared-a / ordinary): " << sa_final[0].coeff_modulus_size() << " / "
                  << ord_final[0].coeff_modulus_size() << " primes" << endl;
        std::cout << "max |shared - ordinary| " << cross_max_err << endl;
    }
    std::cout << "==================================================================" << endl;

    const bool ok = (!run_shared || sa_mean_err < 1e-3) && (!run_ordinary || ord_mean_err < 1e-3);
    if (ok)
        std::cout << "S2C-first bootstrap test PASSED" << endl;
    else
        std::cout << "S2C-first bootstrap test FAILED (error too large)" << endl;
}

/**
 * Stage-isolation probe for the shared-a S2C-first chain. Compares, at each
 * stage, the shared-a batch (decrypted with the aux secrets) against the
 * ordinary chain (decrypted with sk):
 *   S1: after fmt_switch_forward          (sanity)
 *   S2: after Sh-S2C vs ordinary S2C      (clean comparison, pre-modraise)
 *   S3: swap test - Sh-S2C, fmt_switch_backward at the S2C output level, then
 *       the ORDINARY tail (modraise + C2S + EvalMod + correction). If this is
 *       correct, the bug lives in shared-a modraise / Sh-C2S / top-level
 *       backward switch; if not, Sh-S2C itself is broken.
 */
void s2c_first_sh_probe_test(size_t batch_n = 4)
{
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    // FmtSwitch requires P >= Ql at the S2C input level (MaMBo paper Lemma 1);
    // with 4 regular primes left (~189 bits) we use 5 special primes (~255 bits)
    int num_special_primes = 5;

    int secret_key_hamming_weight = 192;

    int remaining_level = 10;
    int boot_level = 14;
    int total_level = remaining_level + boot_level;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
        coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; i++)
        coeff_bit_vec.push_back(logq);
    for (int i = 0; i < num_special_primes; i++)
        coeff_bit_vec.push_back(log_special_prime);

    std::cout << "Setting Parameters... (S2C-first shared-a stage probe)" << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_special_modulus_size(num_special_primes);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    PhantomContext context(parms);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();

    Bootstrapper bootstrapper(10, logn, logN - 1, total_level, scale, 25, 59, 2, 1, &ckks_evaluator);
    BatchBootstrapper pipe(bootstrapper, ckks_evaluator, batch_n);
    pipe.prepare();

    vector<vector<double>> before(batch_n, vector<double>(slot_count, 0.0));
    vector<PhantomCiphertext> cts;
    for (size_t i = 0; i < batch_n; i++)
    {
        vector<double> sparse(sparse_slots, 0.0);
        random_real(sparse, sparse_slots);
        vector<double> full(slot_count, 0.0);
        for (size_t k = 0; k < slot_count; k++)
            full[k] = sparse[k % sparse_slots];
        PhantomPlaintext plain;
        PhantomCiphertext cipher;
        ckks_evaluator.encoder.encode(full, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, cipher);
        for (int lvl = 0; lvl < total_level - 3; lvl++)
            ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
        ckks_evaluator.decryptor.decrypt(cipher, plain);
        ckks_evaluator.encoder.decode(plain, before[i]);
        cts.push_back(std::move(cipher));
    }
    pipe.prepare_shared_keys(secret_key, cts[0].chain_index());

    // ordinary S2C reference
    double kg = 0, xf = 0;
    vector<PhantomCiphertext> ord_s2c_in(cts.begin(), cts.end()), ord_s2c;
    ordinary_sfl_full_3_batch(bootstrapper, ckks_evaluator, ord_s2c, ord_s2c_in, kg, xf, kLowS2cLayer3Factor);

    // shared: forward + Sh-S2C
    moai::SharedACiphertextBatch batch, sh_s2c;
    moai::fmt_switch_forward(context, cts, pipe.fmt_key_bottom, batch, stream);
    bootstrapper.shared_a_sfl_full_3(sh_s2c, batch, *pipe.keygen);
    cudaStreamSynchronize(stream);

    auto decrypt_shared = [&](moai::SharedACiphertextBatch &sh, size_t i, vector<double> &out) {
        PhantomCiphertext member = sh.assemble(context, i, stream);
        PhantomPlaintext plain;
        pipe.keygen->aux_secret_key(i).decrypt(context, member, plain);
        out.assign(slot_count, 0.0);
        ckks_evaluator.encoder.decode(plain, out);
    };
    auto decrypt_ord = [&](PhantomCiphertext &ct, vector<double> &out) {
        PhantomPlaintext plain;
        ckks_evaluator.decryptor.decrypt(ct, plain);
        out.assign(slot_count, 0.0);
        ckks_evaluator.encoder.decode(plain, out);
    };

    // S1: forward-switch sanity
    {
        double max_err = 0;
        for (size_t i = 0; i < batch_n; i++)
        {
            vector<double> dsh;
            decrypt_shared(batch, i, dsh);
            for (long k = 0; k < sparse_slots; k++)
                max_err = max(max_err, abs(dsh[k] - before[i][k]));
        }
        std::cout << "S1 fmt_switch_forward: max |shared - input| = " << max_err << endl;
    }

    // S2: Sh-S2C vs ordinary S2C
    {
        double max_err = 0;
        for (size_t i = 0; i < batch_n; i++)
        {
            vector<double> dsh, dor;
            decrypt_shared(sh_s2c, i, dsh);
            decrypt_ord(ord_s2c[i], dor);
            for (long k = 0; k < sparse_slots; k++)
                max_err = max(max_err, abs(dsh[k] - dor[k]));
        }
        std::cout << "S2 Sh-S2C vs ordinary S2C: max diff = " << max_err
                  << "  (scale sh " << sh_s2c.scale_ << " / ord " << ord_s2c[0].scale() << ")" << endl;
        vector<double> dsh, dor;
        decrypt_shared(sh_s2c, 0, dsh);
        decrypt_ord(ord_s2c[0], dor);
        std::cout << "  first values (shared -> ordinary):" << endl;
        for (int k = 0; k < 6; k++)
            std::cout << "    " << dsh[k] << " -> " << dor[k] << endl;
    }

    // S3: swap test - Sh-S2C, backward at the S2C output level, ordinary tail
    {
        moai::SharedAFormatKey fmt_key_s2c = pipe.keygen->gen_format_key(sh_s2c.chain_index_, stream);
        vector<PhantomCiphertext> back;
        moai::fmt_switch_backward(context, sh_s2c, fmt_key_s2c, back, stream);
        cudaStreamSynchronize(stream);

        vector<PhantomCiphertext> ct1, ct2, fin;
        ordinary_coefftoslot_full_3_batch(bootstrapper, ckks_evaluator, ct1, ct2, back, kg, xf);
        ordinary_evalmod_combine_batch(bootstrapper, ckks_evaluator, fin, ct1, ct2, kg, xf);
        s2c_first_correct_amplitude(ckks_evaluator, fin, kLowS2cLayer3Factor);
        cudaStreamSynchronize(stream);

        double mean_err = 0;
        for (size_t i = 0; i < batch_n; i++)
        {
            vector<double> d;
            decrypt_ord(fin[i], d);
            for (long k = 0; k < sparse_slots; k++)
                mean_err += abs(d[k] - before[i][k]);
        }
        mean_err /= batch_n * sparse_slots;
        std::cout << "S3 swap test (Sh-S2C + ordinary tail): mean |err| = " << mean_err << endl;
        if (mean_err < 1e-3)
            std::cout << "  => Sh-S2C is fine; bug is downstream (shared modraise / Sh-C2S / top backward)"
                      << endl;
        else
            std::cout << "  => Sh-S2C (or the backward switch at this level) is broken" << endl;
    }
    cudaStreamSynchronize(stream);
}

/**
 * Probe for the S2C-first chain: S2C (low level) -> modraise -> C2S (high level)
 * -> recombine, WITHOUT EvalMod. The recombined ciphertext should decrypt to
 * m + q0*K per slot with integer K. Prints:
 *   err_direct : mean |dec - m|                              (garbage expected)
 *   err_frac   : mean |(dec mod q0) - m|                     (small if linear OK)
 *   K spread   : min/max of round((dec - m) / q0)            (integer-ish if OK)
 */
void s2c_first_probe_test()
{
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    // FmtSwitch requires P >= Ql at the S2C input level (MaMBo paper Lemma 1);
    // with 4 regular primes left (~189 bits) we use 5 special primes (~255 bits)
    int num_special_primes = 5;

    int secret_key_hamming_weight = 192;

    int remaining_level = 10;
    int boot_level = 14;
    int total_level = remaining_level + boot_level;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
        coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; i++)
        coeff_bit_vec.push_back(logq);
    for (int i = 0; i < num_special_primes; i++)
        coeff_bit_vec.push_back(log_special_prime);

    std::cout << "Setting Parameters... (S2C-first probe)" << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_special_modulus_size(num_special_primes);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    PhantomContext context(parms);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();

    Bootstrapper bootstrapper(10, logn, logN - 1, total_level, scale, 25, 59, 2, 1, &ckks_evaluator);
    BatchBootstrapper pipe(bootstrapper, ckks_evaluator, 1);
    pipe.prepare();

    // one ciphertext at 4 primes
    vector<double> sparse(sparse_slots, 0.0);
    random_real(sparse, sparse_slots);
    vector<double> full(slot_count, 0.0);
    for (size_t k = 0; k < slot_count; k++)
        full[k] = sparse[k % sparse_slots];

    PhantomPlaintext plain;
    PhantomCiphertext cipher;
    ckks_evaluator.encoder.encode(full, scale, plain);
    ckks_evaluator.encryptor.encrypt(plain, cipher);
    for (int lvl = 0; lvl < total_level - 3; lvl++)
        ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);

    double kg = 0, xf = 0;
    vector<PhantomCiphertext> cts{cipher};

    // S2C at low level
    vector<PhantomCiphertext> s2c_out;
    ordinary_sfl_full_3_batch(bootstrapper, ckks_evaluator, s2c_out, cts, kg, xf);
    std::cout << "after S2C: " << s2c_out[0].coeff_modulus_size() << " primes" << endl;

    // modraise + C2S (modraise is inside ordinary_coefftoslot_full_3_batch)
    vector<PhantomCiphertext> ct1, ct2;
    ordinary_coefftoslot_full_3_batch(bootstrapper, ckks_evaluator, ct1, ct2, s2c_out, kg, xf);
    std::cout << "after C2S: " << ct1[0].coeff_modulus_size() << " primes" << endl;

    // recombine WITHOUT EvalMod
    const long Nh = bootstrapper.Nh;
    complex<double> iunit(0.0, 1.0);
    vector<complex<double>> tmpvec(Nh, 0);
    for (long k = 0; k < Nh; k++)
        tmpvec[k] += iunit;
    PhantomPlaintext tmpplain;
    ckks_evaluator.encoder.encode(tmpvec, 1.0, tmpplain);
    ckks_evaluator.evaluator.mod_switch_to_inplace(tmpplain, ct2[0].params_id());
    PhantomCiphertext mct, rec;
    ckks_evaluator.evaluator.multiply_plain(ct2[0], tmpplain, mct);
    ckks_evaluator.evaluator.add_reduced_error(ct1[0], mct, rec);

    ckks_evaluator.decryptor.decrypt(rec, plain);
    vector<double> dec(slot_count, 0.0);
    ckks_evaluator.encoder.decode(plain, dec);

    const double q0 = static_cast<double>(context.first_context_data().parms().coeff_modulus()[0].value());
    double err_direct = 0, err_frac = 0;
    long kmin = 1 << 30, kmax = -(1 << 30);
    double kfrac_max = 0;
    for (long k = 0; k < sparse_slots; k++)
    {
        err_direct += abs(full[k] - dec[k]);
        const double kr = (dec[k] - full[k]) / q0;
        const long ki = lround(kr);
        if (ki < kmin)
            kmin = ki;
        if (ki > kmax)
            kmax = ki;
        const double kf = abs(kr - (double)ki);
        if (kf > kfrac_max)
            kfrac_max = kf;
        err_frac += abs(dec[k] - q0 * ki - full[k]);
    }
    std::cout << "probe: err_direct " << err_direct / sparse_slots << ", err_frac " << err_frac / sparse_slots
              << ", K in [" << kmin << ", " << kmax << "], max |K - round(K)| " << kfrac_max << endl;
    if (err_frac / sparse_slots < 1e-3)
        std::cout << "probe PASSED: linear chain OK, slots are m + q0*K" << endl;
    else
        std::cout << "probe FAILED: linear chain broken before EvalMod" << endl;
}

/**
 * Probe A for the S2C-first chain: identity round trip WITHOUT modraise.
 * A ciphertext at 7 primes goes S2C (7 -> 4) then C2S (4 -> 1, no modraise)
 * then recombine; the result must decrypt to the original message. This isolates
 * the S2C/C2S composition from the modraise step.
 */
void s2c_first_probe2_test()
{
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    // FmtSwitch requires P >= Ql at the S2C input level (MaMBo paper Lemma 1);
    // with 4 regular primes left (~189 bits) we use 5 special primes (~255 bits)
    int num_special_primes = 5;

    int secret_key_hamming_weight = 192;

    int remaining_level = 10;
    int boot_level = 14;
    int total_level = remaining_level + boot_level;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
        coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; i++)
        coeff_bit_vec.push_back(logq);
    for (int i = 0; i < num_special_primes; i++)
        coeff_bit_vec.push_back(log_special_prime);

    std::cout << "Setting Parameters... (S2C-first probe A: identity round trip)" << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_special_modulus_size(num_special_primes);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    PhantomContext context(parms);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();

    Bootstrapper bootstrapper(10, logn, logN - 1, total_level, scale, 25, 59, 2, 1, &ckks_evaluator);
    BatchBootstrapper pipe(bootstrapper, ckks_evaluator, 1);
    pipe.prepare();

    // one ciphertext at 7 primes
    vector<double> sparse(sparse_slots, 0.0);
    random_real(sparse, sparse_slots);
    vector<double> full(slot_count, 0.0);
    for (size_t k = 0; k < slot_count; k++)
        full[k] = sparse[k % sparse_slots];

    PhantomPlaintext plain;
    PhantomCiphertext cipher;
    ckks_evaluator.encoder.encode(full, scale, plain);
    ckks_evaluator.encryptor.encrypt(plain, cipher);
    for (int lvl = 0; lvl < total_level - 6; lvl++)
        ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
    std::cout << "input: " << cipher.coeff_modulus_size() << " primes" << endl;

    // diagnostics for the third-layer factor analysis
    {
        const auto moduli = context.first_context_data().parms().coeff_modulus();
        std::cout << "diag: scale=" << scale << " initial_scale=" << bootstrapper.initial_scale
                  << " final_scale=" << bootstrapper.final_scale
                  << " q0=" << static_cast<double>(moduli[0].value())
                  << " ct.scale=" << cipher.scale() << endl;
    }

    // factor scan: S2C -> C2S identity round trip for several third-layer factors
    const long Nh = bootstrapper.Nh;
    PhantomGaloisKey &layer_keys = *(ckks_evaluator.galois_keys);
    int div_part1 = floor(logn / 3.0);
    int div_part2 = floor((logn - div_part1) / 2.0);
    int div_part3 = logn - div_part1 - div_part2;
    int totlen1 = (1 << div_part1) - 1, totlen2 = (1 << div_part2) - 1, totlen3 = (1 << div_part3) - 1;
    int basicstep1 = (1 << (logn - div_part1)), basicstep2 = (1 << (logn - div_part1 - div_part2)), basicstep3 = 1;

    vector<double> factor_candidates{1.0, 2.0, 12.5, 25.0, 32.0, 50.0, 64.0};
    for (double cand : factor_candidates)
    {
        double kg = 0, xf = 0;
        vector<PhantomCiphertext> cts{cipher}, s2c_out;
        ordinary_sfl_full_3_batch(bootstrapper, ckks_evaluator, s2c_out, cts, kg, xf, cand);

        // C2S without modraise: 4 -> 1 primes (manual, per-layer keys)
        PhantomCiphertext st1, st2;
        {
            vector<int> steps = collect_rotated_layer_steps(totlen1, basicstep1, Nh);
            ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
            bootstrapper.rotated_bsgs_linear_transform(st1, s2c_out[0], totlen1, basicstep1, logn,
                                                       bootstrapper.invfftcoeff1[bootstrapper.slot_index]);
            ckks_evaluator.evaluator.rescale_to_next_inplace(st1);
            layer_keys = PhantomGaloisKey();
        }
        {
            vector<int> steps = collect_bsgs_layer_steps(totlen2, basicstep2, Nh);
            ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
            bootstrapper.bsgs_linear_transform(st2, st1, totlen2, basicstep2, logn,
                                               bootstrapper.invfftcoeff2[bootstrapper.slot_index]);
            ckks_evaluator.evaluator.rescale_to_next_inplace(st2);
            layer_keys = PhantomGaloisKey();
        }
        {
            vector<int> steps = collect_bsgs_layer_steps(totlen3, basicstep3, Nh);
            ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
            bootstrapper.bsgs_linear_transform(st1, st2, totlen3, basicstep3, logn,
                                               bootstrapper.invfftcoeff3[bootstrapper.slot_index]);
            ckks_evaluator.evaluator.rescale_to_next_inplace(st1);
            layer_keys = PhantomGaloisKey();
        }
        PhantomCiphertext rtn1, rtn2;
        {
            vector<int> steps{0};
            ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
            vector<complex<double>> tmpvec(Nh, complex<double>(0.0, -1.0));
            PhantomPlaintext tmpplain;
            ckks_evaluator.encoder.encode(tmpvec, 1.0, tmpplain);
            ckks_evaluator.evaluator.mod_switch_to_inplace(tmpplain, st1.params_id());
            PhantomCiphertext mct, conj1, conj2;
            ckks_evaluator.evaluator.multiply_plain(st1, tmpplain, mct);
            ckks_evaluator.evaluator.complex_conjugate(st1, layer_keys, conj1);
            ckks_evaluator.evaluator.complex_conjugate(mct, layer_keys, conj2);
            ckks_evaluator.evaluator.add_reduced_error(st1, conj1, rtn1);
            ckks_evaluator.evaluator.add_reduced_error(mct, conj2, rtn2);
            layer_keys = PhantomGaloisKey();
        }

        // recombine rtn1 + i * rtn2 and decrypt
        vector<complex<double>> itmp(Nh, complex<double>(0.0, 1.0));
        PhantomPlaintext iplain;
        ckks_evaluator.encoder.encode(itmp, 1.0, iplain);
        ckks_evaluator.evaluator.mod_switch_to_inplace(iplain, rtn2.params_id());
        PhantomCiphertext mct, rec;
        ckks_evaluator.evaluator.multiply_plain(rtn2, iplain, mct);
        ckks_evaluator.evaluator.add_reduced_error(rtn1, mct, rec);

        ckks_evaluator.decryptor.decrypt(rec, plain);
        vector<double> dec(slot_count, 0.0);
        ckks_evaluator.encoder.decode(plain, dec);

        double err = 0;
        for (long k = 0; k < sparse_slots; k++)
            err += abs(full[k] - dec[k]);
        err /= sparse_slots;
        std::cout << "probe A factor=" << cand << ": mean |err| " << err
                  << "  ratio[0]=" << dec[0] / full[0] << " ratio[1]=" << dec[1] / full[1] << endl;
    }
    cudaStreamSynchronize(stream);

    // ------------------------------------------------------------------
    // probe C: the working-chain order, C2S -> recombine -> S2C, no EvalMod.
    // Tests sfl . sflinv = id (the direction the working bootstrap relies on).
    // ------------------------------------------------------------------
    PhantomCiphertext cipherC;
    ckks_evaluator.encoder.encode(full, scale, plain);
    ckks_evaluator.encryptor.encrypt(plain, cipherC);
    for (int lvl = 0; lvl < total_level - 6; lvl++)
        ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipherC);
    std::cout << "probe C input: " << cipherC.coeff_modulus_size() << " primes" << endl;

    // C2S (sflinv direction): rotated / bsgs / bsgs + conj split
    PhantomCiphertext c1, c2;
    {
        vector<int> steps = collect_rotated_layer_steps(totlen1, basicstep1, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
        bootstrapper.rotated_bsgs_linear_transform(c1, cipherC, totlen1, basicstep1, logn,
                                                   bootstrapper.invfftcoeff1[bootstrapper.slot_index]);
        ckks_evaluator.evaluator.rescale_to_next_inplace(c1);
        layer_keys = PhantomGaloisKey();
    }
    {
        vector<int> steps = collect_bsgs_layer_steps(totlen2, basicstep2, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
        bootstrapper.bsgs_linear_transform(c2, c1, totlen2, basicstep2, logn,
                                           bootstrapper.invfftcoeff2[bootstrapper.slot_index]);
        ckks_evaluator.evaluator.rescale_to_next_inplace(c2);
        layer_keys = PhantomGaloisKey();
    }
    {
        vector<int> steps = collect_bsgs_layer_steps(totlen3, basicstep3, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
        bootstrapper.bsgs_linear_transform(c1, c2, totlen3, basicstep3, logn,
                                           bootstrapper.invfftcoeff3[bootstrapper.slot_index]);
        ckks_evaluator.evaluator.rescale_to_next_inplace(c1);
        layer_keys = PhantomGaloisKey();
    }
    PhantomCiphertext cr1, cr2;
    {
        vector<int> steps{0};
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
        complex<double> iu(0.0, 1.0);
        vector<complex<double>> tv(Nh, 0);
        for (long i = 0; i < Nh; i++)
            tv[i] -= iu;
        PhantomPlaintext tp;
        ckks_evaluator.encoder.encode(tv, 1.0, tp);
        ckks_evaluator.evaluator.mod_switch_to_inplace(tp, c1.params_id());
        PhantomCiphertext mc, cj1, cj2;
        ckks_evaluator.evaluator.multiply_plain(c1, tp, mc);
        ckks_evaluator.evaluator.complex_conjugate(c1, layer_keys, cj1);
        ckks_evaluator.evaluator.complex_conjugate(mc, layer_keys, cj2);
        ckks_evaluator.evaluator.add_reduced_error(c1, cj1, cr1);
        ckks_evaluator.evaluator.add_reduced_error(mc, cj2, cr2);
        layer_keys = PhantomGaloisKey();
    }
    // recombine cr1 + i*cr2 (like the slottocoeff preamble)
    PhantomCiphertext comb;
    {
        complex<double> iu(0.0, 1.0);
        vector<complex<double>> tv(Nh, 0);
        for (long i = 0; i < Nh; i++)
            tv[i] += iu;
        PhantomPlaintext tp;
        ckks_evaluator.encoder.encode(tv, 1.0, tp);
        ckks_evaluator.evaluator.mod_switch_to_inplace(tp, cr2.params_id());
        PhantomCiphertext mc;
        ckks_evaluator.evaluator.multiply_plain(cr2, tp, mc);
        ckks_evaluator.evaluator.add_reduced_error(cr1, mc, comb);
    }
    // S2C (sfl direction): bsgs / bsgs / rotated with the scale factor
    int s_div3 = floor(logn / 3.0);
    int s_div2 = floor((logn - s_div3) / 2.0);
    int s_div1 = logn - s_div3 - s_div2;
    int s_tot1 = (1 << s_div1) - 1, s_tot2 = (1 << s_div2) - 1, s_tot3 = (1 << s_div3) - 1;
    int s_step1 = 1, s_step2 = (1 << s_div1), s_step3 = (1 << (s_div1 + s_div2));
    PhantomCiphertext d1, d2, d3;
    {
        vector<int> steps = collect_bsgs_layer_steps(s_tot1, s_step1, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
        bootstrapper.bsgs_linear_transform(d1, comb, s_tot1, s_step1, logn,
                                           bootstrapper.fftcoeff1[bootstrapper.slot_index]);
        ckks_evaluator.evaluator.rescale_to_next_inplace(d1);
        layer_keys = PhantomGaloisKey();
    }
    {
        vector<int> steps = collect_bsgs_layer_steps(s_tot2, s_step2, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
        bootstrapper.bsgs_linear_transform(d2, d1, s_tot2, s_step2, logn,
                                           bootstrapper.fftcoeff2[bootstrapper.slot_index]);
        ckks_evaluator.evaluator.rescale_to_next_inplace(d2);
        layer_keys = PhantomGaloisKey();
    }
    {
        vector<int> steps = collect_rotated_layer_steps(s_tot3, s_step3, Nh);
        ckks_evaluator.decryptor.create_galois_keys_from_steps(steps, layer_keys);
        const auto modulus = context.first_context_data().parms().coeff_modulus();
        const double mod_zero = static_cast<double>(modulus[0].value());
        auto curr_level = context.get_context_data(d2.params_id()).chain_depth();
        const double curr_mod = static_cast<double>(modulus[curr_level].value());
        const double factor = curr_mod * mod_zero * bootstrapper.final_scale /
                              (d2.scale() * d2.scale() * bootstrapper.initial_scale);
        vector<vector<complex<double>>> fc3(2 * s_tot3 + 1);
        for (int r = 0; r < s_tot2 + 1; r++)
        {
            fc3[r].resize(bootstrapper.n);
            for (long j = 0; j < bootstrapper.n; j++)
                fc3[r][j] = bootstrapper.fftcoeff3[bootstrapper.slot_index][r][j] * factor;
        }
        bootstrapper.rotated_bsgs_linear_transform(d3, d2, s_tot3, s_step3, logn, fc3);
        ckks_evaluator.evaluator.rescale_to_next_inplace(d3);
        d3.scale() = bootstrapper.final_scale;
        layer_keys = PhantomGaloisKey();
    }

    ckks_evaluator.decryptor.decrypt(d3, plain);
    vector<double> decC(slot_count, 0.0);
    ckks_evaluator.encoder.decode(plain, decC);
    double errC = 0;
    for (long k = 0; k < sparse_slots; k++)
        errC += abs(full[k] - decC[k]);
    errC /= sparse_slots;
    std::cout << "probe C (C2S -> S2C identity, working-chain order, no EvalMod): mean |err| " << errC << endl;
    std::cout << "  first values (want -> got):" << endl;
    for (int k = 0; k < 8; k++)
        std::cout << "    " << full[k] << " -> " << decC[k] << endl;
    if (errC < 1e-3)
        std::cout << "probe C PASSED" << endl;
    else
        std::cout << "probe C FAILED" << endl;
    cudaStreamSynchronize(stream);
}

/**
 * Key-equation probe for the shared-a format key at an arbitrary level.
 * Verifies on device, for every (j, i):
 *     T = a_fmt_j * sk'_i + B[j][i] - delta_ij * P * sk   == -e  (small coeffs)
 * by computing T pointwise in the NTT domain on the QlP basis, transforming back
 * to the coefficient domain and checking that every RNS component is small.
 * Splits a fmt-switch failure into "keygen broken" vs "forward switch broken".
 */
void fmt_key_equation_probe(size_t keep_primes = 2)
{
    using namespace phantom::util;

    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    // FmtSwitch requires P >= Ql at the S2C input level (MaMBo paper Lemma 1);
    // with 4 regular primes left (~189 bits) we use 5 special primes (~255 bits)
    int num_special_primes = 5;

    int secret_key_hamming_weight = 192;

    int remaining_level = 10;
    int boot_level = 14;
    int total_level = remaining_level + boot_level;

    const size_t batch_n = 2;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
        coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; i++)
        coeff_bit_vec.push_back(logq);
    for (int i = 0; i < num_special_primes; i++)
        coeff_bit_vec.push_back(log_special_prime);

    std::cout << "Setting Parameters... (fmt key equation probe, keep " << keep_primes << " primes)" << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_special_modulus_size(num_special_primes);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    PhantomContext context(parms);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();

    // dummy ciphertext to locate the chain index of the target level
    PhantomPlaintext plain;
    PhantomCiphertext cipher;
    vector<double> zero(slot_count, 0.0);
    ckks_evaluator.encoder.encode(zero, scale, plain);
    ckks_evaluator.encryptor.encrypt(plain, cipher);
    const int drops = total_level + 1 - static_cast<int>(keep_primes);
    for (int lvl = 0; lvl < drops; lvl++)
        ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
    const size_t chain = cipher.chain_index();
    std::cout << "target chain index: " << chain << ", size_Ql: " << cipher.coeff_modulus_size() << endl;

    SharedAKeyGen keygen(context, secret_key, batch_n);
    SharedAFormatKey fmt_key = keygen.gen_format_key(chain, stream);
    cudaStreamSynchronize(stream);

    const size_t N = poly_modulus_degree;
    const size_t size_P = parms.special_modulus_size();
    const size_t size_QP = parms.coeff_modulus().size();
    const size_t size_Ql = keep_primes;
    const size_t size_QlP = size_Ql + size_P;
    const size_t poly_comp = size_QlP * N;
    const size_t grid = poly_comp / blockDimGlb.x;

    auto &rns_tool = context.get_context_data(chain).gpu_rns_tool();
    auto &ntt_tables = context.gpu_rns_tables();

    // gather the QlP modulus table (regular primes of the level, then special primes)
    auto mod_QlP = make_cuda_auto_ptr<DModulus>(size_QlP, stream);
    const DModulus *global_mod = context.gpu_rns_tables().modulus();
    cudaMemcpyAsync(mod_QlP.get(), global_mod, size_Ql * sizeof(DModulus), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(mod_QlP.get() + size_Ql, global_mod + (size_QP - size_P), size_P * sizeof(DModulus),
                    cudaMemcpyDeviceToDevice, stream);

    // gather sk into the QlP basis
    auto sk_main = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
    cudaMemcpyAsync(sk_main.get(), secret_key.secret_key_poly(), size_Ql * N * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(sk_main.get() + size_Ql * N, secret_key.secret_key_poly() + (size_QP - size_P) * N,
                    size_P * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    vector<cuda_auto_ptr<uint64_t>> sk_aux;
    for (size_t i = 0; i < batch_n; i++)
    {
        sk_aux.push_back(make_cuda_auto_ptr<uint64_t>(poly_comp, stream));
        cudaMemcpyAsync(sk_aux[i].get(), keygen.aux_secret_key(i).secret_key_poly(), size_Ql * N * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(sk_aux[i].get() + size_Ql * N,
                        keygen.aux_secret_key(i).secret_key_poly() + (size_QP - size_P) * N,
                        size_P * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    }

    // host modulus values for the centered-size check
    vector<uint64_t> h_q(size_QlP);
    for (size_t g = 0; g < size_Ql; g++)
        h_q[g] = context.get_context_data(chain).parms().coeff_modulus()[g].value();
    for (size_t g = 0; g < size_P; g++)
        h_q[size_Ql + g] = parms.coeff_modulus()[size_QP - size_P + g].value();

    auto T = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
    auto U = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
    vector<uint64_t> h_T(poly_comp);

    uint64_t worst = 0;
    for (size_t j = 0; j < batch_n; j++)
    {
        for (size_t i = 0; i < batch_n; i++)
        {
            // T = a_fmt_j * sk'_i + B[j][i]  (NTT domain, QlP basis)
            multiply_and_add_rns_poly<<<grid, blockDimGlb, 0, stream>>>(
                    fmt_key.a_fmt_.get() + j * poly_comp, sk_aux[i].get(),
                    fmt_key.B_fmt_.get() + (j * batch_n + i) * poly_comp, mod_QlP.get(), T.get(), N, size_QlP);
            if (i == j)
            {
                // U = P * sk on the regular components, 0 on the special component
                multiply_scalar_rns_poly<<<size_Ql * N / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                        sk_main.get(), rns_tool.bigP_mod_q(), rns_tool.bigP_mod_q_shoup(), mod_QlP.get(), U.get(), N,
                        size_Ql);
                cudaMemsetAsync(U.get() + size_Ql * N, 0, size_P * N * sizeof(uint64_t), stream);
                sub_rns_poly<<<grid, blockDimGlb, 0, stream>>>(T.get(), U.get(), mod_QlP.get(), T.get(), N, size_QlP);
            }
            // back to the coefficient domain
            nwt_2d_radix8_backward_inplace(T.get(), ntt_tables, size_Ql, 0, stream);
            nwt_2d_radix8_backward_inplace_include_special_mod(T.get(), ntt_tables, size_P, size_Ql, size_QP, size_P,
                                                               stream);
            cudaMemcpyAsync(h_T.data(), T.get(), poly_comp * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            uint64_t comp_worst = 0;
            size_t worst_g = 0;
            for (size_t g = 0; g < size_QlP; g++)
            {
                for (size_t k = 0; k < N; k++)
                {
                    const uint64_t v = h_T[g * N + k];
                    const uint64_t c = std::min(v, h_q[g] - v);
                    if (c > comp_worst)
                    {
                        comp_worst = c;
                        worst_g = g;
                    }
                }
            }
            std::cout << "  (j=" << j << ", i=" << i << "): max centered |T| = " << comp_worst
                      << " (worst prime idx " << worst_g << ", q ~ 2^"
                      << (h_q[worst_g] ? 63 - __builtin_clzll(h_q[worst_g]) + 1 : 0) << ")" << endl;
            worst = std::max(worst, comp_worst);
        }
    }
    if (worst < (1 << 20))
        std::cout << "fmt key equation probe PASSED (key equation holds, residual is small)" << endl;
    else
        std::cout << "fmt key equation probe FAILED (key equation broken at this level)" << endl;
    cudaStreamSynchronize(stream);
}

// ---------------------------------------------------------------------------
// forward-switch stage probes (specialized to keep = 2 primes for host CRT)
// ---------------------------------------------------------------------------

// extern declarations for the FmtSwitch.cu-internal kernels
namespace moai {
__global__ void extend_Ql_to_P_kernel(uint64_t *p_comp, const uint64_t *coeff_poly, const uint64_t *inv_g,
                                      const uint64_t *c_g, uint64_t Ql_mod_P, const DModulus *modulus_QlP,
                                      size_t size_Ql, size_t n, size_t p_idx);
}
using moai::extend_Ql_to_P_kernel;

static uint64_t host_modinv_u64(uint64_t a, uint64_t m)
{
    __int128 t = 0, newt = 1;
    __int128 r = m, newr = a % m;
    while (newr != 0)
    {
        __int128 q = r / newr;
        __int128 tmp = t - q * newt;
        t = newt;
        newt = tmp;
        tmp = r - q * newr;
        r = newr;
        newr = tmp;
    }
    if (t < 0)
        t += m;
    return static_cast<uint64_t>(t);
}

/**
 * Stage-by-stage probe of fmt_switch_forward at keep = 2 primes:
 *   A: moddown_from_NTT roundtrip (V = w*P on QlP -> moddown == w)
 *   B: BEHZ extension of a real c1 validated against host-side CRT
 *   C: structural residual of the forward output:
 *      R = (b_i + a*sk'_i) - (c0_i + c1_i*sk)  should equal  -P^{-1} * sum_j X_j e_j
 *      (small, < 2^70), checked by host CRT reconstruction
 */
void fmt_forward_probe()
{
    using namespace phantom::util;

    const size_t keep_primes = 2;
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    // FmtSwitch requires P >= Ql at the S2C input level (MaMBo paper Lemma 1);
    // with 4 regular primes left (~189 bits) we use 5 special primes (~255 bits)
    int num_special_primes = 5;
    int secret_key_hamming_weight = 192;
    int remaining_level = 10;
    int boot_level = 14;
    int total_level = remaining_level + boot_level;

    const size_t batch_n = 2;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
        coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; i++)
        coeff_bit_vec.push_back(logq);
    for (int i = 0; i < num_special_primes; i++)
        coeff_bit_vec.push_back(log_special_prime);

    std::cout << "Setting Parameters... (fmt forward probe, keep 2 primes)" << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_special_modulus_size(num_special_primes);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    PhantomContext context(parms);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;
    PhantomCKKSEncoder encoder(context);
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);
    size_t slot_count = encoder.slot_count();
    const auto &stream = phantom::util::global_variables::default_stream->get_stream();

    // inputs
    vector<PhantomCiphertext> cts;
    for (size_t i = 0; i < batch_n; i++)
    {
        vector<double> sparse(sparse_slots, 0.0);
        random_real(sparse, sparse_slots);
        vector<double> full(slot_count, 0.0);
        for (size_t k = 0; k < slot_count; k++)
            full[k] = sparse[k % sparse_slots];
        PhantomPlaintext plain;
        PhantomCiphertext cipher;
        ckks_evaluator.encoder.encode(full, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, cipher);
        const int drops = total_level + 1 - static_cast<int>(keep_primes);
        for (int lvl = 0; lvl < drops; lvl++)
            ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
        cts.push_back(std::move(cipher));
    }
    const size_t chain = cts[0].chain_index();

    SharedAKeyGen keygen(context, secret_key, batch_n);
    SharedAFormatKey fmt_key = keygen.gen_format_key(chain, stream);

    const size_t N = poly_modulus_degree;
    const size_t size_P = parms.special_modulus_size();
    const size_t size_QP = parms.coeff_modulus().size();
    const size_t size_Ql = keep_primes;
    const size_t size_QlP = size_Ql + size_P;
    const size_t poly_comp = size_QlP * N;
    const size_t size_Ql_n = size_Ql * N;
    const size_t grid_comp = poly_comp / blockDimGlb.x;
    const size_t grid_q = size_Ql_n / blockDimGlb.x;

    auto &rns_tool = context.get_context_data(chain).gpu_rns_tool();
    auto &ntt_tables = context.gpu_rns_tables();
    const auto scheme = parms.scheme();

    auto mod_QlP = make_cuda_auto_ptr<DModulus>(size_QlP, stream);
    const DModulus *global_mod = context.gpu_rns_tables().modulus();
    cudaMemcpyAsync(mod_QlP.get(), global_mod, size_Ql * sizeof(DModulus), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(mod_QlP.get() + size_Ql, global_mod + (size_QP - size_P), size_P * sizeof(DModulus),
                    cudaMemcpyDeviceToDevice, stream);

    SharedACiphertextBatch batch;
    fmt_switch_forward(context, cts, fmt_key, batch, stream);
    cudaStreamSynchronize(stream);

    // ---------------- Probe A: moddown roundtrip ----------------
    {
        auto w = make_cuda_auto_ptr<uint64_t>(size_Ql_n, stream);
        auto seed = make_cuda_auto_ptr<uint8_t>(global_variables::prng_seed_byte_count, stream);
        random_bytes(seed.get(), global_variables::prng_seed_byte_count, stream);
        sample_error_poly<<<grid_q, blockDimGlb, 0, stream>>>(w.get(), seed.get(), mod_QlP.get(), N, size_Ql);
        nwt_2d_radix8_forward_inplace(w.get(), ntt_tables, size_Ql, 0, stream); // w now NTT

        auto V = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
        multiply_scalar_rns_poly<<<grid_q, blockDimGlb, 0, stream>>>(w.get(), rns_tool.bigP_mod_q(),
                                                                     rns_tool.bigP_mod_q_shoup(), mod_QlP.get(),
                                                                     V.get(), N, size_Ql);
        cudaMemsetAsync(V.get() + size_Ql_n, 0, size_P * N * sizeof(uint64_t), stream);

        auto w_out = make_cuda_auto_ptr<uint64_t>(size_Ql_n, stream);
        rns_tool.moddown_from_NTT(w_out.get(), V.get(), ntt_tables, scheme, stream);

        vector<uint64_t> h_w(size_Ql_n), h_wo(size_Ql_n);
        cudaMemcpyAsync(h_w.data(), w.get(), size_Ql_n * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_wo.data(), w_out.get(), size_Ql_n * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        size_t mism = 0;
        for (size_t k = 0; k < size_Ql_n; k++)
            if (h_w[k] != h_wo[k])
                mism++;
        std::cout << "A moddown roundtrip: " << mism << " / " << size_Ql_n << " mismatches"
                  << (mism ? "  <-- FAILED" : "  OK") << endl;
    }

    // ---------------- Probe B: BEHZ extension of c1 ----------------
    {
        const uint64_t *c1 = cts[0].data() + size_Ql_n;
        auto scratch = make_cuda_auto_ptr<uint64_t>(size_Ql_n, stream);
        nwt_2d_radix8_backward(scratch.get(), c1, ntt_tables, size_Ql, 0, stream);

        // host BEHZ constants (same derivation as fmt_switch_forward)
        auto &host_coeff = context.get_context_data(chain).parms().coeff_modulus();
        auto &host_key_coeff = parms.coeff_modulus();
        vector<uint64_t> q(size_Ql);
        for (size_t g = 0; g < size_Ql; g++)
            q[g] = host_coeff[g].value();
        const uint64_t P_value = host_key_coeff[size_QP - size_P].value();
        uint64_t Ql_mod_P = 1;
        for (size_t g = 0; g < size_Ql; g++)
            Ql_mod_P = static_cast<uint64_t>((static_cast<unsigned __int128>(Ql_mod_P) * q[g]) % P_value);
        vector<uint64_t> h_inv_g(size_Ql), h_c_g(size_Ql);
        for (size_t g = 0; g < size_Ql; g++)
        {
            uint64_t qhat_mod_P = 1, qhat_mod_qg = 1;
            for (size_t h = 0; h < size_Ql; h++)
            {
                if (h == g)
                    continue;
                qhat_mod_P = static_cast<uint64_t>((static_cast<unsigned __int128>(qhat_mod_P) * q[h]) % P_value);
                qhat_mod_qg = static_cast<uint64_t>((static_cast<unsigned __int128>(qhat_mod_qg) * q[h]) % q[g]);
            }
            h_inv_g[g] = host_modinv_u64(qhat_mod_qg, q[g]);
            h_c_g[g] = qhat_mod_P; // c_g = QHat_g mod P (kernel multiplies by y_g)
        }
        auto d_inv_g = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
        auto d_c_g = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
        cudaMemcpyAsync(d_inv_g.get(), h_inv_g.data(), size_Ql * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_c_g.get(), h_c_g.data(), size_Ql * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

        auto p_comp = make_cuda_auto_ptr<uint64_t>(N, stream);
        extend_Ql_to_P_kernel<<<(N + blockDimGlb.x - 1) / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                p_comp.get(), scratch.get(), d_inv_g.get(), d_c_g.get(), Ql_mod_P, mod_QlP.get(), size_Ql, N, 0);

        // host-side sanity checks of the BEHZ constants themselves
        {
            const uint64_t chk1 = static_cast<uint64_t>((static_cast<unsigned __int128>(q[1]) * h_inv_g[0]) % q[0]);
            const uint64_t chk2 = static_cast<uint64_t>((static_cast<unsigned __int128>(q[0]) * h_inv_g[1]) % q[1]);
            const uint64_t chk3 = static_cast<uint64_t>(q[1] % P_value);
            const uint64_t chk4 = static_cast<uint64_t>(q[0] % P_value);
            const uint64_t chk5 = static_cast<uint64_t>((static_cast<unsigned __int128>(q[0]) * q[1]) % P_value);
            std::cout << "    BEHZ const checks: q1*inv0 mod q0 = " << chk1 << " (want 1), q0*inv1 mod q1 = " << chk2
                      << " (want 1)" << endl
                      << "      c0 ok: " << (chk3 == h_c_g[0]) << ", c1 ok: " << (chk4 == h_c_g[1])
                      << ", Ql_mod_P ok: " << (chk5 == Ql_mod_P) << endl
                      << "      q0=" << q[0] << " q1=" << q[1] << " P=" << P_value << endl;
        }

        vector<uint64_t> h_scratch(size_Ql_n), h_p(N);
        cudaMemcpyAsync(h_scratch.data(), scratch.get(), size_Ql_n * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_p.data(), p_comp.get(), N * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // host CRT for 2 primes: X = x0 + q0 * t, t = (x1 - x0) * q0^{-1} mod q1
        const uint64_t q0inv = host_modinv_u64(q[0] % q[1], q[1]);
        size_t mism = 0;
        uint64_t first_got = 0, first_want = 0;
        for (size_t k = 0; k < N; k++)
        {
            const uint64_t x0 = h_scratch[k];
            const uint64_t x1 = h_scratch[N + k];
            const uint64_t t = static_cast<uint64_t>((static_cast<unsigned __int128>(
                                                              (x1 + q[1] - (x0 % q[1]))) *
                                                      q0inv) %
                                                     q[1]);
            const uint64_t want = static_cast<uint64_t>(
                    (static_cast<unsigned __int128>(x0) + static_cast<unsigned __int128>(q[0]) * t) % P_value);
            if (h_p[k] != want)
            {
                if (mism == 0)
                {
                    first_got = h_p[k];
                    first_want = want;
                }
                mism++;
            }
            // dump host-side BEHZ intermediates for the first coefficients
            if (k < 3)
            {
                const uint64_t y0 = static_cast<uint64_t>((static_cast<unsigned __int128>(x0) * h_inv_g[0]) % q[0]);
                const uint64_t y1 = static_cast<uint64_t>((static_cast<unsigned __int128>(x1) * h_inv_g[1]) % q[1]);
                const double v = static_cast<double>(y0) / static_cast<double>(q[0]) +
                                 static_cast<double>(y1) / static_cast<double>(q[1]);
                uint64_t alpha = static_cast<uint64_t>(v);
                if (v - static_cast<double>(alpha) > 1.0 - 1e-9)
                    alpha += 1;
                const uint64_t acc = static_cast<uint64_t>(
                        (static_cast<unsigned __int128>(y0) * h_c_g[0] + static_cast<unsigned __int128>(y1) * h_c_g[1]) %
                        P_value);
                const uint64_t vc =
                        static_cast<uint64_t>((static_cast<unsigned __int128>(alpha) * Ql_mod_P) % P_value);
                const uint64_t res = acc >= vc ? acc - vc : acc + P_value - vc;
                std::cout << "    k=" << k << ": x0=" << x0 << " x1=" << x1 << " y0=" << y0 << " y1=" << y1
                          << " v=" << v << " alpha=" << alpha << " acc=" << acc << " res=" << res
                          << " | device=" << h_p[k] << " want(CRT)=" << want << endl;
            }
        }
        std::cout << "B BEHZ extension of c1: " << mism << " / " << N << " mismatches"
                  << (mism ? "  <-- FAILED (got " + to_string(first_got) + " want " + to_string(first_want) + ")"
                           : "  OK")
                  << endl;
    }

    // ---------------- Probe C: structural residual of forward output ----------------
    {
        // R = (b_0 + a*sk'_0) - (c0_0 + c1_0*sk) on Ql (NTT pointwise)
        auto sk_main = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
        cudaMemcpyAsync(sk_main.get(), secret_key.secret_key_poly(), size_Ql_n * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
        auto sk_aux0 = make_cuda_auto_ptr<uint64_t>(poly_comp, stream);
        cudaMemcpyAsync(sk_aux0.get(), keygen.aux_secret_key(0).secret_key_poly(), size_Ql_n * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);

        auto prod = make_cuda_auto_ptr<uint64_t>(size_Ql_n, stream);
        auto R = make_cuda_auto_ptr<uint64_t>(size_Ql_n, stream);
        multiply_rns_poly<<<grid_q, blockDimGlb, 0, stream>>>(batch.a_.get(), sk_aux0.get(), mod_QlP.get(), prod.get(),
                                                              N, size_Ql);
        add_rns_poly<<<grid_q, blockDimGlb, 0, stream>>>(batch.b_[0].get(), prod.get(), mod_QlP.get(), R.get(), N,
                                                         size_Ql);
        sub_rns_poly<<<grid_q, blockDimGlb, 0, stream>>>(R.get(), cts[0].data(), mod_QlP.get(), R.get(), N, size_Ql);
        multiply_rns_poly<<<grid_q, blockDimGlb, 0, stream>>>(cts[0].data() + size_Ql_n, sk_main.get(), mod_QlP.get(),
                                                              prod.get(), N, size_Ql);
        sub_rns_poly<<<grid_q, blockDimGlb, 0, stream>>>(R.get(), prod.get(), mod_QlP.get(), R.get(), N, size_Ql);
        nwt_2d_radix8_backward_inplace(R.get(), ntt_tables, size_Ql, 0, stream);

        vector<uint64_t> h_R(size_Ql_n);
        cudaMemcpyAsync(h_R.data(), R.get(), size_Ql_n * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        auto &host_coeff = context.get_context_data(chain).parms().coeff_modulus();
        const uint64_t q0 = host_coeff[0].value(), q1 = host_coeff[1].value();
        const uint64_t q0inv = host_modinv_u64(q0 % q1, q1);
        const unsigned __int128 Ql = static_cast<unsigned __int128>(q0) * q1;
        long double max_abs = 0;
        for (size_t k = 0; k < N; k++)
        {
            const uint64_t x0 = h_R[k];
            const uint64_t x1 = h_R[N + k];
            const uint64_t t =
                    static_cast<uint64_t>((static_cast<unsigned __int128>((x1 + q1 - (x0 % q1))) * q0inv) % q1);
            const unsigned __int128 X = static_cast<unsigned __int128>(x0) + static_cast<unsigned __int128>(q0) * t;
            const __int128 c = (X > Ql / 2) ? -static_cast<__int128>(Ql - X) : static_cast<__int128>(X);
            const long double a = fabs(static_cast<long double>(c));
            if (a > max_abs)
                max_abs = a;
        }
        std::cout << "C structural residual (b + a*sk') - (c0 + c1*sk): max |R| = " << static_cast<double>(max_abs)
                  << " (log2 ~ " << (max_abs > 0 ? log2l(max_abs) : 0.0L) << ")"
                  << (max_abs < 1e21L ? "  OK (small noise)" : "  <-- BROKEN (garbage)") << endl;
    }
    cudaStreamSynchronize(stream);
}
