#pragma once
// Probe: isolate which primitive breaks when ciphertext "a" parts are drawn
// from a uniform distribution (instead of the a=1 workaround in prng.cu).
#include "include.cuh"
#include "phantom.h"

using namespace std;
using namespace phantom;

namespace
{
double probe_err(CKKSEvaluator &ckks_evaluator, PhantomCiphertext &ct, const vector<double> &ref,
                 size_t sparse_slots)
{
    PhantomPlaintext plain;
    vector<double> decoded(ref.size(), 0.0);
    ckks_evaluator.decryptor.decrypt(ct, plain);
    ckks_evaluator.encoder.decode(plain, decoded);
    double err = 0;
    for (size_t i = 0; i < sparse_slots; i++)
        err += abs(decoded[i] - ref[i]);
    return err / sparse_slots;
}

void probe_show(CKKSEvaluator &ckks_evaluator, PhantomCiphertext &ct, const vector<double> &ref, int k = 4)
{
    PhantomPlaintext plain;
    vector<double> decoded(ref.size(), 0.0);
    ckks_evaluator.decryptor.decrypt(ct, plain);
    ckks_evaluator.encoder.decode(plain, decoded);
    cout << "    decoded: ";
    for (int i = 0; i < k; i++)
        cout << decoded[i] << " ";
    cout << "| expect: ";
    for (int i = 0; i < k; i++)
        cout << ref[i] << " ";
    cout << endl;
}
} // namespace

void uniform_a_probe_test()
{
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;

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
    coeff_bit_vec.push_back(log_special_prime);

    std::cout << "Setting Parameters (uniform-a probe)..." << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
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

    vector<double> sparse(sparse_slots, 0.0);
    vector<double> input(slot_count, 0.0);
    random_real(sparse, sparse_slots);
    for (size_t i = 0; i < slot_count; i++)
        input[i] = sparse[i % sparse_slots];

    PhantomPlaintext plain;
    PhantomCiphertext cipher;
    ckks_evaluator.encoder.encode(input, scale, plain);
    ckks_evaluator.encryptor.encrypt(plain, cipher);

    // P0: fresh encrypt -> decrypt
    {
        PhantomCiphertext ct = cipher;
        cout << "P0 encrypt->decrypt:            err = " << probe_err(ckks_evaluator, ct, input, sparse_slots)
             << endl;
    }

    // P1: modswitch to bottom -> decrypt
    PhantomCiphertext bottom = cipher;
    for (int i = 0; i < total_level; i++)
        ckks_evaluator.evaluator.mod_switch_to_next_inplace(bottom);
    cout << "P1 modswitch-to-bottom:         err = " << probe_err(ckks_evaluator, bottom, input, sparse_slots)
         << endl;

    const auto &stream = phantom::util::global_variables::default_stream->get_stream();
    const size_t N = poly_modulus_degree;

    // stream-ordered device->host copy (plain cudaMemcpy on the legacy stream races with
    // phantom's async stream and can read pre-kernel data)
    auto d2h = [&](void *host, const void *dev, size_t bytes) {
        cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    };

    // P8: NTT round-trip identity at the bottom level (1 prime).
    // For an all-ones vector (the a=1 case) ANY mismatched NTT table still gives identity;
    // a uniform vector only survives if forward/backward tables match.
    {
        PhantomCiphertext ct = bottom; // NTT form, 1 prime
        ckks_evaluator.evaluator.transform_from_ntt_inplace(ct);
        ckks_evaluator.evaluator.transform_to_ntt_inplace(ct);
        vector<uint64_t> h_before(2 * N), h_after(2 * N);
        d2h(h_before.data(), bottom.data(), 2 * N * sizeof(uint64_t));
        d2h(h_after.data(), ct.data(), 2 * N * sizeof(uint64_t));
        size_t bad = 0, first_bad = SIZE_MAX;
        for (size_t i = 0; i < 2 * N; i++)
            if (h_before[i] != h_after[i])
            {
                if (first_bad == SIZE_MAX)
                    first_bad = i;
                bad++;
            }
        cout << "P8 NTT round-trip (1 prime):    mismatches = " << bad << " / " << 2 * N;
        if (bad)
            cout << " | first at " << first_bad << ": " << h_before[first_bad] << " -> " << h_after[first_bad];
        cout << endl;
    }

    // P7: modraise coefficient-level check: c'[poly][j][i] mod q0 == c_coeff[poly][i]
    {
        PhantomCiphertext coeff_ct = bottom;
        ckks_evaluator.evaluator.transform_from_ntt_inplace(coeff_ct); // coeff domain, 1 prime
        vector<uint64_t> h_src(2 * N);
        d2h(h_src.data(), coeff_ct.data(), 2 * N * sizeof(uint64_t));

        Bootstrapper bootstrapper(10, logn, logN - 1, total_level, scale, 25, 59, 2, 1, &ckks_evaluator);
        PhantomCiphertext ct = bottom;
        bootstrapper.modraise_inplace(ct);
        ckks_evaluator.evaluator.transform_from_ntt_inplace(ct); // coeff domain, full level

        const size_t mod_count = ct.coeff_modulus_size();
        vector<uint64_t> h_dst(2 * mod_count * N);
        d2h(h_dst.data(), ct.data(), 2 * mod_count * N * sizeof(uint64_t));

        size_t bad = 0;
        // the resize-bug regression check: after modraise, each poly's prime-0 slot must
        // hold exactly the original bottom-level coefficients (higher primes are the BEHZ
        // centered lift, for which per-prime "mod q0 == src" does NOT hold)
        for (size_t poly = 0; poly < 2; poly++)
            for (size_t i = 0; i < N; i++)
            {
                if (h_dst[poly * mod_count * N + i] != h_src[poly * N + i])
                {
                    if (bad < 5)
                        cout << "    mismatch poly=" << poly << " prime=0 i=" << i << ": got "
                             << h_dst[poly * mod_count * N + i] << " expect " << h_src[poly * N + i] << endl;
                    bad++;
                }
            }
        cout << "P7 modraise prime-0 check:      mismatches = " << bad << " / " << 2 * N << endl;
    }

    // P2: modraise, then decrypt at top level with scale=q0; check I = (dec - m)/q0 is small
    {
        Bootstrapper bootstrapper(10, logn, logN - 1, total_level, scale, 25, 59, 2, 1, &ckks_evaluator);
        PhantomCiphertext ct = bottom;
        bootstrapper.modraise_inplace(ct);
        const double q0 = static_cast<double>(
            ckks_evaluator.context->first_context_data().parms().coeff_modulus()[0].value());
        ct.scale() = q0; // as bootstrap does
        PhantomPlaintext plain2;
        vector<double> dec2(slot_count, 0.0);
        ckks_evaluator.decryptor.decrypt(ct, plain2);
        ckks_evaluator.encoder.decode(plain2, dec2);
        double max_I = 0, mean_I = 0;
        for (size_t i = 0; i < sparse_slots; i++)
        {
            double I = fabs(dec2[i] - input[i]) / q0;
            max_I = max(max_I, I);
            mean_I += I;
        }
        mean_I /= sparse_slots;
        cout << "P2 modraise: max|I| = " << max_I << ", mean|I| = " << mean_I
             << " (should be <= ~boundary_K=25)" << endl;
        cout << "    dec[0..3]: " << dec2[0] << " " << dec2[1] << " " << dec2[2] << " " << dec2[3]
             << " | m[0..3]: " << input[0] << " " << input[1] << " " << input[2] << " " << input[3]
             << " | q0 = " << q0 << endl;
    }

    // P3: rotate by 1024 at top level -> decrypt (try both directions)
    {
        vector<int> steps{1024};
        PhantomGaloisKey rot_keys = secret_key.create_galois_keys_from_steps(context, steps);
        PhantomCiphertext ct = cipher, rot;
        ckks_evaluator.evaluator.rotate_vector(ct, 1024, rot_keys, rot);
        long Nh = 1 << (logN - 1);
        vector<double> ref(slot_count, 0.0);
        for (long i = 0; i < Nh; i++)
            ref[(i + 1024) % Nh] = input[i];
        for (long i = Nh; i < (long)slot_count; i++)
            ref[i] = ref[i - Nh];
        cout << "P3a rotate 1024 (dir +):        err = " << probe_err(ckks_evaluator, rot, ref, sparse_slots)
             << endl;
        vector<double> ref2(slot_count, 0.0);
        for (long i = 0; i < Nh; i++)
            ref2[i] = input[(i + 1024) % Nh];
        for (long i = Nh; i < (long)slot_count; i++)
            ref2[i] = ref2[i - Nh];
        cout << "P3b rotate 1024 (dir -):        err = " << probe_err(ckks_evaluator, rot, ref2, sparse_slots)
             << endl;
        probe_show(ckks_evaluator, rot, ref);
    }

    // P4: complex conjugate -> decrypt
    {
        vector<int> steps{0};
        PhantomGaloisKey conj_keys = secret_key.create_galois_keys_from_steps(context, steps);
        PhantomCiphertext conj;
        ckks_evaluator.evaluator.complex_conjugate(cipher, conj_keys, conj);
        cout << "P4 conjugate:                   err = " << probe_err(ckks_evaluator, conj, input, sparse_slots)
             << " (real input, conj should be identity)" << endl;
    }

    // P5: multiply_plain + rescale -> decrypt
    {
        vector<double> ones(slot_count, 1.0);
        PhantomPlaintext one_plain;
        ckks_evaluator.encoder.encode(ones, scale, one_plain);
        PhantomCiphertext ct;
        ckks_evaluator.evaluator.multiply_plain(cipher, one_plain, ct);
        ckks_evaluator.evaluator.rescale_to_next_inplace(ct);
        ct.scale() = scale;
        cout << "P5 mult-plain + rescale:        err = " << probe_err(ckks_evaluator, ct, input, sparse_slots)
             << endl;
        probe_show(ckks_evaluator, ct, input);
    }

    // P6: ct-ct square + relin + rescale -> decrypt, expect m^2
    {
        PhantomCiphertext ct;
        ckks_evaluator.evaluator.square(cipher, ct);
        ckks_evaluator.evaluator.relinearize_inplace(ct, relin_keys);
        ckks_evaluator.evaluator.rescale_to_next_inplace(ct);
        // true scale is scale^2 / q_last; compute it exactly
        const auto &modulus = ckks_evaluator.context->first_context_data().parms().coeff_modulus();
        const double q_last = static_cast<double>(modulus[ct.coeff_modulus_size()].value());
        ct.scale() = scale * scale / q_last;
        vector<double> ref(slot_count, 0.0);
        for (size_t i = 0; i < slot_count; i++)
            ref[i] = input[i] * input[i];
        cout << "P6 square + relin + rescale:    err = " << probe_err(ckks_evaluator, ct, ref, sparse_slots)
             << endl;
        probe_show(ckks_evaluator, ct, ref);
    }
}
