#include "include.cuh"
// #include "test_ct_pt_matrix_mul.cuh"
// #include "test_phantom_ckks.cuh"
// #include "test_batch_encode_encrypt.cuh"
// #include "test_ct_ct_matrix_mul.cuh"
// #include "test_gelu.cuh"
// #include "test_layernorm.cuh"
// #include "test_softmax.cuh"
// #include "test_single_layer.cuh"
#include <cuda_runtime.h>
using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace moai;

int main(int argc, char **argv)
{
    // mode: "ordinary" (fresh-process ordinary path), "shared" (fresh-process
    // shared-a path), anything else runs both in one process
    const string mode = argc > 1 ? argv[1] : "both";

    if (mode == "sanity")
    {
        // sm_120 debug probe (server only): encrypt at the top of the
        // single-layer chain, decrypt while walking down level by level.
        cout << "sanity probe: full-chain encrypt/decrypt level sweep" << endl;
        const int logp = 46, logq = 51;
        size_t N = (size_t)1 << 16;
        vector<int> bits;
        bits.push_back(logq);
        for (int i = 0; i < 20; i++) bits.push_back(logp);
        for (int i = 0; i < 14; i++) bits.push_back(logq);
        for (int i = 0; i < 5; i++) bits.push_back(51);
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(N);
        parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
        parms.set_special_modulus_size(5);
        parms.set_secret_key_hamming_weight(192);
        parms.set_sparse_slots((size_t)1 << 15);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomCKKSEncoder phantom_encoder(context);
        moai::Encoder encoder(&context, &phantom_encoder);
        Encryptor encryptor(&context, &public_key);
        Evaluator evaluator(&context, &phantom_encoder);
        Decryptor decryptor(&context, &secret_key);
        size_t slots = encoder.slot_count();
        double scale = pow(2.0, logp);
        vector<double> input(slots);
        for (size_t i = 0; i < slots; i++) input[i] = 0.5 * sin((double)i);
        PhantomPlaintext pt;
        encoder.encode(input, scale, pt);
        PhantomCiphertext ct;
        encryptor.encrypt(pt, ct);
        while (true)
        {
            PhantomPlaintext dpt;
            decryptor.decrypt(ct, dpt);
            vector<double> out;
            encoder.decode(dpt, out);
            double maxerr = 0;
            for (size_t i = 0; i < slots; i++) maxerr = max(maxerr, fabs(out[i] - input[i]));
            cout << "primes=" << ct.coeff_modulus_size()
                 << " chain_depth=" << context.get_context_data(ct.params_id()).chain_depth()
                 << " maxerr=" << maxerr << endl;
            if (ct.coeff_modulus_size() <= 2) break;
            evaluator.mod_switch_to_next_inplace(ct);
        }
        return 0;
    }

    // cout << "test Phantom ckks" << endl;
    // phantom_ckks_test();
    // cout << "lib test passed!" << endl;

    // cout << "unit test: Batch encode and encrypt" << endl;
    // batch_input_test();
    // cout << "unit test Batch encode and encrypt passed!" << endl;

    // cout << "unit test: Ct-pt matrix multiplication without preprocessing" << endl;
    // ct_pt_matrix_mul_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: Ct-pt matrix multiplication with preprocessing" << endl;
    // ct_pt_matrix_mul_w_preprocess_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: Ct-ct matrix multiplication" << endl;
    // ct_ct_matrix_mul_test();
    // cout << "unit test Ct-ct matrix multiplication passed!" << endl;

    // cout << "unit test: GeLU" << endl;
    // gelu_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: LayerNorm" << endl;
    // layernorm_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: Softmax" << endl;
    // softmax_test();
    // cout << "unit test passed!" << endl;

    // cout << "uniform-a probe test" << endl;
    // uniform_a_probe_test();
    // cout << "uniform-a probe test done!" << endl;

    // cout << "unit test: Bootstrapping" << endl;
    // bootstrapping_test();
    // cout << "unit test passed!" << endl;

    // cout << "shared-a bootstrap comparison test" << endl;
    // shared_a_bootstrap_test(4);
    // cout << "shared-a bootstrap comparison test done!" << endl;

    // cout << "unit test: Bootstrapping (per-step galois keys)" << endl;
    // bootstrapping_test();
    // cout << "unit test passed!" << endl;

    // cout << "shared-a bootstrap comparison test (n=4)" << endl;
    // shared_a_bootstrap_test(4);
    // cout << "shared-a bootstrap comparison test (n=8)" << endl;
    // shared_a_bootstrap_test(8);
    // cout << "shared-a bootstrap comparison tests done!" << endl;

    if (mode == "sanity2")
    {
        // sm_120 debug probe v2: bisect the read path at a failing level.
        // decrypt -> backward NWT (GPU) -> CRT compose on CPU (NTL) vs GPU compose_array.
        cout << "sanity2: CPU cross-check of decrypt/decode at failing levels" << endl;
        const int logp = 46, logq = 51;
        size_t N = (size_t)1 << 16;
        vector<int> bits;
        bits.push_back(logq);
        for (int i = 0; i < 20; i++) bits.push_back(logp);
        for (int i = 0; i < 14; i++) bits.push_back(logq);
        for (int i = 0; i < 5; i++) bits.push_back(51);
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(N);
        parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
        parms.set_special_modulus_size(5);
        parms.set_secret_key_hamming_weight(192);
        parms.set_sparse_slots((size_t)1 << 15);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomCKKSEncoder phantom_encoder(context);
        moai::Encoder encoder(&context, &phantom_encoder);
        Encryptor encryptor(&context, &public_key);
        Evaluator evaluator(&context, &phantom_encoder);
        Decryptor decryptor(&context, &secret_key);
        size_t slots = encoder.slot_count();
        double scale = pow(2.0, logp);
        vector<double> input(slots);
        for (size_t i = 0; i < slots; i++) input[i] = 0.5 * sin((double)i);
        PhantomPlaintext pt;
        encoder.encode(input, scale, pt);
        PhantomCiphertext ct;
        encryptor.encrypt(pt, ct);

        for (int target : {33, 20, 16})
        {
            while (ct.coeff_modulus_size() > (size_t)target)
                evaluator.mod_switch_to_next_inplace(ct);
            PhantomPlaintext dpt;
            decryptor.decrypt(ct, dpt);
            auto &cd = context.get_context_data(ct.params_id());
            auto &cm = cd.parms().coeff_modulus();
            size_t L = cm.size();

            // reference: full GPU decode (known to be garbage for 17..32)
            vector<double> out;
            encoder.decode(dpt, out);

            // coeff domain via GPU backward NWT on a private copy
            auto buf = phantom::util::make_cuda_auto_ptr<uint64_t>(N * L, cudaStreamLegacy);
            cudaMemcpyAsync(buf.get(), dpt.data(), N * L * sizeof(uint64_t), cudaMemcpyDeviceToDevice, cudaStreamLegacy);
            nwt_2d_radix8_backward_inplace(buf.get(), context.gpu_rns_tables(), L, 0, cudaStreamLegacy);
            vector<uint64_t> h(N * L);
            cudaMemcpyAsync(h.data(), buf.get(), N * L * sizeof(uint64_t), cudaMemcpyDeviceToHost, cudaStreamLegacy);
            cudaStreamSynchronize(cudaStreamLegacy);

            // CPU CRT compose with NTL (correct math)
            NTL::ZZ Q = NTL::ZZ(1);
            for (size_t i = 0; i < L; i++) Q *= NTL::to_ZZ((long)cm[i].value());
            vector<NTL::ZZ> e(L);
            for (size_t i = 0; i < L; i++)
            {
                NTL::ZZ q_i = NTL::to_ZZ((long)cm[i].value());
                NTL::ZZ Qhat = Q / q_i;
                NTL::ZZ inv = NTL::InvMod(Qhat % q_i, q_i);
                e[i] = (Qhat * inv) % Q;
            }
            NTL::ZZ half = Q / 2;
            NTL::RR::SetPrecision(200);
            NTL::ZZ scaleZZ = NTL::to_ZZ((long)llround(scale));
            double maxdiff_compose = 0;
            // GPU compose_array directly on the same coeff-domain buffer
            auto dstc = phantom::util::make_cuda_auto_ptr<cuDoubleComplex>(slots, cudaStreamLegacy);
            auto &uht_h = cd.upper_half_threshold();
            auto uht_d = phantom::util::make_cuda_auto_ptr<uint64_t>(uht_h.size(), cudaStreamLegacy);
            cudaMemcpyAsync(uht_d.get(), uht_h.data(), uht_h.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, cudaStreamLegacy);
            auto &rns_tool = cd.gpu_rns_tool();
            rns_tool.base_Ql().compose_array(dstc.get(), buf.get(), uht_d.get(), 1.0 / scale,
                                             (uint32_t)N, (uint32_t)N, 1, cudaStreamLegacy);
            vector<cuDoubleComplex> hg(slots);
            cudaMemcpyAsync(hg.data(), dstc.get(), slots * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, cudaStreamLegacy);
            cudaStreamSynchronize(cudaStreamLegacy);
            for (size_t j = 0; j < 8; j++)
            {
                NTL::ZZ X = NTL::ZZ(0);
                for (size_t i = 0; i < L; i++)
                    X = (X + NTL::to_ZZ((long)h[i * N + j]) * e[i]) % Q;
                if (X > half) X -= Q;
                double cpu_val = NTL::conv<double>(NTL::conv<NTL::RR>(X) / NTL::conv<NTL::RR>(scaleZZ));
                double gpu_val = (j < slots) ? hg[j].x : hg[j - slots].y;
                maxdiff_compose = max(maxdiff_compose, fabs(cpu_val - gpu_val));
                if (j < 4)
                    cout << "  coeff " << j << ": CPU=" << cpu_val << " GPUcompose=" << gpu_val << endl;
            }
            double maxerr_dec = 0;
            for (size_t i = 0; i < slots; i++) maxerr_dec = max(maxerr_dec, fabs(out[i] - input[i]));
            cout << "target=" << target << " L=" << L << " dpt_scale_log2=" << log2(dpt.scale())
                 << " fulldecode_maxerr=" << maxerr_dec
                 << " compose_CPU_vs_GPU_maxdiff=" << maxdiff_compose << endl;
            {
                // 8-arg compose variant (step1/step1_1/step2) — the one decode_internal actually uses
                auto dstc8 = phantom::util::make_cuda_auto_ptr<cuDoubleComplex>(slots, cudaStreamLegacy);
                rns_tool.base_Ql().compose_array(dstc8.get(), buf.get(), uht_d.get(), 1.0 / scale,
                                                 (uint32_t)N, (uint32_t)N, 1, 1, cudaStreamLegacy);
                vector<cuDoubleComplex> hg8(slots);
                cudaMemcpyAsync(hg8.data(), dstc8.get(), slots * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, cudaStreamLegacy);
                cudaStreamSynchronize(cudaStreamLegacy);
                double md8 = 0; size_t bad8 = 0, first8 = 0;
                for (size_t i = 0; i < slots; i++)
                {
                    double d = max(fabs(hg8[i].x - hg[i].x), fabs(hg8[i].y - hg[i].y));
                    if (d > md8) md8 = d;
                    if (d > 1e-6 * max(1.0, fabs(hg[i].x))) { if (!bad8) first8 = i; bad8++; }
                }
                cout << "  compose8arg_vs_7arg: maxdiff=" << md8 << " bad=" << bad8 << "/" << slots
                     << " first_bad=" << first8 << endl;
            }
            {
                static vector<cuDoubleComplex> ref33;
                static bool have_ref = false;
                if (target == 33) { ref33 = hg; have_ref = true; }
                if (have_ref && target != 33)
                {
                    double md = 0; size_t bad = 0, first_bad = 0;
                    for (size_t i = 0; i < slots; i++)
                    {
                        double d = max(fabs(hg[i].x - ref33[i].x), fabs(hg[i].y - ref33[i].y));
                        if (d > md) md = d;
                        if (d > 1e-6 * max(1.0, fabs(ref33[i].x))) { if (!bad) first_bad = i; bad++; }
                    }
                    cout << "  compose_full_vs_L33: maxdiff=" << md << " bad_complex=" << bad
                         << "/" << slots << " first_bad_idx=" << first_bad << endl;
                }
            }
        }
        return 0;
    }

    if (mode == "sanity3")
    {
        // sm_120 debug probe v3: clean A/B of compose_array 7-arg vs 8-arg variants.
        // Single stream (default), no interleaved decode calls, full level sweep.
        cout << "sanity3: clean compose_array variant A/B sweep" << endl;
        const int logp = 46, logq = 51;
        size_t N = (size_t)1 << 16;
        vector<int> bits;
        bits.push_back(logq);
        for (int i = 0; i < 20; i++) bits.push_back(logp);
        for (int i = 0; i < 14; i++) bits.push_back(logq);
        for (int i = 0; i < 5; i++) bits.push_back(51);
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(N);
        parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
        parms.set_special_modulus_size(5);
        parms.set_secret_key_hamming_weight(192);
        parms.set_sparse_slots((size_t)1 << 15);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomCKKSEncoder phantom_encoder(context);
        moai::Encoder encoder(&context, &phantom_encoder);
        Encryptor encryptor(&context, &public_key);
        Evaluator evaluator(&context, &phantom_encoder);
        Decryptor decryptor(&context, &secret_key);
        size_t slots = encoder.slot_count();
        double scale = pow(2.0, logp);
        vector<double> input(slots);
        for (size_t i = 0; i < slots; i++) input[i] = 0.5 * sin((double)i);
        PhantomPlaintext pt;
        encoder.encode(input, scale, pt);
        PhantomCiphertext ct;
        encryptor.encrypt(pt, ct);
        const auto &ds = phantom::util::global_variables::default_stream->get_stream();

        vector<cuDoubleComplex> ref7;
        bool have_ref = false;
        while (true)
        {
            PhantomPlaintext dpt;
            decryptor.decrypt(ct, dpt);
            auto &cd = context.get_context_data(ct.params_id());
            size_t L = cd.parms().coeff_modulus().size();
            auto &rns_tool = cd.gpu_rns_tool();
            auto &uht_h = cd.upper_half_threshold();
            auto uht_d = phantom::util::make_cuda_auto_ptr<uint64_t>(uht_h.size(), ds);
            cudaMemcpyAsync(uht_d.get(), uht_h.data(), uht_h.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, ds);

            // coeff-domain copy
            auto buf = phantom::util::make_cuda_auto_ptr<uint64_t>(N * L, ds);
            cudaMemcpyAsync(buf.get(), dpt.data(), N * L * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ds);
            nwt_2d_radix8_backward_inplace(buf.get(), context.gpu_rns_tables(), L, 0, ds);

            // 7-arg variant
            auto d7 = phantom::util::make_cuda_auto_ptr<cuDoubleComplex>(slots, ds);
            rns_tool.base_Ql().compose_array(d7.get(), buf.get(), uht_d.get(), 1.0 / scale,
                                             (uint32_t)N, (uint32_t)N, 1, ds);
            vector<cuDoubleComplex> h7(slots);
            cudaMemcpyAsync(h7.data(), d7.get(), slots * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, ds);
            cudaStreamSynchronize(ds);

            // 8-arg variant
            auto d8 = phantom::util::make_cuda_auto_ptr<cuDoubleComplex>(slots, ds);
            rns_tool.base_Ql().compose_array(d8.get(), buf.get(), uht_d.get(), 1.0 / scale,
                                             (uint32_t)N, (uint32_t)N, 1, 1, ds);
            vector<cuDoubleComplex> h8(slots);
            cudaMemcpyAsync(h8.data(), d8.get(), slots * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, ds);
            cudaStreamSynchronize(ds);

            if (!have_ref) { ref7 = h7; have_ref = true; }
            double md7 = 0, md8 = 0;
            for (size_t i = 0; i < slots; i++)
            {
                md7 = max(md7, max(fabs(h7[i].x - ref7[i].x), fabs(h7[i].y - ref7[i].y)));
                md8 = max(md8, max(fabs(h8[i].x - ref7[i].x), fabs(h8[i].y - ref7[i].y)));
            }
            cout << "L=" << L << " 7arg_vs_ref=" << md7 << " 8arg_vs_ref=" << md8 << endl;
            if (ct.coeff_modulus_size() <= 2) break;
            evaluator.mod_switch_to_next_inplace(ct);
        }
        return 0;
    }

    if (mode == "sanity4")
    {
        // sm_120 debug probe v4: level sweep of encode / multiply_plain / rescale
        // using the exact sham layer chain: [51] + 23x46 + 11x51 + 5x51(special).
        cout << "sanity4: encode/multiply/rescale level sweep (sham chain)" << endl;
        const int logp = 46, logq = 51;
        size_t N = (size_t)1 << 16;
        vector<int> bits;
        bits.push_back(logq);
        for (int i = 0; i < 23; i++) bits.push_back(logp);
        for (int i = 0; i < 11; i++) bits.push_back(logq);
        for (int i = 0; i < 5; i++) bits.push_back(51);
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(N);
        parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
        parms.set_special_modulus_size(5);
        parms.set_secret_key_hamming_weight(192);
        parms.set_sparse_slots((size_t)1 << 15);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomCKKSEncoder phantom_encoder(context);
        moai::Encoder encoder(&context, &phantom_encoder);
        Encryptor encryptor(&context, &public_key);
        Evaluator evaluator(&context, &phantom_encoder);
        Decryptor decryptor(&context, &secret_key);
        size_t slots = encoder.slot_count();
        double scale = pow(2.0, logp);
        vector<double> x(slots), w(slots), xw(slots);
        for (size_t i = 0; i < slots; i++)
        {
            x[i] = 0.3 * sin((double)i);
            w[i] = 0.2 * cos((double)i * 0.7);
            xw[i] = x[i] * w[i];
        }
        PhantomPlaintext pt_x;
        encoder.encode(x, scale, pt_x);
        PhantomCiphertext ct_x;
        encryptor.encrypt(pt_x, ct_x);

        for (int L = 34; L >= 4; L--)
        {
            while ((int)ct_x.coeff_modulus_size() > L)
                evaluator.mod_switch_to_next_inplace(ct_x);

            // (1) encode at this chain level, decode right back
            PhantomPlaintext pt_w;
            encoder.encode(w, ct_x.params_id(), scale, pt_w);
            vector<double> w_dec;
            encoder.decode(pt_w, w_dec);
            double err_enc = 0;
            for (size_t i = 0; i < slots; i++) err_enc = max(err_enc, fabs(w_dec[i] - w[i]));

            // (2) multiply_plain, no rescale yet
            PhantomCiphertext ct_m;
            evaluator.multiply_plain(ct_x, pt_w, ct_m);
            PhantomPlaintext dm;
            decryptor.decrypt(ct_m, dm);
            vector<double> m_dec;
            encoder.decode(dm, m_dec);
            double err_mul = 0;
            for (size_t i = 0; i < slots; i++) err_mul = max(err_mul, fabs(m_dec[i] - xw[i]));

            // (3) rescale to next level, force scale like the pipeline does
            evaluator.rescale_to_next_inplace(ct_m);
            ct_m.scale() = scale;
            PhantomPlaintext dr;
            decryptor.decrypt(ct_m, dr);
            vector<double> r_dec;
            encoder.decode(dr, r_dec);
            double err_res = 0;
            for (size_t i = 0; i < slots; i++) err_res = max(err_res, fabs(r_dec[i] - xw[i]));

            cout << "L=" << L << " err_enc=" << err_enc << " err_mul=" << err_mul
                 << " err_rescale=" << err_res << endl;
        }
        return 0;
    }

    if (mode == "sanity5")
    {
        // sm_120 debug probe v5: stress test encode / multiply_plain+add at low
        // levels (sham chain), many iterations, to catch intermittent corruption.
        cout << "sanity5: stress encode / multiply+add at low levels" << endl;
        const int logp = 46, logq = 51;
        size_t N = (size_t)1 << 16;
        vector<int> bits;
        bits.push_back(logq);
        for (int i = 0; i < 23; i++) bits.push_back(logp);
        for (int i = 0; i < 11; i++) bits.push_back(logq);
        for (int i = 0; i < 5; i++) bits.push_back(51);
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(N);
        parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
        parms.set_special_modulus_size(5);
        parms.set_secret_key_hamming_weight(192);
        parms.set_sparse_slots((size_t)1 << 15);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomCKKSEncoder phantom_encoder(context);
        moai::Encoder encoder(&context, &phantom_encoder);
        Encryptor encryptor(&context, &public_key);
        Evaluator evaluator(&context, &phantom_encoder);
        Decryptor decryptor(&context, &secret_key);
        size_t slots = encoder.slot_count();
        double scale = pow(2.0, logp);

        // build a ct at chain level 5 (like att_output going into selfoutput)
        vector<double> x(slots);
        for (size_t i = 0; i < slots; i++) x[i] = 0.3 * sin((double)i);
        PhantomPlaintext pt_x;
        encoder.encode(x, scale, pt_x);
        PhantomCiphertext ct_x;
        encryptor.encrypt(pt_x, ct_x);
        while (ct_x.coeff_modulus_size() > 5)
            evaluator.mod_switch_to_next_inplace(ct_x);

        const int ITER = 300;
        const double w_scalar = 0.123;

        // ---- Loop A: encode-only stress ----
        {
            int bad = 0, first_bad = -1;
            for (int it = 0; it < ITER; it++)
            {
                PhantomPlaintext pt_w;
                encoder.encode(w_scalar, ct_x.params_id(), scale, pt_w);
                vector<double> dec;
                encoder.decode(pt_w, dec);
                double mx = 0;
                for (size_t i = 0; i < slots; i++) mx = max(mx, fabs(dec[i] - w_scalar));
                if (mx > 1e-6) { bad++; if (first_bad < 0) first_bad = it; }
            }
            cout << "LoopA encode-only: bad=" << bad << "/" << ITER << " first_bad=" << first_bad << endl;
        }

        // ---- Loop B: multiply_plain + add_inplace stress (pt encoded once) ----
        {
            PhantomPlaintext pt_w;
            encoder.encode(w_scalar, ct_x.params_id(), scale, pt_w);
            PhantomCiphertext acc;
            evaluator.multiply_plain(ct_x, pt_w, acc); // acc = x*w, scale 2^92
            double expect = w_scalar; // sum multiplier: after it adds, acc ~ (it+2)*x*w
            int bad = 0, first_bad = -1;
            for (int it = 0; it < ITER; it++)
            {
                PhantomCiphertext temp;
                evaluator.multiply_plain(ct_x, pt_w, temp);
                evaluator.add_inplace(acc, temp);
                if ((it % 25) == 0 || it == ITER - 1)
                {
                    PhantomPlaintext dpt;
                    decryptor.decrypt(acc, dpt);
                    vector<double> dec;
                    encoder.decode(dpt, dec);
                    double mx = 0, mexpect = 0;
                    for (size_t i = 0; i < slots; i++)
                    {
                        double e = x[i] * w_scalar * (it + 2);
                        mexpect = max(mexpect, fabs(e));
                        mx = max(mx, fabs(dec[i] - e));
                    }
                    double rel = mx / mexpect;
                    if (rel > 1e-6) { bad++; if (first_bad < 0) first_bad = it; }
                    cout << "  LoopB it=" << it << " relerr=" << rel << endl;
                }
            }
            cout << "LoopB multiply+add: bad=" << bad << " first_bad=" << first_bad << endl;
        }

        // ---- Loop C: encode + multiply_plain + add, exactly like _large_single ----
        {
            PhantomCiphertext acc;
            int bad = 0, first_bad = -1;
            for (int it = 0; it < ITER; it++)
            {
                PhantomPlaintext ecd;
                encoder.encode(w_scalar, ct_x.params_id(), scale, ecd);
                if (it == 0)
                {
                    evaluator.multiply_plain(ct_x, ecd, acc);
                }
                else
                {
                    PhantomCiphertext temp;
                    evaluator.multiply_plain(ct_x, ecd, temp);
                    evaluator.add_inplace(acc, temp);
                }
                if ((it % 25) == 0 || it == ITER - 1)
                {
                    PhantomPlaintext dpt;
                    decryptor.decrypt(acc, dpt);
                    vector<double> dec;
                    encoder.decode(dpt, dec);
                    double mx = 0, mexpect = 0;
                    for (size_t i = 0; i < slots; i++)
                    {
                        double e = x[i] * w_scalar * (it + 1);
                        mexpect = max(mexpect, fabs(e));
                        mx = max(mx, fabs(dec[i] - e));
                    }
                    double rel = mx / mexpect;
                    if (rel > 1e-6) { bad++; if (first_bad < 0) first_bad = it; }
                    cout << "  LoopC it=" << it << " relerr=" << rel << endl;
                }
            }
            cout << "LoopC encode+multiply+add: bad=" << bad << " first_bad=" << first_bad << endl;
        }
        return 0;
    }

    if (mode == "sanity6")
    {
        // sm_120 debug probe v6: pollute the cudaMallocAsync memory pool with
        // 0xFF-filled blocks, then re-run sanity5-style LoopC (encode+multiply+add)
        // and a layernorm-var-like LoopD (square+relin+rescale+add) with
        // per-iteration decryption checks. Tests the "stale pool memory read"
        // hypothesis for the intermittent ~2^151 garbage.
        cout << "sanity6: pool pollution + encode/multiply/add stress" << endl;
        const int logp = 46, logq = 51;
        size_t N = (size_t)1 << 16;
        vector<int> bits;
        bits.push_back(logq);
        for (int i = 0; i < 23; i++) bits.push_back(logp);
        for (int i = 0; i < 11; i++) bits.push_back(logq);
        for (int i = 0; i < 5; i++) bits.push_back(51);
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(N);
        parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
        parms.set_special_modulus_size(5);
        parms.set_secret_key_hamming_weight(192);
        parms.set_sparse_slots((size_t)1 << 15);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomRelinKey relin_key = secret_key.gen_relinkey(context);
        PhantomCKKSEncoder phantom_encoder(context);
        moai::Encoder encoder(&context, &phantom_encoder);
        Encryptor encryptor(&context, &public_key);
        Evaluator evaluator(&context, &phantom_encoder);
        Decryptor decryptor(&context, &secret_key);
        size_t slots = encoder.slot_count();
        double scale = pow(2.0, logp);

        auto pollute_pool = [&](const char *tag)
        {
            size_t total = 0;
            {
                vector<phantom::util::cuda_auto_ptr<uint64_t>> junk;
                for (int i = 0; i < 60; i++)
                {
                    size_t bytes = ((size_t)1 << (20 + (i % 7))) * (1 + (i % 3));
                    auto p = phantom::util::make_cuda_auto_ptr<uint64_t>(bytes / 8, cudaStreamLegacy);
                    cudaMemsetAsync(p.get(), 0xFF, bytes, cudaStreamLegacy);
                    total += bytes;
                    junk.push_back(std::move(p));
                }
                cudaStreamSynchronize(cudaStreamLegacy);
            } // freed back into the pool here
            cout << "[" << tag << "] polluted pool with " << total / 1e6 << " MB of 0xFF" << endl;
        };

        // build a ct at chain level 5 (like att_output going into selfoutput)
        vector<double> x(slots);
        for (size_t i = 0; i < slots; i++) x[i] = 0.3 * sin((double)i);
        PhantomPlaintext pt_x;
        encoder.encode(x, scale, pt_x);
        PhantomCiphertext ct_x;
        encryptor.encrypt(pt_x, ct_x);
        while (ct_x.coeff_modulus_size() > 5)
            evaluator.mod_switch_to_next_inplace(ct_x);

        const int ITER = 300;
        const double w_scalar = 0.123;

        // ---- Loop C2: encode + multiply_plain + add, per-iteration check ----
        pollute_pool("pre-LoopC2");
        {
            PhantomCiphertext acc;
            int bad = 0, first_bad = -1;
            for (int it = 0; it < ITER; it++)
            {
                PhantomPlaintext ecd;
                encoder.encode(w_scalar, ct_x.params_id(), scale, ecd);
                if (it == 0)
                {
                    evaluator.multiply_plain(ct_x, ecd, acc);
                }
                else
                {
                    PhantomCiphertext temp;
                    evaluator.multiply_plain(ct_x, ecd, temp);
                    evaluator.add_inplace(acc, temp);
                }
                PhantomPlaintext dpt;
                decryptor.decrypt(acc, dpt);
                vector<double> dec;
                encoder.decode(dpt, dec);
                double mx = 0, mexpect = 0;
                for (size_t i = 0; i < slots; i++)
                {
                    double e = x[i] * w_scalar * (it + 1);
                    mexpect = max(mexpect, fabs(e));
                    mx = max(mx, fabs(dec[i] - e));
                }
                double rel = mx / mexpect;
                if (rel > 1e-6)
                {
                    bad++;
                    if (first_bad < 0) first_bad = it;
                    cout << "  LoopC2 BAD it=" << it << " relerr=" << rel << " maxdiff=" << mx << endl;
                }
            }
            cout << "LoopC2 (polluted pool): bad=" << bad << "/" << ITER << " first_bad=" << first_bad << endl;
        }

        // ---- LoopD: layernorm-var-like: square + relin + rescale + add chain ----
        pollute_pool("pre-LoopD");
        {
            PhantomCiphertext acc;
            bool acc_init = false;
            int bad = 0, first_bad = -1;
            const int JTER = 192;
            for (int j = 0; j < JTER; j++)
            {
                PhantomCiphertext sq;
                evaluator.square(ct_x, sq);
                evaluator.relinearize_inplace(sq, relin_key);
                evaluator.rescale_to_next_inplace(sq);
                if (!acc_init)
                {
                    acc = sq;
                    acc_init = true;
                }
                else
                {
                    evaluator.add_inplace(acc, sq);
                }
                PhantomPlaintext dpt;
                decryptor.decrypt(acc, dpt);
                vector<double> dec;
                encoder.decode(dpt, dec);
                double mx = 0, mexpect = 0;
                for (size_t i = 0; i < slots; i++)
                {
                    double e = x[i] * x[i] * (j + 1);
                    mexpect = max(mexpect, fabs(e));
                    mx = max(mx, fabs(dec[i] - e));
                }
                double rel = mx / mexpect;
                if (rel > 1e-6)
                {
                    bad++;
                    if (first_bad < 0) first_bad = j;
                    cout << "  LoopD BAD j=" << j << " relerr=" << rel << " maxdiff=" << mx << endl;
                }
            }
            cout << "LoopD (polluted pool): bad=" << bad << "/" << JTER << " first_bad=" << first_bad << endl;
        }
        return 0;
    }

    if (mode == "s2c-probe")
    {
        cout << "S2C-first probe tests" << endl;
        s2c_first_probe2_test(); // identity round trip without modraise
        s2c_first_probe_test();  // full linear chain with modraise, no EvalMod
        cout << "S2C-first probe tests done!" << endl;
        return 0;
    }

    if (mode == "s2c-scan")
    {
        // full-chain factor scan for the low-level S2C third layer (ordinary path only)
        cout << "S2C-first full-chain factor scan" << endl;
        for (double cand : {0.0625, 0.125, 0.25})
        {
            cout << "----- layer3 factor = " << cand << " -----" << endl;
            s2c_first_bootstrap_test(4, 4, true, false, cand);
        }
        cout << "S2C-first full-chain factor scan done!" << endl;
        return 0;
    }

    if (mode == "s2c-shprobe")
    {
        cout << "S2C-first shared-a stage probe" << endl;
        s2c_first_sh_probe_test(4);
        cout << "S2C-first shared-a stage probe done!" << endl;
        return 0;
    }

    if (mode.rfind("fmt-l", 0) == 0)
    {
        // fmt switch unit test at a higher level: ./test fmt-l4 (keep 4 primes)
        const size_t keep = stoul(mode.substr(5));
        cout << "shared-a fmt switch test (keep " << keep << " primes)" << endl;
        shared_a_fmt_switch_test(keep);
        cout << "shared-a fmt switch test done!" << endl;
        return 0;
    }

    if (mode.rfind("fmt-keyprobe", 0) == 0)
    {
        // format key equation probe: ./test fmt-keyprobe2 (keep 2 primes)
        const size_t keep = mode.size() > 12 ? stoul(mode.substr(12)) : 2;
        fmt_key_equation_probe(keep);
        return 0;
    }

    if (mode == "fmt-fwdprobe")
    {
        fmt_forward_probe();
        return 0;
    }

    if (mode.rfind("s2c", 0) == 0)
    {
        // S2C-first comparison: ./test s2c-ordinary | s2c-shared | s2c-both | s2c-sham
        // optional suffixes: s2c-both16 -> batch_n = 16; s2c-both16x768 -> 768 cts
        // s2c-sham = 摊销模式 (cache per-chunk key material); s2c-shared = 不摊销
        size_t batch_n = 4, total_cts = 0;
        const string base = mode.rfind("s2c-both", 0) == 0 ? string("s2c-both")
                          : mode.rfind("s2c-sham", 0) == 0 ? string("s2c-sham")
                          : mode.rfind("s2c-shared", 0) == 0 ? string("s2c-shared")
                                                             : string("s2c-ordinary");
        const bool amortize = base == "s2c-sham";
        const string suffix = mode.substr(base.size());
        const size_t xpos = suffix.find('x');
        if (xpos != string::npos)
        {
            if (xpos > 0)
                batch_n = stoul(suffix.substr(0, xpos));
            total_cts = stoul(suffix.substr(xpos + 1));
        }
        else if (!suffix.empty())
            batch_n = stoul(suffix);
        if (total_cts == 0)
            total_cts = 4 * batch_n;
        cout << "S2C-first bootstrap comparison test (mode: " << mode << ")" << endl;
        s2c_first_bootstrap_test(total_cts, batch_n, base == "s2c-both" || base == "s2c-ordinary",
                                 base != "s2c-ordinary", kLowS2cLayer3Factor, amortize,
                                 (getenv("MOAI_GKEY_GIB") ? (size_t)atoi(getenv("MOAI_GKEY_GIB")) : (size_t)44)
                                     << 30); // default 44 GiB (80GB sweet spot); env MOAI_GKEY_GIB overrides
        cout << "S2C-first bootstrap comparison test done!" << endl;
        return 0;
    }

    if (mode == "layer-orig" || mode == "layer-sha" || mode == "layerq-orig" || mode == "layerq-sha" ||
        mode == "layer-sham" || mode == "layerq-sham")
    {
        // MOAI single-layer inference: original per-ct bootstrap_3 vs batched
        // shared-a S2C-first (the four bootstrap segments). layer* = full
        // 768-wide layer (needs a >=48GB GPU); layerq* = quarter-width 192
        // (fits 16GB; weights/inputs sliced from the full-width files).
        // -sha  = shared-a, 不摊销 (per-chunk keygen); -sham = shared-a, 摊销
        // (first chunk's key material cached and reused by all later chunks).
        const bool quarter = mode.rfind("layerq", 0) == 0;
        const bool use_sha = mode.find("-sha") != string::npos;
        const bool amortize = mode.find("-sham") != string::npos;
        cout << "single layer test (" << (quarter ? "quarter-width 192" : "full-width 768") << ", "
             << (use_sha ? (amortize ? "shared-a S2C-first, amortized keys/摊销" : "shared-a S2C-first, 不摊销")
                         : "original bootstrap_3")
             << ")" << endl;
        single_layer_test(use_sha, quarter ? 192 : 768, amortize);
        cout << "single layer test done!" << endl;
        return 0;
    }

    if (mode == "ln1-repro")
    {
        // standalone reproduction of the run6 L4 LN1 inv_sqrt explosion
        ln1_repro_test();
        return 0;
    }

    if (mode == "layers-orig" || mode == "layers-sham")
    {
        // multi-layer variant: loop the SAME full-width 768 layer N times
        // (N = env MOAI_LAYERS, default 3); layer k>0 takes the previous
        // layer's final output ciphertexts as input. layers-sham uses the
        // amortized shared-a keys, like layer-sham.
        const bool use_sha = mode == "layers-sham";
        const int num_layers = getenv("MOAI_LAYERS") ? atoi(getenv("MOAI_LAYERS")) : 3;
        cout << "multi-layer test (full-width 768, " << num_layers << " layers, "
             << (use_sha ? "shared-a S2C-first, amortized keys" : "original bootstrap_3") << ")" << endl;
        single_layer_test(use_sha, 768, use_sha, num_layers);
        cout << "multi-layer test done!" << endl;
        return 0;
    }

    {
        // MOAI pipeline (C2S-first) comparison: ./test ordinary | shared | both,
        // optional suffix: both16 -> batch_n = 16; both16x768 -> 768 cts
        size_t batch_n = 4, total_cts = 16;
        const string suffix = mode.rfind("both", 0) == 0 ? mode.substr(4)
                            : mode.rfind("shared", 0) == 0 ? mode.substr(6)
                            : mode.rfind("ordinary", 0) == 0 ? mode.substr(8)
                                                             : string();
        const size_t xpos = suffix.find('x');
        if (xpos != string::npos)
        {
            if (xpos > 0)
                batch_n = stoul(suffix.substr(0, xpos));
            total_cts = stoul(suffix.substr(xpos + 1));
        }
        else if (!suffix.empty())
            batch_n = stoul(suffix);
        cout << "MOAI pipeline bootstrap comparison test (mode: " << mode << ")" << endl;
        moai_pipeline_bootstrap_test(total_cts, batch_n, mode != "shared", mode != "ordinary");
        cout << "MOAI pipeline bootstrap comparison test done!" << endl;
    }

    // cout << "unit test: softmax with bootstrapping" << endl;
    // softmax_boot_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: BPmax test" << endl;
    // BPmax_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: BatchLN_test" << endl;
    // BatchLN_test();
    // cout << "unit test passed!" << endl;

    // cout << "single layer test" << endl;
    // single_layer_test();
    // cout << "single layer test passed!" << endl;


    // cout << "Rotary Position Embedding test" << endl;
    // rotary_pos_embed_test();
    // cout << "Rotary Position Embedding test passed!" << endl;

    // cout << "Causal Masked Softmax test" << endl;
    // causal_masked_softmax_test();
    // cout << "Causal Masked Softmax test passed!" << endl;

    // cout << "SiLU coefficient export test" << endl;
    // silu_coeff_verify();
    // cout << "SiLU coefficient export test passed!" << endl;

    // cout << "SiLU coefficient export test" << endl;
    // silu_test();
    // cout << "SiLU coefficient export test passed!" << endl;

    // cout << "shared-a fmt switch test" << endl;
    // shared_a_fmt_switch_test();
    // cout << "shared-a fmt switch test done!" << endl;

    // cout << "shared-a coefftoslot test" << endl;
    // shared_a_coefftoslot_test();
    // cout << "shared-a coefftoslot test done!" << endl;

    // cout << "shared-a bootstrap comparison test" << endl;
    // shared_a_bootstrap_test(4);
    // cout << "shared-a bootstrap comparison test done!" << endl;

    // cout << "RMSNorm test" << endl;
    // RMSNorm_test();
    // cout << "RMSNorm test passed!" << endl;

    return 0;
}
