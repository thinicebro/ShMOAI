#pragma once
#include "include.cuh"

using namespace std;
using namespace phantom;
using namespace moai;

// Standalone reproduction of the run6 L4 LN1 inv_sqrt explosion.
//
// run6 (layers-sham, 12 layers) blew up inside L4's layernorm(): the FHE var
// ciphertext decrypted to 452.177 -- matching the plaintext reconstruction
// (452.202) to noise -- yet invert_sqrt(var) returned ~4e+72, although the
// exact-arithmetic iteration provably converges for all five true variances
// {307.9, 452.2, 432.0, 403.9, 116.7} (divergence needs var >~ 1300).
//
// This test re-encrypts the EXACT reconstructed L4 LN1 input
// (stage_boot1_L4 + stage_input_L4, dumped during run6) with the same CKKS
// parameters and the same chain level (24 primes) that layernorm saw in the
// real run, then calls layernorm() unchanged. If inv_sqrt diverges here, the
// bug is deterministic and bisectable in minutes; if it converges, the run6
// failure was caused by external memory corruption, not by layernorm itself.
//
// Input file: build/ln1_repro_input.txt -- 768 lines x 5 values, value t of
// line i goes to slot t*256 (batch_input layout with num_X = 256).
// Run from build/:  MOAI_DUMP=1 ./test ln1-repro
void ln1_repro_test()
{
  cout << "LN1 repro test (run6 L4 input)" << endl;

  // ---- CKKS params identical to single_layer_test (layers-sham, 768) ----
  const int logp = 46;
  const int logq = 51;
  const int remaining_level = 20;
  const int boot_level = 14;
  const int transition_logp_primes = 3; // shared-a path
  const int num_special_primes = 5;
  const int log_special_prime = 51;
  const int logN = 16;
  const size_t sparse_slots = (size_t)1 << 15;

  vector<int> coeff_bit_vec;
  coeff_bit_vec.push_back(logq);
  for (int i = 0; i < remaining_level; i++)
    coeff_bit_vec.push_back(logp);
  for (int i = 0; i < transition_logp_primes; i++)
    coeff_bit_vec.push_back(logp);
  for (int i = 0; i < boot_level - transition_logp_primes; i++)
    coeff_bit_vec.push_back(logq);
  for (int i = 0; i < num_special_primes; i++)
    coeff_bit_vec.push_back(log_special_prime);

  EncryptionParameters parms(scheme_type::ckks);
  const size_t poly_modulus_degree = (size_t)1 << logN;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
  parms.set_special_modulus_size(num_special_primes);
  parms.set_secret_key_hamming_weight(192);
  parms.set_sparse_slots(sparse_slots);
  const double scale = pow(2.0, logp);

  PhantomContext context(parms);
  print_parameters(context);

  PhantomSecretKey secret_key(context);
  PhantomPublicKey public_key = secret_key.gen_publickey(context);
  PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Encryptor encryptor(&context, &public_key);
  Decryptor decryptor(&context, &secret_key);
  Evaluator evaluator(&context, &phantom_encoder);
  const size_t slot_count = encoder.slot_count();

  // ---- read the reconstructed LN1 input (768 lines x 5 token values) ----
  const int num_col = 768;
  const int num_tok = 5;
  const int stride = 256; // batch_input layout: slot = num_X * token + sample
  ifstream fin("ln1_repro_input.txt");
  if (!fin.is_open())
  {
    cout << "ERROR: cannot open ln1_repro_input.txt (run from build/)" << endl;
    return;
  }
  vector<vector<double>> vals(num_col, vector<double>(num_tok, 0.0));
  for (int i = 0; i < num_col; ++i)
    for (int t = 0; t < num_tok; ++t)
      if (!(fin >> vals[i][t]))
      {
        cout << "ERROR: input file truncated at ct " << i << " token " << t << endl;
        return;
      }
  fin.close();

  // plaintext reference: per-token LN with gamma=1, beta=0
  // layernorm() computes out_i = (n*x_i - sum) * (1/sqrt(var)) * gamma/sqrt(n)
  // with var = sum(x-mean)^2, i.e. out_i = (x_i - mean) / sqrt(popvar).
  vector<double> ref_mean(num_tok, 0.0), ref_popstd(num_tok, 0.0);
  for (int t = 0; t < num_tok; ++t)
  {
    for (int i = 0; i < num_col; ++i)
      ref_mean[t] += vals[i][t];
    ref_mean[t] /= num_col;
    double v = 0;
    for (int i = 0; i < num_col; ++i)
      v += (vals[i][t] - ref_mean[t]) * (vals[i][t] - ref_mean[t]);
    ref_popstd[t] = sqrt(v / num_col);
    cout << "token " << t << ": plaintext var=" << v << " popstd=" << ref_popstd[t] << endl;
  }

  // ---- encrypt at top level, then drop to 24 primes (LN1's entry level) ----
  vector<PhantomCiphertext> x(num_col);
  for (int i = 0; i < num_col; ++i)
  {
    vector<double> vec(slot_count, 0.0);
    for (int t = 0; t < num_tok; ++t)
      vec[t * stride] = vals[i][t];
    PhantomPlaintext pt;
    encoder.encode(vec, scale, pt);
    encryptor.encrypt(pt, x[i]);
  }
  cout << "fresh ct primes = " << x[0].coeff_modulus_size() << endl;
  while (x[0].coeff_modulus_size() > 24)
    for (int i = 0; i < num_col; ++i)
      evaluator.mod_switch_to_next_inplace(x[i]);
  cout << "LN1 input primes = " << x[0].coeff_modulus_size()
       << " log2scale=" << log2(x[0].scale()) << endl;

  vector<int> b_vec(slot_count, 0);
  for (int t = 0; t < num_tok; ++t)
    b_vec[t * stride] = 1;
  vector<double> gamma(num_col, 1.0), beta(num_col, 0.0);

  // ---- the exact call from single_layer_test ----
  vector<PhantomCiphertext> out = layernorm(x, gamma, beta, b_vec, context, relin_keys, secret_key);

  // ---- verify: max abs error vs plaintext LN over valid slots ----
  double max_err = 0;
  for (int i = 0; i < num_col; ++i)
  {
    PhantomPlaintext pt;
    decryptor.decrypt(out[i], pt);
    vector<double> r;
    encoder.decode(pt, r);
    for (int t = 0; t < num_tok; ++t)
    {
      double ref = (vals[i][t] - ref_mean[t]) / ref_popstd[t];
      max_err = max(max_err, fabs(r[t * stride] - ref));
    }
  }
  cout << "LN1 repro: max abs err vs plaintext LN = " << max_err << endl;
  cout << (max_err < 0.1 ? "LN1 repro PASSED (inv_sqrt converged)"
                         : "LN1 repro FAILED (divergence reproduced!)")
       << endl;
}
