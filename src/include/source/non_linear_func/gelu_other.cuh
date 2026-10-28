#include "include.cuh"

using namespace std;
using namespace std::chrono;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;

PhantomCiphertext gelu_v2(PhantomCiphertext &x,
                          PhantomContext &context, PhantomRelinKey &relin_keys, PhantomSecretKey &sk,
                          phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream,
                          const vector<int> *bias_vec = nullptr)
{

  PhantomCKKSEncoder phantom_encoder(context);
  // pack Phantom to SEAL style
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  // for test
  Decryptor decryptor(&context, &sk);

  double scale = x.scale();
  size_t slot_count = encoder.slot_count();

  double coeff_high_to_low[] = {3.18006986e-24, 5.70792114e-22, 3.97205561e-20, 1.31854608e-18,
                                1.64153184e-17, -2.33052347e-16, -9.78309547e-15, -6.72238500e-14,
                                1.43093357e-12, 2.41129634e-11, -4.00991558e-11, -3.06661368e-09,
                                -1.00479838e-08, 2.05368974e-07, 1.25666834e-06, -7.76703686e-06,
                                -6.75419265e-05, 1.62401656e-04, 1.97100905e-03, -1.70511673e-03,
                                -3.22621248e-02, 7.22135066e-03, 3.39374355e-01, 4.92938360e-01, 1.21149468e-02};

  vector<PhantomCiphertext> x_n(25);
  x_n[1] = x;
  double s0 = 0.1;
  PhantomPlaintext inv_e;
  if (bias_vec != nullptr)
  {
    // [pad-mask-gelu] fold the 0/1 padding mask into the s0 prescale
    // plaintext: real slots get x*0.1 as before, padding slots get x*0 = 0,
    // so the 24-degree polynomial never sees drifted padding (observed: L3
    // gelu detonated to 8.5e58 -- inter-out real slots were a clean 4.56,
    // but padding drift makes u^i terms ~e56 whose cancellation error wraps
    // the whole polynomial). Same trick as softmax_boot's exp mask: encoded
    // at the ct scale (2^46, NOT 1.0 -- scale-1.0 encode rounds the mask's
    // ~0.2 IFFT coefficients to the zero polynomial), zero extra multiplies,
    // zero extra modulus levels. Padding output becomes the constant term
    // (~0.012), harmless for the downstream FFN projection.
    vector<double> mask_s0(slot_count, 0.0);
    for (size_t k = 0; k < slot_count; ++k)
    {
      if ((*bias_vec)[k] == 1)
      {
        mask_s0[k] = s0;
      }
    }
    encoder.encode(mask_s0, x_n[1].params_id(), x_n[1].scale(), inv_e, stream);
  }
  else
  {
    encoder.encode(s0, x_n[1].params_id(), x_n[1].scale(), inv_e, stream);
  }
  bridge_to_default(stream);
  evaluator.multiply_plain_inplace(x_n[1], inv_e, stream);
  evaluator.rescale_to_next_inplace(x_n[1], stream);

  double inv_s0 = 1 / s0;
  double temps0 = inv_s0;
  for (int i = 23; i >= 0; --i)
  {
    coeff_high_to_low[i] *= temps0;
    temps0 *= inv_s0;
    // cout <<i<<" "<<coeff_high_to_low[i] <<" "<<temps0<<endl;
  }

  // compute x^2,4,8,16
  evaluator.square(x_n[1], x_n[2], stream);
  evaluator.relinearize_inplace(x_n[2], relin_keys, stream);
  evaluator.rescale_to_next_inplace(x_n[2], stream);

  evaluator.square(x_n[2], x_n[4], stream);
  evaluator.relinearize_inplace(x_n[4], relin_keys, stream);
  evaluator.rescale_to_next_inplace(x_n[4], stream);

  evaluator.square(x_n[4], x_n[8], stream);
  evaluator.relinearize_inplace(x_n[8], relin_keys, stream);
  evaluator.rescale_to_next_inplace(x_n[8], stream);

  evaluator.square(x_n[8], x_n[16], stream);
  evaluator.relinearize_inplace(x_n[16], relin_keys, stream);
  evaluator.rescale_to_next_inplace(x_n[16], stream);

  // cout <<"square. "<<endl;

  // compute x^3,5,9,17
  for (int i = 2; i < 17; i *= 2)
  {
    // cout<<i+1<<" ";
    evaluator.mod_switch_to_inplace(x_n[1], x_n[i].params_id(), stream);
    evaluator.multiply(x_n[1], x_n[i], x_n[i + 1], stream);
    evaluator.relinearize_inplace(x_n[i + 1], relin_keys, stream);
    evaluator.rescale_to_next_inplace(x_n[i + 1], stream);
  }

  // compute x^6,10,18
  for (int i = 4; i < 17; i *= 2)
  {
    // cout<<i+2<<" ";
    evaluator.mod_switch_to_inplace(x_n[2], x_n[i].params_id(), stream);
    evaluator.multiply(x_n[2], x_n[i], x_n[i + 2], stream);
    evaluator.relinearize_inplace(x_n[i + 2], relin_keys, stream);
    evaluator.rescale_to_next_inplace(x_n[i + 2], stream);
  }

  // compute x^7,11,19
  for (int i = 4; i < 17; i *= 2)
  {
    // cout<<i+3<<" ";
    evaluator.mod_switch_to_inplace(x_n[3], x_n[i].params_id(), stream);
    evaluator.multiply(x_n[3], x_n[i], x_n[i + 3], stream);
    evaluator.relinearize_inplace(x_n[i + 3], relin_keys, stream);
    evaluator.rescale_to_next_inplace(x_n[i + 3], stream);
  }

  // compute x^12,20
  for (int i = 8; i < 17; i *= 2)
  {

    evaluator.mod_switch_to_inplace(x_n[4], x_n[i].params_id(), stream);
    evaluator.multiply(x_n[4], x_n[i], x_n[i + 4], stream);
    evaluator.relinearize_inplace(x_n[i + 4], relin_keys, stream);
    evaluator.rescale_to_next_inplace(x_n[i + 4], stream);
  }

  // compute x^13,21
  for (int i = 8; i < 17; i *= 2)
  {
    evaluator.mod_switch_to_inplace(x_n[5], x_n[i].params_id(), stream);
    evaluator.multiply(x_n[5], x_n[i], x_n[i + 5], stream);
    evaluator.relinearize_inplace(x_n[i + 5], relin_keys, stream);
    evaluator.rescale_to_next_inplace(x_n[i + 5], stream);
  }

  // compute x^14,22
  for (int i = 8; i < 17; i *= 2)
  {
    evaluator.mod_switch_to_inplace(x_n[6], x_n[i].params_id(), stream);
    evaluator.multiply(x_n[6], x_n[i], x_n[i + 6], stream);
    evaluator.relinearize_inplace(x_n[i + 6], relin_keys, stream);
    evaluator.rescale_to_next_inplace(x_n[i + 6], stream);
  }

  // compute x^15,23
  for (int i = 8; i < 17; i *= 2)
  {
    evaluator.mod_switch_to_inplace(x_n[7], x_n[i].params_id(), stream);
    evaluator.multiply(x_n[7], x_n[i], x_n[i + 7], stream);
    evaluator.relinearize_inplace(x_n[i + 7], relin_keys, stream);
    evaluator.rescale_to_next_inplace(x_n[i + 7], stream);
  }

  // compute x^24
  evaluator.mod_switch_to_inplace(x_n[8], x_n[16].params_id(), stream);
  evaluator.multiply(x_n[8], x_n[16], x_n[24], stream);
  evaluator.relinearize_inplace(x_n[24], relin_keys, stream);
  evaluator.rescale_to_next_inplace(x_n[24], stream);

  PhantomPlaintext plain_result;
  vector<double> result;
  /*
  decryptor.decrypt(x_n[15],plain_result);
  encoder.decode(plain_result,result);
  cout <<"x^15: ";
  for (int ind = 0 ; ind < 10 ; ++ind){
    cout <<result[ind]<<" ";
  }
  cout <<endl;
*/
  //  cout <<"x^n. "<<endl;

  //  double pt0 = 8*s0;
  //  double pt = pt0;
  //  double ptsum = 0.0;

  // compute \sum a_(24-i)x^i
  PhantomCiphertext res;
  for (int i = 1; i < 25; ++i)
  {
    evaluator.mod_switch_to_inplace(x_n[i], x_n[24].params_id(), stream);
    /*
        decryptor.decrypt(x_n[i],plain_result);
        encoder.decode(plain_result,result);
        cout <<"Ciphertext: x^"<<i<<": ";
          for (int ind = 0 ; ind < 1 ; ++ind){
              cout <<result[ind]<<" ";
          }
          //cout <<endl;
    */
    PhantomPlaintext coeff;
    encoder.encode(coeff_high_to_low[24 - i], x_n[i].params_id(), x_n[i].scale(), coeff, stream);
    evaluator.multiply_plain_inplace(x_n[i], coeff, stream);
    evaluator.rescale_to_next_inplace(x_n[i], stream);
    x_n[i].scale() = scale;
    if (i == 1)
    {
      res = x_n[i];
    }
    else
    {
      evaluator.add_inplace(res, x_n[i], stream);
    }
    /*
        //cout <<"a: "<<coeff_high_to_low[24-i]<<", ";
        encoder.decode(coeff,result);
        cout <<", dcd(ecd(a)): ";
        for (int ind = 0 ; ind < 1 ; ++ind){
            cout <<result[ind]<<" ";
        }
        //cout <<endl;

        decryptor.decrypt(x_n[i],plain_result);
        encoder.decode(plain_result,result);
        cout <<", a_"<<24-i<<" * x^"<<i<<": ";
        for (int ind = 0 ; ind < 1 ; ++ind){
            cout <<result[ind]<<" ";
        }
        //cout <<endl;

        decryptor.decrypt(res,plain_result);
        encoder.decode(plain_result,result);
        cout <<", sum: ";
        for (int ind = 0 ; ind < 1 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;

        cout <<"Plaintext: x^"<<i<<": "<<pt;
        double temppt = pt*coeff_high_to_low[24-i];
        cout <<", a_"<<24-i<<" * x^"<<i<<": "<<temppt;
        ptsum += temppt;
        pt *= pt0;
        cout <<", sum: "<<ptsum<<endl;
      */
  }

  // cout <<"sum. "<<endl;
  // compute res += ecd(a[24])
  PhantomPlaintext coeff0;
  encoder.encode(coeff_high_to_low[24], res.params_id(), res.scale(), coeff0, stream);
  bridge_to_default(stream);
  evaluator.add_plain_inplace(res, coeff0, stream);
  // ptsum += coeff_high_to_low[24];
  // cout <<"pt result = "<<ptsum<<endl;

  return res;
}

// ---------------------------------------------------------------------------
// gelu_cheb: degree-24 GeLU evaluated in the Chebyshev basis on the affine
//   variable u = (x + 3.5) / 16,   fit interval [-19.5, 12.5].
// Needed for the 128-token pipeline: inter activations span [-18.74, 12.04],
// which crosses gelu_v2's x-domain P24 safe zone [-18.14, 10.08], and the P24
// power basis has sum|c_k x^k| ~ 1e8 near the edges (catastrophic cancellation
// amplifies CKKS noise by 8 orders of magnitude). In the Chebyshev basis
// |T_k(u)| <= 1 and sum|d_k| = 14.9, so CKKS noise stays ~1e-7 and the output
// error is the (Lawson near-minimax) fit error: max 0.037 over the interval
// (P24 had 0.078 inside its own narrower band).
// Multiplicative depth is identical to gelu_v2: 1 (affine prescale) + 5 (T_k
// tree) + 1 (final term rescale) = 7 levels, and the same 23 ct-ct multiplies.
// ---------------------------------------------------------------------------
PhantomCiphertext gelu_cheb(PhantomCiphertext &x,
                            PhantomContext &context, PhantomRelinKey &relin_keys, PhantomSecretKey &sk,
                            phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream,
                            const vector<int> *bias_vec = nullptr)
{

  PhantomCKKSEncoder phantom_encoder(context);
  // pack Phantom to SEAL style
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);

  double scale = x.scale();
  size_t slot_count = encoder.slot_count();

  // Lawson near-minimax deg-24 Chebyshev fit of gelu(x) on [-19.5, 12.5],
  // coefficients d_0..d_24 (low to high) for sum_k d_k T_k(u), u=(x+3.5)/16.
  const double cheb_d[25] = {
      3.4550752799e+00, 5.7852067632e+00, 3.1727714170e+00, 7.0234932500e-01, -4.6200569546e-01,
      -3.7817535966e-01, 8.4136420694e-02, 2.2427456987e-01, 2.9451916291e-02, -1.2928072669e-01,
      -6.6659430283e-02, 6.4349502364e-02, 7.1733295078e-02, -1.9882581810e-02, -6.1145844240e-02,
      -8.4827295421e-03, 4.3597254356e-02, 2.3373027950e-02, -2.4790187963e-02, -2.7940100474e-02,
      8.8404026343e-03, 2.3776567058e-02, 8.8171887083e-03, -4.6290425955e-02, -9.7997267900e-03};

  const double s0 = 1.0 / 16.0;  // half-width of [-19.5, 12.5]
  const double b0 = 3.5 / 16.0;  // u = (x + 3.5) / 16

  vector<PhantomCiphertext> T(25); // T[k] = T_k(u), Chebyshev polynomial of the first kind

  // T[1] = u = s0 * x + b0.
  // [pad-mask-gelu] same trick as gelu_v2: with a bias/padding mask the prescale
  // is folded into a 0/1 mask so padding slots get u = 0 + b0 (i.e. x = 0)
  // instead of drifting garbage; |u| <= 1 keeps every T_k bounded there.
  T[1] = x;
  PhantomPlaintext pre;
  if (bias_vec != nullptr)
  {
    vector<double> mask_s0(slot_count, 0.0);
    for (size_t k = 0; k < slot_count; ++k)
    {
      if ((*bias_vec)[k] == 1)
      {
        mask_s0[k] = s0;
      }
    }
    encoder.encode(mask_s0, T[1].params_id(), T[1].scale(), pre, stream);
  }
  else
  {
    encoder.encode(s0, T[1].params_id(), T[1].scale(), pre, stream);
  }
  bridge_to_default(stream);
  evaluator.multiply_plain_inplace(T[1], pre, stream);
  evaluator.rescale_to_next_inplace(T[1], stream);
  PhantomPlaintext shift;
  encoder.encode(b0, T[1].params_id(), T[1].scale(), shift, stream);
  evaluator.add_plain_inplace(T[1], shift, stream);

  // T_{2k} = 2 T_k^2 - 1  (square + relin + rescale, then the affine part is
  // plaintext-only and consumes no extra level)
  auto cheb_double = [&](int k, int dst)
  {
    evaluator.square(T[k], T[dst], stream);
    evaluator.relinearize_inplace(T[dst], relin_keys, stream);
    evaluator.rescale_to_next_inplace(T[dst], stream);
    evaluator.add_inplace(T[dst], T[dst], stream); // 2 T_k^2 (aliasing-safe elementwise add)
    PhantomPlaintext minus_one;
    encoder.encode(-1.0, T[dst].params_id(), T[dst].scale(), minus_one, stream);
    evaluator.add_plain_inplace(T[dst], minus_one, stream);
  };

  // T_{m+n} = 2 T_m T_n - T_{m-n}  (m > n >= 1). Operands may sit at different
  // chain indices; the shallower one is deep-copied and mod-switched down.
  auto cheb_add = [&](int m, int n, int dst)
  {
    if (T[m].chain_index() < T[n].chain_index())
    {
      PhantomCiphertext tm = T[m]; // deep copy
      evaluator.mod_switch_to_inplace(tm, T[n].chain_index(), stream);
      evaluator.multiply(tm, T[n], T[dst], stream);
    }
    else if (T[n].chain_index() < T[m].chain_index())
    {
      PhantomCiphertext tn = T[n]; // deep copy
      evaluator.mod_switch_to_inplace(tn, T[m].chain_index(), stream);
      evaluator.multiply(T[m], tn, T[dst], stream);
    }
    else
    {
      evaluator.multiply(T[m], T[n], T[dst], stream);
    }
    evaluator.relinearize_inplace(T[dst], relin_keys, stream);
    evaluator.rescale_to_next_inplace(T[dst], stream);
    evaluator.add_inplace(T[dst], T[dst], stream); // 2 T_m T_n
    PhantomCiphertext tsub = T[m - n];             // deep copy
    evaluator.mod_switch_to_inplace(tsub, T[dst].chain_index(), stream);
    tsub.scale() = T[dst].scale(); // same scale-forcing convention as the sum below
    evaluator.sub_inplace(T[dst], tsub, stream);
  };

  // T_2 .. T_24, all at multiplicative depth <= 6 (same as gelu_v2's x^24):
  //   evens by doubling, odds via 2 T_{k+1} T_k - T_1
  cheb_double(1, 2);
  cheb_add(2, 1, 3);
  cheb_double(2, 4);
  cheb_add(3, 2, 5);
  cheb_double(3, 6);
  cheb_add(4, 3, 7);
  cheb_double(4, 8);
  cheb_add(5, 4, 9);
  cheb_double(5, 10);
  cheb_add(6, 5, 11);
  cheb_double(6, 12);
  cheb_add(7, 6, 13);
  cheb_double(7, 14);
  cheb_add(8, 7, 15);
  cheb_double(8, 16);
  cheb_add(9, 8, 17);
  cheb_double(9, 18);
  cheb_add(10, 9, 19);
  cheb_double(10, 20);
  cheb_add(11, 10, 21);
  cheb_double(11, 22);
  cheb_add(12, 11, 23);
  cheb_double(12, 24);

  // res = sum_{k=1}^{24} d_k T_k(u) + d_0   (same final-sum pattern as gelu_v2:
  // every term is mod-switched to the deepest level, scaled by d_k, rescaled,
  // and its scale forced back to the input scale before accumulation)
  PhantomCiphertext res;
  for (int k = 1; k < 25; ++k)
  {
    evaluator.mod_switch_to_inplace(T[k], T[24].chain_index(), stream);
    PhantomPlaintext coeff;
    encoder.encode(cheb_d[k], T[k].params_id(), T[k].scale(), coeff, stream);
    evaluator.multiply_plain_inplace(T[k], coeff, stream);
    evaluator.rescale_to_next_inplace(T[k], stream);
    T[k].scale() = scale;
    if (k == 1)
    {
      res = T[k];
    }
    else
    {
      evaluator.add_inplace(res, T[k], stream);
    }
  }

  PhantomPlaintext coeff0;
  encoder.encode(cheb_d[0], res.params_id(), res.scale(), coeff0, stream);
  bridge_to_default(stream);
  evaluator.add_plain_inplace(res, coeff0, stream);

  return res;
}

// PhantomCiphertext gelu_v2(PhantomCiphertext &x,
//                           PhantomContext &context, PhantomRelinKey &relin_keys, PhantomSecretKey &sk)
// {

//   PhantomCKKSEncoder phantom_encoder(context);
//   // pack Phantom to SEAL style
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);
//   // for test
//   Decryptor decryptor(&context, &sk);

//   double scale = x.scale();
//   size_t slot_count = encoder.slot_count();

//   double coeff_high_to_low[] = {3.18006986e-24, 5.70792114e-22, 3.97205561e-20, 1.31854608e-18,
//                                 1.64153184e-17, -2.33052347e-16, -9.78309547e-15, -6.72238500e-14,
//                                 1.43093357e-12, 2.41129634e-11, -4.00991558e-11, -3.06661368e-09,
//                                 -1.00479838e-08, 2.05368974e-07, 1.25666834e-06, -7.76703686e-06,
//                                 -6.75419265e-05, 1.62401656e-04, 1.97100905e-03, -1.70511673e-03,
//                                 -3.22621248e-02, 7.22135066e-03, 3.39374355e-01, 4.92938360e-01, 1.21149468e-02};

//   vector<PhantomCiphertext> x_n(25);
//   x_n[1] = x;
//   double s0 = 0.1;
//   PhantomPlaintext inv_e;
//   encoder.encode(s0, x_n[1].params_id(), x_n[1].scale(), inv_e);
//   evaluator.multiply_plain_inplace(x_n[1], inv_e);
//   evaluator.rescale_to_next_inplace(x_n[1]);

//   double inv_s0 = 1 / s0;
//   double temps0 = inv_s0;
//   for (int i = 23; i >= 0; --i)
//   {
//     coeff_high_to_low[i] *= temps0;
//     temps0 *= inv_s0;
//     // cout <<i<<" "<<coeff_high_to_low[i] <<" "<<temps0<<endl;
//   }

//   // compute x^2,4,8,16
//   evaluator.square(x_n[1], x_n[2]);
//   evaluator.relinearize_inplace(x_n[2], relin_keys);
//   evaluator.rescale_to_next_inplace(x_n[2]);

//   evaluator.square(x_n[2], x_n[4]);
//   evaluator.relinearize_inplace(x_n[4], relin_keys);
//   evaluator.rescale_to_next_inplace(x_n[4]);

//   evaluator.square(x_n[4], x_n[8]);
//   evaluator.relinearize_inplace(x_n[8], relin_keys);
//   evaluator.rescale_to_next_inplace(x_n[8]);

//   evaluator.square(x_n[8], x_n[16]);
//   evaluator.relinearize_inplace(x_n[16], relin_keys);
//   evaluator.rescale_to_next_inplace(x_n[16]);

//   // cout <<"square. "<<endl;

//   // compute x^3,5,9,17
//   for (int i = 2; i < 17; i *= 2)
//   {
//     // cout<<i+1<<" ";
//     evaluator.mod_switch_to_inplace(x_n[1], x_n[i].params_id());
//     evaluator.multiply(x_n[1], x_n[i], x_n[i + 1]);
//     evaluator.relinearize_inplace(x_n[i + 1], relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i + 1]);
//   }

//   // compute x^6,10,18
//   for (int i = 4; i < 17; i *= 2)
//   {
//     // cout<<i+2<<" ";
//     evaluator.mod_switch_to_inplace(x_n[2], x_n[i].params_id());
//     evaluator.multiply(x_n[2], x_n[i], x_n[i + 2]);
//     evaluator.relinearize_inplace(x_n[i + 2], relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i + 2]);
//   }

//   // compute x^7,11,19
//   for (int i = 4; i < 17; i *= 2)
//   {
//     // cout<<i+3<<" ";
//     evaluator.mod_switch_to_inplace(x_n[3], x_n[i].params_id());
//     evaluator.multiply(x_n[3], x_n[i], x_n[i + 3]);
//     evaluator.relinearize_inplace(x_n[i + 3], relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i + 3]);
//   }

//   // compute x^12,20
//   for (int i = 8; i < 17; i *= 2)
//   {

//     evaluator.mod_switch_to_inplace(x_n[4], x_n[i].params_id());
//     evaluator.multiply(x_n[4], x_n[i], x_n[i + 4]);
//     evaluator.relinearize_inplace(x_n[i + 4], relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i + 4]);
//   }

//   // compute x^13,21
//   for (int i = 8; i < 17; i *= 2)
//   {
//     evaluator.mod_switch_to_inplace(x_n[5], x_n[i].params_id());
//     evaluator.multiply(x_n[5], x_n[i], x_n[i + 5]);
//     evaluator.relinearize_inplace(x_n[i + 5], relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i + 5]);
//   }

//   // compute x^14,22
//   for (int i = 8; i < 17; i *= 2)
//   {
//     evaluator.mod_switch_to_inplace(x_n[6], x_n[i].params_id());
//     evaluator.multiply(x_n[6], x_n[i], x_n[i + 6]);
//     evaluator.relinearize_inplace(x_n[i + 6], relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i + 6]);
//   }

//   // compute x^15,23
//   for (int i = 8; i < 17; i *= 2)
//   {
//     evaluator.mod_switch_to_inplace(x_n[7], x_n[i].params_id());
//     evaluator.multiply(x_n[7], x_n[i], x_n[i + 7]);
//     evaluator.relinearize_inplace(x_n[i + 7], relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i + 7]);
//   }

//   // compute x^24
//   evaluator.mod_switch_to_inplace(x_n[8], x_n[16].params_id());
//   evaluator.multiply(x_n[8], x_n[16], x_n[24]);
//   evaluator.relinearize_inplace(x_n[24], relin_keys);
//   evaluator.rescale_to_next_inplace(x_n[24]);

//   PhantomPlaintext plain_result;
//   vector<double> result;
//   /*
//   decryptor.decrypt(x_n[15],plain_result);
//   encoder.decode(plain_result,result);
//   cout <<"x^15: ";
//   for (int ind = 0 ; ind < 10 ; ++ind){
//     cout <<result[ind]<<" ";
//   }
//   cout <<endl;
// */
//   //  cout <<"x^n. "<<endl;

//   //  double pt0 = 8*s0;
//   //  double pt = pt0;
//   //  double ptsum = 0.0;

//   // compute \sum a_(24-i)x^i
//   PhantomCiphertext res;
//   for (int i = 1; i < 25; ++i)
//   {
//     evaluator.mod_switch_to_inplace(x_n[i], x_n[24].params_id());
//     /*
//         decryptor.decrypt(x_n[i],plain_result);
//         encoder.decode(plain_result,result);
//         cout <<"Ciphertext: x^"<<i<<": ";
//           for (int ind = 0 ; ind < 1 ; ++ind){
//               cout <<result[ind]<<" ";
//           }
//           //cout <<endl;
//     */
//     PhantomPlaintext coeff;
//     encoder.encode(coeff_high_to_low[24 - i], x_n[i].params_id(), x_n[i].scale(), coeff);
//     evaluator.multiply_plain_inplace(x_n[i], coeff);
//     evaluator.rescale_to_next_inplace(x_n[i]);
//     x_n[i].scale() = scale;
//     if (i == 1)
//     {
//       res = x_n[i];
//     }
//     else
//     {
//       evaluator.add_inplace(res, x_n[i]);
//     }
//     /*
//         //cout <<"a: "<<coeff_high_to_low[24-i]<<", ";
//         encoder.decode(coeff,result);
//         cout <<", dcd(ecd(a)): ";
//         for (int ind = 0 ; ind < 1 ; ++ind){
//             cout <<result[ind]<<" ";
//         }
//         //cout <<endl;

//         decryptor.decrypt(x_n[i],plain_result);
//         encoder.decode(plain_result,result);
//         cout <<", a_"<<24-i<<" * x^"<<i<<": ";
//         for (int ind = 0 ; ind < 1 ; ++ind){
//             cout <<result[ind]<<" ";
//         }
//         //cout <<endl;

//         decryptor.decrypt(res,plain_result);
//         encoder.decode(plain_result,result);
//         cout <<", sum: ";
//         for (int ind = 0 ; ind < 1 ; ++ind){
//             cout <<result[ind]<<" ";
//         }
//         cout <<endl;

//         cout <<"Plaintext: x^"<<i<<": "<<pt;
//         double temppt = pt*coeff_high_to_low[24-i];
//         cout <<", a_"<<24-i<<" * x^"<<i<<": "<<temppt;
//         ptsum += temppt;
//         pt *= pt0;
//         cout <<", sum: "<<ptsum<<endl;
//       */
//   }

//   // cout <<"sum. "<<endl;
//   // compute res += ecd(a[24])
//   PhantomPlaintext coeff0;
//   encoder.encode(coeff_high_to_low[24], res.params_id(), res.scale(), coeff0);
//   evaluator.add_plain_inplace(res, coeff0);
//   // ptsum += coeff_high_to_low[24];
//   // cout <<"pt result = "<<ptsum<<endl;

//   return res;
// }
