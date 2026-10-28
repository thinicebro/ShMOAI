/**
 * Shared-a (batch) version of the level-3 coefficient-to-slot linear transform.
 *
 * Mirrors Bootstrapper::coefftoslot_3 / sflinv_3 (and the two BSGS linear
 * transforms) but operates on a moai::SharedACiphertextBatch: every rotation of the
 * ordinary algorithm becomes one Sh-Rot (shared-a rotation), so the automorphism +
 * key-switch ModUp of the shared a-part is done once per step for the whole batch
 * instead of once per member. All other operations (diagonal plaintext multiply,
 * add, rescale) are element-wise on the n + 1 polys of the batch.
 *
 * Rotation keys are obtained through SharedAKeyGen::acquire_shared_galois_key:
 * by default (不摊销) they are generated on demand and released right after use,
 * so the peak extra device memory is a single shared-a Galois key; with the
 * cache enabled (摊销) they are kept on the GPU and reused across chunks.
 */
#include "Bootstrapper.cuh"

#include "FmtSwitch.cuh"
#include "ShRotate.cuh"
#include "SharedABatchOps.cuh"

using namespace std;
using namespace phantom;
using namespace moai;

namespace {

/** Rotate the whole batch by step slots (step already reduced into [0, Nh)).
 * step == 0 is the identity and needs no key. */
void sh_rotate_step(const PhantomContext &context, const SharedACiphertextBatch &in, int step, SharedAKeyGen &keygen,
                    SharedACiphertextBatch &out, const cudaStream_t &stream) {
    if (step == 0) {
        shared_a_copy(context, in, out, stream);
        return;
    }
    const SharedAGaloisKey &key = keygen.acquire_shared_galois_key(step, stream);
    shared_a_rotate(context, in, key, out, stream);
}

} // namespace

void Bootstrapper::shared_a_bsgs_linear_transform(
    SharedACiphertextBatch &rtncipher, const SharedACiphertextBatch &cipher, int totlen, int basicstep, int coeff_logn,
    const vector<vector<complex<double>>> &fftcoeff, SharedAKeyGen &keygen)
{
  const auto &stream = phantom::util::global_variables::default_stream->get_stream();
  auto &context = *ckks->context;

  // babyct * diag, in-place semantics of multiply_vector_reduced_error
  auto multiply_diag = [&](const SharedACiphertextBatch &in, const vector<complex<double>> &diag,
                           SharedACiphertextBatch &out) {
    shared_a_copy(context, in, out, stream);
    PhantomPlaintext plain;
    ckks->encoder.encode(diag, out.scale_, plain);
    mod_switch_to_inplace(context, plain, out.chain_index_);
    shared_a_multiply_plain_inplace(context, out, plain, stream);
  };

  int gs1 = giantstep(2 * totlen + 1);
  int basicstart1 = -totlen + gs1 * floor((totlen + 0.0) / (gs1 + 0.0));
  int giantfirst1 = -floor((totlen + 0.0) / (gs1 + 0.0));
  int giantlast1 = floor((2 * totlen + 0.0) / (gs1 + 0.0)) + giantfirst1;

  vector<SharedACiphertextBatch> babyct(gs1);
  SharedACiphertextBatch giantct, tmpct, tmptmpct;
  bool giantbool = false, tmpctbool = false;
  vector<complex<double>> rotatedcoeff;
  rotatedcoeff.reserve(Nh);

  for (int i = basicstart1; i < basicstart1 + gs1; i++)
  {
    if (i == 0)
      shared_a_copy(context, cipher, babyct[i - basicstart1], stream);
    else
      sh_rotate_step(context, cipher, (int)((Nh + i * basicstep) % Nh), keygen, babyct[i - basicstart1], stream);
  }

  for (int i = giantfirst1; i <= giantlast1; i++)
  {
    giantbool = false;
    const int jend = (i != giantlast1) ? basicstart1 + gs1 - 1 : totlen - i * gs1;
    for (int j = basicstart1; j <= jend; j++)
    {
      rotation(coeff_logn, Nh, (-i) * gs1 * basicstep, fftcoeff[(i * gs1 + j) + totlen], rotatedcoeff);
      multiply_diag(babyct[j - basicstart1], rotatedcoeff, tmptmpct);
      if (!giantbool)
      {
        shared_a_copy(context, tmptmpct, giantct, stream);
        giantbool = true;
      }
      else
        shared_a_add_inplace(context, giantct, tmptmpct, stream);
    }

    if (i != 0)
    {
      sh_rotate_step(context, giantct, (int)((Nh + i * gs1 * basicstep) % Nh), keygen, tmptmpct, stream);
      if (!tmpctbool)
      {
        shared_a_copy(context, tmptmpct, tmpct, stream);
        tmpctbool = true;
      }
      else
        shared_a_add_inplace(context, tmpct, tmptmpct, stream);
    }
    else
    {
      if (!tmpctbool)
      {
        shared_a_copy(context, giantct, tmpct, stream);
        tmpctbool = true;
      }
      else
        shared_a_add_inplace(context, tmpct, giantct, stream);
    }
  }
  rtncipher = std::move(tmpct);
}

void Bootstrapper::shared_a_rotated_bsgs_linear_transform(
    SharedACiphertextBatch &rtncipher, const SharedACiphertextBatch &cipher, int totlen, int basicstep, int coeff_logn,
    const vector<vector<complex<double>>> &fftcoeff, SharedAKeyGen &keygen)
{
  const auto &stream = phantom::util::global_variables::default_stream->get_stream();
  auto &context = *ckks->context;

  auto multiply_diag = [&](const SharedACiphertextBatch &in, const vector<complex<double>> &diag,
                           SharedACiphertextBatch &out) {
    shared_a_copy(context, in, out, stream);
    PhantomPlaintext plain;
    ckks->encoder.encode(diag, out.scale_, plain);
    mod_switch_to_inplace(context, plain, out.chain_index_);
    shared_a_multiply_plain_inplace(context, out, plain, stream);
  };

  int gs2 = giantstep(totlen + 1);
  int giantlast2 = floor((totlen + 0.0) / (gs2 + 0.0));

  vector<SharedACiphertextBatch> babyct(gs2);
  SharedACiphertextBatch giantct, tmpct, tmptmpct;
  bool giantbool = false, tmpctbool = false;
  vector<complex<double>> rotatedcoeff;
  rotatedcoeff.reserve(Nh);

  for (int i = 0; i < gs2; i++)
  {
    if (i == 0)
      shared_a_copy(context, cipher, babyct[i], stream);
    else
      sh_rotate_step(context, cipher, (int)((Nh + i * basicstep) % Nh), keygen, babyct[i], stream);
  }

  for (int i = 0; i <= giantlast2; i++)
  {
    giantbool = false;
    const int jend = (i != giantlast2) ? gs2 - 1 : totlen - i * gs2;
    for (int j = 0; j <= jend; j++)
    {
      rotation(coeff_logn, Nh, (-i) * gs2 * basicstep, fftcoeff[i * gs2 + j], rotatedcoeff);
      multiply_diag(babyct[j], rotatedcoeff, tmptmpct);
      if (!giantbool)
      {
        shared_a_copy(context, tmptmpct, giantct, stream);
        giantbool = true;
      }
      else
        shared_a_add_inplace(context, giantct, tmptmpct, stream);
    }

    if (i != 0)
    {
      sh_rotate_step(context, giantct, (int)((Nh + i * gs2 * basicstep) % Nh), keygen, tmptmpct, stream);
      if (!tmpctbool)
      {
        shared_a_copy(context, tmptmpct, tmpct, stream);
        tmpctbool = true;
      }
      else
        shared_a_add_inplace(context, tmpct, tmptmpct, stream);
    }
    else
    {
      if (!tmpctbool)
      {
        shared_a_copy(context, giantct, tmpct, stream);
        tmpctbool = true;
      }
      else
        shared_a_add_inplace(context, tmpct, giantct, stream);
    }
  }
  rtncipher = std::move(tmpct);
}

void Bootstrapper::shared_a_sflinv_3(SharedACiphertextBatch &rtncipher, const SharedACiphertextBatch &cipher,
                                     SharedAKeyGen &keygen)
{
  const auto &stream = phantom::util::global_variables::default_stream->get_stream();

  int div_part1 = floor(logn / 3.0);
  int div_part2 = floor((logn - div_part1) / 2.0);
  int div_part3 = logn - div_part1 - div_part2;

  int totlen1 = (1 << div_part1) - 1;
  int totlen2 = (1 << div_part2) - 1;
  int totlen3 = (1 << div_part3) - 1;

  int basicstep1 = (1 << (logn - div_part1));
  int basicstep2 = (1 << (logn - div_part1 - div_part2));
  int basicstep3 = 1;

  SharedACiphertextBatch tmpct;
  shared_a_rotated_bsgs_linear_transform(tmpct, cipher, totlen1, basicstep1, logn, invfftcoeff1[slot_index], keygen);
  shared_a_rescale_to_next_inplace(*ckks->context, tmpct, stream);

  SharedACiphertextBatch tmpct2;
  shared_a_bsgs_linear_transform(tmpct2, tmpct, totlen2, basicstep2, logn, invfftcoeff2[slot_index], keygen);
  shared_a_rescale_to_next_inplace(*ckks->context, tmpct2, stream);

  shared_a_bsgs_linear_transform(rtncipher, tmpct2, totlen3, basicstep3, logn + 1, invfftcoeff3[slot_index], keygen);
  shared_a_rescale_to_next_inplace(*ckks->context, rtncipher, stream);
}

void Bootstrapper::shared_a_coefftoslot_3(SharedACiphertextBatch &rtncipher, const SharedACiphertextBatch &cipherbatch,
                                          SharedAKeyGen &keygen)
{
  const auto &stream = phantom::util::global_variables::default_stream->get_stream();

  SharedACiphertextBatch tmpct1, tmpct2;
  shared_a_sflinv_3(tmpct1, cipherbatch, keygen);
  // complex conjugation is the Galois automorphism X -> X^{-1}, which is step 0 in
  // phantom's step->elt convention (elt = 2N - 1)
  const SharedAGaloisKey &conj_key = keygen.acquire_shared_galois_key(0, stream);
  shared_a_rotate(*ckks->context, tmpct1, conj_key, tmpct2, stream);
  shared_a_copy(*ckks->context, tmpct1, rtncipher, stream);
  shared_a_add_inplace(*ckks->context, rtncipher, tmpct2, stream);
}

void Bootstrapper::shared_a_sflinv_full_3(SharedACiphertextBatch &rtncipher, const SharedACiphertextBatch &cipher,
                                          SharedAKeyGen &keygen)
{
  const auto &stream = phantom::util::global_variables::default_stream->get_stream();

  int div_part1 = floor(logn / 3.0);
  int div_part2 = floor((logn - div_part1) / 2.0);
  int div_part3 = logn - div_part1 - div_part2;

  int totlen1 = (1 << div_part1) - 1;
  int totlen2 = (1 << div_part2) - 1;
  int totlen3 = (1 << div_part3) - 1;

  int basicstep1 = (1 << (logn - div_part1));
  int basicstep2 = (1 << (logn - div_part1 - div_part2));
  int basicstep3 = 1;

  SharedACiphertextBatch tmpct;
  shared_a_rotated_bsgs_linear_transform(tmpct, cipher, totlen1, basicstep1, logn, invfftcoeff1[slot_index], keygen);
  shared_a_rescale_to_next_inplace(*ckks->context, tmpct, stream);

  SharedACiphertextBatch tmpct2;
  shared_a_bsgs_linear_transform(tmpct2, tmpct, totlen2, basicstep2, logn, invfftcoeff2[slot_index], keygen);
  shared_a_rescale_to_next_inplace(*ckks->context, tmpct2, stream);

  // full-slot variant: the third layer also uses coeff_logn = logn (slotlen == Nh)
  shared_a_bsgs_linear_transform(rtncipher, tmpct2, totlen3, basicstep3, logn, invfftcoeff3[slot_index], keygen);
  shared_a_rescale_to_next_inplace(*ckks->context, rtncipher, stream);
}

void Bootstrapper::shared_a_coefftoslot_full_3(SharedACiphertextBatch &rtncipher1, SharedACiphertextBatch &rtncipher2,
                                               const SharedACiphertextBatch &cipherbatch, SharedAKeyGen &keygen)
{
  const auto &stream = phantom::util::global_variables::default_stream->get_stream();
  auto &context = *ckks->context;

  SharedACiphertextBatch tmpct1;
  shared_a_sflinv_full_3(tmpct1, cipherbatch, keygen);

  // tmpct2 = -i * tmpct1 (plaintext encoded with scale 1.0, like coefftoslot_full_3)
  complex<double> iunit(0.0, 1.0);
  vector<complex<double>> tmpvec(Nh, 0);
  for (int i = 0; i < Nh; i++)
    tmpvec[i] -= iunit;
  PhantomPlaintext tmpplain;
  ckks->encoder.encode(tmpvec, 1.0, tmpplain);
  mod_switch_to_inplace(context, tmpplain, tmpct1.chain_index_);

  SharedACiphertextBatch tmpct2;
  shared_a_copy(context, tmpct1, tmpct2, stream);
  shared_a_multiply_plain_inplace(context, tmpct2, tmpplain, stream);

  // complex conjugation = Galois automorphism X -> X^{-1} = step 0 (elt 2N - 1)
  const SharedAGaloisKey &conj_key = keygen.acquire_shared_galois_key(0, stream);
  SharedACiphertextBatch tmpct3, tmpct4;
  shared_a_rotate(context, tmpct2, conj_key, tmpct3, stream);
  shared_a_rotate(context, tmpct1, conj_key, tmpct4, stream);

  shared_a_copy(context, tmpct1, rtncipher1, stream);
  shared_a_add_inplace(context, rtncipher1, tmpct4, stream);
  shared_a_copy(context, tmpct2, rtncipher2, stream);
  shared_a_add_inplace(context, rtncipher2, tmpct3, stream);
}

/**
 * Shared-a (batch) mirror of Bootstrapper::sfl_full_3: the slot-to-coeff linear
 * transform of S2C-first bootstrapping, executed at a LOW level (before the
 * modraise). Structure identical to sfl_full_3: three BSGS layers with the
 * reversed level split, fftcoeff1/2/3, and a third layer whose coefficient
 * factor (kLowS2cLayer3Factor) sets the S2C-first amplitude convention (see
 * s2c_first_correct_amplitude in BatchBootstrap.cuh).
 */
void Bootstrapper::shared_a_sfl_full_3(SharedACiphertextBatch &rtncipher, const SharedACiphertextBatch &cipher,
                                       SharedAKeyGen &keygen, double layer3_factor)
{
  const auto &stream = phantom::util::global_variables::default_stream->get_stream();

  int div_part3 = floor(logn / 3.0);
  int div_part2 = floor((logn - div_part3) / 2.0);
  int div_part1 = logn - div_part3 - div_part2;

  int totlen1 = (1 << div_part1) - 1;
  int totlen2 = (1 << div_part2) - 1;
  int totlen3 = (1 << div_part3) - 1;

  int basicstep1 = 1;
  int basicstep2 = (1 << div_part1);
  int basicstep3 = (1 << (div_part1 + div_part2));

  SharedACiphertextBatch tmpct;
  shared_a_bsgs_linear_transform(tmpct, cipher, totlen1, basicstep1, logn, fftcoeff1[slot_index], keygen);
  shared_a_rescale_to_next_inplace(*ckks->context, tmpct, stream);

  SharedACiphertextBatch tmpct2;
  shared_a_bsgs_linear_transform(tmpct2, tmpct, totlen2, basicstep2, logn, fftcoeff2[slot_index], keygen);
  shared_a_rescale_to_next_inplace(*ckks->context, tmpct2, stream);

  // Third layer with the S2C-first factor F = layer3_factor (default
  // kLowS2cLayer3Factor = 1/16): the chain's message amplitude is
  // (F/(q0/scale)) * m, restored to m by the post-EvalMod plaintext correction
  // (s2c_first_correct_amplitude, which must receive the SAME factor). Small F
  // keeps the EvalMod fractional part tiny (extraction error ~F^3) and the
  // post-S2C polynomial coefficients far below q0/2. Do NOT use sfl_full_3's
  // working-chain scale-factor formula here (~32 at this level): it pushes the
  // fractional part far outside the EvalMod approximation domain.
  const double factor = layer3_factor;
  vector<vector<complex<double>>> fftcoeff3_scale(2 * totlen3 + 1);
  for (int i = 0; i < totlen2 + 1; i++)
  {
    fftcoeff3_scale[i].resize(n);
    for (long j = 0; j < n; j++)
      fftcoeff3_scale[i][j] = fftcoeff3[slot_index][i][j] * factor;
  }

  shared_a_rotated_bsgs_linear_transform(rtncipher, tmpct2, totlen3, basicstep3, logn, fftcoeff3_scale, keygen);
  shared_a_rescale_to_next_inplace(*ckks->context, rtncipher, stream);
  // keep the natural (preserved) scale: the output feeds the modraise, whose
  // downstream C2S/EvalMod expect the original message scale
  rtncipher.scale_ = tmpct2.scale_;
}
