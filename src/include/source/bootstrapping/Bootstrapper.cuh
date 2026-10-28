#pragma once

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>

#include "../utils.cuh"
#include "ModularReducer.cuh"

using namespace std;
using namespace moai;
using namespace phantom;
using namespace phantom::util;

namespace moai {
struct SharedACiphertextBatch; // SharedACiphertext.cuh
class SharedAKeyGen;           // FmtSwitch.cuh
} // namespace moai

/**
 * Third-layer coefficient factor for the LOW-level S2C transform of S2C-first
 * bootstrapping. The chain's message amplitude is (F / (q0/scale)) * m, so a
 * post-EvalMod plaintext multiply by (q0/scale)/F restores m (see the pipelines
 * in BatchBootstrap.cuh). Small F keeps the EvalMod input's fractional part
 * tiny - its extraction error scales ~F^3, so the corrected error scales ~F^2
 * (scan: F=1/16 -> corrected mean |err| 3.8e-06, better than the C2S-first
 * chain's 2e-05). F also keeps the post-S2C polynomial coefficients
 * (F/2)*scale*|m| far below q0/2, so the modraise introduces no extra error.
 */
constexpr double kLowS2cLayer3Factor = 0.0625; // 1/16

class Bootstrapper
{
public:
    long loge;  //scale的bit数
    long logn;  //多项式环的维度取对数
    long n;     //多项式环维度
    long logNh; //Nh=N/2
    long Nh;
    long L;     //最大乘法深度

    double initial_scale; //初始缩放因子
    double final_scale;   //最终缩放因子

    long boundary_K;      //模约减边界K
    long sin_cos_deg;      //cos阶数
    long scale_factor;    //缩放因子
    long inverse_deg;     //arcsinx阶数

    CKKSEvaluator *ckks = nullptr;  //初始化ckk同态评估器

    vector<long> slot_vec;  //初始化槽位向量
    long slot_index = 0;     
    vector<vector<vector<vector<complex<double>>>>> orig_coeffvec, orig_invcoeffvec; //原始FFT系数矩阵
    vector<vector<vector<complex<double>>>> fftcoeff1, fftcoeff2, fftcoeff3; //正向FFT系数
    vector<vector<vector<complex<double>>>> invfftcoeff1, invfftcoeff2, invfftcoeff3;   //反向FFT系数

    vector<PhantomPlaintext> fftcoeff_plain1, fftcoeff_plain2, invfftcoeff_plain1, invfftcoeff_plain2; //正逆项FFT系数编码后的明文

    ModularReducer *mod_reducer; //模约减器

    Bootstrapper(
        long _loge,
        long _logn,
        long _logNh,
        long _L,
        double _final_scale,
        long _boundary_K,
        long _sin_cos_deg,
        long _scale_factor,
        long _inverse_deg,
        CKKSEvaluator *ckks);

    inline void set_final_scale(double _final_scale)
    {
        final_scale = _final_scale;
    }

    // Add rotation keys needed in bootstrapping (private function)  //用于生成特定的旋转密钥
    void addLeftRotKeys_Linear_to_vector(vector<int> &gal_steps_vector);
    void addLeftRotKeys_Linear_to_vector_3(vector<int> &gal_steps_vector);

    void addLeftRotKeys_Linear_to_vector_3_other_slots(vector<int> &gal_steps_vector, long other_logn);

    void addLeftRotKeys_Linear_to_vector_one_depth(vector<int> &gal_steps_vector);
    void addLeftRotKeys_Linear_to_vector_one_depth_more_depth(vector<int> &gal_steps_vector); 

    // Add rotation keys needed in bootstrapping (public function)
    void addBootKeys(PhantomGaloisKey &gal_keys);  //标准自举所需旋转密钥
    void addBootKeys_3(PhantomGaloisKey &gal_keys);  //level-3自举所需旋转密钥
    void addBootKeys_other_keys(PhantomGaloisKey &gal_keys, vector<int> &other_keys);
    void addBootKeys_3_other_keys(PhantomGaloisKey &gal_keys, vector<int> &other_keys);

    void addBootKeys_3_other_slots(PhantomGaloisKey &gal_keys, vector<long> &other_logn_vec);
    void addBootKeys_3_other_slots_keys(PhantomGaloisKey &gal_keys, vector<long> &other_logn_vec, vector<int> &other_keys);

    void addBootKeys_hoisting(PhantomGaloisKey &gal_keys);  //hoisting提升优化版旋转密钥
    void addBootKeys_one_depth(PhantomGaloisKey &gal_keys);   //单层深度自举的旋转密钥
    void addBootKeys_one_depth_more_depth(PhantomGaloisKey &gal_keys); //更多深度的单层自举旋转密钥

    void change_logn(long new_logn);

    // Prepare the FFT coefficients  
    //用于生成FFT系数
    void genorigcoeff(); 
    void genfftcoeff_one_depth();
    void genfftcoeff_full_one_depth();
    void geninvfftcoeff_one_depth();
    void generate_LT_coefficient_one_depth();

    void genfftcoeff();
    void genfftcoeff_full();
    void geninvfftcoeff();
    void geninvfftcoeff_full();

    void genfftcoeff_3();
    void geninvfftcoeff_3();
    void generate_LT_coefficient();
    void generate_LT_coefficient_3();

    // Prepare the approximate polynomial
    void prepare_mod_polynomial();

    void subsum(double scale, PhantomCiphertext &cipher);

    void bsgs_linear_transform(
        PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn, const vector<vector<complex<double>>> &fftcoeff);
    void rotated_bsgs_linear_transform(
        PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn, const vector<vector<complex<double>>> &fftcoeff);
    void rotated_nobsgs_linear_transform(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int coeff_logn, vector<vector<complex<double>>> fftcoeff);

    void bsgs_linear_transform_hoisting(
        PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn, vector<vector<complex<double>>> fftcoeff);
    void rotated_bsgs_linear_transform_hoisting(
        PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn, vector<vector<complex<double>>> fftcoeff);

    void sfl_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sfl_full_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sflinv_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sflinv_one_depth_more_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void sfl(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sfl_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sflinv(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sflinv_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void sfl_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sfl_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sfl_half_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sfl_full_half_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sflinv_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sflinv_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void sfl_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sfl_full_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sflinv_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void sflinv_full_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    // original bootstrapping
    void coefftoslot(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slottocoeff(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void coefftoslot_full(PhantomCiphertext &rtncipher1, PhantomCiphertext &rtncipher2, PhantomCiphertext &cipher);
    void slottocoeff_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher1, PhantomCiphertext &cipher2);

    // level-3 LT
    void coefftoslot_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slottocoeff_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slottocoeff_half_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    // shared-a (batch) level-3 LT, implemented in SharedACoeffToSlot.cu: applies the
    // same 3-layer BSGS linear transform to every member of a shared-a batch, using
    // shared-a rotation keys (Sh-Rot) so the a-part is rotated/keyswitched only once
    // per step instead of once per member. Rotation keys are generated on demand via
    // keygen (and released after use), so no large key set has to reside on the GPU.
    void shared_a_coefftoslot_3(moai::SharedACiphertextBatch &rtnbatch,
                                const moai::SharedACiphertextBatch &cipherbatch, moai::SharedAKeyGen &keygen);
    void shared_a_sflinv_3(moai::SharedACiphertextBatch &rtnbatch, const moai::SharedACiphertextBatch &cipherbatch,
                           moai::SharedAKeyGen &keygen);
    // full-slot variants (logn == logNh), mirroring sflinv_full_3 / coefftoslot_full_3:
    // the third layer uses coeff_logn = logn (not logn + 1), and the conjugate pair
    // yields two output batches (real parts in rtnbatch1, imaginary parts in rtnbatch2).
    void shared_a_sflinv_full_3(moai::SharedACiphertextBatch &rtnbatch,
                                const moai::SharedACiphertextBatch &cipherbatch, moai::SharedAKeyGen &keygen);
    void shared_a_coefftoslot_full_3(moai::SharedACiphertextBatch &rtnbatch1, moai::SharedACiphertextBatch &rtnbatch2,
                                     const moai::SharedACiphertextBatch &cipherbatch, moai::SharedAKeyGen &keygen);
    // S2C-first bootstrapping: shared-a mirror of sfl_full_3 (slot-to-coeff linear
    // transform executed at a LOW level, before the modraise). layer3_factor is
    // the amplitude convention F baked into the third BSGS layer; it must match
    // the post-EvalMod s2c_first_correct_amplitude factor (both default to
    // kLowS2cLayer3Factor; callers should thread one value through).
    void shared_a_sfl_full_3(moai::SharedACiphertextBatch &rtnbatch,
                             const moai::SharedACiphertextBatch &cipherbatch, moai::SharedAKeyGen &keygen,
                             double layer3_factor = kLowS2cLayer3Factor);
    void shared_a_bsgs_linear_transform(
        moai::SharedACiphertextBatch &rtncipher, const moai::SharedACiphertextBatch &cipher, int totlen,
        int basicstep, int coeff_logn, const vector<vector<complex<double>>> &fftcoeff, moai::SharedAKeyGen &keygen);
    void shared_a_rotated_bsgs_linear_transform(
        moai::SharedACiphertextBatch &rtncipher, const moai::SharedACiphertextBatch &cipher, int totlen,
        int basicstep, int coeff_logn, const vector<vector<complex<double>>> &fftcoeff, moai::SharedAKeyGen &keygen);

    void coefftoslot_full_3(PhantomCiphertext &rtncipher1, PhantomCiphertext &rtncipher2, PhantomCiphertext &cipher);
    void slottocoeff_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher1, PhantomCiphertext &cipher2);
    void slottocoeff_full_half_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher1, PhantomCiphertext &cipher2);
    // original bootstrapping hoisting version
    void coefftoslot_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slottocoeff_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void coefftoslot_full_hoisting(PhantomCiphertext &rtncipher1, PhantomCiphertext &rtncipher2, PhantomCiphertext &cipher);
    void slottocoeff_full_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher1, PhantomCiphertext &cipher2);

    // one depth bootstrapping
    void coefftoslot_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slottocoeff_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void coefftoslot_full_one_depth(PhantomCiphertext &rtncipher1, PhantomCiphertext &rtncipher2, PhantomCiphertext &cipher);
    void slottocoeff_full_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher1, PhantomCiphertext &cipher2);

    // mul_first bootstrapping
    void coefftoslot_full_mul_first(PhantomCiphertext &rtncipher1, PhantomCiphertext &rtncipher2, PhantomCiphertext &cipher);
    void modraise_inplace(PhantomCiphertext &cipher);

    // API bootstrapping
    void bootstrap_sparse(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void bootstrap_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void bootstrap_sparse_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void bootstrap_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void bootstrap_sparse_real_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void bootstrap_full_real_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void bootstrap_sparse_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void bootstrap_full_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void bootstrap_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void bootstrap_more_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void bootstrap(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void bootstrap_inplace(PhantomCiphertext &cipher);

    void bootstrap_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void bootstrap_inplace_3(PhantomCiphertext &cipher);

    void bootstrap_real_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void bootstrap_inplace_real_3(PhantomCiphertext &cipher);

    void bootstrap_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void bootstrap_inplace_hoisting(PhantomCiphertext &cipher);
};
