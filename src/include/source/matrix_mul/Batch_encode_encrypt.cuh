//交错列打包
#include "include.cuh"

using namespace std;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;

// Interlaced batching, column-encoded.
// In BERT, num_X = 128, num_row = 128, num_col = 768
// Output: 768 Ciphertexts
vector<PhantomCiphertext> batch_input(const vector<vector<vector<double>>> &X,
                                      int num_X, int num_row, int num_col, double scale,
                                      PhantomContext &context, PhantomPublicKey &pk)
{

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Encryptor encryptor(&context, &pk);
  size_t slot_count = encoder.slot_count();
  // cout <<"slot count = "<<slot_count<<endl;

  vector<PhantomCiphertext> output(num_col); //用于存放列交错打包的加密结果，数组中每一个元素对应一列的加密结果。

  // #pragma omp parallel for
  for (int i = 0; i < num_col; ++i)
  {
    vector<double> vec(slot_count, 0);
    for (int j = 0; j < num_X; ++j)
    {
      for (int k = 0; k < num_row; ++k)
      {
        vec[num_X * k + j] = X[j][k][i];  //列交错打包，第j个矩阵的第k,i个元素应该放的位置，这里每一列进行一个交错排序，放到槽中。
      }
    }

    PhantomPlaintext ecd_vec;
    encoder.encode(vec, scale, ecd_vec);  //把得到的行交错打包向量，编码为多项式，最后一个参数ecd_vec用于存放编码好的多项式
    encryptor.encrypt(ecd_vec, output[i]); //加密，结果存放到output中
  }
 
  return output;
}

vector<int> bias_vec(const vector<int> &lengths, int num_X, int num_row) //掩码，用于标记有效和无效Token，对于短句子，会用占位符填充，这些填充被标记出来。
{ 
  vector<int> output(num_X * num_row, 0);
  for (int i = 0; i < num_X; ++i)
  {
    // input X_i has size lengths[i] * num_col
    for (int j = 0; j < lengths[i]; ++j)
    {
      output[j * num_X + i] = 1;
    }
  }
  return output;
}//参数const vector<int> &lengths中每一个元素代表第i个样本(句子)中的有效Token数。