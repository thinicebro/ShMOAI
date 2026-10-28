//关于明密文矩阵乘法，整个流程中之用到了列打包乘以明文矩阵；该文件中实现了单、多线程的明密文矩阵乘法；单、多线程的大列维度矩阵乘法；单、多线程的带掩码的大列维度的明密文矩阵乘法。
#pragma once
#include "include.cuh"

using namespace std;
using namespace std::chrono;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;

#include <omp.h>
#include <algorithm>
#include <iostream>
#include <vector>

using std::vector;

inline vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre(
    vector<PhantomCiphertext> &enc_X,
    const vector<vector<double>> &W,
    int col_X, int col_W, int row_W,
    PhantomContext &context)
{
  vector<PhantomCiphertext> output(static_cast<size_t>(col_W)); //static_cast<T>(value)将括号内的值临时转变问尖括号内的类型

  if (enc_X.empty() || W.empty() || col_W <= 0 || row_W <= 0)  //.empty如果一个元素没有返回1。
  {
    std::cout << "ERROR: empty inputs or bad dimensions.\n";
  }
  if (static_cast<int>(enc_X.size()) < row_W)
  {
    std::cout << "ERROR: enc_X size < row_W.\n";
    return output;
  }
  if (static_cast<int>(W.size()) < row_W)
  {
    std::cout << "ERROR: W rows < row_W.\n";
    return output;
  }
  for (int r = 0; r < row_W; ++r)
  {
    if (static_cast<int>(W[r].size()) < col_W)
    {
      std::cout << "ERROR: W row " << r << " has fewer than col_W columns.\n";
      return output;
    }
  }
  if (col_X != row_W)
  {
    std::cout << "ERROR: bad dimensions of X or W.\n";
    return output;
  }
  // cudaSetDevice(1);
  const double scale = enc_X[0].scale(); //返回密文的缩放因子

  {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const double MB = 1024.0 * 1024.0;
    std::cout << "[ct_pt] entry col_W=" << col_W << " row_W=" << row_W
              << " level=" << enc_X[0].coeff_modulus_size()
              << " GPU free=" << free_b / MB << "MB total=" << total_b / MB << "MB" << std::endl;
  }
  const auto fn_t0 = chrono::high_resolution_clock::now();

  const int max_threads = omp_get_max_threads(); //openmp中的函数，返回程序默认允许创建的最大并行线程数。
  const int nthreads = std::max(1, std::min(max_threads, 32)); //将最大线程数约束道1-32之间
  // std::cout << "nums of thread: " << nthreads << std::endl;

  if (stream_pool.size() < static_cast<size_t>(nthreads))  //stream_pool:全局CUDA流容器，每个 OpenMP 线程独占一条独立 CUDA 流，多流并发执行 GPU 编码、同态乘法，提升并行效率，一条流 = GPU 上独立任务队列，流之间可并行执行，互不阻塞。
  {
    stream_pool.reserve(nthreads);
    for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
    {
      stream_pool.emplace_back();
    } 
  }   //如果CUDA流数小于并行线程数，在流容器中补充空流对象。
  if (nthreads == 1)
  {
    stream_pool[0] = *phantom::util::global_variables::default_stream;
  }//单线程特殊处理，新建私有流，不用全局流

  vector<double> time(col_W, 0.0); //计时，专门存放第i个通道的用时

#pragma omp parallel num_threads(nthreads) //openmp的编译指令：开启多线程并行区域。
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);    //local表示为每个线程创造一份编解码器，不共享
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num(); //获取当前进程编号
    auto &stream = stream_pool[tid]; //取出当前的流对象
    // phantom::util::cuda_stream_wrapper stream;
    // if (nthreads == 1){
    //   stream = *phantom::util::global_variables::default_stream;
    // } else {
    //   stream = stream_pool[tid];
    // }

    // All encode/multiply/add/rescale below run on this pool stream so the
    // whole chain is stream-ordered. enc_X may still have pending writes on
    // the shared default stream (produced by the caller), and this thread's
    // encoder twiddle tables were uploaded on the default stream too, so
    // order the default stream before the pool stream once at entry.
    bridge_from_default(stream);

    chrono::high_resolution_clock::time_point start, end; //高精度时间点类型变量，start记录开始的时间，end记录结束时间
    double elapsed; //时间差变量

#pragma omp for schedule(static) //静态均分任务，将下述代码分配给nthreads个OpenMP线程并发执行。
    for (int i = 0; i < col_W; ++i)
    {

      // acc = enc_X[0] * W[0][i]
      PhantomPlaintext ecd_w_0_i;
      // encoder_local.encode(W[0][i], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
      encoder_local.encode(W[0][i], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i, stream); //该函数将结果写入ecd_w_0_i

      // copy enc_X[0] on THIS pool stream: multiply_plain's implicit copy
      // would be enqueued on enc_X's stored (default) stream and race with
      // the pool-stream multiply.
      PhantomCiphertext acc = deep_copy_cipher(enc_X[0], context, stream);
      start = chrono::high_resolution_clock::now(); //记录当前高精度时间戳，精确到微秒
      evaluator_local.multiply_plain_inplace(acc, ecd_w_0_i, stream); //明密文乘法，与encode同流，保证顺序
      end = chrono::high_resolution_clock::now();
      elapsed = duration_cast<duration<double>>(end - start).count();
      // cout << "[DEBUG] time: " << elapsed << endl;
      time[i] += elapsed;
      //上述代码初始化acc。
      for (int j = 1; j < row_W; ++j)
      {
        PhantomPlaintext ecd_w_j_i;
        // encoder_local.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
        encoder_local.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i, stream);

        PhantomCiphertext tmp = deep_copy_cipher(enc_X[j], context, stream);
        start = chrono::high_resolution_clock::now();
        evaluator_local.multiply_plain_inplace(tmp, ecd_w_j_i, stream);

        evaluator_local.add_inplace(acc, tmp, stream);
        end = chrono::high_resolution_clock::now();

        elapsed = duration_cast<duration<double>>(end - start).count();

        time[i] += elapsed;
      }

      start = chrono::high_resolution_clock::now();
      evaluator_local.rescale_to_next_inplace(acc, stream);
      acc.scale() = scale;

      end = chrono::high_resolution_clock::now();
      elapsed = duration_cast<duration<double>>(end - start).count();
      // cout << "[DEBUG] time: " << elapsed << endl;
      time[i] += elapsed;

      output[static_cast<size_t>(i)] = std::move(acc); //用move方法，直接将acc内存转移到outpu中，不需要拷贝复制，减小内存占用。

      if (i % 8 == 0)
      {
#pragma omp critical
        {
          const double el = duration_cast<duration<double>>(chrono::high_resolution_clock::now() - fn_t0).count();
          size_t free_b = 0, total_b = 0;
          cudaMemGetInfo(&free_b, &total_b);
          const double MB = 1024.0 * 1024.0;
          std::cout << "[ct_pt] col " << i << "/" << col_W << " elapsed=" << el << "s free=" << free_b / MB << "MB" << std::endl;
        }
      }
    }
    cudaStreamSynchronize(stream.get_stream());
  }

  // double total = 0.0;
  // for (double t : time){
  //   total += t;
  // }
  // cout << "ct-pt time(without encoding)" << total << " s" << endl;

  // cudaDeviceSynchronize();
  // stream_pool.clear();

  return output;
}

// single thread version
inline vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_single(  
    vector<PhantomCiphertext> &enc_X,
    const vector<vector<double>> &W,
    int col_X, int col_W, int row_W,
    PhantomContext &context)  //单线程版本，无并行运算。
{
  vector<PhantomCiphertext> output(static_cast<size_t>(col_W));

  if (enc_X.empty() || W.empty() || col_W <= 0 || row_W <= 0)
  {
    std::cout << "ERROR: empty inputs or bad dimensions.\n";
  }
  if (static_cast<int>(enc_X.size()) < row_W)
  {
    std::cout << "ERROR: enc_X size < row_W.\n";
    return output;
  }
  if (static_cast<int>(W.size()) < row_W)
  {
    std::cout << "ERROR: W rows < row_W.\n";
    return output;
  }
  for (int r = 0; r < row_W; ++r)
  {
    if (static_cast<int>(W[r].size()) < col_W)
    {
      std::cout << "ERROR: W row " << r << " has fewer than col_W columns.\n";
      return output;
    }
  }
  if (col_X != row_W)
  {
    std::cout << "ERROR: bad dimensions of X or W.\n";
    return output;
  }
  // cudaSetDevice(1);
  const double scale = enc_X[0].scale();

  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, 1));
  // std::cout << "nums of thread: " << nthreads << std::endl;

  vector<double> time(col_W, 0.0);

#pragma omp parallel num_threads(nthreads)
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();


    chrono::high_resolution_clock::time_point start, end;
    double elapsed;

#pragma omp for schedule(static)
    for (int i = 0; i < col_W; ++i)
    {

      // acc = enc_X[0] * W[0][i]
      PhantomPlaintext ecd_w_0_i;
      // encoder_local.encode(W[0][i], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
      encoder_local.encode(W[0][i], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);

      PhantomCiphertext acc;
      start = chrono::high_resolution_clock::now();
      evaluator_local.multiply_plain(enc_X[0], ecd_w_0_i, acc);
      end = chrono::high_resolution_clock::now();
      elapsed = duration_cast<duration<double>>(end - start).count();
      // cout << "[DEBUG] time: " << elapsed << endl;
      time[i] += elapsed;

      for (int j = 1; j < row_W; ++j)
      {
        PhantomPlaintext ecd_w_j_i;
        // encoder_local.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
        encoder_local.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
        // bridge_to_default(stream); // bridge to default stream after each encode

        PhantomCiphertext tmp;
        start = chrono::high_resolution_clock::now();
        evaluator_local.multiply_plain(enc_X[j], ecd_w_j_i, tmp);
        evaluator_local.add_inplace(acc, tmp);
        end = chrono::high_resolution_clock::now();

        elapsed = duration_cast<duration<double>>(end - start).count();

        time[i] += elapsed;
      }

      start = chrono::high_resolution_clock::now();
      evaluator_local.rescale_to_next_inplace(acc);
      acc.scale() = scale;

      end = chrono::high_resolution_clock::now();
      elapsed = duration_cast<duration<double>>(end - start).count();
      // cout << "[DEBUG] time: " << elapsed << endl;
      time[i] += elapsed;

      output[static_cast<size_t>(i)] = std::move(acc);
    }
  }

  // double total = 0.0;
  // for (double t : time){
  //   total += t;
  // }
  // cout << "ct-pt time(without encoding)" << total << " s" << endl;

  // cudaDeviceSynchronize();
  // stream_pool.clear();

  return output;
}

// vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre(vector<PhantomCiphertext> & enc_X,
//   vector<vector<double>> & W, int col_X, int col_W, int row_W,
//   PhantomContext& context){

//   // const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
//   // const auto &stream = stream_wrapper.get_stream();

//   vector<PhantomCiphertext> output(col_W);
//   double scale = enc_X[0].scale();

//   if(col_X != row_W){
//     cout <<"ERROR: bad dimensions of X or W. "<<endl;
//     return output;
//   }

//   PhantomCKKSEncoder phantom_encoder(context);
//   //pack Phantom to SEAL style
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//     // #pragma acc parallel loop
//     for (int i = 0; i < col_W; ++i){
//         // cout << "outer loop <<< " << i << endl;
//         // cout << "========================================================" << endl;
//         //encode w[0][i]
//         PhantomPlaintext ecd_w_0_i;
//         // std::chrono::_V2::system_clock::time_point start_encode = high_resolution_clock::now();
//         encoder.encode(W[0][i], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i); // chain index is default 1
//         // std::chrono::_V2::system_clock::time_point end_encode = high_resolution_clock::now();
//         // std::chrono::duration<double, std::milli> encode_duration = end_encode - start_encode;
//         // cout << "[DEBUG] Encoding duration for W[0][" << i << "]: " << encode_duration.count() << " ms" << endl;

//         // std::chrono::_V2::system_clock::time_point start_mul = high_resolution_clock::now();
//         //enc_X[0]*ecd_w[0][i]
//         evaluator.multiply_plain(enc_X[0], ecd_w_0_i, output[i]);
//         //evaluator.rescale_to_next_inplace(output[i]);
//         // std::chrono::_V2::system_clock::time_point end_mul = high_resolution_clock::now();
//         // std::chrono::duration<double, std::milli> mul_duration = end_mul - start_mul;
//         // cout << "[DEBUG] Multiplication duration for W[0][" << i << "]: " << mul_duration.count() << " ms" << endl;

//         for (int j = 1 ; j < row_W ; ++j){
//           // cout << "inter loop <<< " << j << endl;
//           // cout << "-----------------------------------------------------" << endl;
//           //encode w[j][i]
//           PhantomPlaintext ecd_w_j_i;

//           // std::chrono::_V2::system_clock::time_point start_encode = high_resolution_clock::now();
//           encoder.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
//           // std::chrono::_V2::system_clock::time_point end_encode = high_resolution_clock::now();
//           // std::chrono::duration<double, std::milli> encode_duration = end_encode - start_encode;
//           // cout << "[DEBUG] Encoding duration for W[" << j << "][" << i << "]: " << encode_duration.count() << " ms" << endl;

//           //enc_X[j]*ecd_w[j][i]
//           PhantomCiphertext temp;
//           // std::chrono::_V2::system_clock::time_point start_mul = high_resolution_clock::now();
//           evaluator.multiply_plain(enc_X[j], ecd_w_j_i, temp);
//           // std::chrono::_V2::system_clock::time_point end_mul = high_resolution_clock::now();
//           // std::chrono::duration<double, std::milli> mul_duration = end_mul - start_mul;
//           // cout << "[DEBUG] Multiplication duration for W[" << j << "][" << i << "]: " << mul_duration.count() << " ms" << endl;

//           //evaluator.rescale_to_next_inplace(temp);
//           // std::chrono::_V2::system_clock::time_point start_add = high_resolution_clock::now();
//           evaluator.add_inplace(output[i],temp);
//           // std::chrono::_V2::system_clock::time_point end_add = high_resolution_clock::now();
//           // std::chrono::duration<double, std::milli> add_duration = end_add - start_add;
//           // cout << "[DEBUG] Addition duration for W[" << j << "][" << i << "]: " << add_duration.count() << " ms" << endl;
//         }

//         evaluator.rescale_to_next_inplace(output[i]);
//         output[i].scale()=scale;
//     }

//   return output;

// }

// vector<PhantomCiphertext> ct_pt_matrix_mul(vector<PhantomCiphertext> &enc_X,
//                                            vector<vector<PhantomPlaintext>> &W, int col_X, int col_W, int row_W,
//                                            PhantomContext &context)
// {

//   vector<PhantomCiphertext> output(col_W);

//   if (col_X != row_W)
//   {
//     cout << "ERROR: bad dimensions of X or W. " << endl;
//     return output;
//   }

//   // CKKSEncoder encoder(seal_context);
//   // Evaluator evaluator(seal_context, encoder);
//   PhantomCKKSEncoder phantom_encoder(context);
//   // pack Phantom to SEAL style
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   // #pragma omp parallel for

//   for (int i = 0; i < col_W; ++i)
//   {

//     // encode w[0][i]
//     // Plaintext ecd_w_0_i;
//     // encoder.encode(W[0][i], scale, ecd_w_0_i);

//     // enc_X[0]*ecd_w[0][i]
//     evaluator.multiply_plain(enc_X[0], W[0][i], output[i]);
//     // evaluator.rescale_to_next_inplace(output[i]);

//     for (int j = 1; j < row_W; ++j)
//     {
//       // encode w[j][i]
//       // Plaintext ecd_w_j_i;
//       // encoder.encode(W[j][i], scale, ecd_w_j_i);

//       // enc_X[j]*ecd_w[j][i]
//       PhantomCiphertext temp;
//       evaluator.multiply_plain(enc_X[j], W[j][i], temp);
//       // evaluator.rescale_to_next_inplace(temp);
//       evaluator.add_inplace(output[i], temp);
//     }

//     evaluator.rescale_to_next_inplace(output[i]);
//   }

//   return output;
// }

// CUDA_API_PER_THREAD_DEFAULT_STREAM=1 for testing default stream isolation

vector<PhantomCiphertext> ct_pt_matrix_mul(vector<PhantomCiphertext> &enc_X,
                                           vector<vector<PhantomPlaintext>> &W, int col_X, int col_W, int row_W,
                                           PhantomContext &context) //明文矩阵预编码版本
{

  // vector<PhantomCiphertext> output(col_W);
  vector<PhantomCiphertext> output(static_cast<size_t>(col_W));

  if (col_X != row_W)
  {
    cout << "ERROR: bad dimensions of X or W. " << endl;
    return output;
  } //X的列维度不等于明文矩阵的行维度，就报错


  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, col_W)); 
  // // std::cout << "nums of thread: " << nthreads << std::endl;

  if (stream_pool.size() < static_cast<size_t>(nthreads))
  {
    stream_pool.reserve(nthreads);
    for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
    {
      stream_pool.emplace_back();
    }
  }
  if (nthreads == 1)
  {
    stream_pool[0] = *phantom::util::global_variables::default_stream;
  } //单线程就用默认流

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  cudaStream_t timing_stream = nullptr;
  cudaStreamCreateWithFlags(&timing_stream, cudaStreamNonBlocking); //计算时长
  // CKKSEncoder encoder(seal_context);
  // Evaluator evaluator(seal_context, encoder);
  // PhantomCKKSEncoder phantom_encoder(context);
  // pack Phantom to SEAL style
  // Encoder encoder(&context, &phantom_encoder);
  // Evaluator evaluator(&context, &phantom_encoder);

  // static std::mutex x_locks;
// #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context); //与之前的相比这里少了一个moai::encoder，也即不需要编码环节了。
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid];  //为每个线程指定CUDA流对象

    // std::vector<PhantomCiphertext> X_local(row_W);
    // for (int j = 0; j < row_W; ++j)
    //   X_local[j] = enc_X[j];
    std::vector<PhantomCiphertext> X_local(static_cast<size_t>(row_W));
    // enc_X may have pending writes on the default stream; order them before
    // the pool-stream copies below (missing bridge = race).
    bridge_from_default(stream);
    for (int j =  0; j < row_W; ++j)
    {
      X_local[static_cast<size_t>(j)] = deep_copy_cipher(enc_X[j], context, stream);
    } //每一个线程深拷贝一份矩阵的副本

    cudaStreamWaitEvent(stream.get_stream(), ev_start, 0); //当前线程的所有GPU运算必须等待事件ev_start被记录完毕再进行。

#pragma omp barrier
#pragma omp single
    {
      cudaEventRecord(ev_start, timing_stream ? timing_stream : 0);
    }

#pragma omp for schedule(static)
    for (int i = 0; i < col_W; ++i)
    {
      // PhantomCiphertext x0 = deep_copy_cipher(enc_X[0], context, stream);
      // PhantomPlaintext p0 = W[0][i];
      PhantomCiphertext acc, temp;
      // encode w[0][i]
      // Plaintext ecd_w_0_i;
      // encoder.encode(W[0][i], scale, ecd_w_0_i);

      // enc_X[0]*ecd_w[0][i]
      // std::lock_guard<std::mutex> g(x_locks);
      evaluator_local.multiply_plain(X_local[0], W[0][i], acc, stream);
      // evaluator_local.multiply_plain(x0, p0, acc, stream);
      // evaluator_local.multiply_plain(enc_X[0], W[0][i], acc, stream);
      // bridge_to_default(stream);
      // evaluator.rescale_to_next_inplace(output[i]);

      for (int j = 1; j < row_W; ++j)
      {
        // PhantomCiphertext xj = deep_copy_cipher(enc_X[j], context, stream);
        // PhantomPlaintext pj = W[j][i];
        // encode w[j][i]
        // Plaintext ecd_w_j_i;
        // encoder.encode(W[j][i], scale, ecd_w_j_i);

        // enc_X[j]*ecd_w[j][i]
        // PhantomCiphertext temp;
        // evaluator_local.multiply_plain(enc_X[j], W[j][i], temp, stream);
        // evaluator_local.multiply_plain(xj, pj, temp, stream);
        evaluator_local.multiply_plain(X_local[j], W[j][i], temp, stream);
        // evaluator.rescale_to_next_inplace(temp);
        evaluator_local.add_inplace(acc, temp, stream);
      }

      evaluator_local.rescale_to_next_inplace(acc, stream); //重缩放一下
      // bridge_to_default(stream);
      // cudaStreamSynchronize(stream.get_stream());
      output[static_cast<size_t>(i)] = std::move(acc);
    }
    cudaStreamSynchronize(stream.get_stream());
  }

  std::vector<cudaEvent_t> ev_done(nthreads);
  for (int i = 0; i < nthreads; ++i)
  {
    cudaEventCreateWithFlags(&ev_done[i], cudaEventDisableTiming);
    cudaEventRecord(ev_done[i], stream_pool[i].get_stream());
  }

  for (int i = 0; i < nthreads; ++i)
  {
    cudaStreamWaitEvent(timing_stream ? timing_stream : 0, ev_done[i], 0);
  }

  cudaEventRecord(ev_stop, timing_stream ? timing_stream : 0);
  cudaEventSynchronize(ev_stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop); //计算时间
  cout << "Ct-Pt compute time = " << ms << " ms\n";

  for (auto &e : ev_done)
    cudaEventDestroy(e);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop); //销毁
  if (timing_stream)
    cudaStreamDestroy(timing_stream);

  cudaDeviceSynchronize();
  return output;
}

vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_large(vector<PhantomCiphertext> &enc_X,
                                                        vector<vector<double>> &W, int col_X, int col_W, int row_W,
                                                        PhantomContext &context) //大列维度乘法，通过分组
{

  // const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  // const auto &stream = stream_wrapper.get_stream();

  vector<PhantomCiphertext> output(col_W);
  double scale = enc_X[0].scale();

  if (col_X != row_W)
  {
    cout << "ERROR: bad dimensions of X or W. " << endl;
    return output;
  }

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);

  int col_W_t = col_W / 128;  

  // #pragma omp parallel for
  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, 32));
  // std::cout << "nums of thread: " << nthreads << std::endl;

  if (stream_pool.size() < static_cast<size_t>(nthreads))
  {
    stream_pool.reserve(nthreads);
    for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
    {
      stream_pool.emplace_back();
    }
  }

  vector<double> time(128, 0.0);  //给128个分组每一个建一个计时
#pragma omp parallel num_threads(nthreads)
  {
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid];
    // phantom::util::cuda_stream_wrapper stream;
    // if (nthreads == 1){
    //   stream = *phantom::util::global_variables::default_stream;
    // } else {
    //   stream = stream_pool[tid];
    // }

    // Run the whole encode/multiply/add/rescale chain on this pool stream so
    // it is stream-ordered; order the shared default stream before it once
    // (enc_X may have pending default-stream writes from the caller).
    bridge_from_default(stream);

    chrono::high_resolution_clock::time_point start, end;
    double elapsed;

#pragma omp for schedule(static)
    for (int i = 0; i < 128; ++i)
    {
      for (int k = 0; k < col_W_t; ++k)
      {
        // encode w[0][i]
        PhantomPlaintext ecd_w_0_i;
        // encoder_local.encode(W[0][i * col_W_t + k], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
        encoder_local.encode(W[0][i * col_W_t + k], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i, stream);
        // copy enc_X[0] on THIS pool stream: multiply_plain's implicit copy
        // would be enqueued on enc_X's stored (default) stream and race with
        // the pool-stream multiply.
        output[i * col_W_t + k] = deep_copy_cipher(enc_X[0], context, stream);
        // enc_X[0]*ecd_w[0][i]
        start = chrono::high_resolution_clock::now();
        evaluator_local.multiply_plain_inplace(output[i * col_W_t + k], ecd_w_0_i, stream);
        end = chrono::high_resolution_clock::now();
        elapsed = duration_cast<duration<double>>(end - start).count();
        time[i] += elapsed;
        // evaluator.rescale_to_next_inplace(output[i]);

        for (int j = 1; j < row_W; ++j)
        {
          // encode w[j][i]
          PhantomPlaintext ecd_w_j_i;
          // encoder_local.encode(W[j][i * col_W_t + k], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
          encoder_local.encode(W[j][i * col_W_t + k], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i, stream);

          // enc_X[j]*ecd_w[j][i]
          PhantomCiphertext temp = deep_copy_cipher(enc_X[j], context, stream);
          start = chrono::high_resolution_clock::now();
          evaluator_local.multiply_plain_inplace(temp, ecd_w_j_i, stream);
          // if(i == 0)cout <<log2(temp.scale())<<" "<<log2(output[i*col_W_t+k].scale())<<endl;
          // evaluator.rescale_to_next_inplace(temp);
          evaluator_local.add_inplace(output[i * col_W_t + k], temp, stream);
          end = chrono::high_resolution_clock::now();
          elapsed = duration_cast<duration<double>>(end - start).count();
          time[i] += elapsed;
        }

        evaluator_local.rescale_to_next_inplace(output[i * col_W_t + k], stream);
        output[i * col_W_t + k].scale() = scale;
        // if(i == 0) cout <<log2(output[i*col_W_t+k].scale())<<endl;
      }
    }
    cudaStreamSynchronize(stream.get_stream());
  }

  // double total = 0.0;
  // for (double t : time){
  //   total += t;
  // }
  // cout << "ct-pt time(without encoding)" << total << " s" << endl;

  return output;
}

vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_large_single(vector<PhantomCiphertext> &enc_X,
                                                        vector<vector<double>> &W, int col_X, int col_W, int row_W,
                                                        PhantomContext &context, double &time_p) //大列维度单线程调用这个
{

  // const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  // const auto &stream = stream_wrapper.get_stream();

  vector<PhantomCiphertext> output(col_W);
  double scale = enc_X[0].scale();

  if (col_X != row_W)
  {
    cout << "ERROR: bad dimensions of X or W. " << endl;
    return output;
  }

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);

  // ceil, so widths that are not a multiple of 128 (e.g. 192) still cover
  // every output column; the guard in the loop skips the overrun slots.
  int col_W_t = (col_W + 127) / 128;

  // sm_120 debug: this path was already forced to a single OMP thread; the
  // parallel region and its per-region local encoder/evaluator are removed so
  // everything runs on the caller thread with the function-scope
  // encoder/evaluator (isolating a suspected race tied to the region locals).
  vector<double> time(128, 0.0);
  {
    chrono::high_resolution_clock::time_point start, end;
    double elapsed;

    for (int i = 0; i < 128; ++i)
    {
      for (int k = 0; k < col_W_t; ++k)
      {
        if (i * col_W_t + k >= col_W)
          break; // ceil(col_W/128) overshoots for non-multiple-of-128 widths
        // encode w[0][i]
        PhantomPlaintext ecd_w_0_i;
        encoder.encode(W[0][i * col_W_t + k], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
        // enc_X[0]*ecd_w[0][i]
        start = chrono::high_resolution_clock::now();
        evaluator.multiply_plain(enc_X[0], ecd_w_0_i, output[i * col_W_t + k]);
        end = chrono::high_resolution_clock::now();
        elapsed = duration_cast<duration<double>>(end - start).count();
        time[i] += elapsed;

        for (int j = 1; j < row_W; ++j)
        {
          // encode w[j][i]
          PhantomPlaintext ecd_w_j_i;
          encoder.encode(W[j][i * col_W_t + k], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);

          // enc_X[j]*ecd_w[j][i]
          PhantomCiphertext temp;
          start = chrono::high_resolution_clock::now();
          evaluator.multiply_plain(enc_X[j], ecd_w_j_i, temp);
          evaluator.add_inplace(output[i * col_W_t + k], temp);
          end = chrono::high_resolution_clock::now();
          elapsed = duration_cast<duration<double>>(end - start).count();
          time[i] += elapsed;
        }
        start = chrono::high_resolution_clock::now();
        evaluator.rescale_to_next_inplace(output[i * col_W_t + k]);
        output[i * col_W_t + k].scale() = scale;
        end = chrono::high_resolution_clock::now();
        elapsed = duration_cast<duration<double>>(end - start).count();
        time[i] += elapsed;
      }
    }
  }

  double total = 0.0;
  for (double t : time){
    total += t;
  }
  time_p = total;
  // cout << "ct-pt time(without encoding)" << total << " s" << endl;

  return output;
}

// vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_large(vector<PhantomCiphertext> &enc_X,
//                                                         vector<vector<double>> &W, int col_X, int col_W, int row_W,
//                                                         PhantomContext &context)
// {

//   const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
//   const auto &stream = stream_wrapper.get_stream();

//   vector<PhantomCiphertext> output(col_W);
//   double scale = enc_X[0].scale();

//   if (col_X != row_W)
//   {
//     cout << "ERROR: bad dimensions of X or W. " << endl;
//     return output;
//   }

//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   int col_W_t = col_W / 128;

//   // #pragma omp parallel for

//   for (int i = 0; i < 128; ++i)
//   {
//     for (int k = 0; k < col_W_t; ++k)
//     {
//       // encode w[0][i]
//       PhantomPlaintext ecd_w_0_i;
//       encoder.encode(W[0][i * col_W_t + k], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
//       // enc_X[0]*ecd_w[0][i]
//       evaluator.multiply_plain(enc_X[0], ecd_w_0_i, output[i * col_W_t + k]);
//       // evaluator.rescale_to_next_inplace(output[i]);

//       for (int j = 1; j < row_W; ++j)
//       {
//         // encode w[j][i]
//         PhantomPlaintext ecd_w_j_i;
//         encoder.encode(W[j][i * col_W_t + k], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);

//         // enc_X[j]*ecd_w[j][i]
//         PhantomCiphertext temp;
//         evaluator.multiply_plain(enc_X[j], ecd_w_j_i, temp);
//         // if(i == 0)cout <<log2(temp.scale())<<" "<<log2(output[i*col_W_t+k].scale())<<endl;
//         // evaluator.rescale_to_next_inplace(temp);
//         evaluator.add_inplace(output[i * col_W_t + k], temp);
//       }

//       evaluator.rescale_to_next_inplace(output[i * col_W_t + k]);
//       output[i * col_W_t + k].scale() = scale;
//       // if(i == 0) cout <<log2(output[i*col_W_t+k].scale())<<endl;
//     }
//   }

//   return output;
// }

vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_w_mask(vector<PhantomCiphertext> &enc_X,
                                                         vector<vector<double>> &W, const vector<int> &bias_vec, int col_X, int col_W, int row_W,
                                                         PhantomContext &context) //输入参数增加掩码，有时候场景需要部分特征不参与计算，或者说特征失效，这时候特征所对应的权重位置就可以取0，减小运算。
{

  // const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  // const auto &stream = stream_wrapper.get_stream();

  vector<PhantomCiphertext> output(col_W); //输出列维度个密文组成的向量
  double scale = enc_X[0].scale(); //缩放因子

  if (col_X != row_W)
  {
    cout << "ERROR: bad dimensions of X or W. " << endl;
    return output;
  }

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  size_t slot_count = phantom_encoder.slot_count(); //槽位数

  int col_W_t = col_W / 128;
  // cout <<col_W_t<<endl;

  // #pragma omp parallel for
  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, 32)); //线程数 
  // std::cout << "nums of thread: " << nthreads << std::endl;

  if (stream_pool.size() < static_cast<size_t>(nthreads))
  {
    stream_pool.reserve(nthreads);
    for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
    {
      stream_pool.emplace_back();
    }
  } //CUDA流池补充到和线程数一样

  vector<double> time(128, 0.0); //时间

#pragma omp parallel num_threads(nthreads) //开辟并行域
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local); //每个线程单独的加密、评估器
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num(); 
    auto &stream = stream_pool[tid]; //CUDA对象绑定对应的线程
    // phantom::util::cuda_stream_wrapper stream;
    // if (nthreads == 1){
    //   stream = *phantom::util::global_variables::default_stream;
    // } else {
    //   stream = stream_pool[tid];
    // }
    // Run the whole encode/multiply/add/rescale chain on this pool stream so
    // it is stream-ordered; order the shared default stream before it once
    // (enc_X may have pending default-stream writes from the caller).
    bridge_from_default(stream);

    chrono::high_resolution_clock::time_point start, end;
    double elapsed;

#pragma omp for schedule(static)
    for (int i = 0; i < 128; ++i)
    {
      for (int k = 0; k < col_W_t; ++k)
      {
        // encode w[0][i]
        vector<double> temp(slot_count, 0); //初始化全为0
        for (int j = 0; j < slot_count; ++j)
        {
          if (bias_vec[j] == 1)
          {
            temp[j] = W[0][i * col_W_t + k];
          }
        }
        PhantomPlaintext ecd_w_0_i;
        // encoder_local.encode(temp, enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
        encoder_local.encode(temp, enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i, stream);
        // copy enc_X[0] on THIS pool stream: multiply_plain's implicit copy
        // would be enqueued on enc_X's stored (default) stream and race with
        // the pool-stream multiply.
        output[i * col_W_t + k] = deep_copy_cipher(enc_X[0], context, stream);
        // enc_X[0]*ecd_w[0][i]
        start = chrono::high_resolution_clock::now();
        evaluator_local.multiply_plain_inplace(output[i * col_W_t + k], ecd_w_0_i, stream);
        end = chrono::high_resolution_clock::now();
        elapsed = duration_cast<duration<double>>(end - start).count();
        // cout << "[DEBUG] time: " << elapsed << endl;
        time[i] += elapsed;
        // evaluator.rescale_to_next_inplace(output[i]);
        // cout <<"mul 1."<<endl;

        for (int j = 1; j < row_W; ++j)
        {
          // cout <<j<<" ";
          // encode w[j][i]
          vector<double> tempw(slot_count, 0);
          for (int kk = 0; kk < slot_count; ++kk)
          {
            if (bias_vec[kk] == 1)
            {
              tempw[kk] = W[j][i * col_W_t + k];
            }
          }
          PhantomPlaintext ecd_w_j_i;
          // encoder_local.encode(tempw, enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
          encoder_local.encode(tempw, enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i, stream);
          // enc_X[j]*ecd_w[j][i]
          PhantomCiphertext tempx = deep_copy_cipher(enc_X[j], context, stream);
          start = chrono::high_resolution_clock::now();
          evaluator_local.multiply_plain_inplace(tempx, ecd_w_j_i, stream);

          // cout <<"mul. "<<endl;
          // evaluator.rescale_to_next_inplace(temp);
          evaluator_local.add_inplace(output[i * col_W_t + k], tempx, stream);
          end = chrono::high_resolution_clock::now();
          elapsed = duration_cast<duration<double>>(end - start).count();
          time[i] += elapsed;
          // cout <<"add. "<<endl;
        }

        evaluator_local.rescale_to_next_inplace(output[i * col_W_t + k], stream);
        output[i * col_W_t + k].scale() = scale;
      }
    }
    cudaStreamSynchronize(stream.get_stream());
  }
  // cout <<log(output[0].scale())<<endl;

  // double total = 0.0;
  // for (double t : time){
  //   total += t;
  // }
  // cout << "ct-pt time(without encoding)" << total << " s" << endl;

  return output;
}

vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_w_mask_single(vector<PhantomCiphertext> &enc_X,
                                                         vector<vector<double>> &W, const vector<int> &bias_vec, int col_X, int col_W, int row_W,
                                                         PhantomContext &context, double &time_p) //带掩码的单线程跟之前类似
{

  // const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  // const auto &stream = stream_wrapper.get_stream();

  vector<PhantomCiphertext> output(col_W);
  double scale = enc_X[0].scale();

  if (col_X != row_W)
  {
    cout << "ERROR: bad dimensions of X or W. " << endl;
    return output;
  }

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  size_t slot_count = phantom_encoder.slot_count();

  // ceil, so widths that are not a multiple of 128 (e.g. 192) still cover
  // every output column; the guard in the loop skips the overrun slots.
  int col_W_t = (col_W + 127) / 128;
  // cout <<col_W_t<<endl;

  // #pragma omp parallel for
  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, 1));
  // std::cout << "nums of thread: " << nthreads << std::endl;

  vector<double> time(128, 0.0);

#pragma omp parallel num_threads(nthreads)
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();

    chrono::high_resolution_clock::time_point start, end;
    double elapsed;

#pragma omp for schedule(static)
    for (int i = 0; i < 128; ++i)
    {
      for (int k = 0; k < col_W_t; ++k)
      {
        if (i * col_W_t + k >= col_W)
          break; // ceil(col_W/128) overshoots for non-multiple-of-128 widths
        // encode w[0][i]
        vector<double> temp(slot_count, 0);
        for (int j = 0; j < slot_count; ++j)
        {
          if (bias_vec[j] == 1)
          {
            temp[j] = W[0][i * col_W_t + k];
          }
        }
        PhantomPlaintext ecd_w_0_i;
        // encoder_local.encode(temp, enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
        encoder_local.encode(temp, enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
        // enc_X[0]*ecd_w[0][i]
        start = chrono::high_resolution_clock::now();
        evaluator_local.multiply_plain(enc_X[0], ecd_w_0_i, output[i * col_W_t + k]);
        end = chrono::high_resolution_clock::now();
        elapsed = duration_cast<duration<double>>(end - start).count();
        // cout << "[DEBUG] time: " << elapsed << endl;
        time[i] += elapsed;
        // evaluator.rescale_to_next_inplace(output[i]);
        // cout <<"mul 1."<<endl;

        for (int j = 1; j < row_W; ++j)
        {
          // cout <<j<<" ";
          // encode w[j][i]
          vector<double> tempw(slot_count, 0);
          for (int kk = 0; kk < slot_count; ++kk)
          {
            if (bias_vec[kk] == 1)
            {
              tempw[kk] = W[j][i * col_W_t + k];
            }
          }
          PhantomPlaintext ecd_w_j_i;
          // encoder_local.encode(tempw, enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
          encoder_local.encode(tempw, enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
          // enc_X[j]*ecd_w[j][i]
          PhantomCiphertext tempx;
          start = chrono::high_resolution_clock::now();
          evaluator_local.multiply_plain(enc_X[j], ecd_w_j_i, tempx);

          // cout <<"mul. "<<endl;
          // evaluator.rescale_to_next_inplace(temp);
          evaluator_local.add_inplace(output[i * col_W_t + k], tempx);
          end = chrono::high_resolution_clock::now();
          elapsed = duration_cast<duration<double>>(end - start).count();
          time[i] += elapsed;
          // cout <<"add. "<<endl;
        }
        start = chrono::high_resolution_clock::now();
        evaluator_local.rescale_to_next_inplace(output[i * col_W_t + k]);
        output[i * col_W_t + k].scale() = scale;
        end = chrono::high_resolution_clock::now();
        elapsed = duration_cast<duration<double>>(end - start).count();
        time[i] += elapsed;
      }
    }
  }
  // cout <<log(output[0].scale())<<endl;

  double total = 0.0;
  for (double t : time){
    total += t;
  }
  time_p = total;
  // cout << "ct-pt time(without encoding)" << total << " s" << endl;

  return output;
}


// vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_w_mask(vector<PhantomCiphertext> &enc_X,
//                                                          vector<vector<double>> &W, const vector<int> &bias_vec, int col_X, int col_W, int row_W,
//                                                          PhantomContext &context)
// {

//   const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
//   const auto &stream = stream_wrapper.get_stream();

//   vector<PhantomCiphertext> output(col_W);
//   double scale = enc_X[0].scale();

//   if (col_X != row_W)
//   {
//     cout << "ERROR: bad dimensions of X or W. " << endl;
//     return output;
//   }

//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);
//   size_t slot_count = phantom_encoder.slot_count();

//   int col_W_t = col_W / 128;
//   // cout <<col_W_t<<endl;

//   // #pragma omp parallel for

//   for (int i = 0; i < 128; ++i)
//   {
//     for (int k = 0; k < col_W_t; ++k)
//     {
//       // encode w[0][i]
//       vector<double> temp(slot_count, 0);
//       for (int j = 0; j < slot_count; ++j)
//       {
//         if (bias_vec[j] == 1)
//         {
//           temp[j] = W[0][i * col_W_t + k];
//         }
//       }
//       PhantomPlaintext ecd_w_0_i;
//       encoder.encode(temp, enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
//       // enc_X[0]*ecd_w[0][i]
//       evaluator.multiply_plain(enc_X[0], ecd_w_0_i, output[i * col_W_t + k]);
//       // evaluator.rescale_to_next_inplace(output[i]);
//       // cout <<"mul 1."<<endl;

//       for (int j = 1; j < row_W; ++j)
//       {
//         // cout <<j<<" ";
//         // encode w[j][i]
//         vector<double> tempw(slot_count, 0);
//         for (int kk = 0; kk < slot_count; ++kk)
//         {
//           if (bias_vec[kk] == 1)
//           {
//             tempw[kk] = W[j][i * col_W_t + k];
//           }
//         }
//         PhantomPlaintext ecd_w_j_i;
//         encoder.encode(tempw, enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);

//         // enc_X[j]*ecd_w[j][i]
//         PhantomCiphertext tempx;
//         evaluator.multiply_plain(enc_X[j], ecd_w_j_i, tempx);
//         // cout <<"mul. "<<endl;
//         // evaluator.rescale_to_next_inplace(temp);
//         evaluator.add_inplace(output[i * col_W_t + k], tempx);
//         // cout <<"add. "<<endl;
//       }

//       evaluator.rescale_to_next_inplace(output[i * col_W_t + k]);
//       output[i * col_W_t + k].scale() = scale;
//     }
//   }
//   // cout <<log(output[0].scale())<<endl;

//   return output;
// }

// vector<PhantomCiphertext> ct_pt_matrix_mul(const vector<PhantomCiphertext> & enc_X,
//   const vector<vector<PhantomPlaintext>> & W, int col_X, int col_W, int row_W,
//   PhantomContext& seal_context){

//   const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
//   const auto &stream = stream_wrapper.get_stream();
//   vector<PhantomCiphertext> output(col_W);

//   if(col_X != row_W){
//     cout <<"ERROR: bad dimensions of X or W. "<<endl;
//     return output;
//   }

//   PhantomCKKSEncoder phantom_encoder(seal_context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   // #pragma omp parallel for

//   for (int i = 0; i < col_W; ++i){

//     //encode w[0][i]
//     //Plaintext ecd_w_0_i;
//     //encoder.encode(W[0][i], scale, ecd_w_0_i);

//     //enc_X[0]*ecd_w[0][i]
//     evaluator.multiply_plain(enc_X[0], W[0][i], output[i]);
//     //evaluator.rescale_to_next_inplace(output[i]);

//     for (int j = 1 ; j < row_W ; ++j){
//       //encode w[j][i]
//      // Plaintext ecd_w_j_i;
//      // encoder.encode(W[j][i], scale, ecd_w_j_i);

//       //enc_X[j]*ecd_w[j][i]
//       PhantomCiphertext temp;
//       evaluator.multiply_plain(enc_X[j], W[j][i], temp);
//       //evaluator.rescale_to_next_inplace(temp);
//       evaluator.add_inplace(output[i],temp);
//     }

//     evaluator.rescale_to_next_inplace(output[i]);

//   }

//   return output;

// }