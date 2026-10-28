//模约减函数本身可以用arcsin(sin(2πx))来代替，但是在近似的时候由于cosx区间对称，且倍角公式至于cos相关，所以用arcsin(cos(2πx+π/2))来近似，
//但是为了缩小区间，就先计算cos(1/2^rπ(x+1/4))，然后通过倍角公式再返回来

#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "RemezArcsin.cuh"
#include "RemezCos.cuh"
#include "common/Polynomial.cuh"

using namespace std;

class ModularReducer {
 public:
  long boundary_K;  //雷米兹算法的逼近区间，超过这个区间会系数爆炸
  double log_width;  //区间长度的对数
  long deg;         //多项式阶数
  long num_double_formula;  //倍角公式迭代次数

  double inverse_log_width;  //倒数
  long inverse_deg;  //阶数的倒数

  double scale_inverse_coeff; //全局缩放因数

  CKKSEvaluator *ckks = nullptr; //底层 CKKS 运算器，提供密文加减乘、重线性化、缩放等基础同态接口。

  RemezParam rmparm; //Remez 算法全局配置：区间、阶数、迭代精度等，传给多项式生成器。

  RemezCos *poly_generator; //雷米兹算法生成cos逼近多项式
  RemezArcsin *inverse_poly_generator;  //雷米兹算法生成arcsin逼近多项式 
  //放在下面两个变量中

  boot::Polynomial sin_cos_polynomial;
  boot::Polynomial inverse_sin_polynomial;

  ModularReducer(long _boundary_K, double _log_width, long _deg, long _num_double_formula, long _inverse_deg, CKKSEvaluator *ckks);

  void double_angle_formula(PhantomCiphertext &cipher);  //压缩密文到雷米兹算法收敛的区间[-K,K]
  void double_angle_formula_scaled(PhantomCiphertext &cipher, double scale_coeff); 
  void generate_sin_cos_polynomial(); //生成极大极小多项式
  void generate_inverse_sine_polynomial(); //生成arcsin的极小多项式
  void write_polynomials(); //打印多项式
  void modular_reduction(PhantomCiphertext &rtn, PhantomCiphertext &cipher);  //核心对外接口，输入带噪声cipher，输出干净密文rtn
}; 
