#include "include.cuh"

using namespace std;
using namespace phantom;
using namespace moai;


void RMSNorm_test()
{
    cout << "Task: test the RMSNorm function in CKKS scheme. " << endl;
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40,
                                                                       40, 40, 40, 40, 40, 
                                                                       40, 40, 40, 40, 40, 
                                                                       40, 40, 40, 40, 40, 40, 40, 60}));

    // parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58})); // NEXUS params
    long sparse_slots = 32768;
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, 40);
    //先设置参数，再创建上下文
    PhantomContext context(parms);

    cout << "Set encryption parameters and print" << endl;
    print_parameters(context);
    // cout << "context.total_chain_index(): " << context.get_context_data(0).chain_depth() << endl;
    // KeyGenerator keygen(context);
    // SecretKey secret_key = keygen.secret_key();
    // PublicKey public_key;
    // keygen.create_public_key(public_key);
    // RelinKeys relin_keys;
    // keygen.create_relin_keys(relin_keys);

    PhantomSecretKey secret_key(context); 
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    //根据上下文生成密钥对和重线性化密钥

    // Encryptor encryptor(context, public_key);
    // Decryptor decryptor(context, secret_key);
    // CKKSEncoder encoder(context);
    Encryptor encryptor(&context, &public_key);
    Decryptor decryptor(&context, &secret_key); 
    //根据上下文创建解密器和加密器
    
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);
    //根据上下文、phantom_encoder创建编码器
    
    size_t slot_count = encoder.slot_count();
    //根据编码器获取槽槽数

    struct timeval tstart1, tend1; // 记录时间

    // construct input
    int num_X = 256;
    int num_row = 128;
    int num_col = 4096;
    //参数分别为输入矩阵个数、行维度数、列维度数

    cout << "Number of matrices in one batch = " << num_X << endl;
    vector<vector<vector<double>>> input_x(num_X, vector<vector<double>>(num_row, vector<double>(num_col, 0)));
    for (int i = 0; i < num_X; ++i)
    {
        for (int j = 0; j < num_row; ++j)
        {
            for (int k = 0; k < num_col; ++k)
            {
                input_x[i][j][k] = 0.001 * (double)k;
            }
            /*
            for (int k = 0; k < 10; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<"... ";
            for (int k = num_col-10 ; k<num_col ; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<endl;
            */
        }
    }
    cout << "Matrix X size = " << num_row << " * " << num_col << endl;
    //num_X个输入矩阵，每个矩阵的行维度数为num_row，列维度数为num_col

    // encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context, public_key);
    //输出列密文向量

    cout << "encode and encrypt X. " << endl;
    cout << "Modulus chain index for enc x: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;

    for (int i = 0; i < 5; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th input: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
    cout << "......" << endl;
    for (int i = num_col - 5; i < num_col; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th input: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
    //取出前五个还有后五个密文解密后的前五个和后五个槽内的元素并打印，用于检查加密是否正确


    vector<double> gamma(num_col, 0.5);

    // Real RMSNorm
    vector<vector<double>> real_rmsnorm(num_col, vector<double>(slot_count, 0));
    vector<double> sum_x_square(32768, 0);
    for (int i = 0; i < num_col; ++i)
    {
        for (int j = 0; j < 256; ++j)
        {
            for (int k = 0; k < 128; ++k)
            {
                sum_x_square[128 * j + k] += input_x[j][k][i] * input_x[j][k][i];
            }
        }
    }
    // for (int i = 0; i < 256; ++i)
    // {
    //     cout << "sum_x_square[" << i << "] = " << sum_x_square[i] << endl;
    // }

    for (int i = 0; i < num_col; ++i)
    {
        for (int j = 0; j < 256; ++j)
        {
            for (int k = 0; k < 128; ++k)
            {
                real_rmsnorm[i][128 * j + k] = input_x[j][k][i] / sqrt(sum_x_square[128 * j + k] / 4096.0 + 1e-6) * gamma[i];
            }
        }
    }
    for (int i = 0; i < 10; ++i)
    {
        cout << i + 1 << "-th real_rmsnorm:";
        for (int j = 0; j < 10; ++j)
        {
            cout << real_rmsnorm[i][j] << " ";
        }
        cout << endl;
    }

    gettimeofday(&tstart1, NULL);

    vector<PhantomCiphertext> output = RMSNorm(enc_ecd_x, gamma, context, relin_keys, secret_key);

    gettimeofday(&tend1, NULL);
    double RMSNorm_time = tend1.tv_sec - tstart1.tv_sec + (tend1.tv_usec - tstart1.tv_usec) / 1000000.0;
    cout << "RMSNorm time = " << RMSNorm_time << endl;

    cout << "Modulus chain index for RMSNorm: " << context.get_context_data(output[0].params_id()).chain_depth() << endl;
    append_csv_row("../results.csv", "RMSNorm", RMSNorm_time);
    // decrypt

    for (int i = 0; i < 10; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th ciphertext: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
    cout << "......" << endl;
    for (int i = num_col - 10; i < num_col; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th ciphertext: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
}