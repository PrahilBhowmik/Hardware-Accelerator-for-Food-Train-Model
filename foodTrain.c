#include <math.h>
#include <string.h>
#include "./include/k2c_include.h"
#include "./include/k2c_tensor_include.h"
#include "./include/k2c_activations.h"

size_t i,j;

void foodTrain(k2c_tensor *input_1_input, k2c_tensor *activation_output)
{

	float global_average_pooling2d_output_array[3] = {0};
	k2c_tensor global_average_pooling2d_output = {&global_average_pooling2d_output_array[0], 1, 3, {3, 1, 1, 1, 1}};

	float dense_output_array[101] = {0};
	k2c_tensor dense_output = {&dense_output_array[0], 1, 101, {101, 1, 1, 1, 1}};
	float dense_kernel_array[303] = {
		-1.87843233e-01f,
		-1.91770092e-01f,
		-3.80743816e-02f,
		+4.12032977e-02f,
		+3.44638862e-02f,
		-1.35835424e-01f,
		-2.23663345e-01f,
		-1.54605344e-01f,
		-5.51151372e-02f,
		-8.64779353e-02f,
		+1.37685882e-02f,
		-1.22755021e-02f,
		+5.07389382e-02f,
		-3.88275087e-02f,
		-6.42797947e-02f,
		-2.57585198e-01f,
		-1.24226004e-01f,
		-1.45238280e-01f,
		-1.02441184e-01f,
		+2.28404766e-03f,
		+6.83496818e-02f,
		-2.65808493e-01f,
		+7.01562017e-02f,
		-2.33644515e-01f,
		+1.65310167e-02f,
		+1.03426594e-02f,
		-1.08277909e-02f,
		+4.40208474e-04f,
		-9.41873193e-02f,
		-5.81581183e-02f,
		+4.60407920e-02f,
		+4.48579434e-03f,
		+3.21102291e-02f,
		-7.34007508e-02f,
		-2.66104490e-02f,
		-8.36794153e-02f,
		-7.73259923e-02f,
		+1.16135292e-02f,
		-2.01403033e-02f,
		-7.04157799e-02f,
		+7.08883703e-02f,
		+2.88932677e-02f,
		-6.07328899e-02f,
		-1.58356309e-01f,
		+1.00397039e-02f,
		+2.24528424e-02f,
		-2.04122290e-01f,
		-2.22518310e-01f,
		+1.17797796e-02f,
		+2.10004542e-02f,
		-5.70194460e-02f,
		+3.20323110e-02f,
		+3.73993292e-02f,
		-1.27062216e-01f,
		+4.82432842e-02f,
		-9.70672220e-02f,
		-1.72434747e-01f,
		-6.37026057e-02f,
		-1.96551066e-02f,
		-8.96757171e-02f,
		+3.63082066e-02f,
		-6.59011677e-02f,
		-2.95798201e-02f,
		-5.08528352e-02f,
		+1.60581544e-02f,
		-1.83299676e-01f,
		-1.17054174e-03f,
		-2.37493336e-01f,
		-7.13761896e-02f,
		-1.70412481e-01f,
		-1.64749905e-01f,
		+5.08840457e-02f,
		-9.01959166e-02f,
		-1.82298183e-01f,
		-1.55335814e-01f,
		-1.73572809e-01f,
		+2.57053059e-02f,
		-1.10273167e-01f,
		-7.80063272e-02f,
		+5.64621203e-02f,
		-3.35060060e-03f,
		+3.47420247e-03f,
		+5.50163873e-02f,
		-2.78497159e-01f,
		+7.58440793e-03f,
		-1.56475961e-01f,
		+3.03108562e-02f,
		+3.42821586e-03f,
		-5.92910871e-02f,
		+6.75372183e-02f,
		-2.50637233e-01f,
		+3.56465690e-02f,
		-1.50464639e-01f,
		+7.80406044e-05f,
		-1.44398093e-01f,
		+5.69554754e-02f,
		+7.35012889e-02f,
		+7.64143327e-03f,
		+2.18296889e-02f,
		+4.47822772e-02f,
		-6.88433051e-02f,
		-2.28947133e-01f,
		+3.34395468e-03f,
		+5.51223606e-02f,
		-1.23450756e-01f,
		-1.44355133e-01f,
		+6.34003878e-02f,
		-1.61620528e-01f,
		+5.57066463e-02f,
		+6.40738532e-02f,
		+6.54209778e-02f,
		-1.12469420e-01f,
		-2.09588468e-01f,
		-1.48568839e-01f,
		-1.09218303e-02f,
		+3.42822894e-02f,
		+1.44136161e-01f,
		+4.48055528e-02f,
		+1.17509753e-01f,
		-1.66613329e-02f,
		+5.03426464e-03f,
		-2.33345017e-01f,
		+4.62400205e-02f,
		-2.31327489e-01f,
		-3.98173928e-04f,
		-3.49042788e-02f,
		-1.47224478e-02f,
		+1.96266901e-02f,
		+2.26797396e-03f,
		-3.59588228e-02f,
		+4.98730503e-02f,
		-1.96517438e-01f,
		-3.82629363e-03f,
		-7.40198866e-02f,
		+1.06503731e-02f,
		-2.01174885e-01f,
		-1.21063717e-01f,
		-8.39997008e-02f,
		-1.61368828e-02f,
		+1.74103510e-02f,
		-7.88085461e-02f,
		-2.58565068e-01f,
		-8.28362703e-02f,
		-7.60861114e-02f,
		-3.09121162e-02f,
		-9.02837217e-02f,
		-5.59930205e-02f,
		+5.66986017e-02f,
		+6.83377460e-02f,
		-1.07923955e-01f,
		-4.58556674e-02f,
		-2.12131187e-01f,
		-7.68002644e-02f,
		-8.85022879e-02f,
		+9.65635553e-02f,
		-1.76372379e-01f,
		-1.25959784e-01f,
		-6.45679682e-02f,
		+4.79507037e-02f,
		+2.65059248e-02f,
		+6.35137409e-02f,
		-7.01370388e-02f,
		+7.86477923e-02f,
		+4.58939560e-02f,
		+2.29591993e-03f,
		-1.98789854e-02f,
		+5.70889600e-02f,
		+8.28968454e-03f,
		-1.03584811e-01f,
		-9.70731229e-02f,
		+4.52237800e-02f,
		-7.54457563e-02f,
		-1.54396012e-01f,
		+3.24769504e-02f,
		+1.41167045e-02f,
		+8.80300160e-03f,
		+1.55823911e-02f,
		-4.58798222e-02f,
		-1.43839419e-01f,
		+3.22704837e-02f,
		-1.46353602e-01f,
		-1.23506956e-01f,
		-4.52848636e-02f,
		-2.37667680e-01f,
		+5.46073578e-02f,
		-1.71471313e-01f,
		-2.29701445e-01f,
		-5.97609393e-02f,
		-5.61898714e-03f,
		-7.00161457e-02f,
		-2.10677460e-01f,
		-2.29450822e-01f,
		-1.35312438e-01f,
		+4.43096794e-02f,
		+6.44748798e-03f,
		+4.76491004e-02f,
		-2.40930811e-01f,
		-2.60922551e-01f,
		-9.27727297e-03f,
		-5.18013462e-02f,
		-2.66932011e-01f,
		-1.14895597e-01f,
		-7.58628249e-02f,
		-9.03800130e-02f,
		-1.63174570e-01f,
		-2.22792700e-02f,
		+3.61249335e-02f,
		+7.27530848e-03f,
		-3.16064656e-02f,
		-1.82045460e-01f,
		-1.25855610e-01f,
		-1.23770952e-01f,
		+4.49541546e-02f,
		+2.55333707e-02f,
		-5.07339984e-02f,
		-1.17869370e-01f,
		+3.42550804e-03f,
		-1.67522222e-01f,
		+2.11985465e-02f,
		-9.22046155e-02f,
		+6.56278506e-02f,
		-4.81818430e-02f,
		-9.10999905e-03f,
		-1.96937442e-01f,
		-2.85636424e-03f,
		-8.45131874e-02f,
		-5.88414781e-02f,
		-1.31852269e-01f,
		-8.98038521e-02f,
		-8.04285258e-02f,
		+9.16224197e-02f,
		-3.10200881e-02f,
		+2.99502537e-02f,
		-1.26449555e-01f,
		-9.36518759e-02f,
		-6.51332410e-03f,
		-8.88995826e-03f,
		+5.99248037e-02f,
		+1.01189099e-01f,
		-1.12158075e-01f,
		-1.56086646e-02f,
		-1.66364580e-01f,
		+2.79087145e-02f,
		-3.62415537e-02f,
		+8.70025679e-02f,
		-2.60723680e-02f,
		-1.23904236e-01f,
		-6.19159266e-02f,
		+2.36844588e-02f,
		-1.38188303e-01f,
		-1.00599818e-01f,
		-8.39795247e-02f,
		+6.86116070e-02f,
		-8.04444849e-02f,
		-7.19604269e-02f,
		-4.67331968e-02f,
		+1.09354677e-02f,
		-1.98382631e-01f,
		+4.20262404e-02f,
		+1.46337785e-03f,
		-6.31711930e-02f,
		-2.06141751e-02f,
		-1.07628241e-01f,
		-1.85967296e-01f,
		-1.57540306e-01f,
		+2.53191646e-02f,
		-1.23274609e-01f,
		-2.85047978e-01f,
		-1.36250272e-01f,
		-7.53507614e-02f,
		+4.67402935e-02f,
		-2.59999156e-01f,
		-8.73092115e-02f,
		-3.37680764e-02f,
		-2.40239233e-01f,
		-5.23730069e-02f,
		-4.63679060e-03f,
		-8.18349272e-02f,
		-3.03702541e-02f,
		-2.25928113e-01f,
		+1.59331486e-02f,
		-4.98569719e-02f,
		-6.19260967e-03f,
		+1.91302076e-02f,
		+3.43523324e-02f,
		-2.82724760e-02f,
		-1.13729134e-01f,
		-2.08560973e-01f,
		-1.20373242e-01f,
		-5.21715507e-02f,
		+9.15581584e-02f,
		-1.17085218e-01f,
		+1.76994994e-01f,
		+1.70618612e-02f,
		-2.88816780e-01f,
		-1.52954862e-01f,
		+2.35399120e-02f,
		+2.70289797e-02f,
		-2.50316709e-01f,
		-1.01516232e-01f,
		-3.19317542e-02f,
		+5.94455339e-02f,
		-2.11383045e-01f,
	};
	k2c_tensor dense_kernel = {&dense_kernel_array[0], 2, 303, {3, 101, 1, 1, 1}};
	float dense_bias_array[101] = {
		+0.00000000e+00f,
		+0.00000000e+00f,
		-1.94982097e-01f,
		-1.40707090e-01f,
		-1.47415563e-01f,
		-1.12791322e-01f,
		+0.00000000e+00f,
		-8.66719261e-02f,
		-6.74481392e-02f,
		+1.07404143e-01f,
		-5.78134619e-02f,
		-4.02333867e-03f,
		-1.86622098e-01f,
		-2.18347944e-02f,
		-4.07479741e-02f,
		-7.64924288e-02f,
		-1.24668486e-01f,
		-5.54619916e-02f,
		-3.48083898e-02f,
		-2.07308620e-01f,
		-9.97367948e-02f,
		-6.46368116e-02f,
		-1.24365382e-01f,
		+0.00000000e+00f,
		-6.86646327e-02f,
		-1.20919973e-01f,
		-1.84718177e-01f,
		-6.06359094e-02f,
		-9.15547907e-02f,
		-1.34210974e-01f,
		-9.86446217e-02f,
		-8.41588154e-02f,
		-2.10974753e-01f,
		-1.03337861e-05f,
		+0.00000000e+00f,
		-1.51463654e-02f,
		+7.43952068e-03f,
		-1.60910770e-01f,
		-5.70330843e-02f,
		+0.00000000e+00f,
		-1.20684579e-01f,
		-8.19465145e-02f,
		-8.65291804e-02f,
		+0.00000000e+00f,
		-1.99867063e-03f,
		-1.11379772e-01f,
		-7.45337009e-02f,
		-1.02601111e-01f,
		-8.88725117e-05f,
		-1.06121324e-01f,
		-1.66953206e-02f,
		-1.19611539e-01f,
		-1.87548339e-01f,
		-2.91087963e-02f,
		-9.34922993e-02f,
		+0.00000000e+00f,
		-6.87568026e-05f,
		-1.52223691e-01f,
		-1.37733206e-01f,
		-2.48169973e-02f,
		-1.34387210e-01f,
		-1.26714975e-01f,
		-1.32710382e-01f,
		-1.87121201e-02f,
		-6.32886291e-02f,
		-8.32013935e-02f,
		-1.79690555e-01f,
		+0.00000000e+00f,
		+0.00000000e+00f,
		-4.35407721e-02f,
		+0.00000000e+00f,
		+2.74672318e-04f,
		-2.30993107e-02f,
		+0.00000000e+00f,
		-2.18957663e-02f,
		-4.34563234e-02f,
		-7.79993385e-02f,
		+0.00000000e+00f,
		-5.95356487e-02f,
		-4.97592352e-02f,
		+0.00000000e+00f,
		+2.59862151e-02f,
		-1.20835774e-01f,
		-2.84871776e-02f,
		+0.00000000e+00f,
		+0.00000000e+00f,
		-1.66872412e-01f,
		-2.55907029e-02f,
		+4.91460674e-02f,
		-3.72092836e-02f,
		-2.28403769e-02f,
		-1.98251143e-01f,
		-4.51967455e-02f,
		-1.86710581e-01f,
		-1.32491589e-01f,
		-7.26447701e-02f,
		-1.08614288e-01f,
		-1.88859969e-01f,
		-2.05117062e-01f,
		-1.11412115e-01f,
		+0.00000000e+00f,
	};
	k2c_tensor dense_bias = {&dense_bias_array[0], 1, 101, {101, 1, 1, 1, 1}};
	float dense_fwork[306] = {0};

	k2c_global_avg_pooling(&global_average_pooling2d_output, input_1_input);
	k2c_dense(&dense_output, &global_average_pooling2d_output, &dense_kernel,
			  &dense_bias, k2c_relu, dense_fwork);
	k2c_softmax(dense_output.array, dense_output.numel);
	activation_output->ndim = dense_output.ndim; // copy data into output struct
	activation_output->numel = dense_output.numel;
	// memcpy(activation_output->shape, dense_output.shape, K2C_MAX_NDIM * sizeof(size_t));
	// memcpy(activation_output->array, dense_output.array, activation_output->numel * sizeof(activation_output->array[0]));

	//Copy by loop
	for(i=0;i<K2C_MAX_NDIM;i++){
		activation_output->shape[i] = dense_output.shape[i];
	}
	for(i=0;i<activation_output->numel;i++){
		activation_output->array[i] = dense_output.array[i];
	}
}

void foodTrain_initialize()
{
}

void foodTrain_terminate()
{
}

void k2c_global_avg_pooling(k2c_tensor* output, const k2c_tensor* input) {

    const size_t in_chan = input->shape[input->ndim-1];
    // memset(output->array,0,output->numel*sizeof(input->array[0]));
    // Initaialization by loop
    for(i=0;i<output->numel;i++){
        output->array[i]=0;
    }
    const float num_inv = 1.0f/(input->numel/in_chan);

    for (i=0; i<input->numel; i+=in_chan) {
        for (size_t j=0; j<in_chan; ++j) {
            output->array[j] += input->array[i+j]*num_inv;
        }
    }
}

/**
 * Dense (fully connected) Layer.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param kernel: kernel tensor.
 * :param bias: bias tensor.
 * :param activation: activation function to apply to output.
 * :param fwork: array of working space, size(fwork) = size(input) + size(kernel)
 */
void k2c_dense(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
               const k2c_tensor* bias, k2c_activationType *activation, float * fwork) {

    if (input->ndim <=2) {
        size_t outrows;

        if (input->ndim>1) {
            outrows = input->shape[0];
        }
        else {
            outrows = 1;
        }
        const size_t outcols = kernel->shape[1];
        const size_t innerdim = kernel->shape[0];
        const size_t outsize = outrows*outcols;
        k2c_affine_matmul(output->array,input->array,kernel->array,bias->array,
                          outrows,outcols,innerdim);
        activation(output->array,outsize);
    }
    else {
        const size_t axesA[1] = {input->ndim-1};
        const size_t axesB[1] = {0};
        const size_t naxes = 1;
        const int normalize = 0;

        k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork);
        k2c_bias_add(output, bias);
        activation(output->array, output->numel);
    }
}

/**
 * Just your basic 1d matrix multipication.
 * computes C = A*B
 * assumes A,B,C are all 1d arrays of matrices stored in row major order.
 *
 * :param C: output array.
 * :param A: input array 1.
 * :param B: input array 2.
 * :param outrows: number of rows of C and A.
 * :param outcols: number of cols of C and B.
 * :param innderdim: number of cols of A and rows of B
 */
void k2c_matmul(float * C, const float * A, const float * B, const size_t outrows,
                const size_t outcols, const size_t innerdim) {

    // // make sure output is empty
    // memset(C, 0, outrows*outcols*sizeof(C[0]));

    // Initaialization by loop
    for(i=0;i<outrows*outcols;i++){
        C[i]=0;
    }

    for (i=0 ; i < outrows; ++i) {
        const size_t outrowidx = i*outcols;
        const size_t inneridx = i*innerdim;
        for (size_t k = 0; k < innerdim; ++k) {
            for (size_t j = 0;  j < outcols; ++j) {
                C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
            }
        }
    }
}


/**
 * Affine matrix multiplication.
 * computes C = A*B + d, where d is a vector that is added to each
 row of A*B
 * assumes A,B,C are all 1d arrays of matrices stored in row major order
 *
 * :param C: output array.
 * :param A: input array 1.
 * :param B: input array 2.
 * :param d: input array 3.
 * :param outrows: number of rows of C and A.
 * :param outcols: number of cols of C, B and d.
 * :param innderdim: number of cols of A and rows of B
 */
void k2c_affine_matmul(float * C, const float * A, const float * B, const float * d,
                       const size_t outrows,const size_t outcols, const size_t innerdim) {

    // make sure output is empty
    // memset(C, 0, outrows*outcols*sizeof(C[0]));

    // Initaialization by loop
    for(i=0;i<outrows*outcols;i++){
        C[i]=0;
    }

    for (i=0 ; i < outrows; ++i) {
        const size_t outrowidx = i*outcols;
        const size_t inneridx = i*innerdim;
        for (size_t j = 0;  j < outcols; ++j) {
            for (size_t k = 0; k < innerdim; ++k) {
                C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
            }
            C[outrowidx+j] += d[j];
        }
    }
}


/**
 * Converts subscripts to linear indices in row major order.
 *
 * :param sub: array[ndim] subscript to convert.
 * :param shape: array[ndim] shape of array being indexed.
 * :param ndim: number of dimensions of array being indexed.
 * :return: linear index in row major order.
 */
size_t k2c_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim) {

    size_t idx = 0;
    size_t temp = 0;
    for (i=0; i<ndim; ++i) {
        temp = sub[i];
        for (size_t j=ndim-1; j>i; --j) {
            temp *= shape[j];
        }
        idx += temp;
    }
    return idx;
}


/**
 * Converts linear indices to subscripts in row major order.
 *
 * :param idx: linear index in row major order.
 * :param sub: array[ndim] output subscript.
 * :param shape: array[ndim] shape of array being indexed.
 * :param ndim: number of dimensions of array being indexed.
 */
void k2c_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim) {

    size_t idx2 = idx;
    for (int i=ndim-1; i>=0; --i) {
        sub[i] = idx2%shape[i];
        idx2 /= shape[i];
    }
}


/**
 * Dot product (tensor contraction) between 2 tensors. C=A*B
 *
 * :param C: output tensor.
 * :param A: input tensor 1.
 * :param B: input tensor 2.
 * :param axesA: array[naxes] of axes of A being contracted.
 * :param axesB: array[naxes] of axes of B being contracted.
 * :param naxes: number of axes being contracted from each input.
 * :param normalize: (0,1) whether to L2-normalize samples along the dot product axis before taking the dot product. If set to 1, then the output of the dot product is the cosine proximity between the two samples.
 * :param fwork: array of working space, size(fwork) = size(A) + size(B)
 */
void k2c_dot(k2c_tensor* C, const k2c_tensor* A, const k2c_tensor* B, const size_t * axesA,
             const size_t * axesB, const size_t naxes, const int normalize, float * fwork) {

    size_t permA[K2C_MAX_NDIM];
    size_t permB[K2C_MAX_NDIM];
    size_t prod_axesA = 1;
    size_t prod_axesB = 1;
    size_t free_axesA, free_axesB;
    size_t freeA[K2C_MAX_NDIM];
    size_t freeB[K2C_MAX_NDIM];
    size_t count;
    int isin;
    size_t newshpA[K2C_MAX_NDIM];
    size_t newshpB[K2C_MAX_NDIM];
    const size_t ndimA = A->ndim;
    const size_t ndimB = B->ndim;
    float *reshapeA = &fwork[0];   // temp working storage
    float *reshapeB = &fwork[A->numel];
    size_t Asub[K2C_MAX_NDIM];
    size_t Bsub[K2C_MAX_NDIM];
    // find which axes are free (ie, not being summed over)
    count=0;
    for (i=0; i<ndimA; ++i) {
        isin = 0;
        for (j=0; j<naxes; ++j) {
            if (i==axesA[j]) {
                isin=1;
            }
        }
        if (!isin) {
            freeA[count] = i;
            ++count;
        }
    }
    count=0;
    for (i=0; i<ndimB; ++i) {
        isin = 0;
        for (j=0; j<naxes; ++j) {
            if (i==axesB[j]) {
                isin=1;
            }
        }
        if (!isin) {
            freeB[count] = i;
            ++count;
        }
    }

    // number of elements in inner dimension
    for (i=0; i < naxes; ++i) {
        prod_axesA *= A->shape[axesA[i]];
    }
    for (i=0; i < naxes; ++i) {
        prod_axesB *= B->shape[axesB[i]];
    }
    // number of elements in free dimension
    free_axesA = A->numel/prod_axesA;
    free_axesB = B->numel/prod_axesB;
    // find permutation of axes to get into matmul shape
    for (i=0; i<ndimA-naxes; ++i) {
        permA[i] = freeA[i];
    }
    for (i=ndimA-naxes, j=0; i<ndimA; ++i, ++j) {
        permA[i] = axesA[j];
    }
    for (i=0; i<naxes; ++i) {
        permB[i] = axesB[i];
    }
    for (i=naxes, j=0; i<ndimB; ++i, ++j) {
        permB[i] = freeB[j];
    }



    for (i=0; i<ndimA; ++i) {
        newshpA[i] = A->shape[permA[i]];
    }
    for (i=0; i<ndimB; ++i) {
        newshpB[i] = B->shape[permB[i]];
    }

    // reshape arrays
    for (i=0; i<A->numel; ++i) {
        k2c_idx2sub(i,Asub,A->shape,ndimA);
        for (j=0; j<ndimA; ++j) {
            Bsub[j] = Asub[permA[j]];
        }
        size_t bidx = k2c_sub2idx(Bsub,newshpA,ndimA);
        reshapeA[bidx] = A->array[i];
    }

    for (i=0; i<B->numel; ++i) {
        k2c_idx2sub(i,Bsub,B->shape,ndimB);
        for (j=0; j<ndimB; ++j) {
            Asub[j] = Bsub[permB[j]];
        }
        size_t bidx = k2c_sub2idx(Asub,newshpB,ndimB);
        reshapeB[bidx] = B->array[i];
    }


    if (normalize) {

        float sum;
        float inorm;
        for (i=0; i<free_axesA; ++i) {
            sum = 0;
            for (j=0; j<prod_axesA; ++j) {
                sum += reshapeA[i*prod_axesA + j]*reshapeA[i*prod_axesA + j];
            }
            inorm = 1.0f/sqrtf(sum);
            for (j=0; j<prod_axesA; ++j) {
                reshapeA[i*prod_axesA + j] *= inorm;
            }
        }
        for (i=0; i<free_axesB; ++i) {
            sum = 0;
            for (j=0; j<prod_axesB; ++j) {
                sum += reshapeB[i + free_axesB*j]*reshapeB[i + free_axesB*j];
            }
            inorm = 1.0f/sqrtf(sum);
            for (j=0; j<prod_axesB; ++j) {
                reshapeB[i + free_axesB*j] *= inorm;
            }
        }
    }

    k2c_matmul(C->array, reshapeA, reshapeB, free_axesA,
               free_axesB, prod_axesA);
}


/**
 * Adds bias vector b to tensor A.
 * assumes b is a rank 1 tensor that is added to the last dimension of A.
 *
 * :param A: input tensor. Overwritten with outputs.
 * :param b: bias tensor.
 */
void k2c_bias_add(k2c_tensor* A, const k2c_tensor* b) {

    for (i=0; i<A->numel; i+=b->numel) {
        for (j=0; j<b->numel; ++j) {
            A->array[i+j] += b->array[j];
        }
    }
}