#include "CChanel_AWGN.h"


#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error (%d) at %s:%d\n", x, __FILE__,__LINE__);            \
      exit(0);}} while(0)

#define NORMALIZED_CHANNEL 	1 // REQUIERED FOR SPA DECODER
#define SEQ_LEVEL  			8

__global__ void vectNoise(const float *A, const float *B, float *C, float SigB, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        float x  = sqrt(-2.0 * log( A[i] ));
        float y  = B[i];
        float Ph = x * sin(_2pi * y);
        float Qu = x * cos(_2pi * y);
        C[i]     = -1.0 + Ph * SigB;
        C[i+N]   = -1.0 + Qu * SigB;
    }
}

__global__ void VectNoiseSigmaScaled(const unsigned int *A, const unsigned int *B, float *C, float SigB, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
    {
        float x  = sqrtf(-2.0 * logf( (double)(A[i] & 0x7FFFFFFF) / (double)2147483647.0 ));
        float y  = (double)(B[i] & 0x7FFFFFFF) / (double)2147483647.0;
        float Ph = x * sinf(_2pi * y);
        float Qu = x * cosf(_2pi * y);
        C[i]     = (-1.0 + Ph * SigB) * (2.0f / (1.0f * SigB * SigB));
        C[i+N]   = (-1.0 + Qu * SigB) * (2.0f / (1.0f * SigB * SigB));
    }
}

CChanel_AWGN::CChanel_AWGN(CTrame *t, int _BITS_LLR, bool QPSK, bool Es_N0)
    : CChanel(t, _BITS_LLR, QPSK, Es_N0){

	curandStatus_t Status;
	Status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	CURAND_CALL(Status);

    Status = curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
	CURAND_CALL(Status);
	size_t nb_data = ((size_t)_frames) * ((size_t)_data) / SEQ_LEVEL;
	CUDA_MALLOC_DEVICE(&device_A, nb_data/2,__FILE__, __LINE__);
    CUDA_MALLOC_DEVICE(&device_B, nb_data/2,__FILE__, __LINE__);
    CUDA_MALLOC_DEVICE(&device_R, nb_data  ,__FILE__, __LINE__);
}

CChanel_AWGN::~CChanel_AWGN(){
	cudaError_t Status;
	Status = cudaFree(device_A);
	ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	Status = cudaFree(device_B);
	ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	Status = cudaFree(device_R);
	ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	curandStatus_t eStatus;
    eStatus = curandDestroyGenerator(generator);
	CURAND_CALL(eStatus);
}

void CChanel_AWGN::configure(double _Eb_N0) {
/*
	rendement = (float) (_vars) / (float) (_data);
    if (es_n0) {
        Eb_N0 = _Eb_N0 - 10.0 * log10(2 * rendement);
    } else {
        Eb_N0 = _Eb_N0;
    }
    double interm = 10.0 * log10(rendement);
    interm        = -0.1*((double)Eb_N0+interm);
    SigB          = sqrt(pow(10.0,interm)/2);
*/
    rendement = (float) (_vars) / (float) (_data);
    if (es_n0) {
        Eb_N0 = _Eb_N0 - 10.0 * log10(2 * rendement);
    } else {
        Eb_N0 = _Eb_N0;
    }
    double interm = 10.0 * log10(rendement);
    interm        = -0.1 * ((double)Eb_N0+interm);
    SigB          = sqrt( pow( 10.0, interm ) / 2.0f);
}

#include <limits.h>
#define MAX_RANDOM LONG_MAX    /* Maximum value of random() */


double CChanel_AWGN::awgn(double amp)
{
    return 0.00;
}

#define QPSK 0.707106781
#define BPSK 1.0

#define COMPRESS_MEMORY		1

void CChanel_AWGN::generate()
{
#if NORMALIZED_CHANNEL == 0
		curandStatus_t Status;
		Status = curandGenerateUniform( generator, device_A, _frames*_data/2 );
		CURAND_CALL(Status);
		Status = curandGenerateUniform( generator, device_B, _frames*_data/2 );
		CURAND_CALL(Status);
		int nb_noise_sample = (_frames * _data);
		int threadsPerBlock = 1024;
		int blocksPerGrid   = (nb_noise_sample  + threadsPerBlock - 1) / threadsPerBlock;
		vectNoise<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_R, (float)SigB, nb_noise_sample/2);
		cudaError_t eStatus = cudaMemcpy(&t_noise_data[0], device_R, nb_noise_sample * sizeof(float), cudaMemcpyDeviceToHost);
#else

#if COMPRESS_MEMORY == 1
		size_t nb_data        = (_frames * _data); // data par run
		size_t data_per_round = nb_data / SEQ_LEVEL; // data par run
	for (int i = 0; i < SEQ_LEVEL; i++) {
		curandGenerate(generator, (unsigned int*) device_A, data_per_round / 2);
		curandGenerate(generator, (unsigned int*) device_B, data_per_round / 2);
		int threadsPerBlock = 1024;
		int blocksPerGrid = (data_per_round + threadsPerBlock - 1) / threadsPerBlock;
		VectNoiseSigmaScaled<<<blocksPerGrid, threadsPerBlock>>>((unsigned int*) device_A, (unsigned int*) device_B, device_R, (float) SigB, data_per_round / 2);
		cudaError_t eStatus = cudaMemcpy(&t_noise_data[i * data_per_round], device_R, data_per_round * sizeof(float), cudaMemcpyDeviceToHost);
	}
#else
	size_t nb_data = _frames * _data / 2;
	curandGenerate( generator, (unsigned int*)device_A, nb_data);
	curandGenerate( generator, (unsigned int*)device_B, nb_data);
	int nb_noise_sample = (_frames * _data);
	int threadsPerBlock = 1024;
	int blocksPerGrid = (nb_noise_sample + threadsPerBlock - 1) / threadsPerBlock;
	VectNoiseSigmaScaled<<<blocksPerGrid, threadsPerBlock>>>((unsigned int*)device_A, (unsigned int*)device_B, device_R, (float)SigB, nb_noise_sample/2);
	cudaError_t eStatus = cudaMemcpy(&t_noise_data[0], device_R, nb_noise_sample * sizeof(float), cudaMemcpyDeviceToHost);
#endif
#endif
}
