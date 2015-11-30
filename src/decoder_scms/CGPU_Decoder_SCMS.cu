/*
 *  ldcp_decoder.h
 *  ldpc3
 *
 *  Created by legal on 02/04/11.
 *  Copyright 2011 ENSEIRB. All rights reserved.
 *
 */

/*----------------------------------------------------------------------------*/

#include "CGPU_Decoder_SCMS.h"
#include "../transpose/GPU_Transpose.h"

CGPU_Decoder_SCMS::CGPU_Decoder_SCMS(int _nb_frames, int block_size, unsigned int n, unsigned int k, unsigned int m):
CGPUDecoder(_nb_frames, block_size, n, k, m)
{
	size_t nb_blocks = nb_frames / BLOCK_SIZE;
	printf("(II) Decoder configuration: BLOCK_SIZE = %ld, nb_frames = %ld, nb_blocks = %ld\n", BLOCK_SIZE, nb_frames, nb_blocks);

	struct cudaDeviceProp devProp;
  	cudaGetDeviceProperties(&devProp, 0);
  	struct cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, LDPC_Sched_Stage_1_SCMS);
  	int nMP      = devProp.multiProcessorCount; // NOMBRE DE STREAM PROCESSOR
  	int nWarp    = attr.maxThreadsPerBlock/32;  // PACKET DE THREADs EXECUTABLES EN PARALLELE
  	int nThreads = nWarp * 32;					// NOMBRE DE THREAD MAXI PAR SP
  	int nDOF     = nb_frames;
  	int nBperMP  = 65536 / (attr.numRegs); 	// Nr of blocks per MP
  	int minB     = min(nBperMP*nThreads,1024);
  	int nBlocks  = max(minB/nThreads * nMP, nDOF/nThreads);  //Total number of blocks
  	printf("(II) Nombre de Warp    : %d\n", nWarp);
  	printf("(II) Nombre de Threads           : %d\n", nThreads);
  	printf("(II) LDPC_Sched_Stage_1_SCMS (PTX version %d)\n", attr.ptxVersion);
  	printf("(II) - Nombre de regist/thr : %d\n", attr.numRegs);
  	printf("(II) - Nombre de local/thr  : %ld\n", attr.localSizeBytes);
    printf("(II) - Nombre de shared/thr : %ld\n", attr.sharedSizeBytes);
    printf("(II) - Nombre de pBLOCKs    : %f\n", (float)nb_frames / (float)BLOCK_SIZE);
    printf("(II) - Nombre de pBLOCKs/uP : %f\n", (float)nb_frames / (float)BLOCK_SIZE / (float)devProp.multiProcessorCount);
}


CGPU_Decoder_SCMS::~CGPU_Decoder_SCMS()
{

}

void CGPU_Decoder_SCMS::initialize()
{

}

void CGPU_Decoder_SCMS::decode(float Intrinsic_fix[_N], int Rprime_fix[_N], int nombre_iterations)
{
    cudaError_t Status;
	int nb_blocks = nb_frames / BLOCK_SIZE;
	Status = cudaMemcpy/*Async*/(d_MSG_C_2_V, Intrinsic_fix, sz_nodes * sizeof(float), cudaMemcpyHostToDevice);
    ERROR_CHECK(Status, __FILE__, __LINE__);

	{
    	// ORDERING THE LDPC CODEWORDS FOR DECODING (INTERLEAVING DATA)
		unsigned int NB_TRAMES    = _N;
		unsigned int FRAME_LENGTH = nb_frames;
		dim3 grid(NB_TRAMES/TILE_DIM, FRAME_LENGTH/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);
		transposeDiagonal<<<grid, threads>>>((float*)device_V, (float*)d_MSG_C_2_V, _N, nb_frames);
	}

//	printf("processing decode (%d, %d, %d) !\n", nb_blocks, BLOCK_SIZE, nombre_iterations);
	LDPC_Sched_Stage_1_SCMS<<<nb_blocks, BLOCK_SIZE>>>(device_V, d_MSG_C_2_V, d_transpose, nombre_iterations);

	{
		// REORDERING THE LDPC CODEWORDS
		unsigned int NB_TRAMES    = nb_frames;
		unsigned int FRAME_LENGTH = _N;
		dim3 grid(NB_TRAMES/TILE_DIM, FRAME_LENGTH/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);
		transposeDiagonal_and_hard_decision<<<grid, threads>>>((float*)d_MSG_C_2_V, (float*)device_V, NB_TRAMES, FRAME_LENGTH);
	}

    Status = cudaMemcpy(Rprime_fix, d_MSG_C_2_V, sz_nodes * sizeof(float), cudaMemcpyDeviceToHost);
	ERROR_CHECK(Status, __FILE__, __LINE__);
}
