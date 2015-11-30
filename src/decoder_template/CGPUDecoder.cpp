/*
 *  ldcp_decoder.h
 *  ldpc3
 *
 *  Created by legal on 02/04/11.
 *  Copyright 2011 ENSEIRB. All rights reserved.
 *
 */

/*----------------------------------------------------------------------------*/

#include "CGPUDecoder.h"

CGPUDecoder::CGPUDecoder(int _nb_frames, int block_size, unsigned int n, unsigned int k, unsigned int m)
{
    cudaError_t Status;
	sz_nodes	= n * _nb_frames;
	sz_checks	= k * _nb_frames;
	sz_msgs		= m * _nb_frames;
	nb_frames   = _nb_frames;
    device_V    = NULL;
    d_MSG_C_2_V = NULL;
    BLOCK_SIZE	= block_size;

	size_t s = (2 * ((size_t)n) + ((size_t)m)) * sizeof(float);
	size_t o = ((2 * ((size_t)sz_nodes) + ((size_t)m) + ((size_t)sz_msgs))/1024)*4;

    // DONNEES D'ENTREE DU DECODEUR
    CUDA_MALLOC_DEVICE(&device_V,    sz_nodes, __FILE__, __LINE__);
    CUDA_MALLOC_DEVICE(&d_transpose, m,        __FILE__, __LINE__);
    CUDA_MALLOC_DEVICE(&d_MSG_C_2_V, sz_msgs,  __FILE__, __LINE__);

    Status = cudaMemcpy(d_transpose, PosNoeudsVariable, m * sizeof(unsigned int), cudaMemcpyHostToDevice);
    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

}


CGPUDecoder::~CGPUDecoder()
{
	cudaError_t Status;
	Status = cudaFree(device_V);
    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

	Status = cudaFree(d_transpose);
    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

    Status = cudaFree(d_MSG_C_2_V);
    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
}
