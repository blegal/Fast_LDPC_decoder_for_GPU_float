/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

#include "../../matrix/constantes_gpu.h"

#include "stdio.h"

//#define OFFSET		0.00
#define OFFSET		0.15
#define signc(_X)	((_X<0.0)?0:1)		// 1 => NEGATIF
#define fabsc(_X)	((_X<0.0)?-_X:_X)	//
#define invc(S,_X)	((S==1)?_X:-_X)		// SI 1=> ALORS CHANGEMENT DE SIGNE


__global__ void __launch_bounds__(128, 2) LDPC_Sched_Stage_1_OMS(float var_nodes[_N],
		float var_mesgs[_M], unsigned int PosNoeudsVariable[_M],
   		unsigned int loops
		) {
	int i  = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int ii =                            blockDim.x              * blockDim.y * gridDim.x; // A VERIFIER !!!!
	__shared__ unsigned int iTable[DEG_1];
	float tab_vContr[DEG_1];

	//
	// ON UTILISE UNE PETITE ASTUCE AFIN D'ACCELERER LA SIMULATION DU DECODEUR
	//
	loops -= 1;
	{
		float        *p_msg1w       = var_mesgs + i; // POINTEUR MESG_C_2_V (pour l'�criture)
		unsigned int *p_indice_nod1 = PosNoeudsVariable;
		for (int z = 0; z <DEG_1_COMPUTATIONS; z++) {
			int sign_du_check  = 0;
			float min1 = 1000000.0;
			float min2 = 1000000.0;

			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_1;

			#pragma unroll
			for (int j = 0; j<DEG_1; j++) {
				int iAddr     = iTable[j] * ii + i;		// Ieme INDEX (NODE INDICE)
				tab_vContr[j] = var_nodes[ iAddr ];	    // CALCUL DE LA Ieme CONTRIBUTION
				min2          = fminf(min2, fmaxf(fabsf(tab_vContr[j]), min1));
				min1          = fminf(min1, fabsf(tab_vContr[j]));
				sign_du_check = sign_du_check ^ signc(tab_vContr[j]);
			}

			const float cste_1 = fmaxf(min2 - OFFSET, 0.00);
			const float cste_2 = fmaxf(min1 - OFFSET, 0.00);

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				float tab_esultat;
				if (fabsf(tab_vContr[j]) == min1) {
					tab_esultat = cste_1;
				} else {
					tab_esultat = cste_2;
				}
				int sign_msg = sign_du_check ^ signc(tab_vContr[j]);
				float msg_sortant= invc(sign_msg, tab_esultat);
				*p_msg1w  = msg_sortant;
				p_msg1w  += ii;
				var_nodes[ iTable[j] * ii + i ] = tab_vContr[j] + msg_sortant;
			}
		}
#if NB_DEGRES > 1
		for (int z = 0; z <DEG_2_COMPUTATIONS; z++) {
			int sign_du_check  = 0;
			float min1 = 1000000.0;
			float min2 = 1000000.0;

			__syncthreads();
			if( threadIdx.x < DEG_2){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_2;

			#pragma unroll
			for (int j = 0; j<DEG_2; j++) {
				int iAddr     = iTable[j] * ii + i;			    // Ieme INDEX (NODE INDICE)
				tab_vContr[j] = var_nodes[ iAddr ];	    // CALCUL DE LA Ieme CONTRIBUTION
				min2 = fminf(min2, fmaxf(fabsf(tab_vContr[j]), min1));
				min1 = fminf(min1, fabsf(tab_vContr[j]));
				sign_du_check = sign_du_check ^ signc(tab_vContr[j]);
			}

			float cste_1 = fmaxf(min2 - OFFSET, 0.00);
			float cste_2 = fmaxf(min1 - OFFSET, 0.00);

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {
				float tab_esultat;
				if (fabsf(tab_vContr[j]) == min1) {
					tab_esultat = cste_1;
				} else {
					tab_esultat = cste_2;
				}
				int sign_msg = sign_du_check ^ signc(tab_vContr[j]); //!sign_bit
				float msg_sortant = invc(sign_msg, tab_esultat);
				*p_msg1w  = msg_sortant;
				p_msg1w  += ii;
				var_nodes[ iTable[j] * ii + i ] = tab_vContr[j] + msg_sortant;
			}
		}
#endif
	}

	while( loops-- ){
		float *p_msg1r                    = var_mesgs + i;
		float *p_msg1w                    = var_mesgs + i;
		const unsigned int *p_indice_nod1 = PosNoeudsVariable;
		for (int z = 0; z <DEG_1_COMPUTATIONS; z++) {
			int sign_du_check  = 0;
			float min1 = 1000000.0;
			float min2 = 1000000.0;

			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			} __syncthreads();
			p_indice_nod1 += DEG_1;

			//
			// ON PREFETCH LES DONNEES (VN)
			//
			#pragma unroll
			for (int j = 0; j<DEG_1; j++) {
				int iAddr     = iTable[j] * ii + i;
				tab_vContr[j] = var_nodes[ iAddr ];
			}

			//
			// ON PREFETCH LES DONNEES (MSG)
			//
			float prefetch_msg[DEG_1];
			#pragma unroll
			for (int j = 0; j<DEG_1; j++) {
				prefetch_msg[j] = (*p_msg1r);
				p_msg1r      += ii;
			}

			//
			// ON CALCULE LES CONTRIBUTIONS
			//
			#pragma unroll
			for (int j = 0; j<DEG_1; j++) {
				tab_vContr[j] = tab_vContr[j] - prefetch_msg[j];
			}

			//
			// ON CALCULE LA VALEUR DE CN
			//
			#pragma unroll
			for (int j = 0; j<DEG_1; j++) {
				min2          = fminf(min2, fmaxf(fabsf(tab_vContr[j]), min1));
				min1          = fminf(min1, fabsf(tab_vContr[j]));
				sign_du_check = sign_du_check ^ signc(tab_vContr[j]);
			}
			float cste_1 = fmaxf(min2 - OFFSET, 0.00);
			float cste_2 = fmaxf(min1 - OFFSET, 0.00);
			float output_msg[DEG_1];
			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				float tab_esultat;
				if (fabsf(tab_vContr[j]) == min1) {
					tab_esultat = cste_1;
				} else {
					tab_esultat = cste_2;
				}
				int sign_msg  = sign_du_check ^ signc(tab_vContr[j]);
				output_msg[j] = invc(sign_msg, tab_esultat);
			}

			//
			// ON TRANSMET LES VALEURS DE VN/MSG
			//
			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				int iAddr          = iTable[j] * ii + i;
				var_nodes[ iAddr ] = tab_vContr[j] + output_msg[j];
				*p_msg1w           = output_msg[j];
				p_msg1w           += ii;
			}
		}
#if NB_DEGRES > 1
		for (int z = 0; z <DEG_2_COMPUTATIONS; z++) {
			int sign_du_check  = 0;
			float min1 = 1000000.0;
			float min2 = 1000000.0;

			__syncthreads();
			if( threadIdx.x < DEG_2){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_2;

			#pragma unroll
			for (int j = 0; j<DEG_2; j++) {
				int iAddr     = iTable[j] * ii + i;			            // Ieme INDEX (NODE INDICE)
				tab_vContr[j] = var_nodes[ iAddr ] - (*p_msg1r);	    // CALCUL DE LA Ieme CONTRIBUTION
				p_msg1r += ii;
				min2 = fminf(min2, fmaxf(fabsf(tab_vContr[j]), min1));
				min1 = fminf(min1, fabsf(tab_vContr[j]));
				sign_du_check = sign_du_check ^ signc(tab_vContr[j]);
			}

			const float cste_1 = fmaxf(min2 - OFFSET, 0.00);
			const float cste_2 = fmaxf(min1 - OFFSET, 0.00);

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {
				float tab_esultat;
				if (fabsf(tab_vContr[j]) == min1) {
					tab_esultat = cste_1;
				} else {
					tab_esultat = cste_2;
				}
				int sign_msg      = sign_du_check ^ signc(tab_vContr[j]); //!sign_bit
				float msg_sortant = invc(sign_msg, tab_esultat);
				*p_msg1w          = msg_sortant;
				p_msg1w  += ii;
				var_nodes[ iTable[j] * ii + i ] = tab_vContr[j] + msg_sortant;
			}
		}
#endif
	}
}
