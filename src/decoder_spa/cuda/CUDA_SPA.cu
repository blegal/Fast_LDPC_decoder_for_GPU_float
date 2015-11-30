/*
 * Copyright 2014 Bertrand LE GAL (bertrand.legal@ims-bordeaux.fr).  All rights reserved.
 *
 *  This file is part of LDPC_decoder_float.
 *
 *  LDPC_decoder_float is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  LDPC_decoder_float is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with LDPC_decoder_float.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "../../matrix/constantes_gpu.h"
#include "stdio.h"

#define signc(_X)	((__float_as_int(_X) & 0x80000000) ^ 0x80000000)	  // 1 => NEGATIF
#define fabsc(_X)	__int_as_float(__float_as_int(_X) & 0x7FFFFFFF)       // 1 => NEGATIF
#define invc(S,_X)	__int_as_float(__float_as_int(_X) ^ (S ^ 0x80000000)) // SI 1=> ALORS CHANGEMENT DE SIGNE
#define aSign(S,_X)	__int_as_float(__float_as_int(_X)  ^ S)       // 1 => NEGATIF

__global__ void LDPC_Sched_Stage_1_SPA(float var_nodes[_N],
		float var_mesgs[_M], unsigned int PosNoeudsVariable[_M],
   		unsigned int loops
		) {
	const int i  = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	const int ii =                            blockDim.x              * blockDim.y * gridDim.x; // A VERIFIER !!!!
	__shared__ unsigned int iTable[DEG_1];

	//
	// ON UTILISE UNE PETITE ASTUCE AFIN D'ACCELERER LA SIMULATION DU DECODEUR
	//
	loops -= 1;
	{
		float *p_msg1w                    = var_mesgs + i; // POINTEUR MESG_C_2_V (pour l'�criture)
		const unsigned int *p_indice_nod1 = PosNoeudsVariable;
		for (int z = 0; z <DEG_1_COMPUTATIONS; z++) {
			unsigned int sign_du_check  = 0;
			float BPSum        = 1.0f;
			float tab_vContr   [DEG_1]; // MESG_V_2_C
			float BPVAL        [DEG_1]; // MESG_V_2_C

			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_1;

			#pragma unroll
			for (int j = 0; j<DEG_1; j++) {
				int iAddr     = iTable[j] * ii + i;			            // Ieme INDEX (NODE INDICE)
				tab_vContr[j] = var_nodes[ iAddr ];
				BPVAL[j]      = tanhf( 0.5 * fabsf( tab_vContr[j] ) );
				BPSum        *= BPVAL[j];
				sign_du_check = sign_du_check ^ signc(tab_vContr[j]);
			}

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				float BPArg = BPSum / BPVAL[j];
				BPArg = (BPArg > 0.999999f) ? 0.999999f : BPArg;
				float tab_esultat  = 2.0f * atanhf(BPArg);
				int sign_msg       = sign_du_check ^ signc(tab_vContr[j]); //!sign_bit
				float msg_sortant  = invc(sign_msg, tab_esultat);
				*p_msg1w           = msg_sortant;
				p_msg1w           += ii;
				int iAddr          = iTable[j] * ii + i;			            // Ieme INDEX (NODE INDICE)
				var_nodes[ iAddr ] = tab_vContr[j] + msg_sortant;
			}
		}

#if NB_DEGRES > 1
		for (int z = 0; z <DEG_2_COMPUTATIONS; z++) {
			unsigned int sign_du_check  = 0;
			float BPSum        = 1.0f;
			float tab_vContr   [DEG_1]; // MESG_V_2_C
			float BPVAL        [DEG_1]; // MESG_V_2_C

			__syncthreads();
			if( threadIdx.x < DEG_2){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_2;

			#pragma unroll
			for (int j = 0; j<DEG_2; j++) {
				int iAddr     = iTable[j] * ii + i;			            // Ieme INDEX (NODE INDICE)
				tab_vContr[j] = var_nodes[ iAddr ];	    // CALCUL DE LA Ieme CONTRIBUTION

				BPVAL[j]      = tanhf( 0.5 * fabsf( tab_vContr[j] ) );
				BPSum        *= BPVAL[j];
				sign_du_check = sign_du_check ^ signc(tab_vContr[j]);
			}

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {
				float BPArg = BPSum / BPVAL[j];
				BPArg = (BPArg > 0.999999f) ? 0.999999f : BPArg;
				float tab_esultat = 2.0f * atanhf(BPArg);

				int sign_msg      = sign_du_check ^ signc(tab_vContr[j]); //!sign_bit
				float msg_sortant = invc(sign_msg, tab_esultat);
				*p_msg1w          = msg_sortant;
				p_msg1w          += ii;
				int iAddr     = iTable[j] * ii + i;			            // Ieme INDEX (NODE INDICE)
				var_nodes[ iAddr ] = tab_vContr[j] + msg_sortant;
			}
		}
#endif
	}

	while( loops-- ){
		float *p_msg1r                    = var_mesgs + i; // POINTEUR MESG_C_2_V (pour la lecture, z-1)
		float *p_msg1w                    = var_mesgs + i; // POINTEUR MESG_C_2_V (pour l'�criture)
		const unsigned int *p_indice_nod1 = PosNoeudsVariable;
		for (int z = 0; z <DEG_1_COMPUTATIONS; z++) {
			unsigned int sign_du_check  = 0;
			float BPSum        = 1.0f;
			float tab_vContr   [DEG_1]; // MESG_V_2_C
			float BPVAL        [DEG_1]; // MESG_V_2_C
			unsigned int iTable[DEG_1]; // LIEN ENTRELACEUR

			#pragma unroll
			for (int j = 0; j<DEG_1; j++) {
				iTable[j]     = (*p_indice_nod1++) * ii + i;			    // Ieme INDEX (NODE INDICE)
				tab_vContr[j] = var_nodes[ iTable[j] ] - (*p_msg1r);	    // CALCUL DE LA Ieme CONTRIBUTION
				p_msg1r      += ii;

				BPVAL[j]      = tanhf( 0.5 * fabsf( tab_vContr[j] ) );
				BPSum        *= BPVAL[j];
				sign_du_check = sign_du_check ^ signc(tab_vContr[j]);
			}

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				float BPArg = BPSum / BPVAL[j];
				BPArg = (BPArg > 0.999999f) ? 0.999999f : BPArg;
				float tab_esultat = 2.0f * atanhf(BPArg);

				int sign_msg      = sign_du_check ^ signc(tab_vContr[j]);
				float msg_sortant = invc(sign_msg, tab_esultat);
				*p_msg1w          = msg_sortant;
				p_msg1w          += ii;
				var_nodes[ iTable[j] ] = tab_vContr[j] + msg_sortant;
			}
		}
#if NB_DEGRES > 1
		for (int z = 0; z <DEG_2_COMPUTATIONS; z++) {
			unsigned int sign_du_check  = 0;
			float BPSum        = 1.0f;
			float tab_vContr   [DEG_1]; // MESG_V_2_C
			float BPVAL        [DEG_1]; // MESG_V_2_C
			unsigned int iTable[DEG_1]; // LIEN ENTRELACEUR

			#pragma unroll
			for (int j = 0; j<DEG_2; j++) {
				iTable[j]     = (*p_indice_nod1++) * ii + i;			    // Ieme INDEX (NODE INDICE)
				tab_vContr[j] = var_nodes[ iTable[j] ] - (*p_msg1r);	    // CALCUL DE LA Ieme CONTRIBUTION
				p_msg1r += ii;

				BPVAL[j]      = tanhf( 0.5 * fabsf( tab_vContr[j] ) );
				BPSum        *= BPVAL[j];
				sign_du_check = sign_du_check ^ signc(tab_vContr[j]);
			}

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {
				float BPArg = BPSum / BPVAL[j];
				BPArg = (BPArg > 0.999999f) ? 0.999999f : BPArg;
				float tab_esultat = 2.0f * atanhf(BPArg);

				int sign_msg = sign_du_check ^ signc(tab_vContr[j]); //!sign_bit
				float msg_sortant = invc(sign_msg, tab_esultat);
				*p_msg1w  = msg_sortant;
				p_msg1w  += ii;
				var_nodes[ iTable[j] ] = tab_vContr[j] + msg_sortant;
			}
		}
#endif
	}
}
