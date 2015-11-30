// Includes
#include  <stdio.h>
#include  <stdlib.h>
#include  <iostream>
#include  <cstring>
#include  <math.h>
#include  <time.h>
#include  <string.h>
#include  <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>

using namespace std;

#define pi  3.1415926536

#include "./timer/CTimer.h"
#include "./trame/CTrame.h"
#include "./ber_analyzer/CErrorAnalyzer.h"
#include "./terminal/CTerminal.h"
#include "./matrix/constantes_gpu.h"
#include "./awgn_channel/CChanel_AWGN.h"

#include "./decoder_spa/CGPU_Decoder_SPA.h"
#include "./decoder_ms/CGPU_Decoder_MS.h"
#include "./decoder_oms/CGPU_Decoder_OMS.h"
#include "./decoder_nms/CGPU_Decoder_NMS.h"
#include "./decoder_2nms/CGPU_Decoder_2NMS.h"
#include "./decoder_scms/CGPU_Decoder_SCMS.h"


int    QUICK_STOP           =  true;
bool   BER_SIMULATION_LIMIT =  false;
double BIT_ERROR_LIMIT      =  1e-7;

////////////////////////////////////////////////////////////////////////////////////

double rendement;
double Eb_N0;

////////////////////////////////////////////////////////////////////////////////////

void show_info(){
	struct cudaDeviceProp devProp;
  	cudaGetDeviceProperties(&devProp, 0);
//  	printf("(II) Identifiant du GPU (CUDA)    : %s\n", devProp.name);
  	printf("(II) Nombre de Multi-Processor    : %d\n", devProp.multiProcessorCount);
  	printf("(II) + totalGlobalMem             : %ld Mo\n", (devProp.totalGlobalMem/1024/1024));
  	printf("(II) + sharedMemPerBlock          : %ld Ko\n", (devProp.sharedMemPerBlock/1024));
#ifdef CUDA_6
  	printf("(II) + sharedMemPerMultiprocessor : %ld Ko\n", (devProp.sharedMemPerMultiprocessor/1024));
  	printf("(II) + regsPerMultiprocessor      : %ld\n", devProp.regsPerMultiprocessor);
#endif
  	printf("(II) + regsPerBlock               : %d\n", (int)devProp.regsPerBlock);
  	printf("(II) + warpSize                   : %d\n", (int)devProp.warpSize);
  	printf("(II) + memoryBusWidth             : %d\n", (int)devProp.memoryBusWidth);
  	printf("(II) + memoryClockRate            : %d\n", (int)devProp.memoryClockRate);

  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_SPA);
  	  	printf("(II) LDPC_Sched_Stage_1_SPA  (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_MS);
  	  	printf("(II) LDPC_Sched_Stage_1_MS   (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_OMS);
  	  	printf("(II) LDPC_Sched_Stage_1_OMS  (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_NMS);
  	  	printf("(II) LDPC_Sched_Stage_1_NMS  (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_2NMS);
  	  	printf("(II) LDPC_Sched_Stage_1_2NMS (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_SCMS);
  	  	printf("(II) LDPC_Sched_Stage_1_SCMS (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
	fflush(stdout);
}


int main(int argc, char* argv[])
{
	int p;
    srand( 0 );
	printf("(II) LDPC GPU-based DECODER - Horizontal layered scheduling\n");
	printf("(II) DATA FORMAT (floating point data 32b, IEEE-754)\n");
	printf("(II) GENERATED : %s - %s\n", __DATE__, __TIME__);
	struct cudaDeviceProp devProp;
  	cudaGetDeviceProperties(&devProp, 0);
  	printf("(II) Identifiant du GPU (CUDA)    : %s\n", devProp.name);
	//show_info();

	double MinSignalSurBruit     = 0.50;
	double MaxSignalSurBruit     = 0.51;
	double PasSignalSurBruit     = 0.10;
    int    NOMBRE_ITERATIONS     = 20;
	int    STOP_TIMER_SECOND     = -1;
	bool   QPSK_CHANNEL          = false;
    bool   Es_N0                 = false;
	int    FRAME_ERROR_LIMIT     =  200;
	string dec_type    			 = "MS";

    int    NB_FRAMES_IN_PARALLEL = 128;
    int    THREAD_PER_BLOCK      = 128;

	if( sizeof(int) != 4 ){
		printf("(EE) Error, sizeof(int) = %d != 4\n", sizeof(int));
		exit( 0 );
	}
	if( sizeof(float) != 4 ){
		printf("(EE) Error, sizeof(float) = %d != 4\n", sizeof(float));
		exit( 0 );
	}

    cudaSetDevice        (0);
    cudaDeviceReset      ( );
    cudaThreadSynchronize( );

	//
	// ON VA PARSER LES ARGUMENTS DE LIGNE DE COMMANDE
	//
	for (p=1; p<argc; p++) {
		if( strcmp(argv[p], "-min") == 0 ){
			MinSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-timer") == 0 ){
			STOP_TIMER_SECOND = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-random") == 0 ){
            srand( time(NULL) );

		}else if( strcmp(argv[p], "-max") == 0 ){
			MaxSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-pas") == 0 ){
			PasSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-stop") == 0 ){
			QUICK_STOP = 1;

		}else if( strcmp(argv[p], "-iter") == 0 ){
			NOMBRE_ITERATIONS = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-fer") == 0 ){
			FRAME_ERROR_LIMIT = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-qef") == 0 ){
			BER_SIMULATION_LIMIT =  true;
			BIT_ERROR_LIMIT      = ( atof( argv[p+1] ) );
			p += 1;

		}else if( strcmp(argv[p], "-bpsk") == 0 ){
			QPSK_CHANNEL = false;

		}else if( strcmp(argv[p], "-qpsk") == 0 ){
			QPSK_CHANNEL = true;

		}else if( strcmp(argv[p], "-Eb/N0") == 0 ){
			Es_N0 = false;

		}else if( strcmp(argv[p], "-Es/N0") == 0 ){
			Es_N0 = true;

		}else if( strcmp(argv[p], "-SPA") == 0 ){
			dec_type = "SPA";

		}else if( strcmp(argv[p], "-MS") == 0 ){
			dec_type = "MS";

		}else if( strcmp(argv[p], "-OMS") == 0 ){
			dec_type = "OMS";

		}else if( strcmp(argv[p], "-NMS") == 0 ){
			dec_type = "NMS";

		}else if( strcmp(argv[p], "-2NMS") == 0 ){
			dec_type = "2NMS";

		}else if( strcmp(argv[p], "-SCMS") == 0 ){
			dec_type = "SCMS";

		}else if( strcmp(argv[p], "-n") == 0 ){
			NB_FRAMES_IN_PARALLEL = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-b") == 0 ){
			THREAD_PER_BLOCK = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-info") == 0 ){
			show_info();
			exit( 0 );

		}else{
			printf("(EE) Unknown argument (%d) => [%s]\n", p, argv[p]);
			exit(0);
		}
	}

	if( NB_FRAMES_IN_PARALLEL%THREAD_PER_BLOCK != 0 ){
		printf("(EE) NB_THREADS and THREAD_PER_BLOCK are incompatibles !\n");
		exit( 0 );
	}

	if( THREAD_PER_BLOCK%32 != 0 ){
		printf("(EE) Bad THREAD_PER_BLOCK value !\n");
		exit( 0 );
	}

	rendement = (float)(NmoinsK)/(float)(_N);
	printf("(II) Code LDPC (N, K)     : (%d,%d)\n", _N, _K);
	printf("(II) Rendement du code    : %.3f\n", rendement);
	printf("(II) # ITERATIONs du CODE : %d\n", NOMBRE_ITERATIONS);
    printf("(II) FER LIMIT FOR SIMU   : %d\n", FRAME_ERROR_LIMIT);
	printf("(II) SIMULATION  RANGE    : [%.2f, %.2f], STEP = %.2f\n", MinSignalSurBruit,  MaxSignalSurBruit, PasSignalSurBruit);
	printf("(II) MODE EVALUATION      : %s\n", ((Es_N0)?"Es/N0":"Eb/N0") );
	printf("(II) MIN-SUM ALGORITHM    : %s\n", dec_type.c_str());
	printf("(II) FAST STOP MODE       : %d\n", QUICK_STOP);
	
	CTimer simu_timer(true);

	//
	// ON CREE AUTANT DE TRAMES QUE L'ON A DE THREADS
	//
	CTrame simu_data(_N, _K, NB_FRAMES_IN_PARALLEL);
	CGPUDecoder *decoder = NULL;
	if( strcmp(dec_type.c_str(), "SPA") == 0 ){
		decoder = new CGPU_Decoder_SPA( NB_FRAMES_IN_PARALLEL, THREAD_PER_BLOCK, _N, _K, _M );
	}else if( strcmp(dec_type.c_str(), "MS") == 0 ){
		decoder = new CGPU_Decoder_MS( NB_FRAMES_IN_PARALLEL, THREAD_PER_BLOCK, _N, _K, _M );
	}else if( strcmp(dec_type.c_str(), "OMS") == 0 ){
		decoder = new CGPU_Decoder_OMS( NB_FRAMES_IN_PARALLEL, THREAD_PER_BLOCK, _N, _K, _M );
	}else if( strcmp(dec_type.c_str(), "NMS") == 0 ){
		decoder = new CGPU_Decoder_NMS( NB_FRAMES_IN_PARALLEL, THREAD_PER_BLOCK, _N, _K, _M );
	}else if( strcmp(dec_type.c_str(), "2NMS") == 0 ){
		decoder = new CGPU_Decoder_2NMS( NB_FRAMES_IN_PARALLEL, THREAD_PER_BLOCK, _N, _K, _M );
	}else if( strcmp(dec_type.c_str(), "SCMS") == 0 ){
		decoder = new CGPU_Decoder_SCMS( NB_FRAMES_IN_PARALLEL, THREAD_PER_BLOCK, _N, _K, _M );
	}else{
		printf("(EE) Error the decoder (%s) was not found !\n", dec_type.c_str());
		exit( 0 );
	}
	decoder->initialize();
	CChanel_AWGN noise(&simu_data, 4, QPSK_CHANNEL, Es_N0);

	Eb_N0 = MinSignalSurBruit;
	int temps = 0, fdecoding = 0;
	while (Eb_N0 <= MaxSignalSurBruit){

        //
        // ON CREE UN OBJET POUR LA MESURE DU TEMPS DE SIMULATION (REMISE A ZERO POUR CHAQUE Eb/N0)
        //
        CTimer temps_ecoule(true);
        CTimer term_refresh(true);

		noise.configure( Eb_N0 );

        CErrorAnalyzer errCounters(&simu_data, FRAME_ERROR_LIMIT, false);
        CErrorAnalyzer errCounter (&simu_data, FRAME_ERROR_LIMIT, true);

        //
        // ON CREE L'OBJET EN CHARGE DES INFORMATIONS DANS LE TERMINAL UTILISATEUR
        //
		CTerminal terminal(&errCounters, &temps_ecoule, Eb_N0);

        //
        // ON GENERE LA PREMIERE TRAME BRUITEE
        //
       	noise.generate();
       	errCounter.store_enc_bits();

		while( 1 ){

			//
			//	ON LANCE LE TRAITEMENT SUR PLUSIEURS THREAD...
			//
			CTimer essai(true);
			decoder->decode( simu_data.get_t_noise_data(), simu_data.get_t_decode_data(), NOMBRE_ITERATIONS );
			temps += essai.get_time_ms();
			fdecoding += 1;

			noise.generate();
    		errCounter.generate();
/*
    		#pragma omp sections
			{
				#pragma omp section
				{
				}
				#pragma omp section
			    {
				}
			}
*/
            //
			// ON COMPTE LE NOMBRE D'ERREURS DANS LA TRAME DECODE
            //
			errCounters.reset_internals();
			errCounters.accumulate( &errCounter );

            //
            // ON compare le Frame Error avec la limite imposee par l'utilisateur. Si on depasse
            // alors on affiche les resultats sur Eb/N0 courant.
            //
			if ( errCounters.fe_limit_achieved() == true ){
               break;
            }

            //
            // AFFICHAGE A L'ECRAN DE L'EVOLUTION DE LA SIMULATION SI NECESSAIRE
            //
			if( term_refresh.get_time_sec() >= 1 ){
				term_refresh.reset();
	           	terminal.temp_report();
			}

			if( (simu_timer.get_time_sec() >= STOP_TIMER_SECOND) && (STOP_TIMER_SECOND != -1) ){
        		printf("(II) THE SIMULATION HAS STOP DUE TO THE (USER) TIME CONTRAINT.\n");
        		printf("(II) PERFORMANCE EVALUATION WAS PERFORMED ON %d RUNS, TOTAL TIME = %dms\n", fdecoding, temps);
				temps /= fdecoding;
        		printf("(II) + TIME / RUN = %dms\n", temps);
        		int   workL = NB_FRAMES_IN_PARALLEL;
        		int   kbits = workL * _N / temps ;
        		float mbits = ((float)kbits) / 1000.0;
        		printf("(II) + DECODER LATENCY (ms)     = %d\n", temps);
        		printf("(II) + DECODER THROUGHPUT (Mbps)= %.1f\n", mbits);
        		printf("(II) + (%.2fdB, %dThd : %dCw, %dits) THROUGHPUT = %.1f\n", Eb_N0, NB_FRAMES_IN_PARALLEL, workL, NOMBRE_ITERATIONS, mbits);
				cout << endl << "Temps = " << temps << "ms : " << kbits;
				cout << "kb/s : " << ((float)temps/NB_FRAMES_IN_PARALLEL) << "ms/frame" << endl << endl;
        		break;
			}
		}

		terminal.final_report();


        if( (simu_timer.get_time_sec() >= STOP_TIMER_SECOND) && (STOP_TIMER_SECOND != -1) ){
        	break;
        }

		Eb_N0 = Eb_N0 + PasSignalSurBruit;

        if( BER_SIMULATION_LIMIT == true ){
        	if( errCounters.ber_value() < BIT_ERROR_LIMIT ){
        		printf("(II) THE SIMULATION HAS STOP DUE TO THE (USER) QUASI-ERROR FREE CONTRAINT.\n");
        		break;
        	}
        }
	}

	delete decoder;

	return 0;
}

