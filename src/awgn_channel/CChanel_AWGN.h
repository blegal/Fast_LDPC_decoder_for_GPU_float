#ifndef CLASS_CChanel_AWGN
#define CLASS_CChanel_AWGN

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "CChanel.h"

#include "../custom_api/custom_cuda.h"
#include <curand.h>


class CChanel_AWGN : public CChanel
{
private:
    double awgn(double amp);
    float *device_A;
    float *device_B;
    float *device_R;
	curandGenerator_t generator;

public:
	CChanel_AWGN(CTrame *t, int _BITS_LLR, bool QPSK, bool Es_N0);
    ~CChanel_AWGN();
    virtual void configure(double _Eb_N0);
    virtual void generate();
};

#endif

