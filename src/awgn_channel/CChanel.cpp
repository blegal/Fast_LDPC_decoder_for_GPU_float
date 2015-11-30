#include "CChanel.h"

CChanel::~CChanel(){
}


CChanel::CChanel(CTrame *t, int _BITS_LLR, bool QPSK, bool ES_N0){
    _vars        = t->nb_vars();
    _data        = t->nb_data();
    _checks      = t->nb_checks();
    t_noise_data = t->get_t_noise_data();
    BITS_LLR     = _BITS_LLR;
    qpsk         = QPSK;
    es_n0        = ES_N0;
	_frames		 = t->nb_frames();
    SigB         = 0.0;
    rendement    = 1.0;
    Eb_N0        = 0.0;
}
