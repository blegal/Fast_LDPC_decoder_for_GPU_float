#include "CErrorAnalyzer.h"

CErrorAnalyzer::CErrorAnalyzer(CTrame *t){
    _data         = t->nb_data();
    _vars         = t->nb_vars();
    _frames       = t->nb_frames();
    t_decode_data = t->get_t_decode_data();
//    t_in_bits     = t->get_t_in_bits();
    nb_bit_errors      = 0;
    nb_frame_errors    = 0;
    nb_analyzed_frames = 0;
    _max_fe            = 200;
    _auto_fe_mode      = true;

    ber_per_thread = new int[_frames];
    fer_per_thread = new int[_frames];
	for(int i=0; i<_frames; i++){
		ber_per_thread[i] = 0;
		fer_per_thread[i] = 0;
	}
}

CErrorAnalyzer::CErrorAnalyzer(CTrame *t, int max_fe){
    _data              = t->nb_data();
    _vars              = t->nb_vars();
    _frames            = t->nb_frames();
    t_decode_data      = t->get_t_decode_data();
    nb_bit_errors      = 0;
    nb_frame_errors    = 0;
    nb_analyzed_frames = 0;
    _max_fe            = max_fe;
    _auto_fe_mode      = false;

    ber_per_thread = new int[_frames];
    fer_per_thread = new int[_frames];
	for(int i=0; i<_frames; i++){
		ber_per_thread[i] = 0;
		fer_per_thread[i] = 0;
	}
}

CErrorAnalyzer::CErrorAnalyzer(CTrame *t, int max_fe, bool auto_fe_mode){
    _data              = t->nb_data();
    _vars              = t->nb_vars();
    _frames            = t->nb_frames();
    t_decode_data      = t->get_t_decode_data();
    nb_bit_errors      = 0;
    nb_frame_errors    = 0;
    nb_analyzed_frames = 0;
    _max_fe            = max_fe;
    _auto_fe_mode      = auto_fe_mode;
    ber_per_thread = new int[_frames];
    fer_per_thread = new int[_frames];
	for(int i=0; i<_frames; i++){
		ber_per_thread[i] = 0;
		fer_per_thread[i] = 0;
	}
}

CErrorAnalyzer::~CErrorAnalyzer(){
    delete ber_per_thread;
    delete fer_per_thread;
}

void CErrorAnalyzer::reset_internals()
{
    nb_bit_errors      = 0;
    nb_frame_errors    = 0;
    nb_analyzed_frames = 0;
}

void CErrorAnalyzer::accumulate(CErrorAnalyzer *cErr)
{
    nb_bit_errors      += cErr->nb_bit_errors;
    nb_frame_errors    += cErr->nb_frame_errors;
    nb_analyzed_frames += cErr->nb_analyzed_frames;
}

long int CErrorAnalyzer::fe_limit()
{
    if( _auto_fe_mode == false ){
        return _max_fe;
    }else{
        double tBER = ber_value();
        if( tBER < 1.0e-9){
            return (_max_fe/16);
        }else if( tBER < 1.0e-8){
            return (_max_fe/8);
        }else if( tBER < 1.0e-7){
            return (_max_fe/4);
        }else if( tBER < 1.0e-6){
            return (_max_fe/2);
        }else{
            return (_max_fe);
        }
    }
}

bool CErrorAnalyzer::fe_limit_achieved()
{
    return (nb_fe() >= fe_limit());
}

void CErrorAnalyzer::generate()
{
    for (int z = 0; z < _frames; z++){
        int offset = z * _data;
        int nErrors = 0;

        for (int i=0; i<_vars; i++){
            nErrors += (t_decode_data[offset + i] != 0 /*buf_en_bits[offset + i]*/);
        }
        nb_bit_errors      += nErrors;
        nb_frame_errors    += (nErrors != 0);
        nb_analyzed_frames += 1;
		ber_per_thread[z] +=  nErrors;
		fer_per_thread[z] += (nErrors != 0);
    }
}

void CErrorAnalyzer::generate(int nErrors){
    nb_bit_errors      += nErrors;
    nb_frame_errors    += (nErrors != 0);
    nb_analyzed_frames += 1;
}

void CErrorAnalyzer::store_enc_bits(){
//    for(int i=0; i<_vars; i++){
//        buf_en_bits[i] = 0;// t_in_bits[i];
//    }
}

long int CErrorAnalyzer::nb_processed_frames(){
    return nb_analyzed_frames;
}

long int CErrorAnalyzer::nb_fe(){
    return nb_frame_errors;
}

long int CErrorAnalyzer::nb_be(){
    return nb_bit_errors;
}

double CErrorAnalyzer::fer_value(){
    double tFER = (((double)nb_fe())/(nb_processed_frames()));
    return tFER;
}

double CErrorAnalyzer::ber_value(){
    double tBER = (((double)nb_be())/(nb_processed_frames())/(_vars));
    return tBER;
}

long int CErrorAnalyzer::nb_be(int data){
    nb_bit_errors = data;
    return nb_bit_errors;
}

long int CErrorAnalyzer::nb_processed_frames(int data){
    nb_analyzed_frames = data;
    return nb_analyzed_frames;
}

long int CErrorAnalyzer::nb_fe(int data){
    nb_frame_errors = data;
    return nb_frame_errors;
}

int CErrorAnalyzer::nb_data(){
    return _data;
}

int CErrorAnalyzer::nb_vars(){
    return _vars;
}

int CErrorAnalyzer::nb_checks(){
    return (_data - _vars);
}
