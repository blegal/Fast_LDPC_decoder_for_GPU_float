#include "CTrame.h"

#include "../custom_api/custom_cuda.h"

CTrame::CTrame(int width, int height){
    _width        = width;
    _height       = height;
    _frame        = 1;
    CUDA_MALLOC_HOST(&t_noise_data, nb_data() + 1, __FILE__, __LINE__);
//    CUDA_MALLOC_HOST(&t_fpoint_data, nb_data() + 1, __FILE__, __LINE__);
    CUDA_MALLOC_HOST(&t_decode_data, nb_data() + 1, __FILE__, __LINE__);
}

CTrame::CTrame(int width, int height, int frame){
    _width        = width;
    _height       = height;
    _frame        = frame;
    CUDA_MALLOC_HOST(&t_noise_data,  nb_data()  * frame + 4, __FILE__, __LINE__);
//    CUDA_MALLOC_HOST(&t_fpoint_data, nb_data() * frame + 4, __FILE__, __LINE__);
    CUDA_MALLOC_HOST(&t_decode_data, nb_data() * frame + 4, __FILE__, __LINE__);
}


CTrame::~CTrame(){
	cudaFreeHost(t_noise_data);
//	cudaFreeHost(t_fpoint_data);
	cudaFreeHost(t_decode_data);
}

size_t CTrame::nb_vars(){
    return  (nb_data()-nb_checks());
}

size_t CTrame::nb_frames(){
    return  _frame;
}

size_t CTrame::nb_checks(){
    return _height;
}

size_t CTrame::nb_data(){
    return _width;
}

float* CTrame::get_t_noise_data(){
    return t_noise_data;
}

int* CTrame::get_t_decode_data(){
    return t_decode_data;
}
