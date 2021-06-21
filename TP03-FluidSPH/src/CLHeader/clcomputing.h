#ifndef CL_COMPUTE_H
#define CL_COMPUTE_H

#define NOMINMAX //reset namespace of windows.h
#include "../CLSrc/gpuenv.h"



int gpu_handle(GpuEnvironnment env, int numberiteration, const float* _pos, float* predpos, float* _vel, const int* _type, const int number_of_point, const int* cl_flatten, const int pGrid_Size, const int index_size, const int* flattened_indexes, const int sizeX, const int sizeY, float _h, float _m0, float _d0, float _dt);

#endif // !CL_COMPUTE_H
