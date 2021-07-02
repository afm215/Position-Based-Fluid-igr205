#ifndef CL_COMPUTE_H
#define CL_COMPUTE_H

#define NOMINMAX //reset namespace of windows.h
#include "../CLSrc/gpuenv.h"



int gpu_handle(GpuEnvironnment& env, int numberiteration, const float* _pos, const float* predpos, const float* _vel, const int* _type, const int number_of_point, const int* cl_flatten, const int pGrid_Size, const int index_size, const int* flattened_indexes, float* pos_output, float* vel_output, const int sizeX, const int sizeY, const float _h, const float _m0, const float _d0, const float _dt, const float MIN_X, const float MIN_Y, const float WALL_X, const float MAX_Y, const bool debug);

#endif // !CL_COMPUTE_H
