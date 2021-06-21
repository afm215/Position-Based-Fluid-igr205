#ifndef CL_COMPUTE_H
#define CL_COMPUTE_H

#include "Vector.hpp"
#include "../CLSrc/gpuenv.h"

void()

int gpu_handle(int numberiteration, Vec2f* _pos, Vec2f* _vel, const int number_of_point,   int* cl_flatten, const int pGrid_Size, const int index_size, int* flattened_indexes);

#endif // !CL_COMPUTE_H
