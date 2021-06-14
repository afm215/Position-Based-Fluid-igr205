#ifndef CL_COMPUTE_H
#define CL_COMPUTE_H

#include "Vector.hpp"

void()

void gpu_handle(int numberiteration, Vec2f* _pos, Vec2f* _vel,  int* cl_flatten, const int pGrid_Size, int* flattened_indexes);

#endif // !CL_COMPUTE_H
