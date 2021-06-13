#ifndef GPUENV_H
#define GPUENV_H



#include "defineMacro.h"
    //header part

class GpuEnvironnment {
public:
	cl_kernel kernel;
	cl_program program;
	cl_context context;
	cl_command_queue queue;

};




/*********DEBUG FUNCTION ***************/


const char* getErrorString(cl_int error);
void print_clbuild_errors(cl_program program, cl_device_id device);
unsigned char** read_file(const char* name);
#ifdef _WIN32
int clock_gettime(int, struct timespec* spec);
#endif

#endif // !GPUENV_H