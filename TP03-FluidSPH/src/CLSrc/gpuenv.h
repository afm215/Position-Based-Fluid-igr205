#ifndef GPUENV_H
#define GPUENV_H



#include "defineMacro.h"
    //header part

class GpuEnvironnment {
public:
	cl_kernel computeDensity; 
	cl_kernel computeLambda; 
	cl_kernel computeDp; 
	cl_kernel updatePrediction; 
	cl_kernel updateVelocity; 
	cl_kernel coputeVorticity; 
	cl_kernel applyViscousForce;

	cl_program program;
	cl_context context;
	cl_command_queue queue;

};




/*********DEBUG FUNCTION ***************/


const char* getErrorString(cl_int error);

void checkError(int status, const char* msg);
void print_clbuild_errors(cl_program program, cl_device_id device);
unsigned char** read_file(const char* name);
#ifdef _WIN32
int clock_gettime(int, struct timespec* spec);
#endif

#endif // !GPUENV_H