#include "../CLHeader/clcomputing.h"

int gpu_handle(int numberiteration, Vec2f* _pos, Vec2f* _vel, const int number_of_point, int* cl_flatten, const int pGrid_Size, const int index_size, int* flattened_indexes){
    int a;
    int  size = line * colonne;
    //std::cout << "one entering in the filter function size of the ouput :"<< size <<std::endl;

    cl_mem pos_buf;
    cl_mem output_vel;
    cl_mem neighbour_buff;
    cl_mem index_buffers;
    cl_mem output_pos;



    int status;


    // Input buffers.

    pos_buf = clCreateBuffer(env.context, CL_MEM_READ_ONLY,
        number_of_point * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for pos input");

    output_vel = clCreateBuffer(env.context, CL_MEM_READ_WRITE, number_of_point * sizeof(float), NULL, &status); 
    checkError(status, "Failed to create buffer for vel ");

    neighbour_buff = clCreateBuffer(env.context, CL_MEM_READ_ONLY, pGrid_Size * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for neighbours");

    index_buffers = clCreateBuffer(env.context, CL_MEM_READ_ONLY, index_size * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for indexes");
    // Output buffers.
    output_pos = clCreateBuffer(env.context, CL_MEM_READ_WRITE,
        number_of_point * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for pos output");




    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[3];
    cl_event kernel_event, finish_event;

    status = clEnqueueWriteBuffer(env.queue, pos_buf, CL_FALSE,
        0, number_of_point * sizeof(float), _pos, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input _pos");

    status = clEnqueueWriteBuffer(env.queue, index_buffers, CL_FALSE, 0, index_size * sizeof(float), flattened_indexes, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input indexes");

    status = clEnqueueWriteBuffer(env.queue, neighbour_buff, CL_FALSE, 0, number_of_point * sizeof(float), cl_flatten, 0, NULL, &write_event[2]);
    checkError(status, "Failed to transfer input neighbours");


    clWaitForEvents(4, write_event);
    // Set kernel arguments.
    unsigned argi = 0;
    status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &picture_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &kernel_buff);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");

    status = clSetKernelArg(env.kernel, argi++, sizeof(int), &line);
    checkError(status, "Failed to set argument 4");

    status = clSetKernelArg(env.kernel, argi++, sizeof(int), &colonne);
    checkError(status, "Failed to set argument 5");


    const size_t global_work_size[1] = { number_of_point };
    const size_t local_work_size[2] = { 10, 10 };
    status = clEnqueueNDRangeKernel(env.queue, env.kernel, 2, NULL,
        global_work_size, NULL, 2, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");

    clWaitForEvents(1, &kernel_event);
    // Read the result. This the final operation.



    status = clEnqueueReadBuffer(env.queue, output_buf, CL_TRUE,
        0, size * sizeof(char), output, 1, &kernel_event, &finish_event);
    checkError(status, "Failed to read output buffer");

    /*for (int i = 0; i < size; i++) {
        printf("adrres %c \n", output[i]);
    }*/



    clWaitForEvents(1, &finish_event);
    status = clFinish(env.queue);
    checkError(status, "Queue not finished");

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);

    status = clReleaseEvent(kernel_event);
    checkError(status, "Failed to clean kernel");

    clReleaseEvent(finish_event);
    status = clReleaseMemObject(picture_buf);
    checkError(status, "Failed to clean picture buff");

    clReleaseMemObject(kernel_buff);
    clReleaseMemObject(output_buf);

    //std::cout << "releasing" << std::endl;
    return 0;
}

int resolution(float* inputRe, float* inputIm, unsigned int line, unsigned int colonne, float* outputRe, float* outputIm, GpuEnvironnment& env) {
    int a;
    unsigned int  size = line * colonne;
    unsigned int outsize = 4 * size;

    cl_mem picture_bufRe;
    cl_mem picture_bufIm;
    cl_mem kernel_buff;
    cl_mem output_bufRe;
    cl_mem output_bufIm;



    int status;

    // Input buffers.

    picture_bufRe = clCreateBuffer(env.context, CL_MEM_READ_ONLY,
        size * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");
    picture_bufIm = clCreateBuffer(env.context, CL_MEM_READ_ONLY,
        size * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");
    // Output buffer.
    output_bufRe = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY,
        outsize * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");
    output_bufIm = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY,
        outsize * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");



    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
    cl_event kernel_event, finish_event;

    status = clEnqueueWriteBuffer(env.queue, picture_bufRe, CL_FALSE,
        0, size * sizeof(float), inputRe, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input picture");

    status = clEnqueueWriteBuffer(env.queue, picture_bufIm, CL_FALSE,
        0, size * sizeof(float), inputIm, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input picture");

    clWaitForEvents(2, write_event);
    // Set kernel arguments.
    unsigned argi = 0;
    status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &picture_bufRe);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &picture_bufIm);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &output_bufRe);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &output_bufIm);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(env.kernel, argi++, sizeof(int), &line);
    checkError(status, "Failed to set argument 3");

    status = clSetKernelArg(env.kernel, argi++, sizeof(int), &colonne);
    checkError(status, "Failed to set argument 4");


    const size_t global_work_size[2] = { line, colonne };

    const size_t local_work_size[2] = { 10, 10 };
    status = clEnqueueNDRangeKernel(env.queue, env.kernel, 2, NULL,
        global_work_size, NULL, 2, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");

    clWaitForEvents(1, &kernel_event);
    // Read the result. This the final operation.



    status = clEnqueueReadBuffer(env.queue, output_bufRe, CL_TRUE,
        0, outsize * sizeof(float), outputRe, 1, &kernel_event, &finish_event);
    checkError(status, "Failed to read output buffer");

    status = clEnqueueReadBuffer(env.queue, output_bufIm, CL_TRUE,
        0, outsize * sizeof(float), outputIm, 1, &kernel_event, &finish_event);
    checkError(status, "Failed to read output buffer");

    /*for (int i = 0; i < size; i++) {
        printf("adrres %c \n", output[i]);
    }*/



    clWaitForEvents(1, &finish_event);
    status = clFinish(env.queue);
    checkError(status, "Queue not finished");

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);

    status = clReleaseEvent(kernel_event);
    checkError(status, "Failed to clean kernel");

    clReleaseEvent(finish_event);

    clReleaseMemObject(picture_bufRe);
    clReleaseMemObject(picture_bufIm);

    clReleaseMemObject(output_bufRe);
    clReleaseMemObject(output_bufIm);



    //std::cout << "releasing" << std::endl;
    return 0;
}