#include "../CLHeader/clcomputing.h"

#define NOMINMAX //reset namespace of windows.h

int gpu_handle(GpuEnvironnment env, int numberiteration, const float* _pos, float* predpos, float* _vel, const int* _type, const int number_of_point, const int* cl_flatten, const int pGrid_Size, const int index_size, const int* flattened_indexes, const int sizeX, const int sizeY, float _h, float _m0, float _d0, float _dt) {

    //std::cout << "one entering in the filter function size of the ouput :"<< size <<std::endl;

    cl_mem pos_buf;
    cl_mem output_vel;
    cl_mem neighbour_buff;
    cl_mem index_buffers;
    cl_mem output_pos; // equivalent to pred pos
    cl_mem _d;
    cl_mem _dp;
    cl_mem _lambda;
    cl_mem cl_type;



    int status;


    // Input buffers.

    pos_buf = clCreateBuffer(env.context, CL_MEM_READ_ONLY,
        number_of_point * 2 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for pos input");

    output_vel = clCreateBuffer(env.context, CL_MEM_READ_WRITE, number_of_point * sizeof(float) * 2, NULL, &status); 
    checkError(status, "Failed to create buffer for vel ");

    neighbour_buff = clCreateBuffer(env.context, CL_MEM_READ_ONLY, pGrid_Size * sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for neighbours");

    index_buffers = clCreateBuffer(env.context, CL_MEM_READ_ONLY, index_size * sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for indexes");
    // Output buffers.
    output_pos = clCreateBuffer(env.context, CL_MEM_READ_WRITE,
        number_of_point * sizeof(float) * 2, NULL, &status);
    checkError(status, "Failed to create buffer for pos output");

    _d = clCreateBuffer(env.context, CL_MEM_READ_WRITE, number_of_point * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for density");

    _lambda = clCreateBuffer(env.context, CL_MEM_READ_WRITE, number_of_point * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for lambda");

    _dp = clCreateBuffer(env.context, CL_MEM_READ_WRITE, number_of_point * sizeof(cl_float2), NULL, &status);
    checkError(status, "Failed to create buffer for dp");



    cl_type = clCreateBuffer(env.context, CL_MEM_READ_ONLY, number_of_point * sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for cl_type");




    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[5];
    cl_event kernel_event, finish_event;

    status = clEnqueueWriteBuffer(env.queue, pos_buf, CL_FALSE,
        0, number_of_point * sizeof(float) * 2, _pos, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input _pos");

    status = clEnqueueWriteBuffer(env.queue,output_pos, CL_FALSE,
        0, number_of_point * sizeof(float) * 2, predpos, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer output _pos");

    status = clEnqueueWriteBuffer(env.queue, index_buffers, CL_FALSE, 0, index_size * sizeof(int), flattened_indexes, 0, NULL, &write_event[2]);
    checkError(status, "Failed to transfer input indexes");

    status = clEnqueueWriteBuffer(env.queue, neighbour_buff, CL_FALSE, 0, number_of_point * sizeof(int), cl_flatten, 0, NULL, &write_event[3]);
    checkError(status, "Failed to transfer input neighbours");

    status = clEnqueueWriteBuffer(env.queue, cl_type, CL_FALSE, 0, number_of_point * sizeof(int), _type, 0, NULL, &write_event[4]);
    checkError(status, "Failed to transfer input types");


    clWaitForEvents(5, write_event);

    const size_t global_work_size[1] = { number_of_point };
    unsigned argi = 0;

    int i = 0;
    while (i < numberiteration) {
        //compute Lambda
        argi = 0;
        status = clSetKernelArg(env.computeDensity, argi++, sizeof(cl_mem), &neighbour_buff);
        checkError(status, "Failed to set argument Density");
        
        status = clSetKernelArg(env.computeDensity, argi++, sizeof(cl_mem), &index_buffers);
        checkError(status, "Failed to set argument Density"); 

        status = clSetKernelArg(env.computeDensity, argi++, sizeof(cl_mem), &output_pos);
        checkError(status, "Failed to set argument Density");

        status = clSetKernelArg(env.computeDensity, argi++, sizeof(cl_mem), &_d);
        checkError(status, "Failed to set argument Density");

        status = clSetKernelArg(env.computeDensity, argi++, sizeof(float), &_h);
        checkError(status, "Failed to set argument Density");

        status = clSetKernelArg(env.computeDensity, argi++, sizeof(float), &_m0);
        checkError(status, "Failed to set argument Density");

        status = clSetKernelArg(env.computeDensity, argi++, sizeof(int), &sizeX);
        checkError(status, "Failed to set argument Density");

        status = clSetKernelArg(env.computeDensity, argi++, sizeof(int), &sizeY);
        checkError(status, "Failed to set argument Density");


        status = clEnqueueNDRangeKernel(env.queue, env.computeDensity, 1, NULL,
        global_work_size, NULL, 4, write_event, &kernel_event);
        checkError(status, "Failed to launch Density kernel");

        clWaitForEvents(1, &kernel_event);
        // Read the result. This the final operation.

        //DEBUG stuff
        /*float* tmp_denisty = (float*)malloc(sizeof(float) * number_of_point);
        status = clEnqueueReadBuffer(env.queue, _d, CL_TRUE,
        0, sizeof(float) * number_of_point, tmp_denisty, 1, &kernel_event, &finish_event);
        checkError(status, "Failed to read output buffer");
       
        for (int i = 0; i < number_of_point; i++) {
            std::cout << tmp_denisty[i] << std::endl;
        }*/
        




    //clWaitForEvents(1, &finish_event);


    //compute lambda

    argi = 0;
    status = clSetKernelArg(env.computeLambda, argi++, sizeof(cl_mem), &neighbour_buff);
    checkError(status, "Failed to set argument Lambda");

    status = clSetKernelArg(env.computeLambda, argi++, sizeof(cl_mem), &index_buffers);
    checkError(status, "Failed to set argument Lambda");

    status = clSetKernelArg(env.computeLambda, argi++, sizeof(cl_mem), &output_pos);
    checkError(status, "Failed to set argument Lambda");

    status = clSetKernelArg(env.computeLambda, argi++, sizeof(cl_mem), &_d);
    checkError(status, "Failed to set argument Lambda");

    status = clSetKernelArg(env.computeLambda, argi++, sizeof(cl_mem), &_lambda);
    checkError(status, "Failed to set argument Lambda");

    status = clSetKernelArg(env.computeLambda, argi++, sizeof(float), &_h);
    checkError(status, "Failed to set argument Lambda");

    status = clSetKernelArg(env.computeLambda, argi++, sizeof(float), &_d0);
    checkError(status, "Failed to set argument Lambda");

    status = clSetKernelArg(env.computeLambda, argi++, sizeof(int), &sizeX);
    checkError(status, "Failed to set argument Lambda");

    status = clSetKernelArg(env.computeLambda, argi++, sizeof(int), &sizeY);
    checkError(status, "Failed to set argument Lambda");


    status = clEnqueueNDRangeKernel(env.queue, env.computeLambda, 1, NULL,
        global_work_size, NULL, 1, &kernel_event, &kernel_event);
    checkError(status, "Failed to launch Lambda kernel");

    clWaitForEvents(1, &kernel_event);

    //DEBUG stuff
       /*float* tmp_lambda = (float*)malloc(sizeof(float) * number_of_point);
        status = clEnqueueReadBuffer(env.queue, _lambda, CL_TRUE,
        0, sizeof(float) * number_of_point, tmp_lambda, 1, &kernel_event, &finish_event);
        checkError(status, "Failed to read output buffer");

        for (int i = 0; i < number_of_point; i++) {
            std::cout << tmp_lambda[i] << std::endl;
        }*/





       // clWaitForEvents(1, &finish_event);

    //Compute DP
    argi = 0;
    status = clSetKernelArg(env.computeDp, argi++, sizeof(cl_mem), &neighbour_buff);
    checkError(status, "Failed to set argument Dp");

    status = clSetKernelArg(env.computeDp, argi++, sizeof(cl_mem), &index_buffers);
    checkError(status, "Failed to set argument Dp");

    status = clSetKernelArg(env.computeDp, argi++, sizeof(cl_mem), &output_pos);
    checkError(status, "Failed to set argument Dp");

    status = clSetKernelArg(env.computeDp, argi++, sizeof(cl_mem), &cl_type);
    checkError(status, "Failed to set argument Dp");

    status = clSetKernelArg(env.computeDp, argi++, sizeof(cl_mem), &_lambda);
    checkError(status, "Failed to set argument Dp");

    status = clSetKernelArg(env.computeDp, argi++, sizeof(cl_mem), &_dp);
    checkError(status, "Failed to set argument Dp");

    status = clSetKernelArg(env.computeDp, argi++, sizeof(float), &_d0);
    checkError(status, "Failed to set argument Dp");

    status = clSetKernelArg(env.computeDp, argi++, sizeof(float), &_h);
    checkError(status, "Failed to set argument Dp");



    status = clSetKernelArg(env.computeDp, argi++, sizeof(int), &sizeX);
    checkError(status, "Failed to set argument Dp");

    status = clSetKernelArg(env.computeDp, argi++, sizeof(int), &sizeY);
    checkError(status, "Failed to set argument Dp");


    status = clEnqueueNDRangeKernel(env.queue, env.computeDp, 1, NULL,
        global_work_size, NULL, 1, &kernel_event, &kernel_event);
    checkError(status, "Failed to launch Dp kernel");

    clWaitForEvents(1, &kernel_event);

    /*float* tmp_dp = (float*)malloc(sizeof(float) * number_of_point);
    status = clEnqueueReadBuffer(env.queue, _dp, CL_TRUE,
        0, sizeof(float) * number_of_point, tmp_dp, 1, &kernel_event, &finish_event);
    checkError(status, "Failed to read output buffer");

    for (int i = 0; i < number_of_point; i++) {
        std::cout << tmp_dp[i] << std::endl;
    }

    clWaitForEvents(1, &finish_event);*/

    //update Prediction

    argi = 0;
    status = clSetKernelArg(env.updatePrediction, argi++, sizeof(cl_mem), &_dp);
    checkError(status, "Failed to set argument update Position");

    status = clSetKernelArg(env.updatePrediction, argi++, sizeof(cl_mem), &cl_type);
    checkError(status, "Failed to set argument update Position");

    status = clSetKernelArg(env.updatePrediction, argi++, sizeof(cl_mem), &output_pos);
    checkError(status, "Failed to set argument update Position");

    


    status = clEnqueueNDRangeKernel(env.queue, env.updatePrediction, 1, NULL,
        global_work_size, NULL, 1, &kernel_event, &kernel_event);
    checkError(status, "Failed to launch update Position kernel");

    clWaitForEvents(1, &kernel_event);


    /*float* tmp_pred = (float*)malloc(sizeof(float) * number_of_point);
    status = clEnqueueReadBuffer(env.queue, _dp, CL_TRUE,
        0, sizeof(float) * number_of_point, tmp_pred, 1, &kernel_event, &finish_event);
    checkError(status, "Failed to read output buffer");

    for (int i = 0; i < number_of_point; i++) {
        std::cout << tmp_dp[i] << std::endl;
    }

    clWaitForEvents(1, &finish_event);*/

    
    


        i++;
    }

    //update Velocities
    argi = 0;
    status = clSetKernelArg(env.updateVelocity, argi++, sizeof(cl_mem), &output_pos);
    checkError(status, "Failed to set argument updateVel");

    status = clSetKernelArg(env.updateVelocity, argi++, sizeof(cl_mem), &pos_buf);
    checkError(status, "Failed to set argument updateVel");

    status = clSetKernelArg(env.updateVelocity, argi++, sizeof(cl_mem), &cl_type);
    checkError(status, "Failed to set argument updateVel");

    status = clSetKernelArg(env.updateVelocity, argi++, sizeof(cl_mem), &output_vel);
    checkError(status, "Failed to set argument updateVel");

    status = clSetKernelArg(env.updateVelocity, argi++, sizeof(float), &_dt);
    checkError(status, "Failed to set argument updateVel");

    


    status = clEnqueueNDRangeKernel(env.queue, env.updateVelocity, 1, NULL,
        global_work_size, NULL, 1, &kernel_event, &kernel_event);
    checkError(status, "Failed to launch Velocity kernel");

    clWaitForEvents(1, &kernel_event);
    //compute Vorticity

    argi = 0;
    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(cl_mem), &neighbour_buff);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(cl_mem), &index_buffers);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(cl_mem), &output_pos);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(cl_mem), &output_vel);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(float), &_dt);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(float), &_m0);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(float), &_h);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(int), &sizeX);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(int), &sizeY);
    checkError(status, "Failed to set argument Vorticity");


    status = clEnqueueNDRangeKernel(env.queue, env.coputeVorticity, 1, NULL,
        global_work_size, NULL, 1, &kernel_event, &kernel_event);
    checkError(status, "Failed to launch Vorticity kernel");

    clWaitForEvents(1, &kernel_event);

    //compute viscous Forces

    argi = 0;
    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(cl_mem), &neighbour_buff);
    checkError(status, "Failed to set argument Viscosity");

    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(cl_mem), &index_buffers);
    checkError(status, "Failed to set argument Viscosity");

    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(cl_mem), &output_pos);
    checkError(status, "Failed to set argument Viscosity");

    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(cl_mem), &cl_type);
    checkError(status, "Failed to set argument Viscosity");

    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(cl_mem), &output_vel);
    checkError(status, "Failed to set argument Viscosity");

    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(float), &_dt);
    checkError(status, "Failed to set argument Viscosity");

    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(float), &_h);
    checkError(status, "Failed to set argument Viscosity");

    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(int), &sizeX);
    checkError(status, "Failed to set argument Viscosity");

    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(int), &sizeY);
    checkError(status, "Failed to set argument Viscosity");


    status = clEnqueueNDRangeKernel(env.queue, env.applyViscousForce, 1, NULL,
        global_work_size, NULL, 1, &kernel_event, &kernel_event);
    checkError(status, "Failed to launch Viscosity kernel");

    clWaitForEvents(1, &kernel_event);
    

    status = clEnqueueReadBuffer(env.queue, output_pos, CL_TRUE,
        0, sizeof(float) * number_of_point, predpos, 1, &kernel_event, &finish_event);
    checkError(status, "Failed to write output pos buffer");
    clWaitForEvents(1, &finish_event);

    /*for (int i = 0; i < number_of_point * 2; i++) {
        std::cout << predpos[i] - _pos[i] << std::endl;
    }*/

    /*std::cout << "shuuuuuuuuut" << std::endl;*/

    status = clEnqueueReadBuffer(env.queue, output_vel, CL_TRUE,
        0, sizeof(float) * number_of_point, _vel, 1, &finish_event, &finish_event);
    checkError(status, "Failed to write output vel buffer");
    clWaitForEvents(1, &finish_event);

   /* for (int i = 0; i < number_of_point * 2; i++) {
        std::cout << _vel[i] << std::endl;
    }*/



    
    //// Set kernel arguments.
    //unsigned argi = 0;
    //status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &picture_buf);
    //checkError(status, "Failed to set argument 1");

    //status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &kernel_buff);
    //checkError(status, "Failed to set argument 2");

    //status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &output_buf);
    //checkError(status, "Failed to set argument 3");

    //status = clSetKernelArg(env.kernel, argi++, sizeof(int), &line);
    //checkError(status, "Failed to set argument 4");

    //status = clSetKernelArg(env.kernel, argi++, sizeof(int), &colonne);
    //checkError(status, "Failed to set argument 5");


    //const size_t global_work_size[1] = { number_of_point };
    ////const size_t local_work_size[2] = { 10, 10 };
    //status = clEnqueueNDRangeKernel(env.queue, env.kernel, 2, NULL,
    //    global_work_size, NULL, 3, write_event, &kernel_event);
    //checkError(status, "Failed to launch kernel");

    //clWaitForEvents(1, &kernel_event);
    //// Read the result. This the final operation.



    //status = clEnqueueReadBuffer(env.queue, output_buf, CL_TRUE,
    //    0, size * sizeof(char), output, 1, &kernel_event, &finish_event);
    //checkError(status, "Failed to read output buffer");

    ///*for (int i = 0; i < size; i++) {
    //    printf("adrres %c \n", output[i]);
    //}*/



    //clWaitForEvents(1, &finish_event);
   /* status = clFinish(env.queue);
    checkError(status, "Queue not finished");*/

    //// Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseEvent(write_event[2]);
    clReleaseEvent(write_event[3]);
    clReleaseEvent(write_event[4]);

    /*status = clReleaseEvent(kernel_event);
    checkError(status, "Failed to clean kernel");*/

    //clReleaseEvent(finish_event);
    clReleaseEvent(kernel_event);
    clReleaseEvent(finish_event);

    status = clReleaseMemObject(pos_buf);
    checkError(status, "Failed to clean pos buff");

    status = clReleaseMemObject(output_vel);
    checkError(status, "Failed to clean vel buff");

    status = clReleaseMemObject(output_pos);
    checkError(status, "Failed to clean output pos buff");

    status = clReleaseMemObject(neighbour_buff);
    checkError(status, "Failed to clean neuighbour buff");

    status = clReleaseMemObject(index_buffers);
    checkError(status, "Failed to clean index_buffers buff");

    status = clReleaseMemObject(_d);
    checkError(status, "Failed to clean _d buff");

    status = clReleaseMemObject(_dp);
    checkError(status, "Failed to clean _dp buff");

    status = clReleaseMemObject(_lambda);
    checkError(status, "Failed to clean lambda buff");

    status = clReleaseMemObject(cl_type);
    checkError(status, "Failed to clean type buff");

    

    ////std::cout << "releasing" << std::endl;
    return 0;
}

//int resolution(float* inputRe, float* inputIm, unsigned int line, unsigned int colonne, float* outputRe, float* outputIm, GpuEnvironnment& env) {
//    int a;
//    unsigned int  size = line * colonne;
//    unsigned int outsize = 4 * size;
//
//    cl_mem picture_bufRe;
//    cl_mem picture_bufIm;
//    cl_mem kernel_buff;
//    cl_mem output_bufRe;
//    cl_mem output_bufIm;
//
//
//
//    int status;
//
//    // Input buffers.
//
//    picture_bufRe = clCreateBuffer(env.context, CL_MEM_READ_ONLY,
//        size * sizeof(float), NULL, &status);
//    checkError(status, "Failed to create buffer for input A");
//    picture_bufIm = clCreateBuffer(env.context, CL_MEM_READ_ONLY,
//        size * sizeof(float), NULL, &status);
//    checkError(status, "Failed to create buffer for input A");
//    // Output buffer.
//    output_bufRe = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY,
//        outsize * sizeof(float), NULL, &status);
//    checkError(status, "Failed to create buffer for output");
//    output_bufIm = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY,
//        outsize * sizeof(float), NULL, &status);
//    checkError(status, "Failed to create buffer for output");
//
//
//
//    // Transfer inputs to each device. Each of the host buffers supplied to
//    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
//    // for the host-to-device transfer.
//    cl_event write_event[2];
//    cl_event kernel_event, finish_event;
//
//    status = clEnqueueWriteBuffer(env.queue, picture_bufRe, CL_FALSE,
//        0, size * sizeof(float), inputRe, 0, NULL, &write_event[0]);
//    checkError(status, "Failed to transfer input picture");
//
//    status = clEnqueueWriteBuffer(env.queue, picture_bufIm, CL_FALSE,
//        0, size * sizeof(float), inputIm, 0, NULL, &write_event[1]);
//    checkError(status, "Failed to transfer input picture");
//
//    clWaitForEvents(2, write_event);
//    // Set kernel arguments.
//    unsigned argi = 0;
//    status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &picture_bufRe);
//    checkError(status, "Failed to set argument 1");
//
//    status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &picture_bufIm);
//    checkError(status, "Failed to set argument 1");
//
//    status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &output_bufRe);
//    checkError(status, "Failed to set argument 2");
//
//    status = clSetKernelArg(env.kernel, argi++, sizeof(cl_mem), &output_bufIm);
//    checkError(status, "Failed to set argument 2");
//
//    status = clSetKernelArg(env.kernel, argi++, sizeof(int), &line);
//    checkError(status, "Failed to set argument 3");
//
//    status = clSetKernelArg(env.kernel, argi++, sizeof(int), &colonne);
//    checkError(status, "Failed to set argument 4");
//
//
//    const size_t global_work_size[2] = { line, colonne };
//
//    const size_t local_work_size[2] = { 10, 10 };
//    status = clEnqueueNDRangeKernel(env.queue, env.kernel, 2, NULL,
//        global_work_size, NULL, 2, write_event, &kernel_event);
//    checkError(status, "Failed to launch kernel");
//
//    clWaitForEvents(1, &kernel_event);
//    // Read the result. This the final operation.
//
//
//
//    status = clEnqueueReadBuffer(env.queue, output_bufRe, CL_TRUE,
//        0, outsize * sizeof(float), outputRe, 1, &kernel_event, &finish_event);
//    checkError(status, "Failed to read output buffer");
//
//    status = clEnqueueReadBuffer(env.queue, output_bufIm, CL_TRUE,
//        0, outsize * sizeof(float), outputIm, 1, &kernel_event, &finish_event);
//    checkError(status, "Failed to read output buffer");
//
//    /*for (int i = 0; i < size; i++) {
//        printf("adrres %c \n", output[i]);
//    }*/
//
//
//
//    clWaitForEvents(1, &finish_event);
//    status = clFinish(env.queue);
//    checkError(status, "Queue not finished");
//
//    // Release local events.
//    clReleaseEvent(write_event[0]);
//    clReleaseEvent(write_event[1]);
//
//    status = clReleaseEvent(kernel_event);
//    checkError(status, "Failed to clean kernel");
//
//    clReleaseEvent(finish_event);
//
//    clReleaseMemObject(picture_bufRe);
//    clReleaseMemObject(picture_bufIm);
//
//    clReleaseMemObject(output_bufRe);
//    clReleaseMemObject(output_bufIm);
//
//
//
//    //std::cout << "releasing" << std::endl;
//    return 0;
//}