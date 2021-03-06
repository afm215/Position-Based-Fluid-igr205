#include "../CLHeader/clcomputing.h"

#define NOMINMAX //reset namespace of windows.h

int gpu_handle(GpuEnvironnment& env, int numberiteration, const float* _pos, const float* predpos, const float* _vel, const int* _type, const int number_of_point, const int* cl_flatten, const int pGrid_Size, const int index_size, const int* flattened_indexes, float* pos_output, float* vel_output, const int sizeX, const int sizeY, const float _h, const float _m0, const float _d0, const float _dt, const float MIN_X, const float MIN_Y, const float WALL_X, const float MAX_Y, const bool debug) {

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
    cl_mem w_i_field;

    int debug_to_int = debug;


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

    w_i_field = clCreateBuffer(env.context, CL_MEM_READ_WRITE, number_of_point * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for lambda");



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

    status = clEnqueueWriteBuffer(env.queue, output_pos, CL_FALSE,
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
        //compute Density
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

        status = clWaitForEvents(1, &kernel_event);
        checkError(status, "Failed to launch Density kernel");

        // Read the result. This the final operation.

        //DEBUG stuff
       /* float* tmp_denisty = (float*)malloc(sizeof(float) * number_of_point);
        status = clEnqueueReadBuffer(env.queue, _d, CL_TRUE,
        0, sizeof(float) * number_of_point, tmp_denisty, 1, &kernel_event, &finish_event);
        checkError(status, "Failed to read output buffer");
        std::vector<float > test_vetcor_dens = std::vector<float>(tmp_denisty, tmp_denisty + number_of_point );
        tmp_denisty = nullptr;

        free(tmp_denisty);
    clWaitForEvents(1, &finish_event);*/


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
            global_work_size, NULL, 1, &kernel_event, &finish_event);
        checkError(status, "Failed to launch Lambda kernel");
        clReleaseEvent(kernel_event);
        clWaitForEvents(1, &finish_event);

        //DEBUG stuff
           /*float* tmp_lambda = (float*)malloc(sizeof(float) * number_of_point);
            status = clEnqueueReadBuffer(env.queue, _lambda, CL_TRUE,
            0, sizeof(float) * number_of_point, tmp_lambda, 1, &kernel_event, &finish_event);
            checkError(status, "Failed to read output buffer");
            clWaitForEvents(1, &finish_event);
            std::vector<float > test_vetcor_lambda = std::vector<float>(tmp_lambda, tmp_lambda + number_of_point);*/






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

        status = clSetKernelArg(env.computeDp, argi++, sizeof(int), &debug_to_int);
        checkError(status, "Failed to set argument Dp");



        status = clEnqueueNDRangeKernel(env.queue, env.computeDp, 1, NULL,
            global_work_size, NULL, 1, &finish_event, &kernel_event);
        checkError(status, "Failed to launch Dp kernel");
        clReleaseEvent(finish_event);
        clWaitForEvents(1, &kernel_event);

        //debug

       /* cl_float2* tmp_dp = (cl_float2*)malloc(sizeof(cl_float2 ) * number_of_point);
        status = clEnqueueReadBuffer(env.queue, _dp, CL_TRUE,
            0, sizeof(cl_float2) * number_of_point, tmp_dp, 1, &kernel_event, &finish_event);
        checkError(status, "Failed to read output buffer");

        clWaitForEvents(1, &finish_event);
        std::vector<cl_float2 > test_vetcor_dp = std::vector<cl_float2>(tmp_dp, tmp_dp + number_of_point);*/


        //update Prediction

        argi = 0;
        status = clSetKernelArg(env.updatePrediction, argi++, sizeof(cl_mem), &_dp);
        checkError(status, "Failed to set argument update Position");

        status = clSetKernelArg(env.updatePrediction, argi++, sizeof(cl_mem), &cl_type);
        checkError(status, "Failed to set argument update Position");

        status = clSetKernelArg(env.updatePrediction, argi++, sizeof(cl_mem), &output_pos);
        checkError(status, "Failed to set argument update Position");




        status = clEnqueueNDRangeKernel(env.queue, env.updatePrediction, 1, NULL,
            global_work_size, NULL, 1, &kernel_event, &finish_event);
        checkError(status, "Failed to launch update Position kernel");
        clReleaseEvent(kernel_event);
        clWaitForEvents(1, &finish_event);

        //apply Physical Constraint
        argi = 0;
        status = clSetKernelArg(env.applyPhysicallConstraint, argi++, sizeof(cl_mem), &output_pos);
        checkError(status, "Failed to set argument PhysicallConstraint");

        status = clSetKernelArg(env.applyPhysicallConstraint, argi++, sizeof(cl_mem), &cl_type);
        checkError(status, "Failed to set argument PhysicallConstraint");

        status = clSetKernelArg(env.applyPhysicallConstraint, argi++, sizeof(float), &MIN_X);
        checkError(status, "Failed to set argument PhysicallConstraint");

        status = clSetKernelArg(env.applyPhysicallConstraint, argi++, sizeof(float), &MIN_Y);
        checkError(status, "Failed to set argument PhysicallConstraint");

        status = clSetKernelArg(env.applyPhysicallConstraint, argi++, sizeof(float), &WALL_X);
        checkError(status, "Failed to set argument PhysicallConstraint");

        status = clSetKernelArg(env.applyPhysicallConstraint, argi++, sizeof(float), &MAX_Y);
        checkError(status, "Failed to set argument PhysicallConstraint");




        status = clEnqueueNDRangeKernel(env.queue, env.applyPhysicallConstraint, 1, NULL,
            global_work_size, NULL, 1, &finish_event, &kernel_event);
        checkError(status, "Failed to launch PhysicallConstraint kernel");
        clReleaseEvent(finish_event);
        clWaitForEvents(1, &kernel_event);

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
        global_work_size, NULL, 1, &kernel_event, &finish_event);
    checkError(status, "Failed to launch Velocity kernel");

    clReleaseEvent(kernel_event);

    clWaitForEvents(1, &finish_event);
    //compute Vorticity field
    argi = 0;

    status = clSetKernelArg(env.compute_w_i, argi++, sizeof(cl_mem), &neighbour_buff);
    checkError(status, "Failed to set argument Vorticity field");

    status = clSetKernelArg(env.compute_w_i, argi++, sizeof(cl_mem), &index_buffers);
    checkError(status, "Failed to set argument Vorticity field");

    status = clSetKernelArg(env.compute_w_i, argi++, sizeof(cl_mem), &output_pos);
    checkError(status, "Failed to set argument Vorticity field");

    status = clSetKernelArg(env.compute_w_i, argi++, sizeof(cl_mem), &cl_type);
    checkError(status, "Failed to set argument Vorticity field");

    status = clSetKernelArg(env.compute_w_i, argi++, sizeof(cl_mem), &output_vel);
    checkError(status, "Failed to set argument Vorticity field");

    status = clSetKernelArg(env.compute_w_i, argi++, sizeof(cl_mem), &w_i_field);
    checkError(status, "Failed to set argument Vorticity field");


    status = clSetKernelArg(env.compute_w_i, argi++, sizeof(float), &_h);
    checkError(status, "Failed to set argument Vorticity field");

    status = clSetKernelArg(env.compute_w_i, argi++, sizeof(int), &sizeX);
    checkError(status, "Failed to set argument Vorticity field");

    status = clSetKernelArg(env.compute_w_i, argi++, sizeof(int), &sizeY);
    checkError(status, "Failed to set argument Vorticity field");




    status = clEnqueueNDRangeKernel(env.queue, env.compute_w_i, 1, NULL,
        global_work_size, NULL, 1, &finish_event, &kernel_event);
    checkError(status, "Failed to launch Vorticity field kernel");

    clReleaseEvent(finish_event);

    clWaitForEvents(1, &kernel_event);
    //compute Vorticity forces


    argi = 0;
    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(cl_mem), &neighbour_buff);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(cl_mem), &index_buffers);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(cl_mem), &output_pos);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(cl_mem), &cl_type);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(cl_mem), &output_vel);
    checkError(status, "Failed to set argument Vorticity");

    status = clSetKernelArg(env.coputeVorticity, argi++, sizeof(cl_mem), &w_i_field);
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
        global_work_size, NULL, 1, &kernel_event, &finish_event);
    checkError(status, "Failed to launch Vorticity kernel");

    clReleaseEvent(kernel_event);

    clWaitForEvents(1, &finish_event);




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

    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(float), &_h);
    checkError(status, "Failed to set argument Viscosity");

    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(int), &sizeX);
    checkError(status, "Failed to set argument Viscosity");

    status = clSetKernelArg(env.applyViscousForce, argi++, sizeof(int), &sizeY);
    checkError(status, "Failed to set argument Viscosity");


    status = clEnqueueNDRangeKernel(env.queue, env.applyViscousForce, 1, NULL,
        global_work_size, NULL, 1, &finish_event, &kernel_event);
    checkError(status, "Failed to launch Viscosity kernel");

    clReleaseEvent(finish_event);

    clWaitForEvents(1, &kernel_event);






    status = clEnqueueReadBuffer(env.queue, output_pos, CL_TRUE,
        0, 2 * sizeof(float) * number_of_point, pos_output, 1, &kernel_event, &finish_event);
    checkError(status, "Failed to write output pos buffer");
    clReleaseEvent(kernel_event);
    clWaitForEvents(1, &finish_event);




    status = clEnqueueReadBuffer(env.queue, output_vel, CL_TRUE,
        0, 2 * sizeof(float) * number_of_point, vel_output, 1, &finish_event, &kernel_event);
    checkError(status, "Failed to write output vel buffer");

    clReleaseEvent(finish_event);

    clWaitForEvents(1, &kernel_event);






    status = clFinish(env.queue);
    checkError(status, "Queue not finished");

    if (debug == 1) {


        for (int i = 0; i < number_of_point * 2; i++) {
            if (vel_output[i] != _vel[i]) {
                std::cout << vel_output[i] - _vel[i] << "_vel : " << _vel[i] << "update vel: " << vel_output[i] << std::endl;
            }


        }

    }

    //// Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseEvent(write_event[2]);
    clReleaseEvent(write_event[3]);
    clReleaseEvent(write_event[4]);

    clReleaseEvent(kernel_event);



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

    status = clReleaseMemObject(w_i_field);
    checkError(status, "Failed to clean w_i buff");



    return 0;
}
