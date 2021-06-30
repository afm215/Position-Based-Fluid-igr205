float SPH_EPSILON(){
    return 200.0f;
}

float pabs(const float number){
    if (number < 0){
        return - number;
    }
    else{
        return number;
    }
}

float2 Repulsion(const float2 xij){
    float band = 0.00002f;
    float repulsion_force  = 0.0002;
    float len_xij = length(xij);
    if(len_xij == 0){
        int i = get_global_id(0);
        float global_size =(float) get_global_size(0);
        float theta  = M_PI * i * 2 / global_size;
        return repulsion_force * (float2) (cos(theta), sin(theta));
    }

    // else{
    //     if(len_xij <= band){
    //         return repulsion_force * normalize(xij) *  exp( - len_xij * len_xij / band / band);
    //     }


    // }

    return (float2) (0,0);

}

float kPoly6Factor() {
    return (315.0f / 64.0f /3.141592f);
}

float kSpikyGradFactor() { return (-45.0f / M_PI); }

float plength(const float x , const float y){
    return sqrt(x*x + y*y);
}



float w(const float s, const float h) {
    if (s < 0.0f || s >= h)
        return 0.0f;

    float x = (h * h - s * s) / (h * h * h);
    float result = kPoly6Factor() * x * x * x;
    return result;
}

float2 grad_w(const float2 r, const float h) {
    float r_len  = length(r);
    if (r_len <= 0.0f || r_len >= h)
        return (float2)(0.0f, 0.0f);

    float x = (h - r_len) / (h * h * h);
    float g_factor = kSpikyGradFactor() * x * x;
    float2 result = normalize(r) * g_factor;
    return result;
}

float crossProduct(const float2 a, const float2 b){
    float result = a.x * b.y - a.y * b.x;
    if(a.x != 0 ){
        //printf("%f %f ; %f %f ; %f \n", a.x, a.y, b.x, b.y, result);
    }
    return result;
}





__kernel void proto(__global const int* neighbours, __global const int* indexes_neighbours, __global const float* positions, __global float* update_pos, __global float* update_vel) {




    //int i = 0;

    //while (i < NB_IT)
    //{

    //    //compute lambda_i 
    //    //computeLambda use the comutegradCi function wich use the density of the paricles 
    //    computeDensity();
    //    computeLambda();
    //    //calculate the difference in positions using the lamba_i
    //    computeDp();
    //    //update the position p_i* = p_i* + dp_i
    //    updatePrediction();
    //    //applyPhysicalConstraints();

    //    i++;
    //}
    ////update the velocities v_i = p_i* - p_i 
    //updateVelocity();
    //computeVorticity();
    //applyViscousForce();
    //// use the newly computed velocities to compute vorticity confinement and XSPH viscosity TO DO !!!

    ////modify the position p_i = p_i *
    //updatePosition();
}

// remind that everything is float so size(positions)  = 2 * number_points = size(velocities)
// neighbours is flattened, use indexes to know where are the delimitations of the neighbour arrays

__kernel void computeDensity(__global const int* neighbours, __global const int* indexes_neighbours, __global const float* positions, __global float* _d, const float sr, const float _m0, const int resX, const int resY) {
    
    float sum_m = 0.f;
    int  i = get_global_id(0);
    const float x= positions[2*i];
    const float y = positions[2 * i + 1];

    const int gi_from = (int)(x - sr);
    const int gi_to = (int)(x + sr) + 1;
    const int gj_from = (int)(y - sr);
    const int gj_to = (int)(y + sr) + 1;

    

    for (int gj = max(0, gj_from); gj < min(resY, gj_to); ++gj) {
        for (int gi = max(0, gi_from); gi < min(resX, gi_to); ++gi) {
            const int gidx = gi + gj * resX; //idx1d(gi, gj);

            // each particle in nearby cells
            int neighbour_size = 0;
            if (gidx == 0) {
                neighbour_size = indexes_neighbours[0];
            }
            else {
                neighbour_size = indexes_neighbours[gidx] - indexes_neighbours[gidx - 1];
            }
            for (size_t ni = 0; ni < neighbour_size; ++ni) {
                int j = 0;
                if (gidx == 0) {
                    j = neighbours[ni];
                }
                else{
                    j = neighbours[indexes_neighbours[gidx - 1] + ni];
                }
                if (i == j) continue;
                const float xj =  positions[2 * j];
                const float yj = positions[2 * j + 1];

                const float xij = x - xj;
                const float yij =  y - yj;

                const float len_xij = plength(xij, yij);
                
                sum_m += _m0 * w(len_xij, sr);

            }
        }
    }

    _d[i] = sum_m;

}


float2 computeGradCi(const int i, const int k, const float sr, __global const float* positions, __global const int*neighbours, __global const int* indexes_neighbours, const float _d0, const int resX, const int resY, const float _h) {


    const float2 xi = (float2)(positions[ 2 * i], positions[2*i + 1]);
    float2 result = (float2)(0.f, 0.f);

    
    if (k == i) {

        const int gi_from = (int)(xi.x - sr);
        const int gi_to =(int)(xi.x + sr) + 1;
        const int gj_from = (int)(xi.y - sr);
        const int gj_to = (int)(xi.y + sr) + 1;


        for (int gj = max(0, gj_from); gj < min(resY, gj_to); ++gj) {
            for (int gi = max(0, gi_from); gi < min(resX, gi_to); ++gi) {
                const int gidx = gi + gj * resX; //idx1d(gi, gj);

                // each particle in nearby cells
                int neighbour_size = 0;
                if (gidx == 0) {
                    neighbour_size = indexes_neighbours[0];
                }
                else {
                    neighbour_size = indexes_neighbours[gidx] - indexes_neighbours[gidx - 1];
                }
                for (size_t ni = 0; ni < neighbour_size; ++ni) {
                    int j = 0;
                    if (gidx == 0) {
                        j = neighbours[ni];
                    }
                    else{
                        j = neighbours[indexes_neighbours[gidx - 1] + ni];
                    }
                    if (j == i) continue;
                    const float2 xj = (float2) (positions[ 2 * j], positions[2 * j +1]);
                    const float2 xij = xi - xj;
                    const float len_xij = length(xij);
                    if (len_xij > _h ) continue;

                    result +=  grad_w(xij, _h);

                }
            }
        }

        return result / _d0;
    }

    else{
        const float2 xk = (float2) (positions[2 * k], positions[2 * k + 1]);
        const float2 xik = xi - xk;
        result -= 1 / _d0 * grad_w(xik, _h);
        return result;
    }
        
}

__kernel void computeLambda(__global const int* neighbours, __global const int* indexes_neighbours, __global const float* positions, __global const float* _d, __global float* _lambda, const float sr, const float _d0, const int resX, const int resY) {
    int  i = get_global_id(0);
    float c_i = _d[i] / _d0 - 1;
    _lambda[i] = -c_i;
    

    float sumnormgradCi =  0.f;

    float x_i = positions[2 * i];
    float y_i = positions[ 2* i +1];
    


    const int gi_from = (int)(x_i - sr);
    const int gi_to = (int)(x_i + sr) + 1;
    const int gj_from = (int)(y_i - sr);
    const int gj_to = (int)(y_i + sr) + 1;


    for (int gj = max(0, gj_from); gj < min(resY, gj_to); ++gj) {
        for (int gi = max(0, gi_from); gi < min(resX, gi_to); ++gi) {
            const int gidx = gi + gj * resX;

            int neighbour_size = 0;
            if (gidx == 0) {
                neighbour_size = indexes_neighbours[0];
            }
            else {
                neighbour_size = indexes_neighbours[gidx] - indexes_neighbours[gidx - 1];
            }

            for (size_t ni = 0; ni < neighbour_size; ++ni) {

                int j = 0;
                if (gidx == 0) {
                    j = neighbours[ni];
                }
                else{
                    j = neighbours[indexes_neighbours[gidx - 1] + ni];
                }
                
                if (i == j) continue;
                float2 gradCi = computeGradCi(i, j, sr, positions, neighbours,  indexes_neighbours,  _d0, resX, resY, sr);

                sumnormgradCi += gradCi.x * gradCi.x + gradCi.y * gradCi.y;

            }
        }
    }
    

    _lambda[i] = _lambda[i] / (sumnormgradCi + SPH_EPSILON());



}


__kernel void computeDp(__global const int* neighbours, __global const int* indexes_neighbours, __global const float* positions, __global const int* _type, __global  const float* _lambda,  __global float2* _dp, const float _d0, const float _h, const int resX, const int resY, const int debug) {
                float sr = _h;
                int  i = get_global_id(0);
                if (_type[i] != 1) return;
                float dq = 0.3f; 
                float2 sum_grad_p = (float2)(0.f, 0.f);
                const float2 xi =(float2) (positions[2 * i], positions[2 * i + 1] );

                


                const int gi_from = (int)(xi.x - sr);
                const int gi_to =(int)(xi.x + sr) + 1;
                const int gj_from = (int)(xi.y - sr);
                const int gj_to = (int)(xi.y + sr) + 1;

                for (int gj = max(0, gj_from); gj < min(resY, gj_to); ++gj) {
                    for (int gi = max(0, gi_from); gi < min(resX, gi_to); ++gi) {
                        const int gidx = gi + gj * resX;
                        int neighbour_size = 0;
                        if (gidx == 0) {
                            neighbour_size = indexes_neighbours[0];
                        }
                        else {
                            neighbour_size = indexes_neighbours[gidx] - indexes_neighbours[gidx - 1];
                        }
                        

                        for (size_t ni = 0; ni < neighbour_size; ++ni) {
                            int j = 0;
                            if (gidx == 0) {
                                j = neighbours[ni];
                            }
                            else{
                                j = neighbours[indexes_neighbours[gidx - 1] + ni];
                            }
                            if (_type[j] != 1) continue;
                            const float2 xj = (float2) (positions[2 * j], positions[2 * j + 1] );
                            const float2 xij = xi - xj;
                            
                        
                            float scorr = 0; //-0.001f * pow(w(plength(xij.x, xij.y), _h) / w(dq, _h), 4);
                            float2 debug_vec  = (_lambda[i]+ _lambda[j] + scorr)*grad_w(xij, _h);
                            float len_xij = length(xij);
                             
                            sum_grad_p += (_lambda[i]+ _lambda[j] + scorr)*grad_w(xij, _h) ;
                            if(debug == 1)
                            {

                                if(length(grad_w(xij, _h)) != 0)
                                {
                                   printf(" retour x %f y %f \n", debug_vec.x, debug_vec.y);
                                    
                                }
                            
                        }

                        }
                    }
                }
                _dp[i] = sum_grad_p / _d0;  
              


}


__kernel void updatePrediction(__global const float2* _dp,__global const int* _type, __global float* positions) {
    //WARNING positions have to be CL_MEM_READ_WRITE or duplicate entries?
    int  i = get_global_id(0);
    if (_type[i] != 1) return;
    positions[ 2 * i] += _dp[i].x;
    positions[ 2 * i + 1] += _dp[i].y;


}


__kernel void updateVelocity(__global const float* positions,__global const float* old_positions, __global const int* _type, __global float* update_vel, float const dt) {
    int  i = get_global_id(0);

    
    
    if(_type[i] == 1){
        update_vel[2 * i] = (positions[2 * i] - old_positions[2 * i]) / dt;
        update_vel[2 * i + 1] = (positions[2 * i + 1] - old_positions[2 * i + 1]) / dt;
    }
    else{
        update_vel[2 * i] = 0;
        update_vel [2 * i +1 ] = 0;
    }

    
}

__kernel void compute_w_i(__global const int* neighbours, __global const int* indexes_neighbours , __global const float* positions, __global const float* _vel, __global float* w_i_field,  const float sr, const int resX, const int resY){
    //Explanation, since we are in 2D wi has only a componant with regard to the 'z axis'
    
    int  i = get_global_id(0);
    float2 xi = (float2) (positions[2 * i], positions[2 * i + 1]);
    float2 vi = (float2) (_vel[2 * i], _vel[2 * i + 1]);

    float result = 0.f;

    const int gi_from = (int)(xi.x - sr);
    const int gi_to =(int)(xi.x + sr) + 1;
    const int gj_from = (int)(xi.y - sr);
    const int gj_to = (int)(xi.y + sr) + 1;

    for (int gj = max(0, gj_from); gj < min(resY, gj_to); ++gj) {
            for (int gi = max(0, gi_from); gi < min(resX, gi_to); ++gi) {
                const int gidx = gi + gj * resX;

                int neighbour_size = 0;
                if (gidx == 0) {
                    neighbour_size = indexes_neighbours[0];
                }
                else {
                    neighbour_size = indexes_neighbours[gidx] - indexes_neighbours[gidx - 1];
                }

                for (size_t ni = 0; ni < neighbour_size; ++ni) {
                    int j;

                    if (gidx == 0) {
                        j = neighbours[ni];
                    }
                    else{
                        j = neighbours[indexes_neighbours[gidx - 1] + ni];
                    }

                    if (j == i) continue;
                    
                    const float2 xj = (float2) (positions[2 * j], positions[2 * j + 1] );                    
                    const float2 xij = xi - xj;
                    const float len_xij = plength(xij.x, xij.y);
                    if (len_xij > sr) continue;
                    float2 vj = (float2) (_vel[2 * j], _vel[2 * j + 1] );
                    float2 vij = vj - vi;
                    result += crossProduct(vij, grad_w(xij, sr)); 
                   
                }
            }
        }
    w_i_field[i] = result;
   
    
}

float2 ComputeEta(__global const int* neighbours, __global const int* indexes_neighbours ,const int i,__global const float* positions, __global const float* w_i_field,  const float sr, const int resX, const int resY){
    float2 result = (float2) (0.f,0.f);

    const float2 xi = (float2) (positions[2 * i], positions[2 * i + 1]);
    const float wi = w_i_field[i];
    const float abs_wi = pabs(wi)  ;    

    const int gi_from = (int)(xi.x - sr);
    const int gi_to =(int)(xi.x + sr) + 1;
    const int gj_from = (int)(xi.y - sr);
    const int gj_to = (int)(xi.y + sr) + 1;

    for (int gj = max(0, gj_from); gj < min(resY, gj_to); ++gj) {
            for (int gi = max(0, gi_from); gi < min(resX, gi_to); ++gi) {
                const int gidx = gi + gj * resX;

                int neighbour_size = 0;
                if (gidx == 0) {
                    neighbour_size = indexes_neighbours[0];
                }
                else {
                    neighbour_size = indexes_neighbours[gidx] - indexes_neighbours[gidx - 1];
                }

                for (size_t ni = 0; ni < neighbour_size; ++ni) {
                    int j;

                    if (gidx == 0) {
                        j = neighbours[ni];
                    }
                    else{
                        j = neighbours[indexes_neighbours[gidx - 1] + ni];
                    }

                    if (j == i) continue;
                    
                    const float2 xj = (float2) (positions[2 * j], positions[2 * j + 1] ); 
                    const float wj = w_i_field[j];
                    const float abs_wj = pabs(wj)  ;                 
                    const float2 xji = xj - xi;
                    const float len_xji = plength(xji.x, xji.y);
                    if (len_xji > sr) continue;
                    result += (abs_wj - abs_wi) * grad_w(xji, sr); 
                   
                }
            }
        }
    
    return result;
}
__kernel void computeVorticity(__global const int* neighbours, __global const int* indexes_neighbours ,__global const float* positions, __global float* _vel, __global const float * w_i_field,float const _dt, const float _m0, const float sr, const int resX, const int resY) {

    int  i = get_global_id(0);

    float w_i = w_i_field[i]; //compute_w_i(i,neighbours, indexes_neighbours, positions, _vel, w_i_field, sr, resX, resY);

    

    float2 N = ComputeEta(neighbours, indexes_neighbours,  i,positions,w_i_field,  sr, resX, resY);

    N = normalize(N);
    float2 twoD_cross = (float2) (N.y * w_i, - N.x * w_i);
    float2 vi = _dt /  _m0 * 1.f * twoD_cross;
    if(vi.x != 0){
     //   printf("v_i %f, %f \n", vi.x, vi.y);
    }



    _vel[2 * i] += vi.x;


    _vel[2*i + 1] += vi.y;

}

__kernel void applyViscousForce(__global const int* neighbours, __global const int* indexes_neighbours ,__global const float* positions,__global const int* _type, __global float* update_vel, float const sr, int const resX, int const resY) {
    
    int  i = get_global_id(0);
    
    bool test = _type[i] != 1;
    float c =0.001f;
    if (_type[i] != 1) return;
    float2 sum_acc = (float2) (0, 0);
    const float2 xi =(float2) (positions[2 * i], positions[2 * i + 1] );

    const float2 vi = (float2) (update_vel[2*i], update_vel[2*i +1]);


    const int gi_from = (int)(xi.x - sr);
    const int gi_to =(int)(xi.x + sr) + 1;
    const int gj_from = (int)(xi.y - sr);
    const int gj_to = (int)(xi.y + sr) + 1;

            



    for (int gj = max(0, gj_from); gj < min(resY, gj_to); ++gj) {

        for (int gi = max(0, gi_from); gi < min(resX, gi_to); ++gi) {

            const int gidx = gi + gj * resX;

            int neighbour_size = 0;
            if (gidx == 0) {
                neighbour_size = indexes_neighbours[0];
            }
            else {
                neighbour_size = indexes_neighbours[gidx] - indexes_neighbours[gidx - 1];
            }



            for (size_t ni = 0; ni < neighbour_size; ++ni) {

                int j;

                if (gidx == 0) {
                    j = neighbours[ni];
                }
                else{
                    j = neighbours[indexes_neighbours[gidx - 1] + ni];
                }



                if (i == j) continue;
                const float2 xj = (float2) (positions[2 * j], positions[2 * j + 1] );
                const float2 xij = xi - xj;
                const float2 vj = (float2) (update_vel[2*j], update_vel[2*j +1]);

                const float2 vij = vj - vi;
                const float len_xij = plength(xij.x, xij.y);
                if (len_xij > sr) continue;

                sum_acc += 
                    vij* w(len_xij, sr);
                
            }
        }
    }
    float2 result = c * sum_acc;
    printf("result : %f", result);



    update_vel[2 * i] += result.x;
    update_vel[2 * i + 1] += result.y;

}

