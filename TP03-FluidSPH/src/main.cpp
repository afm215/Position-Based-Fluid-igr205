// ----------------------------------------------------------------------------
// main.cpp
//
//  Created on: Fri Jan 22 20:45:07 2021
//      Author: Kiwon Um
//        Mail: um.kiwon@gmail.com
//
// Description: SPH simulator (DO NOT DISTRIBUTE!)
//
// Copyright 2021 Kiwon Um
//
// The copyright to the computer program(s) herein is the property of Kiwon Um.
// The program(s) may be used and/or copied only with the written permission of
// Kiwon Um or in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
// ----------------------------------------------------------------------------
#define CLOCK_REALTIME 0

#define debugGpu 1
#include <assert.h> 
#define NOMINMAX //reset namespace of windows.h


bool debug = false;

#include "CLSrc/gpuenv.h"

#include "CLHeader/clcomputing.h"

#define SPH_EPSILON 200.0f


#define _USE_MATH_DEFINES

#include <GLFW/glfw3.h>


#include "kernel.hpp"

#ifndef M_PI
#define M_PI 3.141592
#endif

#define NB_IT 10


#include "Vector.hpp"

#include <CL/cl.h>
#include <CL/cl_ext.h>

double inf = std::numeric_limits<double>::infinity();


GpuEnvironnment env; 

// window parameters
GLFWwindow* gWindow = nullptr;
int gWindowWidth = 1024;
int gWindowHeight = 768;

float MAX_X = 160;
float MAX_Y = 160;
float MIN_X = 1;
float MIN_Y = 1;
float WALL_X = MAX_X;

// timer
float gAppTimer = 0.0;
float gAppTimerLastClockTime;
bool gAppTimerStoppedP = true;

// global options
bool gPause = true;
bool gSaveFile = false;
bool gShowGrid = true;
bool gShowVel = false;
int gSavedCnt = 0;

const int kViewScale = 15;

Real _dt;                     // time step


// SPH Kernel function: cubic spline
class CubicSpline {
public:
    explicit CubicSpline(const Real h = 3) : _dim(2)
    {
        setSmoothingLen(h);
    }
    void setSmoothingLen(const Real h)
    {
        const Real h2 = square(h), h3 = h2 * h;
        _h = h;
        _sr = h;
        _c[0] = 2e0 / (3e0 * h);
        _c[1] = 10e0 / (7e0 * M_PI * h2);
        _c[2] = 1e0 / (M_PI * h3);
        _gc[0] = _c[0] / h;
        _gc[1] = _c[1] / h;
        _gc[2] = _c[2] / h;
    }
    Real smoothingLen() const { return _h; }
    Real supportRadius() const { return /*_sr warning after checking the Poly6 code radius is here _h */ _h; }

    Real f(const Real l) const
    {
        const Real q = l / _h;
        if (q < 1e0) return _c[_dim - 1] * (1e0 - 1.5 * square(q) + 0.75 * cube(q));
        else if (q < 2e0) return _c[_dim - 1] * (0.25 * cube(2e0 - q));
        return 0;
    }
    Real derivative_f(const Real l) const
    {
        const Real q = l / _h;
        if (q <= 1e0) return _gc[_dim - 1] * (-3e0 * q + 2.25 * square(q));
        else if (q < 2e0) return -_gc[_dim - 1] * 0.75 * square(2e0 - q);
        return 0;
    }

    Real w(const Vec2f& rij) const { return Poly6Value(rij.length(), _h); }
    Real w(const Real rij) const { return Poly6Value(rij, _h); }
    Vec2f grad_w(const Vec2f& rij) const { return grad_w(rij, rij.length()); }
    Vec2f grad_w(const Vec2f& rij, const Real len) const
    {
        return SpikyGradient(rij, _h);
    }

private:
    unsigned int _dim;
    Real _h, _sr, _c[3], _gc[3];
};

class SphSolver {
public:
    explicit SphSolver(
        const Real nu = 0.08, const Real h = 1, const Real density = 1000,
        const Vec2f g = Vec2f(0, -9.8), const Real eta = 0.01, const Real gamma = 7.0) :
        _kernel(h), _nu(nu), _h(h), _d0(density),
        _g(g), _eta(eta), _gamma(gamma)
    {
        _dt = 0.02;
        _m0 = 1;
        _c = std::fabs(_g.y) / _eta;
        _p0 = _d0 * _c * _c / _gamma;     // k of EOS
    }

    // assume an arbitrary grid with the size of res_x*res_y; a fluid mass fill up
    // the size of f_width, f_height; each cell is sampled with 2x2 particles.
    void initScene(
        const int res_x, const int res_y, const int f_width, const int f_height)
    {
        _pos.clear();
        _pred_pos.clear();

        _resX = res_x;
        _resY = res_y;

        // set wall for boundary
        _l = 0.5 * _h;
        _r = static_cast<Real>(res_x) - 0.5 * _h;
        _b = 0.5 * _h;
        _t = static_cast<Real>(res_y) - 0.5 * _h;

        // sample a fluid mass
        for (int j = 0; j < f_height; ++j) {
            for (int i = 0; i < f_width; ++i) {
                if (i == 0 || j == 0) continue;
                // offset
                int I = i + 1;
                int J = j + 20;
                _pos.push_back(Vec2f(I + 0.25, J + 0.25));
                //_pos.push_back(Vec2f(I + 0.75, J + 0.75));
                _pred_pos.push_back(Vec2f(I + 0.25, J + 0.25));
                //_pred_pos.push_back(Vec2f(I + 0.75, J + 0.75));
               // _type.push_back(1);     // fluid
                _type.push_back(1);
            }
        }


        for (int j = 0; j < res_y; ++j) {
            for (int i = 0; i < res_x; ++i) {
                if (j == 0 && i != 0) {
                    _pos.push_back(Vec2f(i, j + 0.8));
                    _pred_pos.push_back(Vec2f(i, j + 0.8));
                    _type.push_back(0);   // solid
                    _pos.push_back(Vec2f(i + 0.5, j + 0.8));
                    _pred_pos.push_back(Vec2f(i + 0.5, j + 0.8));
                    _type.push_back(0);   // solid
                }
                if (i == 0 && j != 0) {
                    _pos.push_back(Vec2f(i + 0.8, j));
                    _pred_pos.push_back(Vec2f(i + 0.8, j));
                    _type.push_back(0);   // solid
                    _pos.push_back(Vec2f(i + 0.8, j + 0.5));
                    _pred_pos.push_back(Vec2f(i + 0.8, j + 0.5));
                    _type.push_back(0);   // solid
                }
            }
        }

        //solid bars
       /*
       int br=0;
       int j = 7;
       for (int i = 8; i <= 22; ++i) {
           if (i == 22) { j ++; i = 21; br = 1; }
           _pos.push_back(Vec2f(i + 0.25, j + 0.25));
           _pos.push_back(Vec2f(i + 0.75, j + 0.25));
           _pos.push_back(Vec2f(i + 0.25, j + 0.75));
           _pos.push_back(Vec2f(i + 0.75, j + 0.75));
           _pos.push_back(Vec2f(i + 0.5, j + 0.5));
           _pred_pos.push_back(Vec2f(i + 0.25, j + 0.25));
           _pred_pos.push_back(Vec2f(i + 0.75, j + 0.25));
           _pred_pos.push_back(Vec2f(i + 0.25, j + 0.75));
           _pred_pos.push_back(Vec2f(i + 0.75, j + 0.75));
           _pred_pos.push_back(Vec2f(i + 0.5, j + 0.5));
           _type.push_back(0);   // solid
           _type.push_back(0);
           _type.push_back(0);
           _type.push_back(0);
           _type.push_back(0);
           if (br) break;
       }
       */

        _type_data = _type.data();
        
        

        // make sure for the other particle quantities
        _vel = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        flatten_vel = (float*)malloc(2 * _pos.size() * sizeof(float));
        flatten_pos = (float*)malloc(2 * _pos.size() * sizeof(float));
        flatten_pred_pos = (float*)malloc(2 * _pos.size() * sizeof(float));

        _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        _p = std::vector<Real>(_pos.size(), 0);
        _d = std::vector<Real>(_pos.size(), 0);
        _dp = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        _lambda = std::vector<Real>(_pos.size(), 0);
        _w_i_field = std::vector<Real>(_pos.size(), 0);

        _col = std::vector<float>(_pos.size() * 4, 1.0); // RGBA
        _vln = std::vector<float>(_pos.size() * 4, 0.0); // GL_LINES

        updateColor();
    }

    void update()
    {
        std::cout << '.' << std::flush;

        // PBF :
        //apply forces v_i <= v_i + Dt * fext(xi) in this case gravity
        applyBodyForce();
        //predict position p_i* <= p_i + Dt* v_i
        predictPosition();
       // resolveCollision();
        //compute the new neighbours using the predicted 
        applyPhysicalConstraints();
        buildNeighbor();


#ifdef debugGpu 
     
    
        int  index_size = _pidxInGrid.size();
        std::vector<int> indexes;
        std::vector<int> flattenN;
        int end_index = 0;
        for (int i = 0; i < index_size; i++) {
            end_index += _pidxInGrid[i].size();
            indexes.push_back(end_index);

            for (int j = 0; j < _pidxInGrid[i].size(); j++) {
                flattenN.push_back(_pidxInGrid[i][j]);

            }
        }
        const int cl_pGrid_Size = flattenN.size();
        const int cl_index_size = indexes.size();
        int* cl_flatten = flattenN.data();
        int* cl_indexes = indexes.data();


        flattenPosVel();




        /*for (int i = 0; i < index_size; i++) {
            for (int j = 0; j < _pidxInGrid[i].size(); j++) {
                if (i == 0)
                {
                    std::cout << flattenN[j] << std::endl;
                }
                else {
                    std::cout << flattenN[indexes[i - 1] + j] << std::endl;
                    std::cout << cl_flatten[indexes[i - 1] + j] << std::endl;
                    assert(flattenN[indexes[i - 1] + j] == _pidxInGrid[i][j]);
                }


            }
        }*/


        //opencl doesn't override apprently the values so you must define an output vector
        float* output[2];
        output[0] = (float*)malloc(sizeof(float) * 2 * _pos.size());
        output[1] = (float*)malloc(sizeof(float) * 2 * _pos.size());

        gpu_handle(env, NB_IT, flatten_pos, flatten_pred_pos, flatten_vel, _type_data, _pos.size(), cl_flatten, cl_pGrid_Size, cl_index_size, cl_indexes, output[0], output[1], resX(), resY(), _h, _m0, _d0, _dt, MIN_X, MIN_Y, WALL_X, MAX_Y,  debug);

        free(flatten_pred_pos);
        flatten_pred_pos = output[0];

        free(flatten_vel);
        flatten_vel = output[1];



#endif
        

#ifndef debugGpu



        int i = 0;
        


        
       while (i < NB_IT)
        {   
            
            //compute lambda_i 
            //computeLambda use the comutegradCi function wich use the density of the paricles 
            computeDensity();
            computeLambda();
            //calculate the difference in positions using the lamba_i
            computeDp();
            //update the position p_i* = p_i* + dp_i
            updatePrediction();
            applyPhysicalConstraints();

            i++;
       }
        ////update the velocities v_i = p_i* - p_i 
        updateVelocity();
        compute_w_i();
        computeVorticity();
        applyViscousForce();
        // use the newly computed velocities to compute vorticity confinement and XSPH viscosity TO DO !!!
        
#endif // !debugGpu


# ifdef debugGpu)
            updatePosVelFromFlat();

#endif
        //no need to free the flatten indexes as everything is passed by reference;
        

#ifndef debugGpu
            updatePosition();


#endif // !debugGpu

        //modify the position p_i = p_i *



        updateColor();
        if (gShowVel) updateVelLine();


        /*
        * // PSEUDO CODE :
        * 
        for all particles i do
            apply forces vi <= vi + Dtfext(xi)
            predict position x*i <= xi + Dtvi
        end for
        for all particles i do
            find neighboring particles Ni(x*i)
        end for
        while iter < solverIterations do
            for all particles i do
                calculate li
            end for
            for all particles i do
                calculate Dpi
                perform collision detection and response
            end for
            for all particles i do
                update position x*i <= x*i + Dpi
            end for
        end while
        for all particles i do
            update velocity vi <= 1/Dt(x*i - xi)
            apply vorticity confinement and XSPH viscosity
            update position xi <= x*i
        end for
        */


    }

    void flattenPosVel(){
        int array_size = _pos.size();

        #pragma omp parallel for
        for (int i = 0; i < array_size; i++) {
            flatten_pos[2 * i] = _pos[i].x;
            flatten_pos[2 * i + 1] = _pos[i].y;

            flatten_vel[2 * i] = _vel[i].x;
            flatten_vel[2 * i + 1] = _vel[i].y;

            flatten_pred_pos[2 * i] = _pred_pos[i].x;
            flatten_pred_pos[2 * i + 1] = _pred_pos[i].y;

        }

    
    }

    void updatePosVelFromFlat() {
        
        int array_size = _pos.size();
        #pragma omp parallel for
        for (int i = 0; i < array_size; i++) {
            //normally only pre_pos is updated not pos
            _pos[i].x = flatten_pred_pos[2 * i];
            _pos[i].y = flatten_pred_pos[2 * i + 1]; 

            _pred_pos[i].x = flatten_pred_pos[2 * i];
            _pred_pos[i].y = flatten_pred_pos[2 * i + 1];

            _vel[i].x = flatten_vel[2 * i];
            _vel[i].y = flatten_vel[2 * i + 1];
        }

    }



    tIndex particleCount() const { return _pos.size(); }
    const Vec2f& position(const tIndex i) const { return _pred_pos[i]; }
    const float& color(const tIndex i) const { return _col[i]; }
    const float& vline(const tIndex i) const { return _vln[i]; }

    int resX() const { return _resX; }
    int resY() const { return _resY; }

    Real equationOfState(
        const Real d, const Real d0, const Real k,
        const Real gamma = 7.0)
    {
        return k * (std::pow(d / d0, gamma) - 1.0);
    }

private:

    void applyBodyForce()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {

            if (_type[i] == 1) {
                _vel[i] += _dt * _g;
                //assert(!isnan(_vel[i].x) && !isnan(_vel[i].y));
            }
            else {
                _vel[i] = Vec2f(0);
            }

            // simple forward Euler
        }
    }

    void predictPosition()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_type[i] != 1) continue;
            _pred_pos[i] = _pos[i] + _dt * _vel[i];   // simple forward Euler


        }
    }

    void applyPhysicalConstraints()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i)
        {
            if (_type[i] == 1)
            {
                //float randF = (rand()+1) / 10000;
                float rebound = 0.9;
                Vec2f pos = _pred_pos[i];
                if (pos.x < MIN_X) { _pred_pos[i].x = MIN_X + abs(MIN_X - _pred_pos[i].x); }
                if (pos.x > WALL_X) { _pred_pos[i].x = WALL_X - abs(WALL_X - _pred_pos[i].x); }
                if (pos.y < MIN_Y) { _pred_pos[i].y = MIN_Y + abs(MIN_Y - _pred_pos[i].y); }
                if (pos.y > MAX_Y) { _pred_pos[i].y = MAX_Y - abs(MAX_Y - _pred_pos[i].y); }
            }
        }
    }


    void buildNeighbor()
    {
        // particle indices in each cell
        std::vector< std::vector<tIndex> > pidx_in_grid(resX() * resY());

        for (tIndex k = 0; k < particleCount(); ++k) {
            const Vec2f& p = position(k);
            int i = static_cast<int>(p.x), j = static_cast<int>(p.y);
            
            
            const int indice = idx1d(i, j);
            pidx_in_grid[indice].push_back(k);
        }

        _pidxInGrid.swap(pidx_in_grid);
    }


    void computeDensity()
    {


        const Real sr =_h;
        int nb_null = 0;
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            Real sum_m = 0;
            const Vec2f& xi = position(i);

            const int gi_from = static_cast<int>(xi.x - sr);
            const int gi_to = static_cast<int>(xi.x + sr) + 1;
            const int gj_from = static_cast<int>(xi.y - sr);
            const int gj_to = static_cast<int>(xi.y + sr) + 1;

            for (int gj = std::max(0, gj_from); gj < std::min(resY(), gj_to); ++gj) {
                for (int gi = std::max(0, gi_from); gi < std::min(resX(), gi_to); ++gi) {
                    const tIndex gidx = idx1d(gi, gj);

                    // each particle in nearby cells
                    for (size_t ni = 0; ni < _pidxInGrid[gidx].size(); ++ni) {
                        tIndex j = _pidxInGrid[gidx][ni];
                        if (i == j) continue;
                        const Vec2f& xj = position(j);
                        const Vec2f xij = xi - xj;
                        
                        const Real len_xij = xij.length();
                        if (len_xij == 0) {
                            nb_null++;
                        }
                        float value = 1;
                        sum_m +=  _m0 * _kernel.w(xij) ;
                        
                    }
                }
            }

            _d[i] = sum_m;
        }

    }



//    void computePressure()
//    {
//#pragma omp parallel for
//        for (tIndex i = 0; i < particleCount(); ++i) {
//            _p[i] = std::max(equationOfState(_d[i], _d0, _p0, _gamma), Real(0.0));
//        }
//    }

    Vec2f computeGradCi(int i, int k) {


        const Vec2f& xi = position(i);
        const Real sr = _h;
        Vec2f result = Vec2f(0, 0);

        Vec2f p_i = position(i);

      
        if (k == i) {

            const int gi_from = static_cast<int>(p_i.x - sr);
            const int gi_to = static_cast<int>(p_i.x + sr) + 1;
            const int gj_from = static_cast<int>(p_i.y - sr);
            const int gj_to = static_cast<int>(p_i.y + sr) + 1;


            for (int gj = std::max(0, gj_from); gj < std::min(resY(), gj_to); ++gj) {
                for (int gi = std::max(0, gi_from); gi < std::min(resX(), gi_to); ++gi) {
                    const tIndex gidx = idx1d(gi, gj);

                    for (size_t ni = 0; ni < _pidxInGrid[gidx].size(); ++ni) {

                        const tIndex j = _pidxInGrid[gidx][ni];
                        if (j == i) continue;
                        const Vec2f& xj = position(j);
                        const Vec2f xij = xi - xj;
                        const Real len_xij = xij.length();
                        if (len_xij > sr) continue;

                        result += 1 / _d0 * _kernel.grad_w(xij);

                    }
                }
            }

            return result;
        }
    
        else{
            const Vec2f& xk = position(k);
            const Vec2f xik = xi - xk;
            result -= 1 / _d0 * _kernel.grad_w(xik);
            return result;
        }
    }

    

    void computeLambda()
    {
        const Real sr = _h;
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {


            Real c_i = _d[i] / _d0 - 1;
            _lambda[i] = -c_i;
            

            Real sumnormgradCi = 0;

            Vec2f p_i = position(i);
            


            const int gi_from = static_cast<int>(p_i.x - sr);
            const int gi_to = static_cast<int>(p_i.x + sr) + 1;
            const int gj_from = static_cast<int>(p_i.y - sr);
            const int gj_to = static_cast<int>(p_i.y + sr) + 1;


            for (int gj = std::max(0, gj_from); gj < std::min(resY(), gj_to); ++gj) {
                for (int gi = std::max(0, gi_from); gi < std::min(resX(), gi_to); ++gi) {
                    const tIndex gidx = idx1d(gi, gj);

                    for (size_t ni = 0; ni < _pidxInGrid[gidx].size(); ++ni) {

                        tIndex j = _pidxInGrid[gidx][ni];
                        /*auto j = _pidxInGrid[gidx][ni];
                        Vec2f xj = position(j);
                        Vec2f xij = p_i - xj;
                        Vec2f gradCi = _kernel.grad_w(xij) / _d0;
                        
                        grad_sum += gradCi;*/
                        if (i == j) continue;
                        Vec2f gradCi = computeGradCi(i, _pidxInGrid[gidx][ni]);

                        sumnormgradCi += gradCi.dotProduct(gradCi);

                    }
                }
            }
            
            //sumnormgradCi += grad_sum.dotProduct(grad_sum);

            _lambda[i] /= (sumnormgradCi + SPH_EPSILON);


         
        }
    }

    

    void computeDp()
    {
        
        const Real sr = _h;
        Real dq = 0.3 ;

#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_type[i] != 1) continue;
            Vec2f sum_grad_p(0, 0);
            const Vec2f& xi = position(i);

            


            const int gi_from = static_cast<int>(xi.x - sr);
            const int gi_to = static_cast<int>(xi.x + sr) + 1;
            const int gj_from = static_cast<int>(xi.y - sr);
            const int gj_to = static_cast<int>(xi.y + sr) + 1;

            for (int gj = std::max(0, gj_from); gj < std::min(resY(), gj_to); ++gj) {
                for (int gi = std::max(0, gi_from); gi < std::min(resX(), gi_to); ++gi) {
                    const tIndex gidx = idx1d(gi, gj);

                    // each particle in nearby cells
                    for (size_t ni = 0; ni < _pidxInGrid[gidx].size(); ++ni) {
                        const tIndex j = _pidxInGrid[gidx][ni];
                        if (i == j) continue;
                        //if (_type[j] != 1) continue;
                        const Vec2f& xj = position(j);
                        const Vec2f xij = xi - xj;
                       
                        Real scorr = 0;//-0.001f * pow(_kernel.w(xij) / _kernel.w(dq), 4);
                        sum_grad_p += (_lambda[i]+ _lambda[j] + scorr)*_kernel.grad_w(xij);
                    }
                }
            }

            _dp[i] =  sum_grad_p / _d0;   // TODO pas sur du tout changmeent pour debug 
        }
    }

    void updatePrediction()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_type[i] != 1) continue;
            _pred_pos[i] += _dp[i];   // simple forward Euler
        }
    }

    void updateVelocity()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_type[i] == 1) {
                //Vec2f spread = _pred_pos[i] - _pos[i];
                _vel[i] = (_pred_pos[i] - _pos[i]) / _dt;
                //assert(!isnan(_vel[i].x) && !isnan(_vel[i].y));

            }
            else {
                _vel[i] = Vec2f(0);

            }
        }
    }



    void compute_w_i() {
        const Real sr = _h;

#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); i++) {
            const Vec2f xi = position(i);
            Real result = 0;
            const int gi_from = static_cast<int>(xi.x - sr);
            const int gi_to = static_cast<int>(xi.x + sr) + 1;
            const int gj_from = static_cast<int>(xi.y - sr);
            const int gj_to = static_cast<int>(xi.y + sr) + 1;

            for (int gj = std::max(0, gj_from); gj < std::min(resY(), gj_to); ++gj) {
                for (int gi = std::max(0, gi_from); gi < std::min(resX(), gi_to); ++gi) {
                    const tIndex gidx = idx1d(gi, gj);


                    for (size_t ni = 0; ni < _pidxInGrid[gidx].size(); ++ni) {

                        const tIndex j = _pidxInGrid[gidx][ni];
                        if (j == i) continue;
                        const Vec2f& xj = position(j);
                        const Vec2f xij = xi - xj;
                        const Real len_xij = xij.length();
                        if (len_xij > sr) continue;
                        Vec2f vij = _vel[j] - _vel[i];
                        result += vij.crossProduct(_kernel.grad_w(xij));
                        assert(abs(result) != inf);
                        assert(!isnan(result));
                    }
                }
            }
            _w_i_field[i] = result;
        }

    }

    Vec2f ComputeEta(int i) {

        const Real sr = _h;
        const Vec2f xi = position(i);
        const Real abs_wi = abs(_w_i_field[i]);
        const int gi_from = static_cast<int>(xi.x - sr);
        const int gi_to = static_cast<int>(xi.x + sr) + 1;
        const int gj_from = static_cast<int>(xi.y - sr);
        const int gj_to = static_cast<int>(xi.y + sr) + 1;

        Vec2f result = Vec2f(0);

        for (int gj = std::max(0, gj_from); gj < std::min(resY(), gj_to); ++gj) {
            for (int gi = std::max(0, gi_from); gi < std::min(resX(), gi_to); ++gi) {
                const tIndex gidx = idx1d(gi, gj);


                for (size_t ni = 0; ni < _pidxInGrid[gidx].size(); ++ni) {
                    const tIndex j = _pidxInGrid[gidx][ni];
                    const Vec2f xj = position(j);
                    const Real abs_wj = abs(_w_i_field[j]);
                    Vec2f xij = xi - xj;
                    result += (abs_wj - abs_wi) * _kernel.grad_w(xij);
                }
            }
        }

        assert(!isnan(result.x) && !isnan(result.y));
        assert(abs(result.x) != inf && abs(result.y) != inf);

        return result;
    }

    void computeVorticity() {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); i++) {
            Real w_i = _w_i_field[i];
            Vec2f N = ComputeEta(i);

            N = N.normalize();


            Vec2f before = _vel[i];
            Vec2f vi = _dt / _m0 * 1.f * Vec2f(N.y * w_i, -N.x * w_i);//1.f = epsilon vorticity
            _vel[i] += vi;
            assert(!isnan(_vel[i].x) && !isnan(_vel[i].y));
        }
    }

    
    void applyViscousForce()
    {
        const Real sr = _h;
        Real c = 0.001;

#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_type[i] != 1) continue;
            Vec2f sum_acc(0, 0);
            const Vec2f& xi = position(i);

            const int gi_from = static_cast<int>(xi.x - sr);
            const int gi_to = static_cast<int>(xi.x + sr) + 1;
            const int gj_from = static_cast<int>(xi.y - sr);
            const int gj_to = static_cast<int>(xi.y + sr) + 1;

            for (int gj = std::max(0, gj_from); gj < std::min(resY(), gj_to); ++gj) {
                for (int gi = std::max(0, gi_from); gi < std::min(resX(), gi_to); ++gi) {
                    const tIndex gidx = idx1d(gi, gj);

                    // each particle in nearby cells
                    for (size_t ni = 0; ni < _pidxInGrid[gidx].size(); ++ni) {
                        const tIndex j = _pidxInGrid[gidx][ni];
                        if (i == j) continue;
                        const Vec2f& xj = position(j);
                        const Vec2f xij = xi - xj;
                        const Vec2f vij = _vel[j] - _vel[i];
                        const Real len_xij = xij.length();
                        if (len_xij > sr) continue;
                        sum_acc += 
                            vij*_kernel.w(xij);
                            
                    }
                }
            }

            
            _vel[i] += c * sum_acc;

            //assert(!isnan(_vel[i].x) && !isnan(_vel[i].y));

        }
    }

    
    void updatePosition()
    {

        int solverIteration = 10;
        

#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_type[i] != 1) continue;
            _pos[i] = _pred_pos[i];
        }
    }

    

    


    void updateColor()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_type[i] != 1) {
                _col[i * 4 + 0] = 0.8;
                _col[i * 4 + 1] = 0.8;
                _col[i * 4 + 2] = 0.8;
            }
            else {
                _col[i * 4 + 0] = 0.6;
                _col[i * 4 + 1] = 0.6;
                _col[i * 4 + 2] = _d[i] / _d0;
            }
        }
    }

    void updateVelLine()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            _vln[i * 4 + 0] = _pos[i].x;
            _vln[i * 4 + 1] = _pos[i].y;
            _vln[i * 4 + 2] = _pos[i].x + _vel[i].x;
            _vln[i * 4 + 3] = _pos[i].y + _vel[i].y;
        }
    }

    inline tIndex idx1d(const int i, const int j) { return i + j * resX(); }

    CubicSpline _kernel;

    // particle data
    std::vector<int>   _type;
    int* _type_data;// type
    std::vector<Vec2f> _pos;
    float* flatten_pos;// position
    std::vector<Vec2f> _pred_pos;
    std::vector<Real> _w_i_field; //w_i_i field vector colinear to z axis
    float* flatten_pred_pos;// predicted position
    std::vector<Vec2f> _dp;       // position shift
    std::vector<Vec2f> _vel;      // velocity
    float* flatten_vel;
    std::vector<Vec2f> _acc;      // acceleration
    std::vector<Real>  _p;        // pressure
    std::vector<Real>  _lambda;        // density constraint
    std::vector<Real>  _d;        // density

    std::vector< std::vector<tIndex> > _pidxInGrid; // particle neighbor data

    std::vector<float> _col;    // particle color; just for visualization
    std::vector<float> _vln;    // particle velocity lines; just for visualization

    // simulation

    int _resX, _resY;             // background grid resolution

    // wall
    Real _l, _r, _b, _t;          // wall (boundary)

    // SPH coefficients
    Real _nu;                     // viscosity coefficient
    const Real _d0;                     // rest density
    Real _h;                      // particle spacing
    Vec2f _g;                     // gravity

    Real _m0;                     // rest mass
    Real _p0;                     // EOS coefficient

    Real _eta;
    Real _c;                      // speed of sound
    Real _gamma;                  // EOS power factor
};

SphSolver gSolver(80, 1.2, 1, Vec2f(0, -9.8), 0.01, 7.0);
// nu, _h=sr, d0, _g,  eta , gammma
// nu eta and gamma are sph parameters and have no influence here

void printHelp()
{
    std::cout <<
        "> Help:" << std::endl <<
        "    Keyboard commands:" << std::endl <<
        "    * H: print this help" << std::endl <<
        "    * P: toggle simulation" << std::endl <<
        "    * G: toggle grid rendering" << std::endl <<
        "    * V: toggle velocity rendering" << std::endl <<
        "    * S: save current frame into a file" << std::endl <<
        "    * Q: quit the program" << std::endl;
}

// Executed each time the window is resized. Adjust the aspect ratio and the rendering viewport to the current window.
void windowSizeCallback(GLFWwindow* window, int width, int height)
{
    gWindowWidth = width;
    gWindowHeight = height;
    glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

// Executed each time a key is entered.
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS && key == GLFW_KEY_H) {
        printHelp();
    }
    if (action == GLFW_PRESS && key == GLFW_KEY_D) {
        debug = !debug;

    }
    else if (action == GLFW_PRESS && key == GLFW_KEY_S) {
        gSaveFile = !gSaveFile;
    }
    else if (action == GLFW_PRESS && key == GLFW_KEY_G) {
        gShowGrid = !gShowGrid;
    }
    else if (action == GLFW_PRESS && key == GLFW_KEY_V) {
        gShowVel = !gShowVel;
    }
    else if (action == GLFW_PRESS && key == GLFW_KEY_P) {
        gAppTimerStoppedP = !gAppTimerStoppedP;
        if (!gAppTimerStoppedP)
            gAppTimerLastClockTime = static_cast<float>(glfwGetTime());
    }
    else if (action == GLFW_PRESS && key == GLFW_KEY_Q) {
        glfwSetWindowShouldClose(window, true);
    }
}

void initGLFW()
{
    // Initialize GLFW, the library responsible for window management
    if (!glfwInit()) {
        std::cerr << "ERROR: Failed to init GLFW" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Before creating the window, set some option flags
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // only if requesting 3.0 or above
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE); // for OpenGL below 3.2
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

    // Create the window
    gWindowWidth = gSolver.resX() * kViewScale;
    gWindowHeight = gSolver.resY() * kViewScale;
    gWindow = glfwCreateWindow(
        gSolver.resX() * kViewScale, gSolver.resY() * kViewScale,
        "Basic SPH Simulator", nullptr, nullptr);
    if (!gWindow) {
        std::cerr << "ERROR: Failed to open window" << std::endl;
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    // Load the OpenGL context in the GLFW window
    glfwMakeContextCurrent(gWindow);

    // not mandatory for all, but MacOS X
    glfwGetFramebufferSize(gWindow, &gWindowWidth, &gWindowHeight);

    // Connect the callbacks for interactive control
    glfwSetWindowSizeCallback(gWindow, windowSizeCallback);
    glfwSetKeyCallback(gWindow, keyCallback);

    std::cout << "Window created: " <<
        gWindowWidth << ", " << gWindowHeight << std::endl;
#pragma omp parallel
    {
        std::cout << "test" << std::endl;;
    }
}

void clear();

void initOpenGL()
{
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

void init()
{
    gSolver.initScene(MAX_X + 1, MAX_Y + 1, MAX_X / 2 , MAX_Y / 2);



    initGLFW();                   // Windowing system
    initOpenGL();
}

void clear()
{
    glfwDestroyWindow(gWindow);
    glfwTerminate();
}

// The main rendering call
void render()
{
    glClearColor(.4f, .4f, .4f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // grid guides
    glBegin(GL_LINES);
    if (gShowGrid) {

        for (int i = 1; i < gSolver.resX(); ++i) {
            glColor3f(0.3, 0.3, 0.3);
            glVertex2f(static_cast<Real>(i), 0.0);
            glColor3f(0.3, 0.3, 0.3);
            glVertex2f(static_cast<Real>(i), static_cast<Real>(gSolver.resY()));
        }
        for (int j = 1; j < gSolver.resY(); ++j) {
            glColor3f(0.3, 0.3, 0.3);
            glVertex2f(0.0, static_cast<Real>(j));
            glColor3f(0.3, 0.3, 0.3);
            glVertex2f(static_cast<Real>(gSolver.resX()), static_cast<Real>(j));
        }
    }
    glColor3f(0.3, 0.3, 0.3);
    glLineWidth(20);

    glVertex2f(WALL_X + 0.3, MAX_Y + 1);
    glVertex2f(WALL_X + 0.3, 0);
    glEnd();

    // render particles
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glPointSize(0.25f * kViewScale);

    glColorPointer(4, GL_FLOAT, 0, &gSolver.color(0));
    glVertexPointer(2, GL_FLOAT, 0, &gSolver.position(0));
    glDrawArrays(GL_POINTS, 0, gSolver.particleCount());

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    // velocity
    if (gShowVel) {
        glColor4f(0.0f, 0.0f, 0.5f, 0.2f);

        glEnableClientState(GL_VERTEX_ARRAY);

        glVertexPointer(2, GL_FLOAT, 0, &gSolver.vline(0));
        glDrawArrays(GL_LINES, 0, gSolver.particleCount() * 2);

        glDisableClientState(GL_VERTEX_ARRAY);
    }

    if (gSaveFile) {
        std::stringstream fpath;
        fpath << "s" << std::setw(4) << std::setfill('0') << gSavedCnt++ << ".tga";

        std::cout << "Saving file " << fpath.str() << " ... " << std::flush;
        const short int w = gWindowWidth;
        const short int h = gWindowHeight;
        std::vector<int> buf(w * h * 3, 0);
        glReadPixels(0, 0, w, h, GL_BGR_EXT, GL_UNSIGNED_BYTE, &(buf[0]));

        FILE* out = fopen(fpath.str().c_str(), "wb");
        short TGAhead[] = { 0, 2, 0, 0, 0, 0, w, h, 24 };
        fwrite(&TGAhead, sizeof(TGAhead), 1, out);
        fwrite(&(buf[0]), 3 * w * h, 1, out);
        fclose(out);
        gSaveFile = false;

        std::cout << "Done" << std::endl;
    }
}

// Update any accessible variable based on the current time
void update(const float currentTime)
{
    if (!gAppTimerStoppedP) {
        // Animate any entity of the program here
        const float dt = currentTime - gAppTimerLastClockTime;
        gAppTimerLastClockTime = currentTime;
        gAppTimer += dt;
        // <---- Update here what needs to be animated over time ---->
        timespec timer;
        clock_gettime(CLOCK_REALTIME, &timer);
        double start = timer.tv_nsec * pow(10, -9) + timer.tv_sec;
        // solve 10 steps
        for (int i = 0; i < 1; ++i) gSolver.update();
        clock_gettime(CLOCK_REALTIME, &timer);
        double end = timer.tv_nsec * pow(10, -9) + timer.tv_sec;
        std::cout << "Delay of one update" << end - start << std::endl;
       // _dt = end - start;
        WALL_X = MAX_X - 10 + 10 * sin(end / 4);
        
    }
}

int main(int argc, char** argv)
{
    cl_int test;
    

    /*********************************************Context GPU INIT *****************************************/
    std::cout << "running on GPU" << std::endl;
    
    char char_buffer[STRING_BUFFER_LEN];
    cl_platform_id platform;
    cl_device_id device;

   // unsigned char** opencl_program = read_file("operation.cl");
   
    int status;

    clGetPlatformIDs(1, &platform, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    std::cout << "NAME OF PLATEFORM" << std::endl<<char_buffer << std::endl;
    cl_context_properties context_properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 }; //makes it run on windows
    unsigned char** opencl_program = read_file("../../../src/CLSrc/operation.cl");

    context_properties[1] = (cl_context_properties)platform;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    env.context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
    env.queue = clCreateCommandQueue(env.context, device, 0, NULL);

    env.program = clCreateProgramWithSource(env.context, 1, (const char**)opencl_program, NULL, NULL);
    if (env.program == NULL)
    {
        printf("Program creation failed\n");
        return 1;
    }
    int success = clBuildProgram(env.program, 0, NULL, NULL, NULL, NULL);
    if (success != CL_SUCCESS) print_clbuild_errors(env.program, device);

    env.computeDensity = clCreateKernel(env.program, "computeDensity", &success);
    if (success != CL_SUCCESS) checkError(success, "probleme when loading kernel");


    env.computeLambda = clCreateKernel(env.program, "computeLambda", &success);
    if (success != CL_SUCCESS) checkError(success, "probleme when loading kernel");

    env.computeDp = clCreateKernel(env.program, "computeDp", &success);
    if (success != CL_SUCCESS) checkError(success, "probleme when loading kernel");

    env.updatePrediction = clCreateKernel(env.program, "updatePrediction", &success);
    if (success != CL_SUCCESS) checkError(success, "probleme when loading kernel");

    env.applyPhysicallConstraint = clCreateKernel(env.program, "applyPhysicalConstraints", &success);
    if (success != CL_SUCCESS) checkError(success, "probleme when loading kernel");

    env.updateVelocity = clCreateKernel(env.program, "updateVelocity", &success);
    if (success != CL_SUCCESS) checkError(success, "probleme when loading kernel");

    env.compute_w_i = clCreateKernel(env.program, "compute_w_i", &success);
    if (success != CL_SUCCESS) checkError(success, "probleme when loading kernel");

    env.coputeVorticity = clCreateKernel(env.program, "computeVorticity", &success);
    if (success != CL_SUCCESS) checkError(success, "probleme when loading kernel");

    env.applyViscousForce = clCreateKernel(env.program, "applyViscousForce", &success);
    if (success != CL_SUCCESS) checkError(success, "probleme when loading kernel");

    /*************************************END OF GPU INIT ****************************************************/


    init();
   
    while (!glfwWindowShouldClose(gWindow)) {
        update(static_cast<float>(glfwGetTime()));
        render();
        glfwSwapBuffers(gWindow);
        glfwPollEvents();
    }

    status = clFinish(env.queue);
    checkError(status, "Queue not finished");

    status = clReleaseKernel(env.computeDensity);
    checkError(status, "Failed to release kernel");

    status = clReleaseKernel(env.computeLambda);
    checkError(status, "Failed to release kernel");

    status = clReleaseKernel(env.updatePrediction);
    checkError(status, "Failed to release kernel");

    status = clReleaseKernel(env.updateVelocity);
    checkError(status, "Failed to release kernel");

    status = clReleaseKernel(env.coputeVorticity);
    checkError(status, "Failed to release kernel");

    status = clReleaseKernel(env.applyViscousForce);
    checkError(status, "Failed to release kernel");


    status = clReleaseProgram(env.program);
    checkError(status, "Failed to release program");
    status = clReleaseCommandQueue(env.queue);
    checkError(status, "Failed to release queue");

    status = clReleaseContext(env.context);
    checkError(status, "Failed to release context");


    clear();
    std::cout << " > Quit" << std::endl;
    return EXIT_SUCCESS;
}
