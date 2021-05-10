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

#define _USE_MATH_DEFINES

#include <GLFW/glfw3.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.141592
#endif

#define NB_IT 10

#include "Vector.hpp"

// window parameters
GLFWwindow* gWindow = nullptr;
int gWindowWidth = 1024;
int gWindowHeight = 768;

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

// SPH Kernel function: cubic spline
class CubicSpline {
public:
    explicit CubicSpline(const Real h = 1) : _dim(2)
    {
        setSmoothingLen(h);
    }
    void setSmoothingLen(const Real h)
    {
        const Real h2 = square(h), h3 = h2 * h;
        _h = h;
        _sr = 2e0 * h;
        _c[0] = 2e0 / (3e0 * h);
        _c[1] = 10e0 / (7e0 * M_PI * h2);
        _c[2] = 1e0 / (M_PI * h3);
        _gc[0] = _c[0] / h;
        _gc[1] = _c[1] / h;
        _gc[2] = _c[2] / h;
    }
    Real smoothingLen() const { return _h; }
    Real supportRadius() const { return _sr; }

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

    Real w(const Vec2f& rij) const { return f(rij.length()); }
    Vec2f grad_w(const Vec2f& rij) const { return grad_w(rij, rij.length()); }
    Vec2f grad_w(const Vec2f& rij, const Real len) const
    {
        return derivative_f(len) * rij / len;
    }

private:
    unsigned int _dim;
    Real _h, _sr, _c[3], _gc[3];
};

class SphSolver {
public:
    explicit SphSolver(
        const Real nu = 0.08, const Real h = 0.5, const Real density = 1,
        const Vec2f g = Vec2f(0, -9.8), const Real eta = 0.01, const Real gamma = 7.0) :
        _kernel(h), _nu(nu), _h(h), _d0(density),
        _g(g), _eta(eta), _gamma(gamma)
    {
        _dt = 0.0005;
        _m0 = _d0 * _h * _h;
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
                _pos.push_back(Vec2f(i + 0.25, j + 0.25));
                _pos.push_back(Vec2f(i + 0.75, j + 0.25));
                _pos.push_back(Vec2f(i + 0.25, j + 0.75));
                _pos.push_back(Vec2f(i + 0.75, j + 0.75));
                _pred_pos.push_back(Vec2f(i + 0.25, j + 0.25));
                _pred_pos.push_back(Vec2f(i + 0.75, j + 0.25));
                _pred_pos.push_back(Vec2f(i + 0.25, j + 0.75));
                _pred_pos.push_back(Vec2f(i + 0.75, j + 0.75));
                _type.push_back(1);     // fluid
                _type.push_back(1);
                _type.push_back(1);
                _type.push_back(1);
            }
        }

        // solid
        for (int j = 0; j < res_y; ++j) {
            for (int i = 0; i < res_x; ++i) {
                if (i == 0 || j == 0 || i == res_x - 1 || j == res_y - 1) {
                    _pos.push_back(Vec2f(i + 0.25, j + 0.25));
                    _pos.push_back(Vec2f(i + 0.75, j + 0.25));
                    _pos.push_back(Vec2f(i + 0.25, j + 0.75));
                    _pos.push_back(Vec2f(i + 0.75, j + 0.75));
                    _pred_pos.push_back(Vec2f(i + 0.25, j + 0.25));
                    _pred_pos.push_back(Vec2f(i + 0.75, j + 0.25));
                    _pred_pos.push_back(Vec2f(i + 0.25, j + 0.75));
                    _pred_pos.push_back(Vec2f(i + 0.75, j + 0.75));
                    _type.push_back(0);   // solid
                    _type.push_back(0);
                    _type.push_back(0);
                    _type.push_back(0);
                }
            }
        }

        // make sure for the other particle quantities
        _vel = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        _p = std::vector<Real>(_pos.size(), 0);
        _d = std::vector<Real>(_pos.size(), 0);

        _col = std::vector<float>(_pos.size() * 4, 1.0); // RGBA
        _vln = std::vector<float>(_pos.size() * 4, 0.0); // GL_LINES

        updateColor();
    }

    void update()
    {
        std::cout << '.' << std::flush;

        // PBF :

        applyBodyForce();
        predictPosition();
        buildNeighbor();
        int i = 0;
        while (i < NB_IT)
        {
            computeDensity();
            computeLambda();
            computeDp();
            resolveCollision();
            updatePrediction();
            i++;
        }
        updateVelocity();
        applyViscousForce();
        updatePosition();

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
    void buildNeighbor()
    {
        // particle indices in each cell
        std::vector< std::vector<tIndex> > pidx_in_grid(resX() * resY());

        for (tIndex k = 0; k < particleCount(); ++k) {
            const Vec2f& p = position(k);
            const int i = static_cast<int>(p.x), j = static_cast<int>(p.y);
            pidx_in_grid[idx1d(i, j)].push_back(k);
        }

        _pidxInGrid.swap(pidx_in_grid);
    }

    void computeDensity()
    {
        const Real sr = _kernel.supportRadius();

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
                        const Vec2f& xj = position(_pidxInGrid[gidx][ni]);
                        const Real len_xij = (xi - xj).length();
                        sum_m += (len_xij < sr) ? _m0 * _kernel.f(len_xij) : 0;
                    }
                }
            }

            _d[i] = sum_m;
        }
    }



    void computePressure()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            _p[i] = std::max(equationOfState(_d[i], _d0, _p0, _gamma), Real(0.0));
        }
    }

    Vec2f computeGradCi(int i, int k) {


        const Vec2f& xi = position(i);
        const Real sr = _kernel.supportRadius();
        Vec2f result = Vec2f(0, 0);

        Vec2f p_i = position(i);
        int gx = static_cast<int>(p_i.x);
        int gy = static_cast<int>(p_i.y);

        int gidx = idx1d(gx, gy);

        if (k == i) {

            for (size_t ni = 0; ni < _pidxInGrid[gidx].size(); ++ni) {

                const tIndex j = _pidxInGrid[gidx][ni];
                if (j == i) continue;
                const Vec2f& xj = position(j);
                const Vec2f xij = xi - xj;
                const Real len_xij = xij.length();
                if (len_xij > sr) continue;
                
                result += 1 / _d0 * _kernel.grad_w(xij);

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

    Vec2f compute_w_i(int i){
        const Vec2f& xi = position(i);
        const Real sr = _kernel.supportRadius();
        Vec2f result = Vec2f(0, 0);

        

        for (size_t ni = 0; ni < _pidxInGrid[i].size(); ++ni) {

            const tIndex j = _pidxInGrid[i][ni];
            const Vec2f& xj = position(j);
            const Vec2f xij = xi - xj;
            const Real len_xij = xij.length();
            if (len_xij > sr) continue;
            Vec2f vij = _vel[j] - _vel[i];
            result -=  vij * _kernel.grad_w(xij); // WARNING wait to be sure (grad-pj = - grad ??)

        }

        return result;
        
    }


    void computeLambda()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {


            Real c_i = _d[i] / _d0 - 1;
            _lambda.push_back(c_i);

            Real sumnormgradCi = 0;

            Vec2f p_i = position(i);
            int gx = static_cast<int>(p_i.x);
            int gy = static_cast<int>(p_i.y);

            int gidx = idx1d(gx, gy);
            for(int j =0; j < _pidxInGrid[gidx].size(); j ++){
                Vec2f gradCi = computeGradCi(i, _pidxInGrid[gidx][j]);
                sumnormgradCi +=  (square(gradCi.x) + square(gradCi.y));
                
            }


            _lambda[i] /= (sumnormgradCi + DBL_EPSILON);
         
        }
    }

    void computeDp()
    {
        const Real sr = _kernel.supportRadius();

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
                        const Vec2f& xj = position(j);
                        const Vec2f xij = xi - xj;
                        const Real len_xij = xij.length();
                        if (len_xij > sr) continue;

                        sum_grad_p += (_lambda[i]+ _lambda[j])*_kernel.grad_w(xij, len_xij);
                    }
                }
            }

            _dp.push_back(sum_grad_p / _d0);   // TODO pas sur du tout changmeent pour debug 
        }
    }

    void applyBodyForce()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_type[i] != 1) continue;
            _acc[i] += _g;
            _vel[i] += _dt * _acc[i];   // simple forward Euler
        }
    }
    void applyPressureForce()
    {
        const Real sr = _kernel.supportRadius();

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
                        const Vec2f& xj = position(j);
                        const Vec2f xij = xi - xj;
                        const Real len_xij = xij.length();
                        if (len_xij > sr) continue;

                        sum_grad_p += (_p[i] / square(_d[i]) + _p[j] / square(_d[j])) *
                            _kernel.grad_w(xij, len_xij);
                    }
                }
            }

            _acc[i] -= _m0 * sum_grad_p;
        }
    }
    void applyViscousForce()
    {
        const Real sr = _kernel.supportRadius();

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
                        const Vec2f vij = _vel[i] - _vel[j];
                        const Real len_xij = xij.length();
                        if (len_xij > sr) continue;

                        sum_acc += (_m0 / _d[j]) *
                            vij * xij.dotProduct(_kernel.grad_w(xij, len_xij)) /
                            (xij.dotProduct(xij) + 0.01 * square(_h));
                    }
                }
            }

            _acc[i] += 2.0 * _nu * sum_acc;
        }
    }

    void updateVelocity()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_type[i] != 1) continue;
            _vel[i] += (_pred_pos[i] - _pos[i]) / _dt;
        }
    }
    void updatePosition()
    {

        int solverIteration = 10;
        std::vector<Vec2f> old_pos;

#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_type[i] != 1) continue;
            _pos[i] = _pred_pos[i];
        }
    }

    void predictPosition()
    {
#pragma omp parallel for
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_type[i] != 1) continue;
            _pred_pos[i] += _dt * _vel[i];   // simple forward Euler
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

    // simple collision detection/resolution for each particle
    void resolveCollision()
    {
        std::vector<tIndex> need_res;
        for (tIndex i = 0; i < particleCount(); ++i) {
            if (_pos[i].x<_l || _pos[i].y<_b || _pos[i].x>_r || _pos[i].y>_t)
                need_res.push_back(i);
        }

        for (
            std::vector<tIndex>::const_iterator it = need_res.begin();
            it < need_res.end();
            ++it) {
            const Vec2f p0 = _pos[*it];
            _pos[*it].x = clamp(_pos[*it].x, _l, _r);
            _pos[*it].y = clamp(_pos[*it].y, _b, _t);
            _vel[*it] = (_pos[*it] - p0) / _dt;
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
    std::vector<int>   _type;     // type
    std::vector<Vec2f> _pos;      // position
    std::vector<Vec2f> _pred_pos; // predicted position
    std::vector<Vec2f> _dp;       // position shift
    std::vector<Vec2f> _vel;      // velocity
    std::vector<Vec2f> _acc;      // acceleration
    std::vector<Real>  _p;        // pressure
    std::vector<Real>  _lambda;        // density constraint
    std::vector<Real>  _d;        // density

    std::vector< std::vector<tIndex> > _pidxInGrid; // particle neighbor data

    std::vector<float> _col;    // particle color; just for visualization
    std::vector<float> _vln;    // particle velocity lines; just for visualization

    // simulation
    Real _dt;                     // time step

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

SphSolver gSolver(0.08, 0.5, 1, Vec2f(0, -9.8), 0.01, 7.0);

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
    gSolver.initScene(15, 20, 5, 10);

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
    if (gShowGrid) {
        glBegin(GL_LINES);
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
        glEnd();
    }

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

        // solve 10 steps
        for (int i = 0; i < 10; ++i) gSolver.update();
    }
}

int main(int argc, char** argv)
{
    init();
    while (!glfwWindowShouldClose(gWindow)) {
        update(static_cast<float>(glfwGetTime()));
        render();
        glfwSwapBuffers(gWindow);
        glfwPollEvents();
    }
    clear();
    std::cout << " > Quit" << std::endl;
    return EXIT_SUCCESS;
}
