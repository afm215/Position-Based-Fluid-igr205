#ifndef kernel_cuh
#define kernel_cuh

#ifndef M_PI
#define M_PI 3.141592
#endif

#include "Vector.hpp"


constexpr float kPoly6Factor() {
            return (315.0f / 64.0f / M_PI);
        }

      constexpr float kSpikyGradFactor() { return (-45.0f / M_PI); }
     inline float Poly6Value(const float s, const float h) {
        if (s < 0.0f || s >= h)
            return 0.0f;

        float x = (h * h - s * s) / (h * h * h);
        float result = kPoly6Factor() * x * x * x;
        return result;
    }

    inline float Poly6Value(const Vec2f r, const float h) {
        float r_len = r.length();
        return Poly6Value(r_len, h);
    }

   inline Vec2f SpikyGradient(const Vec2f r, const float h) {
        float r_len = r.length();
        if (r_len <= 0.0f || r_len >= h)
            return Vec2f(0.0f);

        float x = (h - r_len) / (h * h * h);
        float g_factor = kSpikyGradFactor() * x * x;
        Vec2f result = r.normalized() * g_factor;
        return result;
    }

#endif