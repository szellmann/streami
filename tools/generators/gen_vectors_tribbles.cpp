#include <fstream>
#include <vector>
#include "vecmath.h"

using namespace vecmath;

box3f worldBounds{{-1.f,-1.f,-1.f},{1.f,1.f,1.f}};
box3f domain{{-1.f,-1.f,-1.f},{1.f,1.f,1.f}};
vec3f center{0.f,0.f,0.f};

inline __device__ bool sample(const vec3f P, vec3f &value) {
  if (!domain.contains(P))
    return false;

  if (!worldBounds.contains(P))
    return false;

  // Move to center:
  const vec3f P0 = P-center;
  const float r = length(P0);
  
  if (r < 1e-6f) {
    value = vec3f(0.f);
    return true;
  }

  // Tribble effect: radial outward + spiral tangential component
  const vec3f radial = normalize(P0);
  
  // Create tangential component using cross product with up vector
  const vec3f up = vec3f(0.f, 0.f, 1.f);
  vec3f tangent = cross(up, P0);
  const float tangent_len = length(tangent);
  if (tangent_len > 1e-6f) {
    tangent = tangent / tangent_len;
  } else {
    // If aligned with up, use different axis
    tangent = cross(vec3f(1.f, 0.f, 0.f), P0);
    tangent = normalize(tangent);
  }
  
  // Mix radial and tangential for fuzzy spiral effect
  // Stronger tangential component creates more "fuzz"
  const float radial_strength = 0.3f;
  const float tangent_strength = 0.7f;
  const float distance_falloff = expf(-r * r * 0.5f); // Gaussian falloff
  
  value = (radial * radial_strength + tangent * tangent_strength) * distance_falloff;

  return true;
}

int main() {
  int dx=128, dy=128, dz=128;
  int dims[] = {dx,dy,dz};
  std::vector<vec3f> out(dx*size_t(dy)*dz);
  for (int z=0;z<dz;z++) {
    for (int y=0;y<dy;y++) {
      for (int x=0;x<dx;x++) {
        vec3f P(x,y,z);
        P /= vec3f(dx-1,dy-1,dz-1);
        P -= 0.5f;
        P *= 2.f;

        size_t index = linearIndex(x,y,z,dims);
        vec3f P1;
        if (!sample(P,P1)) P1 = 0.f;
        out[index] = P1;
      }
    }
  }

  std::ofstream of("out_128_128_128.raw",std::ios::binary);
  of.write((const char *)out.data(),out.size()*sizeof(out[0]));
}
