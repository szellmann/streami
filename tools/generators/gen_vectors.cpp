#include <fstream>
#include <vector>
#include "vecmath.h"

using namespace vecmath;

box3f worldBounds{{-1.f,-1.f,-1.f},{1.f,1.f,1.f}};
box3f domain{{-1.f,-1.f,-1.f},{-0.9f,1.f,1.f}};
vec3f center{0.f,0.f,0.f};

inline __device__ bool sample(const vec3f P, vec3f &value) {
  if (!domain.contains(P))
    return false;

  if (!worldBounds.contains(P))
    return false;

  // Move to center:
  const vec3f P0 = P-center;

  // Compute vector in lon/lat:
  const float r    = length(P0);
  const float lat0 = asinf(P.z/r);
  const float lon0 = atan2f(P.y,P.x);

  // Offset by DEG degree:
#define DEG2RAD(X) (X)*180.f/float(M_PI)
  const float lat1 = lat0+DEG2RAD(1.f);
  const float lon1 = lon0+DEG2RAD(1.f);

  // Back to cartesian:
  const vec3f P1{
    r * cosf(lat1) * cosf(lon1),
      r * cosf(lat1) * sinf(lon1),
      r * sinf(lat1)
  };

  // Compute vector:
  value = P1-P0;

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
