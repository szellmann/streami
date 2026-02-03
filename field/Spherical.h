
#pragma once

namespace streami {

struct SphericalField : public VecField {
  struct DD : public VecField::DD {
    inline __device__ bool sample(const vec3f P, vec3f &value) const {
      if (!mcBounds.contains(P))
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

    vec3f center;
    float radius;
  };

  inline DD getDD(const RankInfo &ri) {
    DD dd;
    dd.center = center;
    dd.radius = radius;
    dd.ri = ri;
    // Base:
    box3f worldBounds(vec3f(+1e30f),vec3f(-1e30f));
    worldBounds.extend(center-vec3f(radius));
    worldBounds.extend(center+vec3f(radius));
    float halo = radius*0.1f;
    dd.buildMCs(worldBounds,vec3i((int)cbrtf(ri.commSize)),ri,halo);
    return dd;
  }

  vec3f center;
  float radius;
};

} // streami


