#pragma once

#include "streami.h"

namespace streami {

struct StructuredField : public VecField {

  struct DD : public VecField::DD {
    inline __device__ bool sample(const vec3f P, vec3f &value) const {
      if (!mc.domain.contains(P))
        return false;
    
      if (!worldBounds.contains(P))
        return false;

      int px(P.x);
      int py(P.y);
      int pz(P.z);
      size_t idx = linearIndex(px,py,pz,(int *)&dims);
      if (idx>=dims.x*size_t(dims.y)*dims.z) {
        printf("out-of-range access: %f,%f,%f\n",P.x,P.y,P.z);
        return false;
      }
      value = values[idx];
      return true;
    }

    vec3f *values;
    vec3i dims, org;
  };

  /* host interface for field */
  StructuredField(vec3f *values, vec3i dims, vec3i org = {0,0,0});
  ~StructuredField();

  box3f computeWorldBounds() const override;

  DD getDD(const RankInfo &ri);

private:
  vec3f *d_values{nullptr};
  vec3i dims{0,0,0};
  vec3i org{0,0,0};
};

} // streami
