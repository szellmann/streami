#pragma once

#include "streami.h"

namespace streami {

struct StructuredField : public VecField {

  typedef std::shared_ptr<StructuredField> SP;

  struct DD : public VecField::DD {
    inline __device__ bool sample(const vec3f P, vec3f &value) const {
      if (!mc.domain.contains(P))
        return false;
    
      if (!worldBounds.contains(P))
        return false;

      vec3f PP = (P-org)/spacing;
      int px(PP.x);
      int py(PP.y);
      int pz(PP.z);
      size_t idx = linearIndex(px,py,pz,(int *)&dims);
      if (idx>=dims.x*size_t(dims.y)*dims.z) {
        printf("out-of-range access: %f,%f,%f\n",P.x,P.y,P.z);
        return false;
      }
      value = values[idx];
      return true;
    }

    vec3f *values;
    vec3i dims;
    vec3f org, spacing;
  };

  /* host interface for field */
  StructuredField(vec3f *values,
                  vec3i dims,
                  vec3f org = {0.f,0.f,0.f},
                  vec3f spacing = {1.f,1.f,1.f});
  ~StructuredField();

  box3f computeWorldBounds() const override;

  DD getDD(const RankInfo &ri);

private:
  vec3f *d_values{nullptr};
  vec3i dims{0,0,0};
  vec3f org{0,0,0};
  vec3f spacing{0,0,0};
};

} // streami
