
#include "StructuredField.h"

namespace streami {

StructuredField::StructuredField(vec3f *values, vec3i dims, vec3f org, vec3f spacing)
  : dims(dims), org(org), spacing(spacing)
{
  size_t numVoxels = dims.x*size_t(dims.y)*dims.z;
  CUDA_SAFE_CALL(cudaMalloc(&d_values,sizeof(d_values[0])*numVoxels));
  CUDA_SAFE_CALL(cudaMemcpy(d_values,
                            values,
                            sizeof(d_values[0])*numVoxels,
                            cudaMemcpyHostToDevice));
}

StructuredField::~StructuredField()
{
  CUDA_SAFE_CALL(cudaFree(d_values));
}

box3f StructuredField::computeWorldBounds() const
{
  return {
    {org.x,org.y,org.z},
    {org.x+dims.x*spacing.x,org.y+dims.y*spacing.y,org.z+dims.z*spacing.z}
  };
}

StructuredField::DD StructuredField::getDD(const RankInfo &ri)
{
  DD dd;
  (VecField::DD &)dd = VecField::getDD(ri);
  dd.values  = d_values;
  dd.dims    = dims;
  dd.org     = org;
  dd.spacing = spacing;
  return dd;
}

} // streami



