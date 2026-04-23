// ours
#include "kernels.cuh"
#include "kernels.h"

#define CONFIG_KERNEL_512(kernel,n) kernel<<<iDivUp(n,512),512>>>
#define CONFIG_KERNEL_1024(kernel,n) kernel<<<iDivUp(n,1024),1024>>>
#define CONFIG_KERNEL CONFIG_KERNEL_1024

namespace streami {

// ========================================================
// Dispatch / specializations:
// ========================================================

void call_generateRandomSeeds(const VecField::SP field,
                              rafi::DeviceInterface<Particle> rafi,
                              Particle *output, // to dump to file
                              int numParticles,
                              box3f *roi,
                              bool roiIsSpherical)
{
  if (numParticles <= 0)
    return;

  RankInfo ri{rafi.mpi.rank,rafi.mpi.size};
  MacroCell mc = field->mc;

  CONFIG_KERNEL(generateRandomSeeds,numParticles)(
      ri,mc,rafi,output,numParticles,roi,roiIsSpherical);
}

void call_update_StructuredField(const VecField::SP field,
                                 rafi::DeviceInterface<Particle> rafi,
                                 Particle *output, // to dump to file
                                 int numParticles,
                                 float stepSize,
                                 float minLength,
                                 box1f *magnitudeRange/*for diagnostic*/)
{
  if (numParticles <= 0)
    return;

  RankInfo ri{rafi.mpi.rank,rafi.mpi.size};

  const StructuredField::SP &sfield = (const StructuredField::SP &)field;
  const StructuredField::DD &fieldDD = sfield->getDD(ri);

  CONFIG_KERNEL(update,numParticles)(
      fieldDD,rafi,output,numParticles,stepSize,minLength,0);
}

void call_update_UMeshField(const VecField::SP field,
                            rafi::DeviceInterface<Particle> rafi,
                            Particle *output, // to dump to file
                            int numParticles,
                            float stepSize,
                            float minLength,
                            box1f *magnitudeRange/*for diagnostic*/)
{
  if (numParticles <= 0)
    return;

  RankInfo ri{rafi.mpi.rank,rafi.mpi.size};

  const UMeshField::SP &sfield = (const UMeshField::SP &)field;
  const UMeshField::DD &fieldDD = sfield->getDD(ri);

  CONFIG_KERNEL(update,numParticles)(
      fieldDD,rafi,output,numParticles,stepSize,minLength,0);
}

} // namespace streami


