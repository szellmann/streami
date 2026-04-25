#pragma once

// rafi
#include "rafi/implementation.h"
// ours
#include "field/StructuredField.h"
#include "field/UMeshField.h"

namespace streami {
//=========================================================
// seed generation kernels
//=========================================================

__global__
static void generateRandomSeeds(RankInfo ri,
                                MacroCell mc,
                                rafi::DeviceInterface<Particle> rafi,
                                Particle *output, // to dump to file
                                int numParticles,
                                int batchOffset=0,
                                box3f *roi=nullptr,
                                bool roiIsSpherical=false)
{
  int particleID = threadIdx.x+blockIdx.x*blockDim.x;
  if (particleID >= numParticles) return;

  Particle p;
  p.ID = batchOffset+numParticles*ri.rankID+particleID;

  if (roi && !roiIsSpherical && !roi->overlaps(mc.bounds)) {
    p.P = {NAN,NAN,NAN};
    output[particleID] = p;
    return;
  }

  Random rand(particleID,numParticles);
  do {
    p.P.x = rand()*mc.bounds.size().x+mc.bounds.lower.x;
    p.P.y = rand()*mc.bounds.size().y+mc.bounds.lower.y;
    p.P.z = rand()*mc.bounds.size().z+mc.bounds.lower.z;
    if (!roi) break;
    if (roiIsSpherical) {
      float r = length(p.P);
      float lat = asinf(p.P.z/r);
      float lon = atan2f(p.P.y, p.P.x);
      if (roi->contains({r,lat,lon})) break;
    } else {
      if (roi->contains(p.P)) break;
    }
  } while (true);
  rafi.emitOutgoing(p,ri.rankID); // only on ours!
  //printf("generate Random; rank: %i -- position: %f,%f,%f\n",ri.rankID,p.P.x,p.P.y,p.P.z);
  // for dumping:
  output[particleID] = p;
}

//=========================================================
// update kernel
//=========================================================

template<typename Field>
__global__
static void update(const Field field,
                   rafi::DeviceInterface<Particle> rafi,
                   Particle *output, // to dump to file
                   int numParticles,
                   float stepSize,
                   float minLength,
                   box1f *magnitudeRange=0/*for diagnostic*/)
{
  int particleID = threadIdx.x+blockIdx.x*blockDim.x;
  if (particleID >= numParticles) return;

  Particle p = rafi.getIncoming(particleID);
  const vec3f P0 = p.P;
//  printf("ID: %i, pos: %f,%f,%f\n",p.ID,P0.x,P0.y,P0.z);

  if (isnan(P0.x) || isnan(P0.y) || isnan(P0.z))
    return;

  vec3f k1;
  if (!field.sample(P0,k1))
    return;
  k1 *= stepSize;
  const vec3f P1 = P0+k1*0.5f;

  vec3f k2;
  if (!field.sample(P1,k2))
    return;
  k2 *= stepSize;
  const vec3f P2 = P0+k2*0.5f;

  vec3f k3;
  if (!field.sample(P2,k3))
    return;
  k3 *= stepSize;
  const vec3f P3 = P0+k3;

  vec3f k4;
  if (!field.sample(P3,k4))
    return;
  k4 *= stepSize;
  
  const vec3f P = P0 + 1/6.f*(k1+2.f*k2+2.f*k3+k4);
  //printf("%i => %f,%f,%f\n",field.ri.rankID,P.x,P.y,P.z);

  if (magnitudeRange) {
    atomicMin(&magnitudeRange->lower,length(P-P0));
    atomicMax(&magnitudeRange->upper,length(P-P0));
  }

  if (!field.worldBounds.contains(P) || length(P-P0) < minLength) {
    return;
  }

  p.P = P;
  int dest = field.destinationID(P);
  //printf("%i => %i\n",field.ri.rankID,dest);
  rafi.emitOutgoing(p,dest);
  // for dumping:
  output[particleID] = p;
}

} // namespace streami



