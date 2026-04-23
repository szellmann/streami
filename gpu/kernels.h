
#pragma once

// rafi
#include "rafi/implementation.h"
// ours
#include "field/StructuredField.h"
#include "field/UMeshField.h"

namespace streami {

void call_generateRandomSeeds(const VecField::SP field,
                              rafi::DeviceInterface<Particle> rafi,
                              Particle *output, // to dump to file
                              int numParticles,
                              int idOffset=0,
                              box3f *roi=nullptr,
                              bool roiIsSpherical=false);

void call_update_UMeshField(const VecField::SP field,
                            rafi::DeviceInterface<Particle> rafi,
                            Particle *output, // to dump to file
                            int numParticles,
                            float stepSize,
                            float minLength,
                            box1f *magnitudeRange=0/*for diagnostic*/);

void call_update_StructuredField(const VecField::SP field,
                                 rafi::DeviceInterface<Particle> rafi,
                                 Particle *output, // to dump to file
                                 int numParticles,
                                 float stepSize,
                                 float minLength,
                                 box1f *magnitudeRange=0/*for diagnostic*/);

} // namespace streami


