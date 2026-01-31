// std
#include <cassert>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
// cuda
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
// ours
#include "common.h"
#include "vecmath.h"

namespace streami {

using namespace vecmath;

// ========================================================
// Particle
// ========================================================

struct Particle {
  int ID;
  vec3f P;
};

struct particle_valid {
  __both__
  inline bool operator()(const Particle &p) {
    const vec3f P = p.P;
    return !(isnan(P.x)||isnan(P.y)||isnan(P.z));
  }
};

struct ParticleIO {
  typedef std::vector<Particle> Line;
  std::vector<Line> lines;

  void append(const Particle *particles, int numParticles) {
    std::vector<Particle> thisGen(numParticles);
    CUDA_SAFE_CALL(cudaMemcpy(thisGen.data(),
                              particles,
                              sizeof(particles[0])*numParticles,
                              cudaMemcpyDefault));
    if (lines.empty()) {
      lines.resize(numParticles);
    }

    std::sort(thisGen.begin(),thisGen.end(),
        [](const Particle &p0, const Particle &p1){ return p0.ID<p1.ID; });

    for (int i=0; i<numParticles; ++i) {
      assert(lines.size() > thisGen[i].ID);
      lines[thisGen[i].ID].push_back(thisGen[i]);
    }
  }

  void saveOBJ(std::string fileName) {
    std::ofstream out(fileName);
    std::vector<std::vector<int>> IDs;
    int vID=1;
    for (int i=0; i<lines.size(); ++i) {
      IDs.emplace_back();
      for (int j=0; j<lines[i].size(); ++j) {
        const vec3f P = lines[i][j].P;
        if (isnan(P.x) || isnan(P.y) || isnan(P.z))
          break;
        out << "v " << P.x << ' ' << P.y << ' ' << P.z << '\n';
        IDs[i].push_back(vID++);
      }
    }

    for (int i=0; i<IDs.size(); ++i) {
      if (IDs[i].size() <= 1) continue;

      out << "l";
      for (int j=0; j<IDs[i].size(); ++j) {
        out << ' ' << IDs[i][j];
      }
      out << '\n';
    }
  }
};

// ========================================================
// Fields
// ========================================================

struct VecField {
  struct DD {
    box3f worldBounds;
    vec3i mcID;
    box3f mcBounds;
  };
};

struct SphericalField : public VecField{
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

    inline void computeBounds() {
      worldBounds = box3f(vec3f(+1e30f),vec3f(-1e30f));
      worldBounds.extend(center-vec3f(radius));
      worldBounds.extend(center+vec3f(radius));

      mcBounds = worldBounds;
    }

    vec3f center;
    float radius;
  };

  inline DD getDD() {
    DD dd;
    dd.center = center;
    dd.radius = radius;
    dd.computeBounds();
    return dd;
  }

  vec3f center;
  float radius;
};


// ========================================================
// Kernels
// ========================================================

__global__ void generateRandomSeeds(
    const box3f bounds, Particle *particles, int numParticles)
{
  int particleID = threadIdx.x+blockIdx.x*blockDim.x;
  if (particleID >= numParticles) return;

  Random rand(particleID,numParticles);
  Particle &p = particles[particleID];
  p.ID = particleID;
  p.P.x = rand()*bounds.size().x+bounds.lower.x;
  p.P.y = rand()*bounds.size().y+bounds.lower.y;
  p.P.z = rand()*bounds.size().z+bounds.lower.z;
}

template<typename Field>
__global__ void update(const Field &field,
                       Particle *thisGen, /* particle positions before */
                       Particle *nextGen, /* particle positions after */
                       int numParticles,
                       float stepSize,
                       float minLength)
{
  int particleID = threadIdx.x+blockIdx.x*blockDim.x;
  if (particleID >= numParticles) return;

  nextGen[particleID].ID = thisGen[particleID].ID;

  const vec3f P0 = thisGen[particleID].P;

  if (isnan(P0.x) || isnan(P0.y) || isnan(P0.z)) {
    nextGen[particleID].P = vec3f(NAN);
    return;
  }

  bool valid{true};
  vec3f k1;
  valid &= field.sample(P0,k1);
  k1 *= stepSize;
  const vec3f P1 = P0+k1*0.5f;

  vec3f k2;
  valid &= field.sample(P1,k2);
  k2 *= stepSize;
  const vec3f P2 = P0+k2*0.5f;

  vec3f k3;
  valid &= field.sample(P2,k3);
  k3 *= stepSize;
  const vec3f P3 = P0+k3;

  vec3f k4;
  valid &= field.sample(P3,k4);
  k4 *= stepSize;
  
  const vec3f P = P0 + 1/6.f*(k1+2.f*k2+2.f*k3+k4);

  if (!field.worldBounds.contains(P) || length(P-P0) < minLength) {
    nextGen[particleID].P = vec3f(NAN);
    return;
  }

  nextGen[particleID].P = P;
}

} // streami

int main(int argc, char **argv) {
  using namespace streami;
  SphericalField field;
  field.center = vec3f(0.f);
  field.radius = 1.f;

  SphericalField::DD fieldDD = field.getDD();

  Particle *thisGen{nullptr};
  Particle *nextGen{nullptr};
  int N=5000;
  CUDA_SAFE_CALL(cudaMalloc(&thisGen,sizeof(Particle)*N));
  CUDA_SAFE_CALL(cudaMalloc(&nextGen,sizeof(Particle)*N));

  ParticleIO io;

  #define CONFIG_KERNEL(kernel,n) kernel<<<iDivUp(n,1024),1024>>>
  CONFIG_KERNEL(generateRandomSeeds,N)(fieldDD.worldBounds,thisGen,N);
  io.append(thisGen,N);

  int activeN=N;
  int steps=1000;
  std::cout << "Computing " << steps << " Runge-Kutta steps for "
      << activeN << " particles...\n";
  int i=0;
  for (; i<steps; ++i) {
    CONFIG_KERNEL(update,activeN)(fieldDD,thisGen,nextGen,activeN,0.1f,1e-10f);
    io.append(nextGen,activeN);
    auto it = thrust::copy_if(thrust::device,
                              nextGen,
                              nextGen+activeN,
                              thisGen,
                              particle_valid());
    activeN = it-thisGen;
    if (activeN<1) break;
  }
  std::cout << "Done after " << i
      << " steps, particles still active: " << activeN << '\n';

  std::cout << "Saving out to obj..\n";
  io.saveOBJ("streamlines.obj");
  std::cout << "Done.. bye!\n";
}




