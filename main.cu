// std
#include <cassert>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
// rafi
#include "rafi/implementation.h"
// ours
#include "streami.h"
#include "field/Spherical.h"

namespace streami {

// ========================================================
// Helper class, writes particles to obj file:
// ========================================================

struct ParticleIO {
  typedef std::vector<Particle> Line;
  std::vector<Line> lines;

  void append(const Particle *particles, int numParticles) {
    if (numParticles <= 0)
      return;

    std::vector<Particle> thisGen(numParticles);
    CUDA_SAFE_CALL(cudaMemcpy(thisGen.data(),
                              particles,
                              sizeof(particles[0])*numParticles,
                              cudaMemcpyDefault));
    std::sort(thisGen.begin(),thisGen.end(),
        [](const Particle &p0, const Particle &p1){ return p0.ID<p1.ID; });

    lines.resize(std::max((int)lines.size(),thisGen.back().ID+1));

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
// Kernels
// ========================================================

template<typename Field>
__global__ void generateRandomSeeds(const Field &field,
                                    rafi::DeviceInterface<Particle> rafi,
                                    Particle *output, // to dump to file
                                    int numParticles)
{
  int particleID = threadIdx.x+blockIdx.x*blockDim.x;
  if (particleID >= numParticles) return;

  Random rand(particleID,numParticles);
  Particle p;
  p.ID = numParticles*field.ri.rankID+particleID;
  p.P.x = rand()*field.mcBounds.size().x+field.mcBounds.lower.x;
  p.P.y = rand()*field.mcBounds.size().y+field.mcBounds.lower.y;
  p.P.z = rand()*field.mcBounds.size().z+field.mcBounds.lower.z;
  rafi.emitOutgoing(p,field.ri.rankID); // only on ours!
  //printf("%i -- %f,%f,%f\n",field.ri.rankID,P0.x,P0.y,P0.z);
  // for dumping:
  output[particleID] = p;
}

template<typename Field>
__global__ void update(const Field &field,
                       rafi::DeviceInterface<Particle> rafi,
                       Particle *output, // to dump to file
                       int numParticles,
                       float stepSize,
                       float minLength)
{
  int particleID = threadIdx.x+blockIdx.x*blockDim.x;
  if (particleID >= numParticles) return;

  Particle p = rafi.getIncoming(particleID);
  const vec3f P0 = p.P;

  if (isnan(P0.x) || isnan(P0.y) || isnan(P0.z))
    return;

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
  //printf("%i => %f,%f,%f\n",field.ri.rankID,P.x,P.y,P.z);

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

} // streami

int main(int argc, char **argv) {
  using namespace streami;

  RAFI_CUDA_CALL(Free(0));
  RAFI_MPI_CALL(Init(&argc,&argv));

  rafi::HostContext<Particle> *rafi = rafi::createContext<Particle>(MPI_COMM_WORLD, 0);

  RankInfo ri{rafi->mpi.rank,rafi->mpi.size};

  SphericalField field;
  field.center = vec3f(0.f);
  field.radius = 1.f;

  SphericalField::DD fieldDD = field.getDD(ri);

  int N=5000;
  int localN=iDivUp(N,ri.commSize);
  N *= ri.commSize;
  rafi->resizeRayQueues(N);

  // for file I/O:
  ParticleIO io;
  Particle *output{nullptr};
  CUDA_SAFE_CALL(cudaMalloc(&output,sizeof(Particle)*N));

  #define CONFIG_KERNEL(kernel,n) kernel<<<iDivUp(n,1024),1024>>>
  CONFIG_KERNEL(generateRandomSeeds,localN)(
      fieldDD,rafi->getDeviceInterface(),output,localN);
  rafi->forwardRays();
  io.append(output,localN);

  int steps=1000;

  std::cout << "Computing " << steps << " Runge-Kutta steps for "
      << localN << " out of " << N << " particles on rank " << ri.rankID << "...\n";

  int i=0;
  for (; i<steps; ++i) {
    if (localN) {
      CONFIG_KERNEL(update,localN)(
          fieldDD,rafi->getDeviceInterface(),output,localN,0.1f,1e-10f);
    }
    rafi::ForwardResult result = rafi->forwardRays();
    io.append(output,localN);
    localN = result.numRaysInIncomingQueueThisRank;
    std::cout << "rank " << ri.rankID << " in queue: " << localN << '\n';
  }
  std::cout << "Done\n";

  std::string fileName = "streamlines";
  fileName += std::to_string(ri.rankID);
  fileName += ".obj";
  std::cout << "Saving out to obj..\n";
  io.saveOBJ(fileName);
  std::cout << "Done.. bye!\n";

  RAFI_MPI_CALL(Finalize());
}




