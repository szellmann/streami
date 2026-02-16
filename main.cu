// std
#include <cassert>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
// umesh
#include "umesh/UMesh.h"
// rafi
#include "rafi/implementation.h"
// ours
#include "streami.h"
#include "field/Spherical.h"
#include "field/UMeshField.h"

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
__global__ void generateRandomSeeds(const Field field,
                                    rafi::DeviceInterface<Particle> rafi,
                                    Particle *output, // to dump to file
                                    int numParticles)
{
  int particleID = threadIdx.x+blockIdx.x*blockDim.x;
  if (particleID >= numParticles) return;

  Random rand(particleID,numParticles);
  Particle p;
  p.ID = numParticles*field.ri.rankID+particleID;
  p.P.x = rand()*field.mc.bounds.size().x+field.mc.bounds.lower.x;
  p.P.y = rand()*field.mc.bounds.size().y+field.mc.bounds.lower.y;
  p.P.z = rand()*field.mc.bounds.size().z+field.mc.bounds.lower.z;
  rafi.emitOutgoing(p,field.ri.rankID); // only on ours!
  //printf("%i -- %f,%f,%f\n",field.ri.rankID,P0.x,P0.y,P0.z);
  // for dumping:
  output[particleID] = p;
}

template<typename Field>
__global__ void update(const Field field,
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

#define CONFIG_KERNEL_512(kernel,n) kernel<<<iDivUp(n,512),512>>>
#define CONFIG_KERNEL_1024(kernel,n) kernel<<<iDivUp(n,1024),1024>>>
#define CONFIG_KERNEL CONFIG_KERNEL_1024



// ========================================================
// test/use cases:
// ========================================================

// random seeds on a procedural vector field:
void main_Spherical(int argc, char **argv, rafi::HostContext<Particle> *rafi) {
  RankInfo ri{rafi->mpi.rank,rafi->mpi.size};

  SphericalField field;
  field.center = vec3f(0.f);
  field.radius = 1.f;

  field.numMCs = (int)cbrtf(ri.commSize);

  float halo = field.radius*0.1f;
  field.mc = makeMacroCell(field.computeWorldBounds(),field.numMCs,ri,halo);

  SphericalField::DD fieldDD = field.getDD(ri);

  int N=5000;
  int localN=iDivUp(N,ri.commSize);
  N *= ri.commSize;
  rafi->resizeRayQueues(N);

  // for file I/O:
  ParticleIO io;
  Particle *output{nullptr};
  CUDA_SAFE_CALL(cudaMalloc(&output,sizeof(Particle)*N));

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
}

void main_UMesh(int argc, char **argv, rafi::HostContext<Particle> *rafi) {
  using namespace streami;

  if (argc < 2) {
    std::cerr << "UMesh filename missing..\n";
    return;
  }

  umesh::UMesh::SP inMesh = umesh::UMesh::loadFrom(argv[1]);

  vec3i gridSize{1};
  for (int i=2;i<argc;i++) {
    std::string arg(argv[i]);
    if (arg[0] == '-') {
      if (arg == "-gx") {
        gridSize.x = std::stoi(argv[++i]);
      }
      if (arg == "-gy") {
        gridSize.y = std::stoi(argv[++i]);
      }
      if (arg == "-gz") {
        gridSize.z = std::stoi(argv[++i]);
      }
    }
  }

  RankInfo ri{rafi->mpi.rank,rafi->mpi.size};

  int mcCount = gridSize.x*gridSize.y*gridSize.z;
  if (mcCount != ri.commSize) {
    std::cerr << "# macro cells (" << mcCount << ") and # MPI ranks ("
        << ri.commSize << ") don't match..\n";
    return;
  }

  auto umeshBounds = inMesh->getBounds();
  box3f worldBounds{
    {umeshBounds.lower.x,umeshBounds.lower.y,umeshBounds.lower.z},
    {umeshBounds.upper.x,umeshBounds.upper.y,umeshBounds.upper.z}
  };

  MacroCell localMC = makeMacroCell(worldBounds,gridSize,ri,worldBounds.size().x/10.f);

  std::vector<vec3f> vertices;
  std::vector<int> indices;
  std::vector<int> cellIndices;
  std::vector<vec3f> uvw;

  std::vector<int> old2new(inMesh->vertices.size(),-1);

  for (size_t i=0; i<inMesh->wedges.size(); ++i) {
    bool ours = false;
    for (int j=0; j<6; ++j) {
      int vertID = inMesh->wedges[i][j];
      auto vv = inMesh->vertices[vertID];
      vec3f v(vv.x,vv.y,vv.z);
      if (localMC.domain.contains(v)) {
        ours = true;
        break;
      }
    }

    if (!ours) continue;

    for (int j=0; j<6; ++j) {
      int vertID = inMesh->wedges[i][j];
      auto vv = inMesh->vertices[vertID];
      vec3f v(vv.x,vv.y,vv.z);
      if (old2new[vertID] < 0) {
        old2new[vertID] = (int)vertices.size();
        vertices.push_back(v);
      }
    }
  }

  // TODO: for now only wedges...

  for (size_t i=0, cellIndex=0; i<inMesh->wedges.size(); ++i) {
    int I[6];
    for (int j=0; j<6; ++j) {
      I[j] = old2new[inMesh->wedges[i][j]];
    }
    if (I[0]<0 || I[1]<0 || I[2]<0 || I[3]<0 || I[4]<0 || I[5]<0) // not ours!
      continue;

    indices.push_back(I[0]);
    indices.push_back(I[1]);
    indices.push_back(I[2]);
    indices.push_back(I[3]);
    indices.push_back(I[4]);
    indices.push_back(I[5]);
    cellIndices.push_back(cellIndex);
    cellIndex += 6;
    // u/v/w direction vectors stored in
    // the first three vertices:
    float u = inMesh->perVertex->values[inMesh->wedges[i][0]];
    float v = inMesh->perVertex->values[inMesh->wedges[i][1]];
    float w = inMesh->perVertex->values[inMesh->wedges[i][2]];
    uvw.push_back({u,v,w});
  }

  std::cout << "rank #" << ri.rankID << " gets " << vertices.size()
    << " out of " << inMesh->vertices.size() << " vertices\n";

  std::cout << "rank #" << ri.rankID << " gets " << uvw.size()
    << " out of " << inMesh->wedges.size() << " wedge cells\n";

  UMeshField field(vertices.data(),
                   indices.data(),
                   cellIndices.data(),
                   uvw.data(),
                   vertices.size(),
                   indices.size(),
                   cellIndices.size());

  field.numMCs = gridSize;
  field.mc = localMC;

  UMeshField::DD fieldDD = field.getDD(ri);

  int N=100000;
  int localN=iDivUp(N,ri.commSize);
  N *= ri.commSize;
  rafi->resizeRayQueues(N);

  // for file I/O:
  ParticleIO io;
  Particle *output{nullptr};
  CUDA_SAFE_CALL(cudaMalloc(&output,sizeof(Particle)*N));

  CONFIG_KERNEL(generateRandomSeeds,localN)(
      fieldDD,rafi->getDeviceInterface(),output,localN);
  rafi->forwardRays();
  io.append(output,localN);

  int steps=100000;

  std::cout << "Computing " << steps << " Runge-Kutta steps for "
      << localN << " out of " << N << " particles on rank " << ri.rankID << "...\n";

  int i=0;
  for (; i<steps; ++i) {
    if (localN) {
      CONFIG_KERNEL_512(update,localN)(
          fieldDD,rafi->getDeviceInterface(),output,localN,100.f,1.f);
    }
    rafi::ForwardResult result = rafi->forwardRays();
    io.append(output,localN);
    localN = result.numRaysInIncomingQueueThisRank;
    std::cout << "rank " << ri.rankID << " in queue: " << localN << '\n';

    int globalN = 0;
    MPI_SAFE_CALL(MPI_Allreduce(&localN,&globalN,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD));
    if (globalN == 0)
      break;
  }
  std::cout << "Done\n";

  std::string fileName = "streamlines";
  fileName += std::to_string(ri.rankID);
  fileName += ".obj";
  std::cout << "Saving out to obj..\n";
  io.saveOBJ(fileName);
  std::cout << "Done.. bye!\n";
}

} // streami


// ========================================================
// main dispatch:
// ========================================================

int main(int argc, char **argv) {
  using namespace streami;

  RAFI_CUDA_CALL(Free(0));
  RAFI_MPI_CALL(Init(&argc,&argv));

  rafi::HostContext<Particle> *rafi = rafi::createContext<Particle>(MPI_COMM_WORLD, 0);
  //main_Spherical(argc,argv,rafi);
  main_UMesh(argc,argv,rafi);

  RAFI_MPI_CALL(Finalize());
}




