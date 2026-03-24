// std
#include <atomic>
// rafi
#include "rafi/implementation.h"
// ours
#include "streami.h"
#include "field/StructuredField.h"
#include "field/UMeshField.h"
#include "gpu/kernels.h"

namespace streami {

// ========================================================
// Helpers
// ========================================================

__host__ __device__
inline vec3f randomColor(unsigned idx)
{
  unsigned int r = (unsigned int)(idx*13*17 + 0x234235);
  unsigned int g = (unsigned int)(idx*7*3*5 + 0x773477);
  unsigned int b = (unsigned int)(idx*11*19 + 0x223766);
  return vec3f{(r&255)/255.f,
               (g&255)/255.f,
               (b&255)/255.f};
}

static
std::vector<vec3f> randomColorsPerID(size_t length)
{
  std::vector<vec3f> result(length);
  for (size_t i=0; i<length; ++i) {
    result[i] = randomColor((unsigned)i);
  }
  return result;
}


std::atomic<TimeStamp> g_timeStamp{0ull};
TimeStamp newTimeStamp() {
  return ++g_timeStamp;
}

// ========================================================
// Context
// ========================================================

Context::Context(int argc, char **argv)
{
  CUDA_SAFE_CALL(cudaFree(0));
  MPI_SAFE_CALL(MPI_Init(&argc,&argv));
}

Context::~Context()
{
  MPI_SAFE_CALL(MPI_Finalize());
}

MPI_Comm Context::newComm()
{
  return MPI_COMM_WORLD; // todo
}



// ========================================================
// Tracer
// ========================================================

Tracer::Tracer(Context &ctx, const Tracer::Params &p)
  : context(ctx), params(p)
{
  rafi = rafi::createContext<Particle>(ctx.newComm(), 0);
  lastInitRequest = newTimeStamp();
}


void Tracer::setField(const StructuredField::SP &f)
{
  field = f;
  fieldType = Structured;
  lastInitRequest = newTimeStamp();
}

void Tracer::setField(const UMeshField::SP &f)
{
  field = f;
  fieldType = UMesh;
  lastInitRequest = newTimeStamp();
}

void Tracer::setParams(const Params &p)
{
  params = p;
  lastInitRequest = newTimeStamp();
}

bool Tracer::step()
{
  if (lastInitRequest >= lastInitCall)
    init();

  if (fieldType == Structured) {
    call_update_StructuredField(
        field,rafi->getDeviceInterface(),dOutput,localN,params.stepSize,params.minLength);
  }

  if (fieldType == UMesh) {
    call_update_UMeshField(
        field,rafi->getDeviceInterface(),dOutput,localN,params.stepSize,params.minLength);
  }

  rafi::ForwardResult result = rafi->forwardRays();

  auto colors = randomColorsPerID(globalN);
  appendOutput(colors);

  localN = result.numRaysInIncomingQueueThisRank;

  int particlesLeft = result.numRaysAliveAcrossAllRanks;

  return particlesLeft > 0;
}

void Tracer::trace()
{
  for (int i=0; i<params.maxSteps; ++i) {
    if (!step()) break;
  }
}

std::vector<Tracer::Line> Tracer::getLines()
{
  if (lastInitRequest >= lastInitCall)
    init();

  return hLines;
}

void Tracer::init()
{
  lastInitCall = newTimeStamp();

  RankInfo ri{rafi->mpi.rank,rafi->mpi.size};

  hLines.clear();

  globalN=params.numParticles;
  localN=iDivUp(globalN,ri.commSize);
  globalN = localN*ri.commSize;
  rafi->resizeRayQueues(globalN);

  CUDA_SAFE_CALL(cudaFree(dOutput));
  CUDA_SAFE_CALL(cudaMalloc(&dOutput,sizeof(Particle)*globalN));

  box3f *d_roi=nullptr;
  if (!params.roi.bounds.empty()) {
    CUDA_SAFE_CALL(cudaMalloc(&d_roi,sizeof(box3f)));
    CUDA_SAFE_CALL(cudaMemcpy(d_roi,
                              &params.roi.bounds,
                              sizeof(params.roi.bounds),
                              cudaMemcpyHostToDevice));
  }

  call_generateRandomSeeds(
      field,rafi->getDeviceInterface(),dOutput,localN,d_roi,params.roi.isSpherical);

  if (d_roi) {
    CUDA_SAFE_CALL(cudaFree(d_roi));
  }

  rafi->forwardRays();
}

void Tracer::appendOutput(const std::vector<vec3f> &vertexColors)
{
  if (localN <= 0)
    return;

  std::vector<Particle> thisGen(localN);
  CUDA_SAFE_CALL(cudaMemcpy(thisGen.data(),
                            dOutput,
                            sizeof(dOutput[0])*localN,
                            cudaMemcpyDefault));
  std::sort(thisGen.begin(),thisGen.end(),
      [](const Particle &p0, const Particle &p1){ return p0.ID<p1.ID; });

  hLines.resize(std::max((int)hLines.size(),thisGen.back().ID+1));

  for (int i=0; i<localN; ++i) {
    assert(hLines.size() > thisGen[i].ID);
    vec3f color
        = vertexColors.empty() ? vec3f(0.0f) : vertexColors[thisGen[i].ID];
    hLines[thisGen[i].ID].push_back({thisGen[i],color});
  }
}

} // namespace streami



