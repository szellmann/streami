
#pragma once

// std
#include <memory>
#include <vector>
// mpi
#include <mpi.h>
// ours
#include "common.h"
#include "vecmath.h"

namespace rafi {
template<typename RayT>
struct HostContext;
} // namespace rafi

namespace streami {

struct StructuredField;
struct UMeshField;

typedef uint64_t TimeStamp;
TimeStamp newTimeStamp();

// import vector math types:
using namespace vecmath;

// ========================================================
// Particle
// ========================================================

struct Particle {
  int ID;
  vec3f P;
#if 1//ndef NDEBUG
  bool dbg; // compat with rafi..
#endif
};

// ========================================================
// MPI
// ========================================================

struct RankInfo {
  int rankID;
  int commSize;
};

// ========================================================
// Macrocell
// ========================================================

struct MacroCell {
  vec3i mcID;
  // tight
  box3f bounds;
  // with halos
  box3f domain;
};

inline
MacroCell makeMacroCell(const box3f worldBounds,
                        vec3i numMCs,
                        RankInfo ri,
                        float halo=0.f)
{
  MacroCell mc;

  mc.mcID.x = ri.rankID%numMCs.x;
  mc.mcID.y = (ri.rankID/numMCs.x)%numMCs.y;
  mc.mcID.z = ri.rankID/(numMCs.x*numMCs.y);

  vec3f mcSize = worldBounds.size()/vec3f(numMCs);
  mc.bounds.lower = worldBounds.lower+vec3f(mc.mcID)*mcSize;
  mc.bounds.upper = worldBounds.lower+vec3f(mc.mcID)*mcSize+mcSize;

  mc.domain = mc.bounds;
  mc.domain.lower -= halo;
  mc.domain.upper += halo;

  return mc;
}

// ========================================================
// Fields
// ========================================================

struct VecField {
  typedef std::shared_ptr<VecField> SP;

  struct DD {
    box3f worldBounds;
    vec3i numMCs;
    MacroCell mc;
    RankInfo ri;

    inline __device__
    int flattened_mcID(const vec3i ID) const {
      return ID.x+ID.y*numMCs.x+ID.z*numMCs.x*numMCs.y;
    }

    inline __device__
    int destinationID(vec3f P) const {
      vec3f mcSize = worldBounds.size()/vec3f(numMCs);
      P -= worldBounds.lower;
      vec3f idf(P/mcSize);
      vec3i id(idf.x,idf.y,idf.z);
      return flattened_mcID(id);
    }

  };

  virtual box3f computeWorldBounds() const = 0;

  inline DD getDD(const RankInfo &ri) {
    DD dd;
    dd.worldBounds = computeWorldBounds();
    dd.numMCs      = numMCs;
    dd.mc          = mc;
    dd.ri          = ri;
    return dd;
  }

  vec3i numMCs;
  MacroCell mc;
};


// ========================================================
// Context
// ========================================================

struct Context {
  Context(int argc, char **argv);
  ~Context();

  MPI_Comm newComm();
 private:
};


// ========================================================
// Tracer
// ========================================================

struct Tracer {
  struct Params {
    enum Mode { Streamlines, Streaklines, Undefined, };
    Mode mode{Undefined};
    int numParticles{0};
    int maxSteps{0};
    float stepSize{0.f};
    float minLength{0.f};
    struct {
      box3f bounds{vec3f{1e20f},vec3f{-1e20f}};
      bool isSpherical{false};
    } roi;
  };

  struct Vertex {
    Particle p;
    vec3f color;
  };
  typedef std::vector<Vertex> Line;

  Tracer(Context &ctx, const Params &p);

  //=======================================================
  // Set field for time step
  //=======================================================
  void setField(const std::shared_ptr<StructuredField> &field, size_t timeStep=0);
  void setField(const std::shared_ptr<UMeshField> &field, size_t timeStep=0);

  void setParams(const Params &p);

  bool step();

  void trace();

  std::vector<Line> getLines();

 private:
  enum FieldType { Structured, UMesh, Undefined, };
  void init();
  int generateNewParticles();
  void resizeRayQueues(size_t N);
  void insertField(const VecField::SP &field, size_t timeStep);
  void doTimeStep();
  void appendOutput(const std::vector<vec3f> &vertexColors);

  Context &context;
  Params params;
  std::vector<VecField::SP> fields;
  FieldType fieldType{Undefined};

  std::vector<Line> hLines;

  int globalN, localN, maxN;
  int injectionCount{0};
  size_t currentTimeStep{0ull};
  Particle *dOutput{nullptr};

  TimeStamp lastInitCall{0}, lastInitRequest{0};

  rafi::HostContext<Particle> *rafi{nullptr};
};

} // streami


