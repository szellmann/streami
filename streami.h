
#pragma once

// ours
#include "common.h"
#include "vecmath.h"

namespace streami {

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
// Fields
// ========================================================

struct VecField {
  struct DD {
    box3f worldBounds;
    vec3i numMCs;
    vec3i mcID;
    box3f mcBounds;

    RankInfo ri;

    inline __both__
    void buildMCs(const box3f &worldBounds, vec3i numMCs, RankInfo ri) {
      this->worldBounds = worldBounds;
      this->numMCs = numMCs;
      this->ri = ri;

      mcID.x = ri.rankID%numMCs.x;
      mcID.y = (ri.rankID/numMCs.x)%numMCs.y;
      mcID.z = ri.rankID/(numMCs.x*numMCs.y);

      vec3f mcSize = worldBounds.size()/vec3f(numMCs);
      mcBounds.lower = worldBounds.lower+vec3f(mcID)*mcSize;
      mcBounds.upper = worldBounds.lower+vec3f(mcID)*mcSize+mcSize;
    }

    inline __both__
    int flattened_mcID(const vec3i ID, const vec3i gridSize) const {
      return ID.x+ID.y*numMCs.x+ID.z*numMCs.x*numMCs.y;
    }

    inline __both__
    int destinationID(vec3f P) const {
      int gridSize(cbrtf(ri.commSize));
      vec3f mcSize = worldBounds.size()/vec3f(gridSize);
      P -= worldBounds.lower;
      vec3f idf(P/mcSize);
      vec3i id(idf.x,idf.y,idf.z);
      return flattened_mcID(id,gridSize);
    }

  };
};

} // streami


