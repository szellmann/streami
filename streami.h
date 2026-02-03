
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
    // tight
    box3f mcBounds;
    // with halos
    box3f mcDomain;
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

  inline void buildMCs(vec3i numMCs, RankInfo ri, float halo=0.f) {
    this->worldBounds = computeWorldBounds();
    this->numMCs = numMCs;

    mcID.x = ri.rankID%numMCs.x;
    mcID.y = (ri.rankID/numMCs.x)%numMCs.y;
    mcID.z = ri.rankID/(numMCs.x*numMCs.y);

    vec3f mcSize = worldBounds.size()/vec3f(numMCs);
    mcBounds.lower = worldBounds.lower+vec3f(mcID)*mcSize;
    mcBounds.upper = worldBounds.lower+vec3f(mcID)*mcSize+mcSize;

    mcDomain = mcBounds;
    mcDomain.lower -= halo;
    mcDomain.upper += halo;
  }

  inline DD getDD(const RankInfo &ri) {
    DD dd;
    dd.worldBounds = worldBounds;
    dd.numMCs      = numMCs;
    dd.mcID        = mcID;
    dd.mcBounds    = mcBounds;
    dd.mcDomain    = mcDomain;
    dd.ri          = ri;
    return dd;
  }

  box3f worldBounds;
  vec3i numMCs;
  vec3i mcID;
  // tight
  box3f mcBounds;
  // with halos
  box3f mcDomain;
};

} // streami


