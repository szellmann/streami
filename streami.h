
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

} // streami


