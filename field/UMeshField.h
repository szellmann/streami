
#pragma once

// cuBQL
#include "cuBQL/traversal/fixedBoxQuery.h"
// ours
#include "streami.h"
#include "UElems.h"

namespace streami {

struct UMeshField : public VecField {
#if 1
    using bvh_t  = cuBQL::BinaryBVH<float,3>;
#else
    enum { BVH_WIDTH = 4 };
    using bvh_t  = cuBQL::WideBVH<float,3,BVH_WIDTH>;
#endif

    using node_t = typename bvh_t::Node;
  struct DD : public VecField::DD {
    inline __device__ bool sample(const vec3f P, vec3f &value) const {
      typename bvh_t::box_t box; box.lower = box.upper = {P.x,P.y,P.z};

      bool hit{false};
      auto lambda = [this,P,&hit,&value]
        (const uint32_t primID)
      {
        const int *I = indices + cellIndices[primID];
        const vec3f dir = uvw[primID];
        // Hack direction vector into w:
        const vec4f v0(vertices[I[0]],dir.x);
        const vec4f v1(vertices[I[1]],dir.y);
        const vec4f v2(vertices[I[2]],dir.z);
        const vec4f v3(vertices[I[3]],dir.x);
        const vec4f v4(vertices[I[4]],dir.y);
        const vec4f v5(vertices[I[5]],dir.z);
        if (intersectWedgeEXT(value,P,v0,v1,v2,v3,v4,v5)) {
          hit = true;
          return CUBQL_TERMINATE_TRAVERSAL;
        }
        return CUBQL_CONTINUE_TRAVERSAL;
      };
      cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
      return hit;
    }

    vec3f *vertices;
    int   *indices;
    int   *cellIndices;
    vec3f *uvw;
    int numVertices;
    int numIndices;
    int numCells;

    bvh_t bvh;
  };

  /* host interface for field */
  UMeshField(vec3f *vertices, int *indices, int *cellIndices, vec3f *uvw,
             int numVertices, int numIndices, int numCells);
  ~UMeshField();

  DD getDD(const RankInfo &ri);

 private:
  vec3f *d_vertices{nullptr};
  int   *d_indices{nullptr};
  int   *d_cellIndices{nullptr};
  vec3f *d_uvw{nullptr};
  int numVertices{0};
  int numIndices{0};
  int numCells{0};
  bvh_t bvh = {0,0,0,0};
  box3f worldBounds;
};

} // streami


