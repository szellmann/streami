
#include "UMeshField.h"

namespace streami {

__global__ void computeBounds(box3f *primBounds,
                              box3f *localWorldBounds,
                              const vec3f *vertices,
                              const int *indices,
                              const int *cellIndices,
                              const uint8_t *cellTypes,
                              size_t numCells)
{
  size_t cellID = threadIdx.x+blockIdx.x*blockDim.x;
  if (cellID >= numCells) return;

  int numIndices = -1;
  switch (cellTypes[cellID]) {
  case VTK_TET_:   numIndices = 4; break;
  case VTK_PYR_:   numIndices = 5; break;
  case VTK_WEDGE_: numIndices = 6; break;
  case VTK_HEX_:   numIndices = 8; break;
  default: break;
  }
  assert(numIndices>=4 && numIndices<=8);

  const int *I = indices + cellIndices[cellID];
  vec3f V[8];
  for (int i=0; i<numIndices; ++i) {
    V[i] = vertices[I[i]];
  }

  // primBounds:
  primBounds[cellID] = box3f(vec3f(FLT_MAX),vec3f(-FLT_MAX));
  for (int i=0; i<numIndices; ++i) {
    primBounds[cellID].extend(V[i]);
  }

  // localWorldBounds:
  atomicMin(&localWorldBounds->lower.x,primBounds[cellID].lower.x);
  atomicMin(&localWorldBounds->lower.y,primBounds[cellID].lower.y);
  atomicMin(&localWorldBounds->lower.z,primBounds[cellID].lower.z);
  atomicMax(&localWorldBounds->upper.x,primBounds[cellID].upper.x);
  atomicMax(&localWorldBounds->upper.y,primBounds[cellID].upper.y);
  atomicMax(&localWorldBounds->upper.z,primBounds[cellID].upper.z);
}

UMeshField::UMeshField(const box3f &worldBounds,
                       vec3f *vertices, int *indices, int *cellIndices, uint8_t *cellTypes,
                       vec3f *uvw, size_t numVertices, size_t numIndices, size_t numCells)
  : worldBounds(worldBounds), numVertices(numVertices), numIndices(numIndices), numCells(numCells)
{
  CUDA_SAFE_CALL(cudaMalloc(&d_vertices, sizeof(vertices[0])*numVertices));
  CUDA_SAFE_CALL(cudaMemcpy(d_vertices,
                            vertices,
                            sizeof(vertices[0])*numVertices,
                            cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaMalloc(&d_indices, sizeof(indices[0])*numIndices));
  CUDA_SAFE_CALL(cudaMemcpy(d_indices,
                            indices,
                            sizeof(indices[0])*numIndices,
                            cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaMalloc(&d_cellIndices, sizeof(cellIndices[0])*numCells));
  CUDA_SAFE_CALL(cudaMemcpy(d_cellIndices,
                            cellIndices,
                            sizeof(cellIndices[0])*numCells,
                            cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaMalloc(&d_cellTypes, sizeof(cellTypes[0])*numCells));
  CUDA_SAFE_CALL(cudaMemcpy(d_cellTypes,
                            cellTypes,
                            sizeof(cellTypes[0])*numCells,
                            cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaMalloc(&d_uvw, sizeof(uvw[0])*numCells));
  CUDA_SAFE_CALL(cudaMemcpy(d_uvw,
                            uvw,
                            sizeof(uvw[0])*numCells,
                            cudaMemcpyHostToDevice));

  box3f localWorldBounds(vec3f(FLT_MAX),vec3f(-FLT_MAX));

  box3f *primBounds;
  CUDA_SAFE_CALL(cudaMalloc(&primBounds, sizeof(primBounds[0])*numCells));

  box3f *dWorldBounds;
  CUDA_SAFE_CALL(cudaMalloc(&dWorldBounds, sizeof(*dWorldBounds)));
  CUDA_SAFE_CALL(cudaMemcpy(dWorldBounds,&localWorldBounds,sizeof(localWorldBounds),
                            cudaMemcpyHostToDevice));

  computeBounds<<<iDivUp(numCells,1024),1024>>>(primBounds,
                                                dWorldBounds,
                                                d_vertices,
                                                d_indices,
                                                d_cellIndices,
                                                d_cellTypes,
                                                numCells);

  CUDA_SAFE_CALL(cudaMemcpy(&localWorldBounds,dWorldBounds,sizeof(localWorldBounds),
                            cudaMemcpyDeviceToHost));
  std::cout << "LOCAL WORLD BOUNDS: " << localWorldBounds << '\n';

  cuBQL::DeviceMemoryResource memResource;
  cuBQL::gpuBuilder(bvh,
                    (const cuBQL::box_t<float,3>*)primBounds,
                    numCells,
                    cuBQL::BuildConfig(),
                    0,
                    memResource);

  CUDA_SAFE_CALL(cudaFree(primBounds));
}

UMeshField::~UMeshField()
{
  CUDA_SAFE_CALL(cudaFree(d_vertices));
  CUDA_SAFE_CALL(cudaFree(d_indices));
  CUDA_SAFE_CALL(cudaFree(d_cellIndices));
  CUDA_SAFE_CALL(cudaFree(d_cellTypes));
}

box3f UMeshField::computeWorldBounds() const
{
  return worldBounds;
}

UMeshField::DD UMeshField::getDD(const RankInfo &ri)
{
  DD dd;
  (VecField::DD &)dd = VecField::getDD(ri);
  dd.vertices    = d_vertices;
  dd.indices     = d_indices;
  dd.cellIndices = d_cellIndices;
  dd.cellTypes   = d_cellTypes;
  dd.uvw         = d_uvw;
  dd.numVertices = numVertices;
  dd.numIndices  = numIndices;
  dd.numCells    = numCells;
  dd.bvh         = bvh;
  return dd;
}

} // streami



