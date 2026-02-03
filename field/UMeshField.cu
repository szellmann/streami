
#include "UMeshField.h"

namespace streami {

__global__ void computeBounds(box3f *primBounds,
                              box3f *worldBounds,
                              const vec3f *vertices,
                              const int *indices,
                              const int *cellIndices,
                              int numCells)
{
  int cellID = threadIdx.x+blockIdx.x*blockDim.x;
  if (cellID >= numCells) return;

  const int *I = indices + cellIndices[cellID];
  const vec3f v0 = vertices[I[0]];
  const vec3f v1 = vertices[I[1]];
  const vec3f v2 = vertices[I[2]];
  const vec3f v3 = vertices[I[3]];
  const vec3f v4 = vertices[I[4]];
  const vec3f v5 = vertices[I[5]];

  // primBounds:
  primBounds[cellID] = box3f(vec3f(FLT_MAX),vec3f(-FLT_MAX));
  primBounds[cellID].extend(v0);
  primBounds[cellID].extend(v1);
  primBounds[cellID].extend(v2);
  primBounds[cellID].extend(v3);
  primBounds[cellID].extend(v4);
  primBounds[cellID].extend(v5);

  // worldBounds:
  atomicMin(&worldBounds->lower.x,primBounds[cellID].lower.x);
  atomicMin(&worldBounds->lower.y,primBounds[cellID].lower.y);
  atomicMin(&worldBounds->lower.z,primBounds[cellID].lower.z);
  atomicMax(&worldBounds->upper.x,primBounds[cellID].upper.x);
  atomicMax(&worldBounds->upper.y,primBounds[cellID].upper.y);
  atomicMax(&worldBounds->upper.z,primBounds[cellID].upper.z);
}

UMeshField::UMeshField(vec3f *vertices, int *indices, int *cellIndices, vec3f *uvw,
                       int numVertices, int numIndices, int numCells)
  : numVertices(numVertices), numIndices(numIndices), numCells(numCells)
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

  CUDA_SAFE_CALL(cudaMalloc(&d_uvw, sizeof(uvw[0])*numCells));
  CUDA_SAFE_CALL(cudaMemcpy(d_uvw,
                            uvw,
                            sizeof(uvw[0])*numCells,
                            cudaMemcpyHostToDevice));

  worldBounds = box3f(vec3f(FLT_MAX),vec3f(-FLT_MAX));

  box3f *primBounds;
  CUDA_SAFE_CALL(cudaMalloc(&primBounds, sizeof(primBounds[0])*numCells));

  box3f *dWorldBounds;
  CUDA_SAFE_CALL(cudaMalloc(&dWorldBounds, sizeof(*dWorldBounds)));
  CUDA_SAFE_CALL(cudaMemcpy(dWorldBounds,&worldBounds,sizeof(worldBounds),
                            cudaMemcpyHostToDevice));

  computeBounds<<<iDivUp(numCells,1024),1024>>>(primBounds,
                                                dWorldBounds,
                                                vertices,
                                                indices,
                                                cellIndices,
                                                numCells);

  CUDA_SAFE_CALL(cudaMemcpy(&worldBounds,dWorldBounds,sizeof(worldBounds),
                            cudaMemcpyDeviceToHost));
  std::cout << "WORLD BOUNDS: " << worldBounds << '\n';

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
}

UMeshField::DD UMeshField::getDD(const RankInfo &ri)
{
  DD dd;
  dd.vertices    = d_vertices;
  dd.indices     = d_indices;
  dd.cellIndices = d_cellIndices;
  dd.uvw         = d_uvw;
  dd.numVertices = numVertices;
  dd.numIndices  = numIndices;
  dd.numCells    = numCells;
  dd.bvh         = bvh;
  // Base:
  dd.worldBounds = worldBounds;
  float halo = 0.f;
  dd.buildMCs(worldBounds,vec3i(1),ri,halo);
  return dd;
}

} // streami



