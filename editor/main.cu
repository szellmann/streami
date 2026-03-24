#include <fstream>

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
// anari-math
#include <anari/anari_cpp/ext/linalg.h>
// common
#include <dvr_course-common.h>
// umesh
#include "umesh/UMesh.h"
// streami:
#include "streami.h"
#include "field/Spherical.h"
#include "field/StructuredField.h"
#include "field/UMeshField.h"

using box3_t = std::array<anari::math::float3, 2>;
namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(box3_t, ANARI_FLOAT32_BOX3);
ANARI_TYPEFOR_DEFINITION(box3_t);
} // namespace anari

using namespace dvr_course;

inline __host__ __device__ vec3f toSpherical(const vec3f cartesian)
{
  float r = length(cartesian);
  float lat = asinf(cartesian.z/r);
  float lon = atan2f(cartesian.y, cartesian.x);
  return {r,lat,lon};
}

inline __host__ __device__ vec3f toCartesian(const vec3f spherical)
{
  const float r = spherical.x;
  const float lat = spherical.y;
  const float lon = spherical.z;

#ifdef __CUDA_ARCH__
  float x = r * __cosf(lat) * __cosf(lon);
  float y = r * __cosf(lat) * __sinf(lon);
  float z = r * __sinf(lat);
#else
  float x = r * cosf(lat) * cosf(lon);
  float y = r * cosf(lat) * sinf(lon);
  float z = r * sinf(lat);
#endif
  return {x,y,z};
}

struct {
  Transfunc transfunc;
  anari::SpatialField field{nullptr};
  anari::Volume volume{nullptr};
  anari::Geometry roiGeom{nullptr};
  anari::Geometry lineGeom{nullptr};
  box1f valueRange{0.f,1.f};
  float cylRadius{1.f};
  struct {
    box3f bounds{vec3f{1e20f},vec3f{-1e20f}};
    bool isSpherical{false};
  } roi;
} g_appState;

// ========================================================
// Log ANARI errors
// ========================================================
static void statusFunc(const void * /*userData*/,
    ANARIDevice /*device*/,
    ANARIObject source,
    ANARIDataType /*sourceType*/,
    ANARIStatusSeverity severity,
    ANARIStatusCode /*code*/,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
  } else if (severity == ANARI_SEVERITY_WARNING) {
    fprintf(stderr, "[WARN ][%p] %s\n", source, message);
  } else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING) {
    fprintf(stderr, "[PERF ][%p] %s\n", source, message);
  }
  // Ignore INFO/DEBUG messages
}

static anari::World generateWorld(anari::Device device, anari::Volume volume)
{
  g_appState.roiGeom = anari::newObject<anari::Geometry>(device, "cylinder");
  anari::commitParameters(device, g_appState.roiGeom);

  g_appState.lineGeom = anari::newObject<anari::Geometry>(device, "cylinder");
  anari::commitParameters(device, g_appState.lineGeom);

  auto mat = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, mat, "color", "color");
  anari::setParameter(device, mat, "alphaMode", "blend");
  anari::commitParameters(device, mat);

  auto roiSurface = anari::newObject<anari::Surface>(device);
  anari::setParameter(device, roiSurface, "geometry", g_appState.roiGeom);
  anari::setParameter(device, roiSurface, "material", mat);
  anari::commitParameters(device, roiSurface);

  auto lineSurface = anari::newObject<anari::Surface>(device);
  anari::setParameter(device, lineSurface, "geometry", g_appState.lineGeom);
  anari::setParameter(device, lineSurface, "material", mat);
  anari::commitParameters(device, lineSurface);

  // Create World //

  std::vector<anari::Surface> surface;
  surface.push_back(roiSurface);
  surface.push_back(lineSurface);

  auto group = anari::newObject<anari::Group>(device);
  anari::setAndReleaseParameter(
      device, group, "volume", anari::newArray1D(device, &volume));
  anari::setAndReleaseParameter(
      device, group, "surface", anari::newArray1D(device, surface.data(), surface.size()));
  anari::commitParameters(device, group);

  auto inst = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, inst, "group", group);
  anari::commitParameters(device, inst);

  anari::World world = anari::newObject<anari::World>(device);
  anari::setAndReleaseParameter(
      device, world, "instance", anari::newArray1D(device, &inst));

  anari::commitParameters(device, world);

  return world;
}

static anari::World generateScene(anari::Device device,
                                  const std::vector<vec3f> &vertices,
                                  const std::vector<int> &indices,
                                  const std::vector<int> &cellIndices,
                                  const std::vector<vec3f> &uvw)
{
  float minValue{1e30f};
  float maxValue{-1e30f};
  std::vector<float> magnitude(uvw.size());
  for (size_t i=0; i<uvw.size(); ++i) {
    magnitude[i] = length(uvw[i]);
    minValue = fminf(minValue,magnitude[i]);
    maxValue = fmaxf(maxValue,magnitude[i]);
  }

  enum {
    VTK_TET_ = 10,
    VTK_HEX_ = 12,
    VTK_WEDGE_ = 13,
    VTK_PYR_ = 14,
    VTK_BEZIER_HEX_ = 79,
  };
  std::vector<uint8_t> cellType(cellIndices.size());
  for (size_t i=0; i<cellType.size(); ++i) {
    cellType[i] = VTK_WEDGE_;
  }

  g_appState.field = anari::newObject<anari::SpatialField>(device, "unstructured");

  anari::setParameterArray1D(device,
      g_appState.field,
      "vertex.position",
      ANARI_FLOAT32_VEC3,
      vertices.data(),
      vertices.size());

  anari::setParameterArray1D(device,
      g_appState.field,
      "cell.type",
      ANARI_UINT8,
      cellType.data(),
      cellType.size());

  anari::setParameterArray1D(device,
      g_appState.field,
      "index",
      ANARI_UINT32,
      indices.data(),
      indices.size());

  anari::setParameterArray1D(device,
      g_appState.field,
      "cell.index",
      ANARI_UINT32,
      cellIndices.data(),
      cellIndices.size());

  anari::setParameterArray1D(device,
      g_appState.field,
      "cell.data",
      ANARI_FLOAT32,
      magnitude.data(),
      magnitude.size());

  anari::commitParameters(device, g_appState.field);

  auto &volume = g_appState.volume;
  volume = anari::newObject<anari::Volume>(device, "transferFunction1D");
  anari::setParameter(device, volume, "value", g_appState.field);

  std::vector<anari::math::float3> colors;
  std::vector<float> opacities;

  anari::math::float2 voxelRange(minValue,maxValue);
  anariSetParameter(
      device, volume, "valueRange", ANARI_FLOAT32_BOX1, &voxelRange);
  float unitDistance{100.f};
  anariSetParameter(
      device, volume, "unitDistance", ANARI_FLOAT32, &unitDistance);

  g_appState.valueRange = {voxelRange.x,voxelRange.y};

  anari::commitParameters(device, volume);

  return generateWorld(device, volume);
}

static anari::World generateScene(anari::Device device,
                                  const std::vector<vec3f> &values,
                                  vec3i org, vec3i dims)
{
  float minValue{1e30f};
  float maxValue{-1e30f};
  std::vector<float> magnitude(values.size());
  for (size_t i=0; i<values.size(); ++i) {
    magnitude[i] = length(values[i]);
    minValue = fminf(minValue,magnitude[i]);
    maxValue = fmaxf(maxValue,magnitude[i]);
  }

  g_appState.field = anari::newObject<anari::SpatialField>(device, "structuredRegular");

  auto scalar = anariNewArray3D(device,
      magnitude.data(),
      0,
      0,
      ANARI_FLOAT32,
      dims.x,
      dims.y,
      dims.z);
  anari::setAndReleaseParameter(device, g_appState.field, "data", scalar);

  anari::commitParameters(device, g_appState.field);

  auto &volume = g_appState.volume;
  volume = anari::newObject<anari::Volume>(device, "transferFunction1D");
  anari::setParameter(device, volume, "value", g_appState.field);

  std::vector<anari::math::float3> colors;
  std::vector<float> opacities;

  anari::math::float2 voxelRange(minValue,maxValue);
  anariSetParameter(
      device, volume, "valueRange", ANARI_FLOAT32_BOX1, &voxelRange);
  float unitDistance{0.02f};
  anariSetParameter(
      device, volume, "unitDistance", ANARI_FLOAT32, &unitDistance);

  g_appState.valueRange = {voxelRange.x,voxelRange.y};

  anari::commitParameters(device, volume);

  return generateWorld(device, volume);
}

static void drawROI(
    anari::Device device, vec3f p0, vec3f p1, bool spherical, bool visible)
{
  vec3f lower = min(p0,p1);
  vec3f upper = max(p0,p1);

  if (!visible) {
    anari::unsetParameter(device, g_appState.roiGeom, "vertex.position");
    anari::unsetParameter(device, g_appState.roiGeom, "primitive.index");
    anari::unsetParameter(device, g_appState.roiGeom, "vertex.color");
    anari::setParameter(device, g_appState.roiGeom, "radius", 0.f);
  } else if (spherical) {
    std::vector<anari::math::float3> vertices;
    std::vector<anari::math::uint2> indices;

    auto toCartesian = [](const vec3f &v) {
      auto vv = ::toCartesian(v);
      return anari::math::float3{vv.x,vv.y,vv.z};
    };

    int LAT=150, LON=150;

    float latInc = (upper.y-lower.y)/LAT;
    float lonInc = (upper.z-lower.z)/LON;

    // LAT
    vertices.push_back(toCartesian({lower.x,lower.y,lower.z}));
    for (int lat=1; lat<=LAT; ++lat) {
      vertices.push_back(toCartesian({lower.x,lower.y+lat*latInc,lower.z}));
      indices.push_back({vertices.size()-2,vertices.size()-1});
    }
    vertices.push_back(toCartesian({lower.x,lower.y,upper.z}));
    for (int lat=1; lat<=LAT; ++lat) {
      vertices.push_back(toCartesian({lower.x,lower.y+lat*latInc,upper.z}));
      indices.push_back({vertices.size()-2,vertices.size()-1});
    }
    vertices.push_back(toCartesian({upper.x,lower.y,upper.z}));
    for (int lat=1; lat<=LAT; ++lat) {
      vertices.push_back(toCartesian({upper.x,lower.y+lat*latInc,upper.z}));
      indices.push_back({vertices.size()-2,vertices.size()-1});
    }
    vertices.push_back(toCartesian({upper.x,lower.y,lower.z}));
    for (int lat=1; lat<=LAT; ++lat) {
      vertices.push_back(toCartesian({upper.x,lower.y+lat*latInc,lower.z}));
      indices.push_back({vertices.size()-2,vertices.size()-1});
    }
    // LON
    vertices.push_back(toCartesian({lower.x,lower.y,lower.z}));
    for (int lon=1; lon<=LON; ++lon) {
      vertices.push_back(toCartesian({lower.x,lower.y,lower.z+lon*lonInc}));
      indices.push_back({vertices.size()-2,vertices.size()-1});
    }
    vertices.push_back(toCartesian({lower.x,upper.y,lower.z}));
    for (int lon=1; lon<=LON; ++lon) {
      vertices.push_back(toCartesian({lower.x,upper.y,lower.z+lon*lonInc}));
      indices.push_back({vertices.size()-2,vertices.size()-1});
    }
    vertices.push_back(toCartesian({upper.x,upper.y,lower.z}));
    for (int lon=1; lon<=LON; ++lon) {
      vertices.push_back(toCartesian({upper.x,upper.y,lower.z+lon*lonInc}));
      indices.push_back({vertices.size()-2,vertices.size()-1});
    }
    vertices.push_back(toCartesian({upper.x,lower.y,lower.z}));
    for (int lon=1; lon<=LON; ++lon) {
      vertices.push_back(toCartesian({upper.x,lower.y,lower.z+lon*lonInc}));
      indices.push_back({vertices.size()-2,vertices.size()-1});
    }

    anari::setParameterArray1D(device,
        g_appState.roiGeom,
        "vertex.position",
        ANARI_FLOAT32_VEC3,
        vertices.data(),
        vertices.size());
    anari::setParameterArray1D(device,
        g_appState.roiGeom,
        "primitive.index",
        ANARI_UINT32_VEC2,
        indices.data(),
        indices.size());
    anari::setParameter(device, g_appState.roiGeom, "radius", g_appState.cylRadius);
  } else {
    anari::math::float3 vertices[8] = {
      {lower.x,lower.y,lower.z},
      {upper.x,lower.y,lower.z},
      {upper.x,upper.y,lower.z},
      {lower.x,upper.y,lower.z},
      {lower.x,lower.y,upper.z},
      {upper.x,lower.y,upper.z},
      {upper.x,upper.y,upper.z},
      {lower.x,upper.y,upper.z},
    };

    anari::math::uint2 indices[12] = {
      {0,1}, {1,2}, {2,3}, {3,0},
      {4,5}, {5,6}, {6,7}, {7,4},
      {0,4},
      {1,5},
      {2,6},
      {3,7},
    };

    anari::setParameterArray1D(device,
        g_appState.roiGeom,
        "vertex.position",
        ANARI_FLOAT32_VEC3,
        vertices,
        sizeof(vertices)/sizeof(vertices[0]));
    anari::setParameterArray1D(device,
        g_appState.roiGeom,
        "primitive.index",
        ANARI_UINT32_VEC2,
        indices,
        sizeof(indices)/sizeof(indices[0]));
    anari::setParameter(device, g_appState.roiGeom, "radius", g_appState.cylRadius);
  }
  anari::commitParameters(device, g_appState.roiGeom);
}

static void drawStreamlines(anari::Device device,
                            const std::vector<streami::Tracer::Line> &lines)
{
  std::vector<anari::math::float3> vertices;
  std::vector<anari::math::uint2> indices;
  std::vector<anari::math::float3> colors;

  for (size_t i=0; i<lines.size(); ++i) {
    if (lines[i].empty()) continue;
    vec3f v0 = lines[i][0].p.P;
    vec3f c0 = lines[i][0].color;
    vertices.push_back({v0.x,v0.y,v0.z});
    colors.push_back({c0.x,c0.y,c0.z});
    for (size_t j=1; j<lines[i].size(); ++j) {
      vec3f v = lines[i][j].p.P;
      vec3f c = lines[i][j].color;
      vertices.push_back({v.x,v.y,v.z});
      colors.push_back({c.x,c.y,c.z});
      indices.push_back({vertices.size()-2,vertices.size()-1});
    }
  }

  if (!vertices.empty() && !indices.empty()) {
    anari::setParameterArray1D(device,
        g_appState.lineGeom,
        "vertex.position",
        ANARI_FLOAT32_VEC3,
        vertices.data(),
        vertices.size());
    anari::setParameterArray1D(device,
        g_appState.lineGeom,
        "primitive.index",
        ANARI_UINT32_VEC2,
        indices.data(),
        indices.size());
    anari::setParameterArray1D(device,
        g_appState.lineGeom,
        "vertex.color",
        ANARI_FLOAT32_VEC3,
        colors.data(),
        colors.size());
    anari::setParameter(device, g_appState.lineGeom, "radius", g_appState.cylRadius);
  }

  anari::commitParameters(device, g_appState.lineGeom);
}

static void updateLUT(anari::Device device, const Transfunc &tf) {

  auto &volume = g_appState.volume;

  std::vector<anari::math::float3> colors;
  std::vector<float> opacities;
  auto lut = tf.getLUT();
  for (int i=0; i<lut.size(); ++i) {
    colors.push_back(*(anari::math::float3 *)&lut[i].xyz);
    opacities.push_back(lut[i].w);
  }
  anari::setAndReleaseParameter(device,
      volume,
      "color",
      anari::newArray1D(device, colors.data(), colors.size()));
  anari::setAndReleaseParameter(device,
      volume,
      "opacity",
      anari::newArray1D(device, opacities.data(), opacities.size()));

  anari::commitParameters(device, volume);
}

int main(int argc, char *argv[]) {

  streami::Context ctx(argc, argv);

  streami::Tracer::Params parms;
  parms.numParticles=5;
  parms.maxSteps=1000;
  parms.stepSize=0.5f;
  parms.minLength=1e-3f;

  streami::Tracer tracer(ctx,parms);

  Pipeline pl(argc, argv, "ex00_hello_dvr_course");

  std::vector<std::string> fileNames;
  streami::Tracer::Params::Mode mode{streami::Tracer::Params::Streamlines};
  vec3i org=0.f, dims=0.f;
  for (int i=1;i<argc;i++) {
    std::string arg(argv[i]);
    if (arg[0] == '-') {
      if (arg == "-dims") {
        dims.x = std::stoi(argv[++i]);
        dims.y = std::stoi(argv[++i]);
        dims.z = std::stoi(argv[++i]);
      }
      if (arg == "-org") {
        org.x = std::stoi(argv[++i]);
        org.y = std::stoi(argv[++i]);
        org.z = std::stoi(argv[++i]);
      }
      if (arg == "-streaklines") {
        mode = streami::Tracer::Params::Streaklines;
      }
      if (arg == "-roi") {
        g_appState.roi.bounds.lower.x = atof(argv[++i]);
        g_appState.roi.bounds.lower.y = atof(argv[++i]);
        g_appState.roi.bounds.lower.z = atof(argv[++i]);
        g_appState.roi.bounds.upper.x = atof(argv[++i]);
        g_appState.roi.bounds.upper.y = atof(argv[++i]);
        g_appState.roi.bounds.upper.z = atof(argv[++i]);
        g_appState.roi.isSpherical = false;
      }
      if (arg == "-stepsize") {
        parms.stepSize = atof(argv[++i]);
      }
      if (arg == "-minlength") {
        parms.minlength = atof(argv[++i]);
      }
      if (arg == "-numparticles") {
        parms.numParticles = std::stoi(argv[++i]);
      }
      if (arg == "-numsteps") {
        parms.maxSteps = std::stoi(argv[++i]);
      }
      if (arg == "-tuberadius") {
        g_appState.cylRadius = atof(argv[++i]);
      }
    } else {
      fileNames.push_back(arg);
    }
  }

  auto library = anari::loadLibrary("environment", statusFunc);
  auto device = anari::newDevice(library, "default");

  anari::World world{nullptr};
  
  streami::VecField::SP field{nullptr};

  streami::RankInfo ri{0,1};//rafi->mpi.rank,rafi->mpi.size};

  box3f worldBounds, sphericalBounds;
  if (endsWith(fileNames[0],".umesh")) {
    for (size_t f=0; f<fileNames.size(); ++f) {
      std::string fileName = fileNames[f];
      umesh::UMesh::SP inMesh = umesh::UMesh::loadFrom(fileName);

      auto umeshBounds = inMesh->getBounds();
      worldBounds = box3f{
        {umeshBounds.lower.x,umeshBounds.lower.y,umeshBounds.lower.z},
        {umeshBounds.upper.x,umeshBounds.upper.y,umeshBounds.upper.z}
      };

      vec3i gridSize(1);
      streami::MacroCell localMC
          = makeMacroCell(worldBounds,gridSize,ri,worldBounds.size().x/10.f);

      std::vector<vec3f> vertices;
      std::vector<int> indices;
      std::vector<int> cellIndices;
      std::vector<vec3f> uvw;

      std::vector<int> old2new(inMesh->vertices.size(),-1);

      sphericalBounds = box3f(vec3f(1e30f),vec3f(-1e30f));

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
            sphericalBounds.extend(toSpherical(v));
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

      field = std::make_shared<streami::UMeshField>(vertices.data(),
                                                    indices.data(),
                                                    cellIndices.data(),
                                                    uvw.data(),
                                                    vertices.size(),
                                                    indices.size(),
                                                    cellIndices.size());
      tracer.setField((const streami::UMeshField::SP &)field, f);

      if (f == 0) {
        world = generateScene(device,vertices,indices,cellIndices,uvw);
      }
    }
  } else {
    if (dims.x*dims.y*dims.z<=0) {
      std::cerr << "-dims invalid\n";
      return 0;
    }

    for (size_t f=0; f<fileNames.size(); ++f) {
      std::string fileName = fileNames[f];
      std::ifstream in(fileName);
      std::vector<vec3f> values(dims.x*size_t(dims.y)*dims.z);
      in.read((char *)values.data(),sizeof(values[0])*values.size());

      worldBounds = box3f{
        {(float)org.x,(float)org.y,(float)org.z},
        {(float)dims.x,(float)dims.y,(float)dims.z}
      };

      vec3i gridSize(1);
      float halo(20.f);
      streami::MacroCell localMC = streami::makeMacroCell(worldBounds,gridSize,ri,halo);

      field = std::make_shared<streami::StructuredField>(values.data(),dims,org);
      field->numMCs = gridSize;
      field->mc = localMC;
      tracer.setField((const streami::StructuredField::SP &)field, f);

      if (f == 0) {
        world = generateScene(device,values,org,dims);
      }
    }
  }

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(&fb);

  if (!pl.transfuncValid()) {
    auto &tf = g_appState.transfunc;
    std::vector<vec4f> tfValues({
      {0.f,0.f,1.f,0.1f },
      {0.f,1.f,0.f,0.1f }
    });
    tf.valueRange = g_appState.valueRange;
    tf.setLUT(tfValues);
    pl.setTransfunc(&tf);
  }

  pl.setTransfuncUpdateHandler([&](const Transfunc *tf,int) {
    updateLUT(device,*tf);
  });

  auto renderer = anari::newObject<anari::Renderer>(device, "default");
  const anari::math::float4 backgroundColor = {0.1f, 0.1f, 0.1f, 1.f};
  anari::setParameter(device, renderer, "background", backgroundColor);
  const anari::math::float3 ambientColor = {1.f, 1.f, 1.f};
  anari::setParameter(device, renderer, "ambientColor", ambientColor);
  anari::setParameter(device, renderer, "ambientRadiance", 0.2f);
  anari::setParameter(device, renderer, "pixelSamples", 1);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);

  anari::math::uint2 imageSize = {imgWidth, imgHeight};
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);

  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "renderer", renderer);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");

  anari::commitParameters(device, camera);
  anari::setParameter(device, frame, "camera", camera);
  anari::commitParameters(device, frame);

  box3_t bounds;
  anari::getProperty(device, world, "bounds", bounds, ANARI_WAIT);

  Camera cam;
  cam.viewAll(*(box3f *)&bounds);
  pl.setCamera(&cam);

  //box3f worldBounds = *(box3f*)&bounds;

  bool roiVisible=true, prevRoiVisible=false;
  pl.uiParam("roi.visible", &roiVisible);

  bool sphericalROI=false, prevSphericalROI=false;
  pl.uiParam("roi.spherical", &sphericalROI);

  box3f roiBounds=worldBounds;
  if (!g_appState.roi.bounds.empty()) {
    roiBounds = g_appState.roi.bounds;
  }
  if (sphericalROI) {
    roiBounds=sphericalBounds;
  }
  vec3f lower=roiBounds.lower, prevLower=roiBounds.lower;
  vec3f size=roiBounds.size(), prevSize=roiBounds.size();
  pl.uiParam("roi.lo", &lower, roiBounds.lower, roiBounds.upper);
  pl.uiParam("roi.size", &size, vec3f(1.f), roiBounds.size());

  int numParticles=parms.numParticles, prevNumParticles=parms.numParticles;
  pl.uiParam("# particles", &numParticles, 1, 1<<16);

  float stepSize=parms.stepSize, prevStepSize=parms.stepSize;
  pl.uiParam("step size", &stepSize, 1e-3f, 16.f);

  float minLength=parms.minLength, prevMinLength=parms.minLength;
  pl.uiParam("min length", &minLength, 1e-6f, 1.f);

  pl.uiParam("STEP", [&](){ tracer.step(); drawStreamlines(device,tracer.getLines()); });

  do {
    struct {
      vec3f lower_left, horizontal, vertical;
    } screen;
    cam.getScreen(screen.lower_left,screen.horizontal,screen.vertical);

    if (lower != prevLower || size != prevSize || numParticles != prevNumParticles ||
        stepSize != prevStepSize || minLength != prevMinLength ||
        sphericalROI != prevSphericalROI || roiVisible != prevRoiVisible)
    {
      parms.mode             = mode;
      parms.roi.bounds.lower = lower;
      parms.roi.bounds.upper = lower+size;
      parms.numParticles     = numParticles;
      parms.stepSize         = stepSize;
      parms.minLength        = minLength;
      tracer.setParams(parms);

      prevLower        = lower;
      prevSize         = size;
      prevNumParticles = numParticles;
      prevStepSize     = stepSize;
      prevMinLength    = minLength;
      prevSphericalROI = sphericalROI;
      prevRoiVisible   = roiVisible;

      auto p0 = parms.roi.bounds.lower;
      auto p1 = parms.roi.bounds.upper;

      drawROI(device,p0,p1,sphericalROI,roiVisible);

      drawStreamlines(device,tracer.getLines());

      std::cout << "-roi " << p0.x << ' ' << p0.y << ' ' << p0.z << ' ' << p1.x << ' ' << p1.y << ' ' << p1.z << '\n';
    }

    anari::math::float3 eye(cam.getPosition().x,cam.getPosition().y,cam.getPosition().z);
    anari::math::float3 dir(cam.getPOI().x,cam.getPOI().y,cam.getPOI().z);
    dir -= eye;
    anari::math::float3 up(cam.getUp().x,cam.getUp().y,cam.getUp().z);

    anari::setParameter(device, camera, "position", eye);
    anari::setParameter(device, camera, "direction", dir);
    anari::setParameter(device, camera, "up", up);
    anari::setParameter(device, camera, "fovy", cam.getFovyInRadians());
    anari::setParameter(device, camera, "aspect", (float)fb.width/fb.height);
    anari::commitParameters(device, camera);

    anari::math::uint2 imageSize = {fb.width, fb.height};
    anari::setParameter(device, frame, "size", imageSize);
    anari::commitParameters(device, frame);

    anari::render(device, frame);
    anari::wait(device, frame);

    auto af = anari::map<uint32_t>(device, frame, "channel.color");

    memcpy(fb.fbPointer,af.data,sizeof(uint32_t)*af.width*af.height);
  
    anari::unmap(device, frame, "channel.color");

    pl.launch();
    pl.present();
  } while (pl.isRunning());
}



