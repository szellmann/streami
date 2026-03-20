#include <fstream>

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
// anari-math
#include <anari/anari_cpp/ext/linalg.h>

#include <dvr_course-common.h>

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

struct {
  Transfunc transfunc;
  anari::SpatialField field{nullptr};
  anari::Volume volume{nullptr};
  anari::Geometry roiGeom{nullptr};
  anari::Geometry lineGeom{nullptr};
  box1f valueRange{0.f,1.f};
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
  //anari::setParameterArray3D(device,
  //    g_appState.field,
  //    "data",
  //    ANARI_FLOAT32,
  //    magnitude.data(),
  //    dims.x,dims.y,dims.z);

  anari::commitParameters(device, g_appState.field);

  auto &volume = g_appState.volume;
  volume = anari::newObject<anari::Volume>(device, "transferFunction1D");
  anari::setParameter(device, volume, "value", g_appState.field);

  std::vector<anari::math::float3> colors;
  std::vector<float> opacities;

  //anari::setAndReleaseParameter(device,
  //    volume,
  //    "color",
  //    anari::newArray1D(device, colors.data(), colors.size()));
  //anari::setAndReleaseParameter(device,
  //    volume,
  //    "opacity",
  //    anari::newArray1D(device, opacities.data(), opacities.size()));
  anari::math::float2 voxelRange(minValue,maxValue);
  anariSetParameter(
      device, volume, "valueRange", ANARI_FLOAT32_BOX1, &voxelRange);
  float unitDistance{0.02f};
  anariSetParameter(
      device, volume, "unitDistance", ANARI_FLOAT32, &unitDistance);

  g_appState.valueRange = {voxelRange.x,voxelRange.y};

  anari::commitParameters(device, volume);

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

static void drawROI(anari::Device device, vec3f p0, vec3f p1) {
  vec3f lower = min(p0,p1);
  vec3f upper = max(p0,p1);
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
  anari::setParameter(device, g_appState.roiGeom, "radius", 1.f);
  anari::commitParameters(device, g_appState.roiGeom);
}

static void drawStreamlines(anari::Device device,
                            const std::vector<streami::Tracer::Line> &lines)
{
  std::vector<anari::math::float3> vertices;
  std::vector<anari::math::uint2> indices;

  for (size_t i=0; i<lines.size(); ++i) {
    if (lines[i].empty()) continue;
    vec3f v0 = lines[i][0].p.P;
    vertices.push_back({v0.x,v0.y,v0.z});
    for (size_t j=1; j<lines[i].size(); ++j) {
      vec3f v = lines[i][j].p.P;
      vertices.push_back({v.x,v.y,v.z});
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
    anari::setParameter(device, g_appState.lineGeom, "radius", 1.f);
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

  Pipeline pl(argc, argv, "ex00_hello_dvr_course");

  std::string fileName;
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
    } else {
      fileName = arg;
    }
  }

  if (dims.x*dims.y*dims.z<=0) {
    std::cerr << "-dims invalid\n";
    return 0;
  }

  std::ifstream in(fileName);
  std::vector<vec3f> values(dims.x*size_t(dims.y)*dims.z);
  in.read((char *)values.data(),sizeof(values[0])*values.size());

  box3f worldBounds{
    {(float)org.x,(float)org.y,(float)org.z},
    {(float)dims.x,(float)dims.y,(float)dims.z}
  };

  streami::RankInfo ri{0,1};//rafi->mpi.rank,rafi->mpi.size};

  vec3i gridSize(1);
  float halo(20.f);
  streami::MacroCell localMC = streami::makeMacroCell(worldBounds,gridSize,ri,halo);

  auto field = std::make_shared<streami::StructuredField>(values.data(),dims,org);
  field->numMCs = gridSize;
  field->mc = localMC;

  streami::Tracer::Params parms;
  parms.numParticles=5;
  parms.maxSteps=1000;
  parms.stepSize=0.5f;
  parms.minLength=1e-3f;

  streami::Tracer tracer(ctx,parms);

  tracer.setField(field);

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(&fb);

  auto library = anari::loadLibrary("environment", statusFunc);
  auto device = anari::newDevice(library, "default");

  auto world = generateScene(device,values,org,dims);

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

  vec3f lower=worldBounds.lower, prevLower=worldBounds.lower;
  pl.uiParam("roi.lo", &lower, worldBounds.lower, worldBounds.upper);

  vec3f size=worldBounds.size(), prevSize=worldBounds.size();
  pl.uiParam("roi.size", &size, vec3f(1.f), worldBounds.size());

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
        stepSize != prevStepSize || minLength != prevMinLength)
    {
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

      auto p0 = parms.roi.bounds.lower;
      auto p1 = parms.roi.bounds.upper;

      drawROI(device,p0,p1);
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



