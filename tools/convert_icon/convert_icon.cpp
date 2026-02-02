// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <netcdf.h>
#include "umesh/UMesh.h"

struct {
  std::string horizontalGridFile; // -hgrid
  std::string hsurfFile; // -hsurf
  std::vector<std::string> hhlFiles; // -hhl
  std::vector<std::string> uFiles; // -u (zonal wind, from E to W)
  std::vector<std::string> vFiles; // -v (meridional wind, from N to S)
  std::vector<std::string> wFiles; // -w (vertical wind)
  std::string outfileBase{"out"};
  bool convertToUMesh{true};
  int maxLayers{4};
} g_appState;

static void printHelp() {
  std::cout << "SYNOPSIS (TODO)\n\n";
}

inline int div_up(int a, int b) {
  return (a+b-1)/b;
}

inline umesh::vec3f toCartesian(const umesh::vec3f spherical)
{
  const float r = spherical.x;
  const float lat = spherical.y;
  const float lon = spherical.z;

  float x = r * cosf(lat) * cosf(lon);
  float y = r * cosf(lat) * sinf(lon);
  float z = r * sinf(lat);
  return {x,y,z};
}

static size_t readDimLength(int ncid, std::string name) {
  int retval, dimid;
  if ((retval != nc_inq_dimid(ncid, name.c_str(), &dimid)) != NC_NOERR) {
    fprintf(stderr, "dim %s not found: %s\n", name.c_str(), nc_strerror(retval));
    return ~0ull;
  }

  size_t result;
  if ((retval = nc_inq_dimlen(ncid, dimid, &result)) != NC_NOERR) {
    fprintf(stderr, "variable %s found but size mismatch\n", name.c_str());
    return ~0ull;
  }

  return result;
}

static std::vector<int> readIntVar(int ncid, std::string name, size_t len) {
  int retval, varid;
  if ((retval = nc_inq_varid(ncid, name.c_str(), &varid)) != NC_NOERR) {
    fprintf(stderr, "variable %s not found\n", name.c_str());
    return {};
  }

  std::vector<int> result(len);

  if ((retval = nc_get_var_int(ncid, varid, result.data())) != NC_NOERR) {
    fprintf(stderr, "cannot read from variable %s\n", name.c_str());
    return {};
  }

  if (result.size() != len) {
    fprintf(stderr, "variable %s found but size mismatch\n", name.c_str());
    return {};
  }

  return result;
}

static std::vector<double> readDoubleVar(int ncid, std::string name, size_t len) {
  int retval, varid;
  if ((retval = nc_inq_varid(ncid, name.c_str(), &varid)) != NC_NOERR) {
    fprintf(stderr, "variable %s not found\n", name.c_str());
    return {};
  }

  std::vector<double> result(len);

  if ((retval = nc_get_var_double(ncid, varid, result.data())) != NC_NOERR) {
    fprintf(stderr, "cannot read from variable %s\n", name.c_str());
    return {};
  }

  if (result.size() != len) {
    fprintf(stderr, "variable %s found but size mismatch\n", name.c_str());
    return {};
  }

  return result;
}


static void parseCommandLine(int argc, char *argv[]) {
  enum Mode { Hgrid, Hsurf, Hhl, U, V, W, None, };
  Mode mode{None};

  for (int i=1; i<argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-') {
      if (mode == Hgrid) {
        g_appState.horizontalGridFile = arg;
      }
      else if (mode == Hsurf) {
        g_appState.hsurfFile = arg;
      }
      else if (mode == Hhl) {
        g_appState.hhlFiles.push_back(arg);
      }
      else if (mode == U) {
        g_appState.uFiles.push_back(arg);
      }
      else if (mode == V) {
        g_appState.vFiles.push_back(arg);
      }
      else if (mode == W) {
        g_appState.wFiles.push_back(arg);
      }
      else {
        fprintf(stderr, "Unknown parm: %s\n", argv[i]);
        break;
      }
    }
    else if (arg == "-hgrid") {
      mode = Hgrid;
    }
    else if (arg == "-hsurf") {
      mode = Hsurf;
    }
    else if (arg == "-hhl") {
      mode = Hhl;
    }
    else if (arg == "-u") {
      mode = U;
    }
    else if (arg == "-v") {
      mode = V;
    }
    else if (arg == "-w") {
      mode = W;
    }
    else if (arg == "-o") {
      g_appState.outfileBase = argv[++i];
    }
  }
}

static void printUsage() {
  fprintf(stderr, "%s\n",
    "Usage: ./convert_icon -hgrid <hg.nc> -hsurf <hs.nc> -hhl [hh.nc*] -data [df.nc*]");
}

int main(int argc, char *argv[]) {
  if (argc < 3 || std::string(argv[1]) == "help" ) {
    printHelp();
    return 1;
  }

  parseCommandLine(argc, argv);

  if (g_appState.horizontalGridFile.empty() ||
      g_appState.hsurfFile.empty() ||
      g_appState.hhlFiles.empty() || g_appState.hsurfFile.empty()) {
    printUsage();
    return 1;
  }

  int ncid, retval;

  // Horizontal grid file:

  if ((retval = nc_open(g_appState.horizontalGridFile.c_str(), NC_NOWRITE, &ncid)) != NC_NOERR) {
    printf("Error opening file: %s\n", nc_strerror(retval));
    return 1;
  }

  // read number of cells:
  size_t cell = readDimLength(ncid, "cell");
  printf("number of cells: %i\n",(int)cell);

  // read number of vertices:
  size_t vertex = readDimLength(ncid, "vertex");
  printf("number of vertices: %i\n",(int)vertex);

  // read clon_vertices & clat_vertices:

  auto clon_vertices = readDoubleVar(ncid, "clon_vertices", cell*3);
  auto clat_vertices = readDoubleVar(ncid, "clat_vertices", cell*3);

  nc_close(ncid);

  if (clon_vertices.empty() || clat_vertices.empty()) {
    fprintf(stderr, "%s\n", "Cannot proceed as lon/lat coordinates missing");
    nc_close(ncid);
    return 1;
  }

  struct DataField {
    int height{0};
    std::vector<float> value;
  };

  struct VectorField {
    int height{0};
    std::vector<umesh::vec3f> value;
  };

  // HSURF file:

  if ((retval = nc_open(g_appState.hsurfFile.c_str(), NC_NOWRITE, &ncid)) != NC_NOERR) {
    printf("Error opening file: %s\n", nc_strerror(retval));
    return 1;
  }

  auto hsurf = readDoubleVar(ncid, "HSURF", cell);

  if (clon_vertices.empty() || clat_vertices.empty()) {
    fprintf(stderr, "%s\n", "Cannot proceed as HSURF is ill-formed");
    nc_close(ncid);
    return 1;
  }


  // HHL files:

  std::vector<DataField> hhl(g_appState.hhlFiles.size());
  int hhlHeightBounds[2] = {INT_MAX, INT_MIN};

  for (int i=0; i<g_appState.hhlFiles.size(); ++i) {
    std::string hhlFile = g_appState.hhlFiles[i];
    if ((retval = nc_open(hhlFile.c_str(), NC_NOWRITE, &ncid)) != NC_NOERR) {
      printf("Error opening file: %s\n", nc_strerror(retval));
      return 1;
    }

    DataField &field = hhl[i];

    auto height = readDoubleVar(ncid, "height", 1);
    if (height.empty()) {
      fprintf(stderr, "No height found in %s, aborting...\n", hhlFile.c_str());
      nc_close(ncid);
      return 1;
    }

    field.height = (int)height[0];

    hhlHeightBounds[0] = std::min(hhlHeightBounds[0],field.height);
    hhlHeightBounds[1] = std::max(hhlHeightBounds[1],field.height);

    auto hhlI = readDoubleVar(ncid, "HHL", cell);
    if (hhlI.empty()) {
      fprintf(stderr, "HHL not propery read from %s, aborting...\n", hhlFile.c_str());
      nc_close(ncid);
      return 1;
    }

    for (int j=0; j<cell; ++j) {
      field.value.push_back((float)hhlI[j]);
    }

    nc_close(ncid);
  }

  std::sort(hhl.begin(), hhl.end(), [](auto &a, auto &b) { return a.height > b.height; });


  // U/V/W files:

  std::vector<VectorField> values(g_appState.uFiles.size());
  int valuesHeightBounds[2] = {INT_MAX, INT_MIN};

  if (g_appState.uFiles.size() != g_appState.vFiles.size() ||
      g_appState.uFiles.size() != g_appState.wFiles.size()) {
    printf("Need exactly the same number of u/v/w files, but got: %i %i %i\n",
           (int)g_appState.uFiles.size(),
           (int)g_appState.vFiles.size(),
           (int)g_appState.wFiles.size());
  }

  for (int i=0; i<g_appState.uFiles.size(); ++i) {
    int ncid_u, ncid_v, ncid_w;
    std::string uFile = g_appState.uFiles[i];
    if ((retval = nc_open(uFile.c_str(), NC_NOWRITE, &ncid_u)) != NC_NOERR) {
      printf("Error opening file: %s\n", nc_strerror(retval));
      return 1;
    }

    std::string vFile = g_appState.vFiles[i];
    if ((retval = nc_open(vFile.c_str(), NC_NOWRITE, &ncid_v)) != NC_NOERR) {
      printf("Error opening file: %s\n", nc_strerror(retval));
      return 1;
    }

    std::string wFile = g_appState.wFiles[i];
    if ((retval = nc_open(wFile.c_str(), NC_NOWRITE, &ncid_w)) != NC_NOERR) {
      printf("Error opening file: %s\n", nc_strerror(retval));
      return 1;
    }

    VectorField &field = values[i];

    // read number of cells:
    size_t ncells = readDimLength(ncid_u, "ncells");
    printf("number of cells IN DATA FILE: %i\n",(int)ncells);

    auto height = readDoubleVar(ncid_u, "height", 1);
    if (height.empty()) {
      fprintf(stderr, "No height found in %s, aborting...\n", uFile.c_str());
      nc_close(ncid_u);
      nc_close(ncid_v);
      nc_close(ncid_w);
      return 1;
    }

    field.height = (int)height[0];

    valuesHeightBounds[0] = std::min(valuesHeightBounds[0],field.height);
    valuesHeightBounds[1] = std::max(valuesHeightBounds[1],field.height);

    // read vars:
    auto uvar = readDoubleVar(ncid_u, "u", ncells);
    if (uvar.empty()) {
      fprintf(stderr, "Error reading variable u, error: %s\n", nc_strerror(retval));
      nc_close(ncid_u);
      return 1;
    }

    auto vvar = readDoubleVar(ncid_v, "v", ncells);
    if (vvar.empty()) {
      fprintf(stderr, "Error reading variable v, error: %s\n", nc_strerror(retval));
      nc_close(ncid_v);
      return 1;
    }

    auto wvar = readDoubleVar(ncid_w, "wz", ncells);
    if (wvar.empty()) {
      fprintf(stderr, "Error reading variable wz, error: %s\n", nc_strerror(retval));
      nc_close(ncid_w);
      return 1;
    }

    for (int j=0; j<cell; ++j) {
      field.value.push_back({(float)uvar[j],(float)vvar[j],(float)wvar[j]});
    }

    nc_close(ncid_u);
    nc_close(ncid_v);
    nc_close(ncid_w);
  }

  std::sort(values.begin(), values.end(), [](auto &a, auto &b) { return a.height > b.height; });

  if (!(hhlHeightBounds[0] == valuesHeightBounds[0] && hhlHeightBounds[1] == valuesHeightBounds[1])) {
    fprintf(stderr, "%s\n", "Heights of HHL and data field don't match, aborting...");
  }

  int heightBounds[2] = { hhlHeightBounds[0], hhlHeightBounds[1] };

  int numLayers(g_appState.uFiles.size());

  if (numLayers > g_appState.maxLayers) {
    numLayers = g_appState.maxLayers;
  }

  if (g_appState.convertToUMesh) {
    using namespace umesh;
    auto output = std::make_shared<UMesh>();
    output->perVertex = std::make_shared<Attribute>();
    for (int cellID=0; cellID<cell; ++cellID) {
      float lat[3]{(float)clat_vertices[cellID*3],(float)clat_vertices[cellID*3+1],(float)clat_vertices[cellID*3+2]};
      float lon[3]{(float)clon_vertices[cellID*3],(float)clon_vertices[cellID*3+1],(float)clon_vertices[cellID*3+2]};;
      constexpr float R = 6.371229E6f;
      constexpr float scale = 1.f;
      for (int j=0; j<numLayers; ++j) {
        float h1 = j==0 ? R + hsurf[cellID]*scale
                     : R + (hhl[j].value[cellID]-hsurf[cellID])*scale;
        float h2 = R + (hhl[j+1].value[cellID]-hsurf[cellID])*scale;

        // bottom triangle vertices
        vec3f bv1 = toCartesian({h1,lat[0],lon[0]});
        vec3f bv2 = toCartesian({h1,lat[1],lon[1]});
        vec3f bv3 = toCartesian({h1,lat[2],lon[2]});
        // bottom value
        umesh::vec3f bot = values[j].value[cellID];

        // top triangle vertices
        vec3f tv1 = toCartesian({h2,lat[0],lon[0]});
        vec3f tv2 = toCartesian({h2,lat[1],lon[1]});
        vec3f tv3 = toCartesian({h2,lat[2],lon[2]});
        // top value
        umesh::vec3f top = values[j].value[cellID];

        // one vector per cell, we store the vector components 
        // as per-vertex scalars (bot/top are the same, this is redundant)
        output->vertices.push_back(bv1); output->perVertex->values.push_back(bot.x);
        output->vertices.push_back(bv2); output->perVertex->values.push_back(bot.y);
        output->vertices.push_back(bv3); output->perVertex->values.push_back(bot.z);
        output->vertices.push_back(tv1); output->perVertex->values.push_back(top.x);
        output->vertices.push_back(tv2); output->perVertex->values.push_back(top.y);
        output->vertices.push_back(tv3); output->perVertex->values.push_back(top.z);

        UMesh::Wedge wedge;
        wedge[0] = (int)output->vertices.size()-6;
        wedge[1] = (int)output->vertices.size()-5;
        wedge[2] = (int)output->vertices.size()-4;
        wedge[3] = (int)output->vertices.size()-3;
        wedge[4] = (int)output->vertices.size()-2;
        wedge[5] = (int)output->vertices.size()-1;

        output->wedges.push_back(wedge);

        h1 = h2;
      }
    }

    output->finalize();
    std::cout << output->vertices.size() << '\n';
    std::cout << output->wedges.size() << '\n';
    std::string outfileName = g_appState.outfileBase + ".umesh";
    output->saveTo(outfileName);
  }
}
