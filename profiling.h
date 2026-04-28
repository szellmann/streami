#pragma once

#include <map>
#include <string>
#include <vector>
#include <mpi.h>
#include "common.h"

namespace profiling {
  std::map<std::string,std::vector<double>> timings;
  std::vector<std::string> names;

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank{-1};

  inline
  void record_time(std::string s) {
    if (rank < 0) {
      MPI_SAFE_CALL(MPI_Comm_rank(comm,&rank));
    }

    if (rank == 0) {
      auto it = timings.find(s);
      if (it == timings.end()) {
        timings.insert({s,{}});
        names.push_back(s);
      }
      it = timings.find(s);
      it->second.push_back(MPI_Wtime());
    }
  }

  inline
  void print_report() {
    for (size_t n=1; n<names.size(); ++n) {
      auto &tv1 = timings[names[n-1]];
      auto &tv2 = timings[names[n]];

      double avg = 0.;
      for (size_t i=0; i<std::min(tv1.size(),tv2.size()); ++i) {
        avg += tv2[i]-tv1[i];
      }
      avg /= std::min(tv1.size(),tv2.size());
      std::cout << "elapsed [" << names[n-1] << '/' << names[n] << "]: " << avg << '\n';
    }
  }
}

#ifdef PROFILING
#define RECORD_TIME(S) profiling::record_time(S);
#define PRINT_PROFILING_REPORT() profiling::print_report();
#else
#define RECORD_TIME(S)
#define PRINT_PROFILING_REPORT()
#endif



