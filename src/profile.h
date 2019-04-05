#ifdef MSSM_PROF
#ifndef MSSM_PROF_H
#define MSSM_PROF_H
#include <string>
#include <atomic>

class profiler {
  static std::atomic<bool> running_profiler;

public:
  profiler(const std::string&);

  ~profiler();
};

#endif
#endif
