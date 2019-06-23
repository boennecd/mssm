#include "thread_pool.h"

#ifdef USE_THREAD_LOCAL
thread_local work_stealing_queue* thread_pool::local_work_queue = nullptr;
thread_local unsigned thread_pool::my_index = 0L;
#endif
