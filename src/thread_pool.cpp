#include "thread_pool.h"

#ifdef USE_THREAD_LOCAL
thread_local thread_pool::queues_class* thread_pool::local_work_queue;
thread_local unsigned thread_pool::my_index;
#endif
