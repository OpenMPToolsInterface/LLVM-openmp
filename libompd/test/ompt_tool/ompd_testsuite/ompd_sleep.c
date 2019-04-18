#include <unistd.h>

#ifdef  __cplusplus
extern "C"
#endif
void* ompd_tool_test(void* n)
{
  sleep(1000);
  return NULL;
}
