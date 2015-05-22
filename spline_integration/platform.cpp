#include "platform.h"

void Platform::initialize()
{
    cuda_initialize();
    cuda_device_info();
}

void Platform::finalize()
{
    cuda_device_reset();
}
