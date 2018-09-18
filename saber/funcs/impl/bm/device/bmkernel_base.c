#include "bmkernel_base.h"
#include "bm_config.h"
#include "bmk_conv.c"
#include <stdio.h>
/**
 * bmkernel_func is the user entry to BMKERNEL just like "main" to some applications.
 * 
 * \param args - Pointer to arguments that user sends from host.
 *               op - Flag to determine the operation type.             
 */

int bmkernel_func(void *args)
{
    bmkernel_api_base* param = (bmkernel_api_base *)args;
    switch (param->op) {
        case ACTIVATION:
            // bm_activation_fwd(param)
            break;
        case CONV:
            bm_api_conv_forward_t* api = (bm_api_conv_forward_t*)param->opParam;
            bm_conv_fwd(*api);
            break;
        default:
            printf("op %d is not supported by BM yet.\n", param->op);
    }
    return 0;
}
