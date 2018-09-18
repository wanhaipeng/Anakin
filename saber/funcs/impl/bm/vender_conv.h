#ifndef ANAKIN_SABER_FUNCS_IMPL_BM_CONV2D_H
#define ANAKIN_SABER_FUNCS_IMPL_BM_CONV2D_H

#include "saber/funcs/impl/impl_conv.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class VenderConv2D<BM, OpDtype> : public ImplBase<
        BM, OpDtype, ConvParam<BM> > {
            
public:
    VenderConv2D(): _handle(NULL) {}
    ~VenderConv2D() {}

    virtual SaberStatus init(const std::vector<Tensor<BM> *>& inputs,
                             std::vector<Tensor<BM> *>& outputs,
                             ConvParam<BM>& param, Context<BM>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<BM> *>& inputs,
                               std::vector<Tensor<BM> *>& outputs,
                               ConvParam<BM>& param, Context<BM>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<BM>*>& inputs,
                                 std::vector<Tensor<BM>*>& outputs,
                                 ConvParam<BM>& param);

private:
    bm_handle_t _handle;
};

typedef struct tensor_4d_t {
    int n;
    int c;
    int h;
    int w;
} bm_tensor_4d_t;

typedef struct kernel_param{
    int g;
    int oc;
    int ic;
    int h;
    int w;
} bm_kernel_param_t;

typedef struct bm_conv_param{
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int dilation_h;
    int dilation_w;
    bool result_add;
} bm_conv_param_t;

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_BM_CONV2D_H
