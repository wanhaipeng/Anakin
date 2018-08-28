
#include "saber/funcs/impl/bm/vender_conv.h"

namespace anakin
{
namespace saber
{

// FP32 part
template <>
SaberStatus VenderConv2D<BM, AK_FLOAT>::\
    create(const std::vector<Tensor<BM> *>& inputs,
            std::vector<Tensor<BM> *>& outputs,
            ConvParam<BM>& param, Context<BM>& ctx)
{
}

template <>
SaberStatus VenderConv2D<BM, AK_FLOAT>::\
    init(const std::vector<Tensor<BM> *> &inputs,
         std::vector<Tensor<BM> *> &outputs,
         ConvParam<BM> &param, Context<BM> &ctx)
{

    _handle = ctx.get_handle();
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderConv2D<BM, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<BM>*>& inputs,
                std::vector<Tensor<BM>*>& outputs,
                ConvParam<BM>& param)
{

    const BM_mem_addr in_data = (const BM_mem_addr) inputs[0]->data();
    BM_mem_addr out_data = (BM_mem_addr) outputs[0]->mutable_data();
    const BM_mem_addr weight = (const BM_mem_addr) param.weight()->data();

    int input_n = inputs[0]->num();
    int input_c = inputs[0]->channel();
    int input_h = inputs[0]->height();
    int input_w = inputs[0]->width();

    int output_n = outputs[0]->num();
    int output_c = outputs[0]->channel();
    int output_h = outputs[0]->height();
    int output_w = outputs[0]->width();

    int group = param.group;
    int kh = param.weight()->height();
    int kw = param.weight()->width();
    int pad_h = param.pad_h;
    int pad_w = param.pad_w;
    int stride_h = param.stride_h;
    int stride_w = param.stride_w;
    int dilation_h = param.dilation_h;
    int dilation_w = param.dilation_w;

    bool with_bias = param.bias()->size() > 0;
    const bm_mem_desc bias = with_bias ? (const BM_mem_addr)param.bias()->data() : BM_MEM_NULL;

    bm_tensor_4d_t input_shape = {
        input_n,
        input_c,
        input_h,
        input_w};

    bm_tensor_4d_t output_shape = {
        output_n,
        output_c,
        output_h,
        output_w};

    bm_kernel_param_t kernel_param = {
        group,
        output_c,
        input_c,
        kh,
        kw};

    bm_conv_param_t conv_param = {
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        0};

    if (param.activation_param.has_active && param.activation_param.active == Active_relu) {
        // TODO: bmdnn_conv_relu_forward
    }
    else{
        BMDNN_CHECK(bmdnn_conv_forward(_handle, in_data, weight, bias, input_shape,
                                   kernel_param, output_shape, conv_param, with_bias, out_data));
    }

    return SaberSuccess;
}

// INT8 part
// TODO: BM1684

template class VenderConv2D<BM, AK_FLOAT>;
} // namespace saber
} // namespace anakin