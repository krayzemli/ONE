/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "BinaryArithmeticLayer.h"

#include <cker/operation/BinaryArithmeticOps.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

namespace
{

struct FloatOpFunctor
{
  nnfw::cker::BinaryArithmeticOpParamFloat m_op_params;
  void (*m_elementwiseFunc)(int, const nnfw::cker::BinaryArithmeticOpParamFloat &, const float *, const float *, float *);
  void (*m_swappedArgsElementwiseFunc)(int, const nnfw::cker::BinaryArithmeticOpParamFloat &, const float *, const float *, float *);
  void (*m_lhsBroadcastFunc)(int, const nnfw::cker::BinaryArithmeticOpParamFloat &, const float, const float *, float *);
  void (*m_rhsBroadcastFunc)(int, const nnfw::cker::BinaryArithmeticOpParamFloat &, const float, const float *, float *);
  std::function<float(const float &, const float &)> m_genericFunc;

  template<typename OPERATOR, typename SWAPPEDARGSOPERATOR>
  FloatOpFunctor(const nnfw::cker::BinaryArithmeticOpParamFloat & op_params, const OPERATOR &, const SWAPPEDARGSOPERATOR &)
    : m_op_params(op_params)
  {
    auto op_func1 = nnfw::cker::optimized::getBinaryOpWithActivationImplFloat<OPERATOR>(m_op_params);
    auto op_func2 = nnfw::cker::optimized::getBinaryOpWithActivationImplFloat<SWAPPEDARGSOPERATOR>(m_op_params);
    m_elementwiseFunc = op_func1.first;
    m_swappedArgsElementwiseFunc = op_func2.first;
    m_lhsBroadcastFunc = op_func1.second;
    m_rhsBroadcastFunc = op_func2.second;
    m_genericFunc = [](const float &a, const float &b) -> float { return OPERATOR::calculate(a, b); };
  }

  void operator()(const IPortableTensor *lhs, const IPortableTensor *rhs, IPortableTensor *output)
  {
    const auto lhsShape = getTensorShape(lhs);
    const auto rhsShape = getTensorShape(rhs);
    const bool need_broadcast = nnfw::cker::ProcessBroadcastShapes(lhsShape, rhsShape, &m_op_params);
    const auto outputShape = getTensorShape(output);
    const float * const lhsBuffer = reinterpret_cast<const float *>(lhs->buffer());
    const float * const rhsBuffer = reinterpret_cast<const float *>(rhs->buffer());
    float * const outputBuffer = reinterpret_cast<float *>(output->buffer());
    if (need_broadcast)
    {
      if (m_op_params.broadcast_category == nnfw::cker::BroadcastableOpCategory::kFirstInputBroadcastsFast)
      {
        nnfw::cker::optimized::BinaryBroadcastFiveFold(m_op_params, false, lhsShape, lhsBuffer, rhsShape, rhsBuffer,
                            outputShape, outputBuffer, m_elementwiseFunc, m_lhsBroadcastFunc);
      }
      else if (m_op_params.broadcast_category == nnfw::cker::BroadcastableOpCategory::kSecondInputBroadcastsFast)
      {
        nnfw::cker::optimized::BinaryBroadcastFiveFold(m_op_params, true, lhsShape, lhsBuffer, rhsShape, rhsBuffer,
                            outputShape, outputBuffer, m_swappedArgsElementwiseFunc, m_rhsBroadcastFunc);
      }
      else
      {          
        nnfw::cker::reference::BroadcastBinaryArithmeticOpSlow(m_op_params, lhsShape, lhsBuffer, rhsShape,
                                               rhsBuffer, outputShape, outputBuffer, m_genericFunc);
      }          
    }
    else
    {
      const int flat_size = nnfw::cker::MatchingElementsSize(lhsShape, rhsShape, outputShape);
      m_elementwiseFunc(flat_size, m_op_params, lhsBuffer, rhsBuffer, outputBuffer);
    }
  }
};

template <nnfw::cker::BinaryArithmeticOpType arithmetic_type, typename T, typename OPERATORPARAMS>
void eval(const IPortableTensor *lhs, const IPortableTensor *rhs, IPortableTensor *output,
          OPERATORPARAMS op_params)
{
  const auto lhsShape = getTensorShape(lhs);
  const auto rhsShape = getTensorShape(rhs);
  const bool need_broadcast = nnfw::cker::ProcessBroadcastShapes(lhsShape, rhsShape, &op_params);
  if (need_broadcast)
  {
    nnfw::cker::BroadcastBinaryArithmeticOp<arithmetic_type>(
        op_params, lhsShape, reinterpret_cast<const T *>(lhs->buffer()), rhsShape,
        reinterpret_cast<const T *>(rhs->buffer()), getTensorShape(output),
        reinterpret_cast<T *>(output->buffer()));
    return;
  }

  nnfw::cker::BinaryArithmeticOp<arithmetic_type>(
      op_params, lhsShape, reinterpret_cast<const T *>(lhs->buffer()), rhsShape,
      reinterpret_cast<const T *>(rhs->buffer()), getTensorShape(output),
      reinterpret_cast<T *>(output->buffer()));
}

void setAddOrSubQuant8Params(const IPortableTensor *lhs, const IPortableTensor *rhs,
                             IPortableTensor *output, ir::Activation activation,
                             nnfw::cker::BinaryArithmeticOpParamQuantized *params)
{
  int32_t output_activation_min, output_activation_max;
  CalculateActivationRangeUint8(activation, output, &output_activation_min, &output_activation_max);
  nnfw::cker::BinaryArithmeticOpParamQuantized &op_params = *params;
  op_params.quantized_activation_max = output_activation_max;
  op_params.quantized_activation_min = output_activation_min;
  // Parameters for scaled quantized computation
  op_params.left_shift = 20;
  // Zero-points of input and output tensors
  op_params.input1_offset = -lhs->data_offset();
  op_params.input2_offset = -rhs->data_offset();
  op_params.output_offset = output->data_offset();
  assert((op_params.input1_offset >= 0) && (op_params.input1_offset <= 255));
  assert((op_params.input2_offset >= 0) && (op_params.input2_offset <= 255));
  assert((op_params.output_offset >= 0) && (op_params.output_offset <= 255));

  // Compute normalized scale for _lhs and _rhs values,
  // and represent in 32-bit fixed point
  const double norm_max_scale = 2 * std::max(lhs->data_scale(), rhs->data_scale());
  const double real_lhs_scale = lhs->data_scale() / norm_max_scale;
  const double real_rhs_scale = rhs->data_scale() / norm_max_scale;
  // output scale is used to normalize final result, so we invert the scale here
  const double real_output_scale =
      norm_max_scale / (output->data_scale() * (1 << op_params.left_shift));

  // Represent the scales as fixed int32_t multipliers, and int32_t shifts
  QuantizeMultiplier(real_lhs_scale, &op_params.input1_multiplier, &op_params.input1_shift);
  QuantizeMultiplier(real_rhs_scale, &op_params.input2_multiplier, &op_params.input2_shift);
  QuantizeMultiplier(real_output_scale, &op_params.output_multiplier, &op_params.output_shift);
}

void setMulQuant8Params(const IPortableTensor *lhs, const IPortableTensor *rhs,
                        IPortableTensor *output, ir::Activation activation,
                        nnfw::cker::BinaryArithmeticOpParamQuantized *params)
{
  int32_t output_activation_min, output_activation_max;
  CalculateActivationRangeUint8(activation, output, &output_activation_min, &output_activation_max);
  nnfw::cker::BinaryArithmeticOpParamQuantized &op_params = *params;

  op_params.quantized_activation_max = output_activation_max;
  op_params.quantized_activation_min = output_activation_min;
  op_params.input1_offset = -lhs->data_offset();
  op_params.input2_offset = -rhs->data_offset();
  op_params.output_offset = output->data_offset();

  double real_multiplier = lhs->data_scale() * rhs->data_scale() / output->data_scale();
  QuantizeMultiplier(real_multiplier, &op_params.output_multiplier, &op_params.output_shift);
}

} // namespace

void BinaryArithmeticLayer::configure(const IPortableTensor *lhs, const IPortableTensor *rhs,
                                      IPortableTensor *output, const ir::Activation activation,
                                      const ArithmeticType arithmetic_type)
{
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  assert(output != nullptr);

  _lhs = lhs;
  _rhs = rhs;
  _output = output;

  switch (_lhs->data_type())
  {
  case OperandType::QUANT_UINT8_ASYMM:
    {
      nnfw::cker::BinaryArithmeticOpParamQuantized op_params;
      switch (arithmetic_type)
      {
      case ArithmeticType::kAdd:
        setAddOrSubQuant8Params(_lhs, _rhs, _output, activation, &op_params);
        _kernel = std::bind(&eval<nnfw::cker::BinaryArithmeticOpType::ADD, uint8_t, nnfw::cker::BinaryArithmeticOpParamQuantized>,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            op_params);
        break;
      case ArithmeticType::kSub:
        setAddOrSubQuant8Params(_lhs, _rhs, _output, activation, &op_params);
        op_params.input2_multiplier *= -1;
        _kernel = std::bind(&eval<nnfw::cker::BinaryArithmeticOpType::SUB, uint8_t, nnfw::cker::BinaryArithmeticOpParamQuantized>,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            op_params);
        break;
      case ArithmeticType::kMul:
        setMulQuant8Params(_lhs, _rhs, _output, activation, &op_params);
        _kernel = std::bind(&eval<nnfw::cker::BinaryArithmeticOpType::MUL, uint8_t, nnfw::cker::BinaryArithmeticOpParamQuantized>,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            op_params);
        break;
      case ArithmeticType::kDiv:
        throw std::runtime_error{ "BinaryArithmetic(Div): Div operation does not support quantization"};

      default:
        throw std::runtime_error{"BinaryArithmetic: Unsupported BinaryArithmetic type"};
      }
    }
    break;

  case OperandType::FLOAT32:
    {
      using namespace nnfw::cker::optimized;
      nnfw::cker::BinaryArithmeticOpParamFloat op_params;
      float output_activation_min = 0, output_activation_max = 0;
      CalculateActivationRange(activation, &output_activation_min, &output_activation_max);
      op_params.float_activation_max = output_activation_max;
      op_params.float_activation_min = output_activation_min;

      switch (arithmetic_type)
      {
      case ArithmeticType::kAdd:
        _kernel = FloatOpFunctor(op_params, BinaryOpFuncAddFloat(), BinaryOpFuncAddFloat());
        break;
      case ArithmeticType::kSub:
        _kernel = FloatOpFunctor(op_params, BinaryOpFuncSubFloat(), BinaryOpFuncSwapArgs<BinaryOpFuncSubFloat>());
        break;
      case ArithmeticType::kMul:
        _kernel = FloatOpFunctor(op_params, BinaryOpFuncMulFloat(), BinaryOpFuncMulFloat());
        break;
      case ArithmeticType::kDiv:
#ifdef __aarch64     
        _kernel = FloatOpFunctor(op_params, BinaryOpFuncDivFloat(), BinaryOpFuncSwapArgs<BinaryOpFuncDivFloat>());
#else
        _kernel = std::bind(eval<nnfw::cker::BinaryArithmeticOpType::DIV, float, nnfw::cker::BinaryArithmeticOpParamFloat>,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, op_params);
#endif
        break;
      default:
        throw std::runtime_error{"BinaryArithmetic: Unsupported BinaryArithmetic type"};
      }
    }
    break;

  case OperandType::INT32:
    {
      nnfw::cker::BinaryArithmeticOpParamQuantized op_params;
      int32_t output_activation_min = 0, output_activation_max = 0;
      CalculateActivationRange(activation, &output_activation_min, &output_activation_max);
      op_params.quantized_activation_max = output_activation_max;
      op_params.quantized_activation_min = output_activation_min;
      switch (arithmetic_type)
      {
      case ArithmeticType::kAdd:
        _kernel = std::bind(eval<nnfw::cker::BinaryArithmeticOpType::ADD, int32_t, nnfw::cker::BinaryArithmeticOpParamQuantized>,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, op_params);
        break;
      case ArithmeticType::kSub:
        _kernel = std::bind(eval<nnfw::cker::BinaryArithmeticOpType::SUB, int32_t, nnfw::cker::BinaryArithmeticOpParamQuantized>,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, op_params);
        break;
      case ArithmeticType::kMul:
        _kernel = std::bind(eval<nnfw::cker::BinaryArithmeticOpType::MUL, int32_t, nnfw::cker::BinaryArithmeticOpParamQuantized>,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, op_params);
        break;
      case ArithmeticType::kDiv:
        _kernel = std::bind(eval<nnfw::cker::BinaryArithmeticOpType::DIV, int32_t, nnfw::cker::BinaryArithmeticOpParamQuantized>,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, op_params);
        break;
      default:
        throw std::runtime_error{"BinaryArithmetic: Unsupported BinaryArithmetic type"};
      }
    }
    break;

  default:
    throw std::runtime_error{"BinaryArithmetic: Unsupported data type"};
  }
}

void BinaryArithmeticLayer::run() { _kernel(_lhs, _rhs, _output); }

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
