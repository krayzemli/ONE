/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "luci/Pass/ResolveCustomOpMatMulPass.h"

#include "flatbuffers/flexbuffers.h"
#include <loco/IR/DataTypeTraits.h>

#include <luci/IR/CircleNodes.h>

#include <loco.h>
#include <oops/InternalExn.h>
#include <loco/Service/ShapeInference.h>
#include <loco/Service/TypeInference.h>

namespace
{

template <typename T>
luci::CircleConst *create_const_node(loco::Graph *g, const loco::DataType dtype,
                                     const std::vector<uint32_t> &shape,
                                     const std::vector<T> &values)
{
  auto node = g->nodes()->create<luci::CircleConst>();
  node->dtype(dtype);
  node->rank(shape.size());

  uint32_t size = 1;
  for (uint32_t i = 0; i < shape.size(); ++i)
  {
    node->dim(i) = shape.at(i);
    size *= shape.at(i);
  }

#define INIT_VALUES(DT)                          \
  {                                              \
    node->size<DT>(size);                        \
    for (uint32_t i = 0; i < values.size(); ++i) \
      node->at<DT>(i) = values[i];               \
  }

  switch (dtype)
  {
    case loco::DataType::U8:
      INIT_VALUES(loco::DataType::U8);
      break;
    case loco::DataType::S16:
      INIT_VALUES(loco::DataType::S16);
      break;
    case loco::DataType::S32:
      INIT_VALUES(loco::DataType::S32);
      break;
    case loco::DataType::FLOAT32:
      INIT_VALUES(loco::DataType::FLOAT32)
      break;
    default:
      INTERNAL_EXN("create_const_node called with unsupported type");
      break;
  }
  return node;
}

bool resolve_matmul(luci::CircleCustom *cop)
{
#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;
#define CHECK_OR_THROW(condition, message) \
  if (not(condition))                      \
    INTERNAL_EXN(message);

  auto graph = cop->graph();
  const std::vector<uint8_t> custom_options = cop->custom_options();
  auto map = flexbuffers::GetRoot(custom_options).AsMap();
  const auto U8 = loco::DataType::U8;
  const auto S16 = loco::DataType::S16;
  const auto S32 = loco::DataType::S32;
  const auto FLOAT32 = loco::DataType::FLOAT32;

  bool transpose_a = map["transpose_a"].AsBool();
  bool transpose_b = map["transpose_b"].AsBool();

  loco::Node *lhs = cop->inputs(0);
  loco::Node *rhs = cop->inputs(1);

  // Check that the type of the first input is known
  CHECK_OR_FALSE(loco::dtype_known(lhs));
  auto lhs_dtype = loco::dtype_get(cop->inputs(0));

  // If transpose of first input is requested, its shape must be known
  CHECK_OR_FALSE(!transpose_a || loco::shape_known(lhs));
  // and its rank should be at least 2
  CHECK_OR_FALSE(!transpose_a || loco::shape_get(lhs).as<loco::TensorShape>().rank() >= 2);
  // Check that the shape of the 2nd input is known
  CHECK_OR_FALSE(loco::shape_known(rhs));
  // TODO as of 06/23/20 TFLite only supports rank 2 for 2nd input. Fix this once that changes!
  CHECK_OR_FALSE(loco::shape_get(rhs).as<loco::TensorShape>().rank() == 2);
  // Check that input data type is supported
  CHECK_OR_THROW(lhs_dtype == U8 || lhs_dtype == S16 || lhs_dtype == FLOAT32,
                 "Only UInt8, Int16 and Float32 data types are supported by MatMul");

  if (transpose_a)
  {
    auto a_shape = loco::shape_get(lhs).as<loco::TensorShape>();
    // Create a permutation constant node
    std::vector<uint32_t> perm;
    for (uint32_t i = 0; i < a_shape.rank(); ++i)
      perm.push_back(i);
    std::swap(perm[a_shape.rank() - 1], perm[a_shape.rank() - 2]);
    auto perm_node = create_const_node(graph, S32, {a_shape.rank()}, perm);
    // Now make a transpose node
    auto transpose_node = graph->nodes()->create<luci::CircleTranspose>();
    transpose_node->a(lhs);
    transpose_node->perm(perm_node);
    lhs = transpose_node;
  }

  // Transpose the second input if needed. TFLite FullyConnected operator
  // assumes the second input is in column-major order, but the input is
  // in row-major order, thus we need to convert between them.
  if (!transpose_b)
  {
    const std::vector<uint32_t> perm{1, 0};
    auto perm_node = create_const_node(graph, S32, {2}, perm);
    auto transpose_node = graph->nodes()->create<luci::CircleTranspose>();
    transpose_node->a(rhs);
    transpose_node->perm(perm_node);
    rhs = transpose_node;
  }

  // Make a constant zero-filled bias node
  auto b_shape = loco::shape_get(cop->inputs(1)).as<loco::TensorShape>();
  uint32_t bias_size = b_shape.dim(transpose_b ? 1 : 0).value();
  const std::vector<float> val(bias_size, .0f);
  auto bias_node = create_const_node(graph, lhs_dtype, {bias_size}, val);
  auto fc_node = graph->nodes()->create<luci::CircleFullyConnected>();
  fc_node->input(lhs);
  fc_node->weights(rhs);
  fc_node->bias(bias_node);
  fc_node->fusedActivationFunction(luci::FusedActFunc::NONE);

  replace(cop).with(fc_node);
  return true;
}

} // namespace

namespace luci
{

bool ResolveCustomOpMatMulPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto cop = dynamic_cast<luci::CircleCustom *>(node);
    if (not cop)
      continue;

    if (cop->custom_code() != "MatMul")
      continue;

    if (!resolve_matmul(cop))
      continue;

    changed = true;
  }

  return changed;
}

} // namespace luci
