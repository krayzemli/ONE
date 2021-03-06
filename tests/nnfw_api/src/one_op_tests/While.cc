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

#include "GenModelTest.h"

#include <memory>

TEST_F(GenModelTest, OneOp_While)
{
  // The model looks just like the below pseudocode
  //
  // function model(x)
  // {
  //   while (x < 100.0)
  //   {
  //     x = x + 10.0;
  //   }
  //   return x
  // }

  CircleGen cgen;
  std::vector<float> incr_data{10};
  uint32_t incr_buf = cgen.addBuffer(incr_data);
  std::vector<float> end_data{100};
  uint32_t end_buf = cgen.addBuffer(end_data);

  // primary subgraph
  {
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorWhile({{x_in}, {x_out}}, 1, 2);
    cgen.setInputsAndOutputs({x_in}, {x_out});
  }

  // cond subgraph
  {
    cgen.nextSubgraph();
    int x = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end = cgen.addTensor({{1}, circle::TensorType_FLOAT32, end_buf});
    int result = cgen.addTensor({{1}, circle::TensorType_BOOL});
    cgen.addOperatorLess({{x, end}, {result}});
    cgen.setInputsAndOutputs({x}, {result});
  }

  // body subgraph
  {
    cgen.nextSubgraph();
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int incr = cgen.addTensor({{1}, circle::TensorType_FLOAT32, incr_buf});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorAdd({{x_in, incr}, {x_out}}, circle::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({x_in}, {x_out});
  }

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0}}, {{100}}));
  _context->addTestCase(uniformTCD<float>({{2}}, {{102}}));
  _context->addTestCase(uniformTCD<float>({{22}}, {{102}}));
  _context->addTestCase(uniformTCD<float>({{100}}, {{100}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_While_TwoInputs)
{
  // The model looks just like the below pseudocode
  //
  // function model(x, end)
  // {
  //   while (x < end)
  //   {
  //     x = x + 10.0
  //   }
  //   return x
  // }

  CircleGen cgen;
  std::vector<float> incr_data{10};
  uint32_t incr_buf = cgen.addBuffer(incr_data);

  // primary subgraph
  {
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorWhile({{x_in, end_in}, {x_out, end_out}}, 1, 2);
    cgen.setInputsAndOutputs({x_in, end_in}, {x_out});
  }

  // cond subgraph
  {
    cgen.nextSubgraph();
    int x = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int result = cgen.addTensor({{1}, circle::TensorType_BOOL});
    cgen.addOperatorLess({{x, end}, {result}});
    cgen.setInputsAndOutputs({x, end}, {result});
  }

  // body subgraph
  {
    cgen.nextSubgraph();
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int incr = cgen.addTensor({{1}, circle::TensorType_FLOAT32, incr_buf});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorAdd({{x_in, incr}, {x_out}}, circle::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({x_in, end}, {x_out, end});
  }

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0}, {20}}, {{20}}));
  _context->addTestCase(uniformTCD<float>({{5}, {30}}, {{35}}));
  _context->addTestCase(uniformTCD<float>({{20}, {10}}, {{20}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

class WhileWrongSubgraphIndex : public GenModelTest,
                                public ::testing::WithParamInterface<std::pair<int, int>>
{
};

TEST_P(WhileWrongSubgraphIndex, neg_Test)
{
  // These values must be less than 0 or greater than 2
  int cond_subg = GetParam().first;
  int body_subg = GetParam().second;

  // When While operation's subgraph index is invalid

  CircleGen cgen;

  // constant buffers
  std::vector<float> incr_data{10};
  uint32_t incr_buf = cgen.addBuffer(incr_data);

  // primary subgraph
  {
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorWhile({{x_in, end_in}, {x_out, end_out}}, cond_subg, body_subg);
    cgen.setInputsAndOutputs({x_in, end_in}, {x_out});
  }

  // cond subgraph
  {
    cgen.nextSubgraph();
    int x = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int result = cgen.addTensor({{1}, circle::TensorType_BOOL});
    cgen.addOperatorLess({{x, end}, {result}});
    cgen.setInputsAndOutputs({x, end}, {result});
  }

  // body subgraph
  {
    cgen.nextSubgraph();
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int incr = cgen.addTensor({{1}, circle::TensorType_FLOAT32, incr_buf});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorAdd({{x_in, incr}, {x_out}}, circle::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({x_in, end}, {x_out, end});
  }

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

INSTANTIATE_TEST_CASE_P(GenModelTest, WhileWrongSubgraphIndex,
                        ::testing::Values(std::make_pair(99, 2), std::make_pair(-1, 2),
                                          std::make_pair(1, 99), std::make_pair(1, -99),
                                          std::make_pair(-99, 99)));
