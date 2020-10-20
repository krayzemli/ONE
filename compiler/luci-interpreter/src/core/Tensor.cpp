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

#include "luci_interpreter/core/Tensor.h"

#include <cstring>
#include <stdexcept>
#include <iostream>

static size_t s_memsize = 0;

namespace luci_interpreter
{

Tensor::Tensor(DataType element_type, Shape shape, AffineQuantization quantization,
               std::string name)
    : _element_type(element_type), _shape(std::move(shape)), _quantization(std::move(quantization)),
      _name(std::move(name)), _memsize(0)
{
}

Tensor::Tensor(Tensor &&t2)
    : _element_type(t2._element_type), _shape(t2._shape), _quantization(t2._quantization),
      _name(t2._name), _memsize(t2._memsize)
{
  std::cerr << "Move " << t2._name << std::endl;
  t2._name = "-dummy-";
  t2._memsize = 0;
  _data = std::move(t2._data);
  t2._data = nullptr;
}

Tensor::~Tensor()
{
  s_memsize -= _memsize;
  std::cerr << "Destruct " << _name << ": " << _memsize << " -> 0 : total = " << s_memsize
            << std::endl;
}

void Tensor::allocate()
{
  const size_t element_size = getDataTypeSize(_element_type);
  const int32_t num_elements = _shape.num_elements();
  size_t old_mems = _memsize;
  size_t new_mems = num_elements * element_size;
  s_memsize = s_memsize + new_mems - old_mems;
  _memsize = new_mems;
  _data = std::make_unique<uint8_t[]>(new_mems);
  std::cerr << "Alloc " << _name << ": " << old_mems << " -> " << new_mems
            << " : total = " << s_memsize << std::endl;
}

void Tensor::deallocate()
{
  _data = nullptr;
  size_t new_mems = 0;

  size_t old_mems = _memsize;
  s_memsize = s_memsize + new_mems - old_mems;
  _memsize = new_mems;
  std::cerr << "Dealloc " << _name << ": " << old_mems << " -> " << new_mems
            << " : total = " << s_memsize << std::endl;
}

void Tensor::readData(void *data_ptr, size_t data_size) const
{
  const size_t element_size = getDataTypeSize(element_type());
  const int32_t num_elements = shape().num_elements();
  if (data_size != num_elements * element_size)
  {
    throw std::invalid_argument("Invalid data size.");
  }
  assert(data_ptr != nullptr);
  std::memcpy(data_ptr, data<void>(), data_size);
}

void Tensor::writeData(const void *data_ptr, size_t data_size)
{
  const size_t element_size = getDataTypeSize(element_type());
  const int32_t num_elements = shape().num_elements();
  if (data_size != num_elements * element_size)
  {
    throw std::invalid_argument("Invalid data size.");
  }
  assert(data_ptr != nullptr);
  std::memcpy(data<void>(), data_ptr, data_size);
}

void Tensor::resize(const Shape &new_shape)
{
  deallocate();
  _shape = new_shape;
}

} // namespace luci_interpreter
