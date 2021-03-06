#!/bin/bash

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# See https://github.com/Samsung/ONE/issues/4155 for information

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd $SCRIPT_PATH > /dev/null

if [[ ! -s "inception_v3.pb" ]]; then
    rm -rf inception_v3_2018_04_27.tgz
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz
    tar zxvf inception_v3_2018_04_27.tgz
fi

if [[ ! -s "while_3.pbtxt" ]]; then
    rm -rf while_3.zip
    wget https://github.com/Samsung/ONE/files/5095630/while_3.zip
    unzip while_3.zip
fi

if [[ ! -s "inception_v3_test_data.h5" ]]; then
    rm -rf inception_v3_test_data.zip
    wget https://github.com/Samsung/ONE/files/5139370/inception_v3_test_data.zip
    unzip inception_v3_test_data.zip
fi

if [[ ! -s "mobilenet_test_data.h5" ]]; then
    rm -rf mobilenet_test_data.zip
    wget https://github.com/Samsung/ONE/files/5139460/mobilenet_test_data.zip
    unzip mobilenet_test_data.zip
fi

if [[ ! -s "bcq.pb" ]]; then
    rm -rf bcq.pb.zip
    wget https://github.com/Samsung/ONE/files/5153842/bcq.pb.zip
    unzip bcq.pb.zip
fi

# prepare 'inception_v3.circle' file used for quantizatio test
inputfile="./inception_v3.pb"
outputfile="./inception_v3.circle"

if [[ ! -s ${outputfile} ]]; then
  ../bin/one-import-tf \
  --input_path ${inputfile} \
  --output_path ${outputfile} \
  --input_arrays input --input_shapes "1,299,299,3" \
  --output_arrays InceptionV3/Predictions/Reshape_1
fi

popd > /dev/null
