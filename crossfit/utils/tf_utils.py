# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


def parse_dtype(data_type):
    if isinstance(data_type, tf.DType):
        return data_type

    if data_type == int:
        return tf.int32
    elif data_type == float:
        return tf.float32
    elif data_type == bool:
        return tf.bool
    elif data_type == str:
        return tf.string
    elif data_type == complex:
        return tf.complex64
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def get_dtype_kind(dtype):
    if dtype.name == "float32":
        return "f"
    elif dtype.name == "float64":
        return "f"
    elif dtype.name == "int8":
        return "i"
    elif dtype.name == "int16":
        return "i"
    elif dtype.name == "int32":
        return "i"
    elif dtype.name == "int64":
        return "i"
    elif dtype.name == "uint8":
        return "u"
    elif dtype.name == "uint16":
        return "u"
    elif dtype.name == "bool":
        return "b"
    elif dtype.name == "string":
        return "S"
    elif dtype.name == "complex64":
        return "c"
    elif dtype.name == "complex128":
        return "c"
    elif dtype.name == "float16":
        return "f"
    elif dtype.name == "bfloat16":
        return "f"
