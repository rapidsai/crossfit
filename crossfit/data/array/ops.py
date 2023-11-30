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

import numpy as np

from crossfit.data.array.dispatch import np_backend_dispatch, with_dispatch

dtype = with_dispatch(np.dtype)
errstate = np.errstate
asarray = with_dispatch(np.asarray)
result_type = with_dispatch(np.result_type)

# alen = with_dispatch(np.alen)
all = with_dispatch(np.all)
# alltrue = with_dispatch(np.alltrue)
amax = with_dispatch(np.amax)
amin = with_dispatch(np.amin)
any = with_dispatch(np.any)
argmax = with_dispatch(np.argmax)
argmin = with_dispatch(np.argmin)
argpartition = with_dispatch(np.argpartition)
argsort = with_dispatch(np.argsort)
around = with_dispatch(np.around)
choose = with_dispatch(np.choose)
clip = with_dispatch(np.clip)
compress = with_dispatch(np.compress)
cumprod = with_dispatch(np.cumprod)
cumproduct = with_dispatch(np.cumproduct)
cumsum = with_dispatch(np.cumsum)
diagonal = with_dispatch(np.diagonal)
mean = with_dispatch(np.mean)
ndim = with_dispatch(np.ndim)
nonzero = with_dispatch(np.nonzero)
partition = with_dispatch(np.partition)
prod = with_dispatch(np.prod)
product = with_dispatch(np.product)
ptp = with_dispatch(np.ptp)
put = with_dispatch(np.put)
ravel = with_dispatch(np.ravel)
repeat = with_dispatch(np.repeat)
reshape = with_dispatch(np.reshape)
resize = with_dispatch(np.resize)
round_ = with_dispatch(np.round_)
searchsorted = with_dispatch(np.searchsorted)
shape = with_dispatch(np.shape)
size = with_dispatch(np.size)
sometrue = with_dispatch(np.sometrue)
sort = with_dispatch(np.sort)
squeeze = with_dispatch(np.squeeze)
std = with_dispatch(np.std)
sum = with_dispatch(np.sum)
swapaxes = with_dispatch(np.swapaxes)
take = with_dispatch(np.take)
trace = with_dispatch(np.trace)
transpose = with_dispatch(np.transpose)
var = with_dispatch(np.var)

average = with_dispatch(np.average)
piecewise = with_dispatch(np.piecewise)
select = with_dispatch(np.select)
copy = with_dispatch(np.copy)
interp = with_dispatch(np.interp)
angle = with_dispatch(np.angle)
unwrap = with_dispatch(np.unwrap)
sort_complex = with_dispatch(np.sort_complex)
trim_zeros = with_dispatch(np.trim_zeros)
extract = with_dispatch(np.extract)
place = with_dispatch(np.place)
cov = with_dispatch(np.cov)
corrcoef = with_dispatch(np.corrcoef)
percentile = with_dispatch(np.percentile)
quantile = with_dispatch(np.quantile)
trapz = with_dispatch(np.trapz)
meshgrid = with_dispatch(np.meshgrid)
delete = with_dispatch(np.delete)
insert = with_dispatch(np.insert)
append = with_dispatch(np.append)
median = with_dispatch(np.median)

ediff1d = with_dispatch(np.ediff1d)
setxor1d = with_dispatch(np.setxor1d)
union1d = with_dispatch(np.union1d)
setdiff1d = with_dispatch(np.setdiff1d)
unique = with_dispatch(np.unique)
in1d = with_dispatch(np.in1d)
isin = with_dispatch(np.isin)


absolute = with_dispatch(np.absolute)
add = with_dispatch(np.add)
arccos = with_dispatch(np.arccos)
arccosh = with_dispatch(np.arccosh)
arcsin = with_dispatch(np.arcsin)
arcsinh = with_dispatch(np.arcsinh)
arctan2 = with_dispatch(np.arctan2)
arctan = with_dispatch(np.arctan)
arctanh = with_dispatch(np.arctanh)
bitwise_and = with_dispatch(np.bitwise_and)
bitwise_not = with_dispatch(np.bitwise_not)
bitwise_or = with_dispatch(np.bitwise_or)
bitwise_xor = with_dispatch(np.bitwise_xor)
cbrt = with_dispatch(np.cbrt)
ceil = with_dispatch(np.ceil)
conj = with_dispatch(np.conj)
conjugate = with_dispatch(np.conjugate)
copysign = with_dispatch(np.copysign)
cos = with_dispatch(np.cos)
cosh = with_dispatch(np.cosh)
deg2rad = with_dispatch(np.deg2rad)
degrees = with_dispatch(np.degrees)
divide = with_dispatch(np.divide)
divmod = with_dispatch(np.divmod)
equal = with_dispatch(np.equal)
exp2 = with_dispatch(np.exp2)
exp = with_dispatch(np.exp)
expm1 = with_dispatch(np.expm1)
fabs = with_dispatch(np.fabs)
float_power = with_dispatch(np.float_power)
floor = with_dispatch(np.floor)
floor_divide = with_dispatch(np.floor_divide)
fmax = with_dispatch(np.fmax)
fmin = with_dispatch(np.fmin)
fmod = with_dispatch(np.fmod)
frexp = with_dispatch(np.frexp)
gcd = with_dispatch(np.gcd)
greater = with_dispatch(np.greater)
greater_equal = with_dispatch(np.greater_equal)
heaviside = with_dispatch(np.heaviside)
hypot = with_dispatch(np.hypot)
invert = with_dispatch(np.invert)
isfinite = with_dispatch(np.isfinite)
isinf = with_dispatch(np.isinf)
isnan = with_dispatch(np.isnan)
isnat = with_dispatch(np.isnat)
lcm = with_dispatch(np.lcm)
ldexp = with_dispatch(np.ldexp)
left_shift = with_dispatch(np.left_shift)
less = with_dispatch(np.less)
less_equal = with_dispatch(np.less_equal)
log10 = with_dispatch(np.log10)
log1p = with_dispatch(np.log1p)
log2 = with_dispatch(np.log2)
log = with_dispatch(np.log)
logaddexp2 = with_dispatch(np.logaddexp2)
logaddexp = with_dispatch(np.logaddexp)
logical_and = with_dispatch(np.logical_and)
logical_not = with_dispatch(np.logical_not)
logical_or = with_dispatch(np.logical_or)
logical_xor = with_dispatch(np.logical_xor)
matmul = with_dispatch(np.matmul)
maximum = with_dispatch(np.maximum)
minimum = with_dispatch(np.minimum)
mod = with_dispatch(np.mod)
modf = with_dispatch(np.modf)
multiply = with_dispatch(np.multiply)
negative = with_dispatch(np.negative)
nextafter = with_dispatch(np.nextafter)
not_equal = with_dispatch(np.not_equal)
positive = with_dispatch(np.positive)
power = with_dispatch(np.power)
rad2deg = with_dispatch(np.rad2deg)
radians = with_dispatch(np.radians)
reciprocal = with_dispatch(np.reciprocal)
remainder = with_dispatch(np.remainder)
right_shift = with_dispatch(np.right_shift)
rint = with_dispatch(np.rint)
sign = with_dispatch(np.sign)
signbit = with_dispatch(np.signbit)
sin = with_dispatch(np.sin)
sinh = with_dispatch(np.sinh)
spacing = with_dispatch(np.spacing)
sqrt = with_dispatch(np.sqrt)
square = with_dispatch(np.square)
subtract = with_dispatch(np.subtract)
tan = with_dispatch(np.tan)
tanh = with_dispatch(np.tanh)
true_divide = with_dispatch(np.true_divide)
trunc = with_dispatch(np.trunc)

abs = absolute


def concatenate(arrays: list, axis=0):
    """Dispatched version of np.concatenate"""
    if len(arrays) == 0:
        raise ValueError("Zero-length list passed to concatenate")
    try:
        backend = np_backend_dispatch.dispatch(type(arrays[0]))
    except TypeError:
        return np.concatenate(arrays, axis=axis)
    return backend(np.concatenate, arrays, axis=axis)


__all__ = [
    "dtype",
    "errstate",
    # "alen",
    "all",
    # "alltrue",
    "amax",
    "amin",
    "any",
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "around",
    "choose",
    "clip",
    "compress",
    "concatenate",
    "cumprod",
    "cumproduct",
    "cumsum",
    "diagonal",
    "mean",
    "ndim",
    "nonzero",
    "partition",
    "prod",
    "product",
    "ptp",
    "put",
    "ravel",
    "repeat",
    "reshape",
    "resize",
    "round_",
    "searchsorted",
    "shape",
    "size",
    "sometrue",
    "sort",
    "squeeze",
    "std",
    "sum",
    "swapaxes",
    "take",
    "trace",
    "transpose",
    "var",
    "average",
    "piecewise",
    "select",
    "copy",
    "interp",
    "angle",
    "unwrap",
    "sort_complex",
    "trim_zeros",
    "extract",
    "place",
    "cov",
    "corrcoef",
    "percentile",
    "quantile",
    "trapz",
    "meshgrid",
    "delete",
    "insert",
    "append",
    "median",
    "ediff1d",
    "setxor1d",
    "union1d",
    "setdiff1d",
    "unique",
    "in1d",
    "isin",
    "absolute",
    "add",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan2",
    "arctan",
    "arctanh",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "cbrt",
    "ceil",
    "conj",
    "conjugate",
    "copysign",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "divide",
    "divmod",
    "equal",
    "exp2",
    "exp",
    "expm1",
    "fabs",
    "float_power",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "invert",
    "isfinite",
    "isinf",
    "isnan",
    "isnat",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "log10",
    "log1p",
    "log2",
    "log",
    "logaddexp2",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "maximum",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "negative",
    "nextafter",
    "not_equal",
    "positive",
    "power",
    "rad2deg",
    "radians",
    "reciprocal",
    "remainder",
    "right_shift",
    "rint",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spacing",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "true_divide",
    "trunc",
    "abs",
    "asarray",
    "result_type",
]
