/* Copyright 2022 The ml_dtypes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ML_DTYPES_COMMON_H_
#define ML_DTYPES_COMMON_H_

// Must be included first
// clang-format off
#include "_src/numpy.h"
// clang-format on

#include <Python.h>

#include <complex>  //NOLINT

#include "Eigen/Core"

namespace ml_dtypes {

struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

// Safe container for an owned PyObject. On destruction, the reference count of
// the contained object will be decremented.
using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
inline Safe_PyObjectPtr make_safe(PyObject* object) {
  return Safe_PyObjectPtr(object);
}

template <typename T, typename Enable = void>
struct TypeDescriptor {
  // typedef ... T;  // Representation type in memory for NumPy values of type
  // static constexpr int legacy_type_num = NPY_...; }  // Numpy type number for
  // T
};

template <>
struct TypeDescriptor<unsigned char> {
  typedef unsigned char T;
  static constexpr int legacy_type_num = NPY_UBYTE;
  static PyArray_DTypeMeta* Dtype() { return &PyArray_UByteDType; }
};

template <>
struct TypeDescriptor<unsigned short> {  // NOLINT
  typedef unsigned short T;              // NOLINT
  static constexpr int legacy_type_num = NPY_USHORT;
};

// We register "int", "long", and "long long" types for portability across
// Linux, where "int" and "long" are the same type, and Windows, where "long"
// and "longlong" are the same type.
template <>
struct TypeDescriptor<unsigned int> {
  typedef unsigned int T;
  static constexpr int legacy_type_num = NPY_UINT;
};

template <>
struct TypeDescriptor<unsigned long> {  // NOLINT
  typedef unsigned long T;              // NOLINT
  static constexpr int legacy_type_num = NPY_ULONG;
};

template <>
struct TypeDescriptor<unsigned long long> {  // NOLINT
  typedef unsigned long long T;              // NOLINT
  static constexpr int legacy_type_num = NPY_ULONGLONG;
};

template <>
struct TypeDescriptor<signed char> {
  typedef signed char T;
  static constexpr int legacy_type_num = NPY_BYTE;
};

template <>
struct TypeDescriptor<short> {  // NOLINT
  typedef short T;              // NOLINT
  static constexpr int legacy_type_num = NPY_SHORT;
};

template <>
struct TypeDescriptor<int> {
  typedef int T;
  static constexpr int legacy_type_num = NPY_INT;
  static PyArray_DTypeMeta* Dtype() { return &PyArray_IntDType; }
};

template <>
struct TypeDescriptor<long> {  // NOLINT
  typedef long T;              // NOLINT
  static constexpr int legacy_type_num = NPY_LONG;
};

template <>
struct TypeDescriptor<long long> {  // NOLINT
  typedef long long T;              // NOLINT
  static constexpr int legacy_type_num = NPY_LONGLONG;
};

template <>
struct TypeDescriptor<bool> {
  typedef unsigned char T;
  static constexpr int legacy_type_num = NPY_BOOL;
  static PyArray_DTypeMeta* Dtype() { return &PyArray_BoolDType; }
};

template <>
struct TypeDescriptor<Eigen::half> {
  typedef Eigen::half T;
  static constexpr int legacy_type_num = NPY_HALF;
};

template <>
struct TypeDescriptor<float> {
  typedef float T;
  static constexpr int legacy_type_num = NPY_FLOAT;
};

template <>
struct TypeDescriptor<double> {
  typedef double T;
  static constexpr int legacy_type_num = NPY_DOUBLE;
};

template <>
struct TypeDescriptor<long double> {
  typedef long double T;
  static constexpr int legacy_type_num = NPY_LONGDOUBLE;
};

template <>
struct TypeDescriptor<std::complex<float>> {
  typedef std::complex<float> T;
  static constexpr int legacy_type_num = NPY_CFLOAT;
};

template <>
struct TypeDescriptor<std::complex<double>> {
  typedef std::complex<double> T;
  static constexpr int legacy_type_num = NPY_CDOUBLE;
};

template <>
struct TypeDescriptor<std::complex<long double>> {
  typedef std::complex<long double> T;
  static constexpr int legacy_type_num = NPY_CLONGDOUBLE;
};

template <class T>
struct is_complex : std::false_type {};
template <class T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

}  // namespace ml_dtypes

#endif  // ML_DTYPES_COMMON_H_
