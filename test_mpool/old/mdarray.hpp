// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file mdarray.hpp
 *
 *  \brief Contains implementation of multidimensional array class.
 */

#ifndef __MDARRAY_HPP__
#define __MDARRAY_HPP__

#include <signal.h>
#include <cassert>
#include <memory>
#include <string>
#include <atomic>
#include <vector>
#include <array>
#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <functional>
#include "memory_pool.hpp"

// TODO: now .at() method is templated over the device type; it would make more sense to template over
//       the memory type, i.e. instead of array.at<GPU>() we would write array.at<memory_t::device>()
//       Later,  move device_t to typedefs.hpp or to a SDDK namespace or to linear algebra and FFT backends.

// TODO: now .at() method is templated over the device type; it would make more sense to template over
//       the memory type, i.e. instead of array.at<GPU>() we would write array.at<memory_t::device>()

namespace sddk {

//#ifdef __GPU
//extern "C" void add_checksum_gpu(cuDoubleComplex* wf__,
//                                 int num_rows_loc__,
//                                 int nwf__,
//                                 cuDoubleComplex* result__);
//#endif

#ifdef NDEBUG
#define mdarray_assert(condition__)
#else
#define mdarray_assert(condition__)                             \
{                                                               \
    if (!(condition__)) {                                       \
        printf("Assertion (%s) failed ", #condition__);         \
        printf("at line %i of file %s\n", __LINE__, __FILE__);  \
        printf("array label: %s\n", label_.c_str());            \
        for (int i = 0; i < N; i++)                             \
            printf("dim[%i].size = %li\n", i, dims_[i].size()); \
        raise(SIGTERM);                                         \
        exit(-13);                                              \
    }                                                           \
}
#endif

// TODO: change to enum class
/// Type of the main processing unit.
/** List the processing units on which the code can run. */
enum device_t
{
    /// CPU device.
    CPU = 0,

    /// GPU device (with CUDA programming model).
    GPU = 1
};

// TODO: remove operator|() and operator&(): their usage is very limited and complicates the code
inline constexpr memory_t operator&(memory_t a__, memory_t b__) noexcept
{
    return static_cast<memory_t>(static_cast<unsigned int>(a__) & static_cast<unsigned int>(b__));
}

inline constexpr memory_t operator|(memory_t a__, memory_t b__) noexcept
{
    return static_cast<memory_t>(static_cast<unsigned int>(a__) | static_cast<unsigned int>(b__));
}

inline constexpr bool on_device(memory_t mem_type__) noexcept
{
    return (mem_type__ & memory_t::device) == memory_t::device ? true : false;
}

/// Mapping between a memory type and a device type.
template <memory_t mem_type>
struct device;

template<>
struct device<memory_t::host> {
    static const device_t type{device_t::CPU};
};

template<>
struct device<memory_t::host_pinned> {
    static const device_t type{device_t::CPU};
};

template<>
struct device<memory_t::device> {
    static const device_t type{device_t::GPU};
};

/// Mapping between a device type and a memory type.
template <device_t dev_type>
struct memory;

template<>
struct memory<device_t::CPU> {
    static const memory_t type{memory_t::host};
};

template<>
struct memory<device_t::GPU> {
    static const memory_t type{memory_t::device};
};

/// Index descriptor of mdarray.
class mdarray_index_descriptor
{
  public:
    //typedef int64_t index_type;
    using index_type = int64_t;

  private:
    /// Beginning of index.
    index_type begin_{0};

    /// End of index.
    index_type end_{-1};

    /// Size of index.
    size_t size_{0};

  public:

    /// Constructor of empty descriptor.
    mdarray_index_descriptor()
    {
    }

    /// Constructor for index range [0, size).
    mdarray_index_descriptor(size_t const size__)
        : end_(size__ - 1)
        , size_(size__)
    {
    }

    /// Constructor for index range [begin, end]
    mdarray_index_descriptor(index_type const begin__, index_type const end__)
        : begin_(begin__)
        , end_(end__)
        , size_(end_ - begin_ + 1)
    {
        assert(end_ >= begin_);
    };

    /// Constructor for index range [begin, end]
    mdarray_index_descriptor(std::pair<int, int> const range__)
        : begin_(range__.first)
        , end_(range__.second)
        , size_(end_ - begin_ + 1)
    {
        assert(end_ >= begin_);
    };

    /// Return first index value.
    inline index_type begin() const
    {
        return begin_;
    }

    /// Return last index value.
    inline index_type end() const
    {
        return end_;
    }

    /// Return index size.
    inline size_t size() const
    {
        return size_;
    }
};

struct mdarray_mem_count
{
    static std::atomic<int64_t>& allocated()
    {
        static std::atomic<int64_t> allocated_{0};
        return allocated_;
    }

    static std::atomic<int64_t>& allocated_max()
    {
        static std::atomic<int64_t> allocated_max_{0};
        return allocated_max_;
    }
};

/// Simple memory manager handler which keeps track of allocated and deallocated memory.
template <typename T>
struct mdarray_mem_mgr
{
    /// Number of elements of the current allocation.
    size_t size_{0};

    /// Type of allocated memory.
    memory_t mode_{memory_t::none};

    mdarray_mem_mgr()
    {
    }

    mdarray_mem_mgr(size_t const size__, memory_t mode__)
        : size_(size__)
        , mode_(mode__)
    {
        if ((mode_ & memory_t::host) == memory_t::host) {
            mdarray_mem_count::allocated() += size_ * sizeof(T);
            mdarray_mem_count::allocated_max() = std::max(mdarray_mem_count::allocated().load(),
                                                          mdarray_mem_count::allocated_max().load());
        }
    }

    /// Called by std::unique_ptr when the object is destroyed.
    void operator()(T* p__) const
    {
        if ((mode_ & memory_t::host) == memory_t::host) {
            mdarray_mem_count::allocated() -= size_ * sizeof(T);
            /* call destructor for non-primitive objects */
            if (!std::is_pod<T>::value) {
                for (size_t i = 0; i < size_; i++) {
                    (p__ + i)->~T();
                }
            }
        }

        /* host memory can be of two types */
        if ((mode_ & memory_t::host) == memory_t::host) {
            /* check if the memory is host pinned */
            if ((mode_ & memory_t::host_pinned) == memory_t::host_pinned) {
#ifdef __GPU
                if (acc::num_devices() > 0) {
                    acc::deallocate_host(p__);
                } else {
                    free(p__);
                }
#endif
            } else {
                free(p__);
            }
        }

        if ((mode_ & memory_t::device) == memory_t::device) {
#ifdef __GPU
            acc::deallocate(p__);
#endif
        }
    }
};

/// Multidimensional array with the column-major (Fortran) order.
/** The implementation supports two memory pointers: one is accessible by CPU and second is accessible by a device. */
template <typename T, int N>
class mdarray
{
  public:
    typedef mdarray_index_descriptor::index_type index_type;

  private:
    /// Optional array label.
    std::string label_;

    /// Unique pointer to the allocated memory.
    std::unique_ptr<T[], mdarray_mem_mgr<T>> unique_ptr_{nullptr};

    /// Unique pointer in case of memory pool allocation.
    memory_pool::unique_ptr<T> unique_pool_ptr_{nullptr};

    /// Raw pointer.
    T* raw_ptr_{nullptr};
#ifdef __GPU
    /// Unique pointer to the allocated GPU memory.
    std::unique_ptr<T[], mdarray_mem_mgr<T>> unique_ptr_device_{nullptr};

    /// Unique pointer in case of memory pool allocation.
    memory_pool::unique_ptr<T> unique_pool_ptr_device_{nullptr};

    /// Raw pointer to GPU memory
    T* raw_ptr_device_{nullptr};
#endif
    /// Array dimensions.
    std::array<mdarray_index_descriptor, N> dims_;

    /// List of offsets to compute the element location by dimension indices.
    std::array<index_type, N> offsets_;

    /// Initialize the offsets used to compute the index of the elements.
    void init_dimensions(std::array<mdarray_index_descriptor, N> const dims__)
    {
        dims_ = dims__;

        offsets_[0] = -dims_[0].begin();
        size_t ld{1};
        for (int i = 1; i < N; i++) {
            ld *= dims_[i - 1].size();
            offsets_[i] = ld;
            offsets_[0] -= ld * dims_[i].begin();
        }
    }

    template <typename... Args>
    inline index_type idx(Args... args) const
    {
        static_assert(N == sizeof...(args), "wrong number of dimensions");
        std::array<index_type, N> i = {args...};

        for (int j = 0; j < N; j++) {
            mdarray_assert(i[j] >= dims_[j].begin() && i[j] <= dims_[j].end());
        }

        size_t idx = offsets_[0] + i[0];
        for (int j = 1; j < N; j++) {
            idx += i[j] * offsets_[j];
        }
        mdarray_assert(idx >= 0 && idx < size());
        return idx;
    }

    template <device_t pu>
    inline T* at_idx(index_type const idx__)
    {
        switch (pu) {
            case CPU: {
                mdarray_assert(raw_ptr_ != nullptr);
                return &raw_ptr_[idx__];
            }
            case GPU: {
#ifdef __GPU
                mdarray_assert(raw_ptr_device_ != nullptr);
                return &raw_ptr_device_[idx__];
#else
                printf("error at line %i of file %s: not compiled with GPU support\n", __LINE__, __FILE__);
                exit(0);
#endif
            }
        }
        return nullptr;
    }

    template <device_t pu>
    inline T const* at_idx(index_type const idx__) const
    {
        switch (pu) {
            case CPU: {
                mdarray_assert(raw_ptr_ != nullptr);
                return &raw_ptr_[idx__];
            }
            case GPU: {
#ifdef __GPU
                mdarray_assert(raw_ptr_device_ != nullptr);
                return &raw_ptr_device_[idx__];
#else
                printf("error at line %i of file %s: not compiled with GPU support\n", __LINE__, __FILE__);
                exit(0);
#endif
            }
        }
        return nullptr;
    }

    /// Copy constructor is forbidden
    mdarray(mdarray<T, N> const& src) = delete;

    /// Assignment operator is forbidden
    mdarray<T, N>& operator=(mdarray<T, N> const& src) = delete;

  public:

    /// Move constructor
    mdarray(mdarray<T, N>&& src)
        : label_(src.label_)
        , unique_ptr_(std::move(src.unique_ptr_))
        , unique_pool_ptr_(std::move(src.unique_pool_ptr_))
        , raw_ptr_(src.raw_ptr_)
#ifdef __GPU
        , unique_ptr_device_(std::move(src.unique_ptr_device_))
        , unique_pool_ptr_device_(std::move(src.unique_pool_ptr_device_))
        , raw_ptr_device_(src.raw_ptr_device_)
#endif
    {
        for (int i = 0; i < N; i++) {
            dims_[i]    = src.dims_[i];
            offsets_[i] = src.offsets_[i];
        }
        src.raw_ptr_ = nullptr;
#ifdef __GPU
        src.raw_ptr_device_ = nullptr;
#endif
    }

    /// Move assigment operator
    inline mdarray<T, N>& operator=(mdarray<T, N>&& src)
    {
        if (this != &src) {
            label_           = src.label_;
            unique_ptr_      = std::move(src.unique_ptr_);
            unique_pool_ptr_ = std::move(src.unique_pool_ptr_);
            raw_ptr_         = src.raw_ptr_;
            src.raw_ptr_     = nullptr;
#ifdef __GPU
            unique_ptr_device_      = std::move(src.unique_ptr_device_);
            unique_pool_ptr_device_ = std::move(src.unique_pool_ptr_device_);
            raw_ptr_device_         = src.raw_ptr_device_;
            src.raw_ptr_device_     = nullptr;
#endif
            for (int i = 0; i < N; i++) {
                dims_[i]    = src.dims_[i];
                offsets_[i] = src.offsets_[i];
            }
        }
        return *this;
    }

    /// Allocate memory for array.
    void allocate(memory_t memory__)
    {
        size_t sz = size();
        /* do nothing for zero-sized array */
        if (!sz) {
            return;
        }
#ifndef __GPU
        if ((memory__ & memory_t::host_pinned) == memory_t::host_pinned) {
            /* CPU only code, no point in using page-locked memory */
            memory__ = memory_t::host;
        }
#else
        /* GPU enabled code, check if there is a CUDA capable device */
        if ((memory__ & memory_t::host_pinned) == memory_t::host_pinned) {
            if (acc::num_devices() == 0) {
                /* there is no cuda card, don't use page-locked memory */
                memory__ = memory_t::host;
            }
        }
#endif
        /* host allocation */
        if ((memory__ & memory_t::host) == memory_t::host) {
            /* page-locked memory */
            if ((memory__ & memory_t::host_pinned) == memory_t::host_pinned) {
#ifdef __GPU
                raw_ptr_    = acc::allocate_host<T>(sz);
                unique_ptr_ = std::unique_ptr<T[], mdarray_mem_mgr<T>>(raw_ptr_, mdarray_mem_mgr<T>(sz, memory_t::host_pinned));
#endif
            } else { /* regular mameory */
                raw_ptr_    = static_cast<T*>(malloc(sz * sizeof(T)));
                unique_ptr_ = std::unique_ptr<T[], mdarray_mem_mgr<T>>(raw_ptr_, mdarray_mem_mgr<T>(sz, memory_t::host));
            }

            /* call constructor on non-trivial data */
            if (raw_ptr_ != nullptr && !std::is_pod<T>::value) {
                for (size_t i = 0; i < sz; i++) {
                    new (raw_ptr_ + i) T();
                }
            }
        }

        /* device allocation */
#ifdef __GPU
        if ((memory__ & memory_t::device) == memory_t::device) {
            raw_ptr_device_    = acc::allocate<T>(sz);
            unique_ptr_device_ = std::unique_ptr<T[], mdarray_mem_mgr<T>>(raw_ptr_device_, mdarray_mem_mgr<T>(sz, memory_t::device));
        }
#endif
    }

    void deallocate(memory_t memory__)
    {
        if ((memory__ & memory_t::host) == memory_t::host) {
            if (unique_ptr_) {
                unique_ptr_.reset(nullptr);
                unique_pool_ptr_.reset(nullptr);
                raw_ptr_ = nullptr;
            }
        }
#ifdef __GPU
        if ((memory__ & memory_t::device) == memory_t::device) {
            if (unique_ptr_device_) {
                unique_ptr_device_.reset(nullptr);
                unique_pool_ptr_device_.reset(nullptr);
                raw_ptr_device_ = nullptr;
            }
        }
#endif
    }

    inline void allocate(memory_pool& mp__)
    {
        switch (mp__.memory_type()) {
            case memory_t::host:
            case memory_t::host_pinned: {
                this->unique_pool_ptr_ = mp__.get_unique_ptr<T>(size());
                this->raw_ptr_ = this->unique_pool_ptr_.get();
                break;
            }
            case memory_t::device: {
#ifdef __GPU
                this->unique_pool_ptr_device_ = mp__.get_unique_ptr<T>(size());
                this->raw_ptr_device_ = this->unique_pool_ptr_device_.get();
#endif
                break;
            }
            default: {
                throw std::runtime_error("unsupported memory type");
            }
        }
    }

    template <typename... Args>
    inline T& operator()(Args... args)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(args...)];
    }

    template <typename... Args>
    inline T const& operator()(Args... args) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(args...)];
    }

    inline T& operator[](size_t const idx__)
    {
        mdarray_assert(idx__ >= 0 && idx__ < size());
        return raw_ptr_[idx__];
    }

    inline T const& operator[](size_t const idx__) const
    {
        assert(idx__ >= 0 && idx__ < size());
        return raw_ptr_[idx__];
    }

    template <device_t pu>
    inline T* at()
    {
        return at_idx<pu>(0);
    }

    template <device_t pu>
    inline T const* at() const
    {
        return at_idx<pu>(0);
    }

    template <device_t pu, typename... Args>
    inline T* at(Args... args)
    {
        return at_idx<pu>(idx(args...));
    }

    template <device_t pu, typename... Args>
    inline T const* at(Args... args) const
    {
        return at_idx<pu>(idx(args...));
    }

    template <device_t pu>
    typename std::enable_if<pu == device_t::CPU, T*>::type data()
    {
        return raw_ptr_;
    }

    /// Return total size (number of elements) of the array.
    inline size_t size() const
    {
        size_t size_{1};

        for (int i = 0; i < N; i++) {
            size_ *= dims_[i].size();
        }

        return size_;
    }

    /// Return size of particular dimension.
    inline size_t size(int i) const
    {
        mdarray_assert(i < N);
        return dims_[i].size();
    }

    inline mdarray_index_descriptor dim(int i) const
    {
        mdarray_assert(i < N);
        return dims_[i];
    }

    /// Return leading dimension size.
    inline uint32_t ld() const
    {
        mdarray_assert(dims_[0].size() < size_t(1 << 31));

        return (int32_t)dims_[0].size();
    }

    /// Compute hash of the array
    /** Example: printf("hash(h) : %16llX\n", h.hash()); */
    inline uint64_t hash(uint64_t h__ = 5381) const
    {
        for (size_t i = 0; i < size() * sizeof(T); i++) {
            h__ = ((h__ << 5) + h__) + ((unsigned char*)raw_ptr_)[i];
        }

        return h__;
    }

    inline T checksum_w(size_t idx0__, size_t size__) const
    {
        T cs{0};
        for (size_t i = 0; i < size__; i++) {
            cs += raw_ptr_[idx0__ + i] * static_cast<double>((i & 0xF) - 8);
        }
        return cs;
    }

    inline T checksum(size_t idx0__, size_t size__) const
    {
        T cs{0};
        for (size_t i = 0; i < size__; i++) {
            cs += raw_ptr_[idx0__ + i];
        }
        return cs;
    }

    inline T checksum() const
    {
        return checksum(0, size());
    }

    //== template <device_t pu>
    //== inline T checksum() const
    //== {
    //==     switch (pu) {
    //==         case CPU: {
    //==             return checksum();
    //==         }
    //==         case GPU: {
    //==            auto cs = acc::allocate<T>(1);
    //==            acc::zero(cs, 1);
    //==            add_checksum_gpu(raw_ptr_device_, (int)size(), 1, cs);
    //==            T cs1;
    //==            acc::copyout(&cs1, cs, 1);
    //==            acc::deallocate(cs);
    //==            return cs1;
    //==         }
    //==     }
    //== }

    /// Copy the content of the array to dest
    void operator>>(mdarray<T, N>& dest__) const
    {
        for (int i = 0; i < N; i++) {
            if (dest__.dims_[i].begin() != dims_[i].begin() || dest__.dims_[i].end() != dims_[i].end()) {
                printf("error at line %i of file %s: array dimensions don't match\n", __LINE__, __FILE__);
                raise(SIGTERM);
                exit(-1);
            }
        }
        std::memcpy(dest__.raw_ptr_, raw_ptr_, size() * sizeof(T));
    }

    /// Copy n elements starting from idx0.
    template <memory_t from__, memory_t to__>
    inline void copy(size_t idx0__, size_t n__, int stream_id__ = -1)
    {
#ifdef __GPU
        mdarray_assert(raw_ptr_ != nullptr);
        mdarray_assert(raw_ptr_device_ != nullptr);
        mdarray_assert(idx0__ + n__ <= size());

        if ((from__ & memory_t::host) == memory_t::host && (to__ & memory_t::device) == memory_t::device) {
            if (stream_id__ == -1) {
                acc::copyin(&raw_ptr_device_[idx0__], &raw_ptr_[idx0__], n__);
            } else {
                acc::copyin(&raw_ptr_device_[idx0__], &raw_ptr_[idx0__], n__, stream_id__);
            }
        }

        if ((from__ & memory_t::device) == memory_t::device && (to__ & memory_t::host) == memory_t::host) {
            if (stream_id__ == -1) {
                acc::copyout(&raw_ptr_[idx0__], &raw_ptr_device_[idx0__], n__);
            } else {
                acc::copyout(&raw_ptr_[idx0__], &raw_ptr_device_[idx0__], n__, stream_id__);
            }
        }
#endif
    }

    template <memory_t from__, memory_t to__>
    inline void copy(size_t n__)
    {
        copy<from__, to__>(0, n__);
    }

    template <memory_t from__, memory_t to__>
    inline void async_copy(size_t n__, int stream_id__)
    {
        copy<from__, to__>(0, n__, stream_id__);
    }

    template <memory_t from__, memory_t to__>
    inline void copy()
    {
        copy<from__, to__>(0, size());
    }

    template <memory_t from__, memory_t to__>
    inline void async_copy(int stream_id__)
    {
        copy<from__, to__>(0, size(), stream_id__);
    }

    /// Zero n elements starting from idx0.
    template <memory_t mem_type__>
    inline void zero(size_t idx0__, size_t n__)
    {
        mdarray_assert(idx0__ + n__ <= size());
        if (((mem_type__ & memory_t::host) == memory_t::host) && n__) {
            mdarray_assert(raw_ptr_ != nullptr);
            std::memset(&raw_ptr_[idx0__], 0, n__ * sizeof(T));
        }
#ifdef __GPU
        if (((mem_type__ & memory_t::device) == memory_t::device) && on_device() && n__) {
            mdarray_assert(raw_ptr_device_ != nullptr);
            acc::zero(&raw_ptr_device_[idx0__], n__);
        }
#endif
    }

    /// Zero the entire array.
    template <memory_t mem_type__ = memory_t::host>
    inline void zero()
    {
        zero<mem_type__>(0, size());
    }

    inline bool on_device() const
    {
#ifdef __GPU
        return (raw_ptr_device_ != nullptr);
#else
        return false;
#endif
    }

    mdarray()
    {
    }

    /// N-dimensional array with index bounds.
    mdarray(std::array<mdarray_index_descriptor, N> const dims__,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        this->label_ = label__;
        this->init_dimensions(dims__);
        this->allocate(memory__);
    }

    /// 1D array with memory allocation.
    mdarray(mdarray_index_descriptor const& d0,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0});
        this->allocate(memory__);
    }

    /// 2D array with memory allocation.
    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1});
        this->allocate(memory__);
    }

    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 3, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2});
        this->allocate(memory__);
    }

    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 4, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3});
        this->allocate(memory__);
    }

    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            mdarray_index_descriptor const& d4,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 5, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3, d4});
        this->allocate(memory__);
    }

    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            mdarray_index_descriptor const& d4,
            mdarray_index_descriptor const& d5,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 6, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3, d4, d5});
        this->allocate(memory__);
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0});
        this->raw_ptr_ = ptr__;
    }

    mdarray(memory_pool& mp__,
            mdarray_index_descriptor const& d0,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0});
        this->unique_pool_ptr_ = mp__.get_unique_ptr<T>(size());
        this->raw_ptr_ = this->unique_pool_ptr_.get();
    }

    mdarray(T* ptr__,
            T* ptr_device__,
            mdarray_index_descriptor const& d0,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0});
        this->raw_ptr_ = ptr__;
#ifdef __GPU
        this->raw_ptr_device_ = ptr_device__;
#endif
    }

    mdarray(memory_pool& mp__,
            memory_pool& mpd__,
            mdarray_index_descriptor const& d0,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0});
        this->unique_pool_ptr_ = mp__.get_unique_ptr<T>(size());
        this->raw_ptr_ = this->unique_pool_ptr_.get();
#ifdef __GPU
        this->unique_pool_ptr_device_ = mpd__.get_unique_ptr<T>(size());
        this->raw_ptr_device_ = this->unique_pool_ptr_device_.get();
#endif
    }

    mdarray(T* ptr__,
            memory_pool& mpd__,
            mdarray_index_descriptor const& d0,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0});
        this->raw_ptr_ = ptr__;
#ifdef __GPU
        this->unique_pool_ptr_device_ = mpd__.get_unique_ptr<T>(size());
        this->raw_ptr_device_ = this->unique_pool_ptr_device_.get();
#endif
    }

    /// Wrap a pointer into 2D array.
    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1});
        this->raw_ptr_ = ptr__;
    }

    mdarray(memory_pool& mp__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1});
        this->unique_pool_ptr_ = std::move(mp__.get_unique_ptr<T>(size()));
        this->raw_ptr_ = this->unique_pool_ptr_.get();
    }

    mdarray(T* ptr__,
            T* ptr_device__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1});
        this->raw_ptr_ = ptr__;
#ifdef __GPU
        this->raw_ptr_device_ = ptr_device__;
#endif
    }

    mdarray(memory_pool& mp__,
            memory_pool& mpd__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1});
        this->unique_pool_ptr_ = mp__.get_unique_ptr<T>(size());
        this->raw_ptr_ = this->unique_pool_ptr_.get();
#ifdef __GPU
        this->unique_pool_ptr_device_ = mpd__.get_unique_ptr<T>(size());
        this->raw_ptr_device_ = this->unique_pool_ptr_device_.get();
#endif
    }

    mdarray(T* ptr__,
            memory_pool& mpd__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1});
        this->raw_ptr_ = ptr__;
#ifdef __GPU
        this->unique_pool_ptr_device_ = mpd__.get_unique_ptr<T>(size());
        this->raw_ptr_device_ = this->unique_pool_ptr_device_.get();
#endif
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            std::string label__ = "")
    {
        static_assert(N == 3, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2});
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__,
            T* ptr_device__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            std::string label__ = "")
    {
        static_assert(N == 3, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2});
        this->raw_ptr_ = ptr__;
#ifdef __GPU
        this->raw_ptr_device_ = ptr_device__;
#endif
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            std::string label__ = "")
    {
        static_assert(N == 4, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3});
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            mdarray_index_descriptor const& d4,
            std::string label__ = "")
    {
        static_assert(N == 5, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3, d4});
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            mdarray_index_descriptor const& d4,
            mdarray_index_descriptor const& d5,
            std::string label__ = "")
    {
        static_assert(N == 6, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3, d4, d5});
        this->raw_ptr_ = ptr__;
    }

    mdarray<T, N>& operator=(std::function<T(index_type)> f__)
    {
        static_assert(N == 1, "wrong number of dimensions");

        for (index_type i0 = this->dims_[0].begin(); i0 <= this->dims_[0].end(); i0++) {
            (*this)(i0) = f__(i0);
        }
        return *this;
    }

    mdarray<T, N>& operator=(std::function<T(index_type, index_type)> f__)
    {
        static_assert(N == 2, "wrong number of dimensions");

        for (index_type i1 = this->dims_[1].begin(); i1 <= this->dims_[1].end(); i1++) {
            for (index_type i0 = this->dims_[0].begin(); i0 <= this->dims_[0].end(); i0++) {
                (*this)(i0, i1) = f__(i0, i1);
            }
        }
        return *this;
    }
};

// Alias for matrix
template <typename T>
using matrix = mdarray<T, 2>;

/// Serialize to std::ostream
template <typename T, int N>
std::ostream& operator<<(std::ostream& out, mdarray<T, N>& v)
{
    if (v.size()) {
        out << v[0];
        for (size_t i = 1; i < v.size(); i++) {
            out << std::string(" ") << v[i];
        }
    }
    return out;
}

} // namespace sddk

#endif // __MDARRAY_HPP__
