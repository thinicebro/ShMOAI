#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define PHANTOM_CHECK_CUDA(val) check((val), #val, __FILE__, __LINE__)
#define PHANTOM_CHECK_CUDA_LAST() checkLast(__FILE__, __LINE__)

template <typename T>
inline void check(T err, const char *const func, const char *const file,
                  const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << std::endl
                  << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkLast(const char *const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

namespace phantom::util
{

    class cuda_stream_wrapper
    {
    public:
        cuda_stream_wrapper()
        {
            cudaSetDevice(0); // TODO: remove me
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        }

        ~cuda_stream_wrapper()
        {
            cudaStreamDestroy(stream);
        }

        [[nodiscard]] auto &get_stream() const
        {
            return stream;
        }

    private:
        cudaStream_t stream{};
    };

    namespace detail
    {
        // [EXP quarantine] Deferred-free quarantine for the async mempool.
        //
        // Root-cause experiment for the intermittent garbage-ciphertext
        // explosions (L1/L4/L6/L8, deterministic consequences, drifting
        // victims): cudaFreeAsync(ptr, stream) only orders the free against
        // work on `stream`, but ciphertext buffers are routinely consumed on
        // OTHER streams (multi-stream matmul/gelu/softmax with per-thread
        // stream tags). The mempool can then recycle a block while another
        // stream is still reading/writing it -> cross-stream
        // use-after-free.
        //
        // Instead of freeing immediately, reset() parks the block in a
        // quarantine. Blocks are only returned to the pool after a
        // device-wide synchronize, at which point NO stream can still be
        // using them. This closes the hazard regardless of which stream tag
        // a buffer carries, while keeping cudaMallocAsync pool performance:
        // the device sync is amortized over QUARANTINE_FLUSH_BYTES of frees
        // instead of happening per free (per-free cudaFree made the 12-layer
        // run ~10x slower; this should stay within ~1.2x of baseline).
        struct FreeQuarantine
        {
            std::mutex mu;
            std::vector<std::pair<void *, cudaStream_t>> items;
            size_t bytes = 0;
        };

        inline FreeQuarantine &free_quarantine()
        {
            static FreeQuarantine q;
            return q;
        }

        // Flush threshold: 2 GiB of quarantined blocks. Peak live usage in
        // the 768-wide run leaves ~29 GiB free, so 2 GiB of dead-but-
        // unreusable memory is safe and keeps syncs infrequent.
        inline constexpr size_t QUARANTINE_FLUSH_BYTES = (size_t)2 << 30;

        inline void quarantine_flush()
        {
            auto &q = free_quarantine();
            std::vector<std::pair<void *, cudaStream_t>> todo;
            {
                std::lock_guard<std::mutex> lk(q.mu);
                todo.swap(q.items);
                q.bytes = 0;
            }
            if (todo.empty())
            {
                return;
            }
            // After a device-wide sync no stream can still touch the
            // quarantined blocks, so returning them to the pool is safe on
            // any stream tag.
            PHANTOM_CHECK_CUDA(cudaDeviceSynchronize());
            for (auto &it : todo)
            {
                PHANTOM_CHECK_CUDA(cudaFreeAsync(it.first, it.second ? it.second : (cudaStream_t)cudaStreamLegacy));
            }
        }

        inline void quarantined_free(void *ptr, size_t bytes, cudaStream_t stream)
        {
            bool need_flush;
            {
                auto &q = free_quarantine();
                std::lock_guard<std::mutex> lk(q.mu);
                q.items.emplace_back(ptr, stream);
                q.bytes += bytes;
                need_flush = q.bytes >= QUARANTINE_FLUSH_BYTES;
            }
            if (need_flush)
            {
                quarantine_flush();
            }
        }
    } // namespace detail

    // Flush all quarantined frees (device sync + return blocks to the pool).
    // May be called at natural pipeline sync points to bound memory growth.
    inline void quarantine_flush()
    {
        detail::quarantine_flush();
    }

    template <class T>
    class cuda_auto_ptr
    {

    private:
        T *ptr_ = nullptr;
        size_t n_ = 0;
        cudaStream_t cudaStream_ = nullptr;

    public:
        cuda_auto_ptr() = default;

        explicit cuda_auto_ptr(T *ptr, size_t n, const cudaStream_t &stream)
        {
            ptr_ = ptr;
            n_ = n;
            cudaStream_ = stream;
        }

        // copy constructor
        cuda_auto_ptr(const cuda_auto_ptr &obj)
        {
            // The copy is enqueued on the SOURCE object's stream; without a
            // sync here, consumers operating on a different stream can read
            // the destination buffer before the copy lands (observed as
            // intermittent whole-ciphertext garbage on sm_120). Sync the
            // source stream so the copy is complete on return.
            PHANTOM_CHECK_CUDA(cudaMallocAsync(&this->ptr_, obj.n_ * sizeof(T), obj.cudaStream_));
            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(this->ptr_, obj.ptr_, obj.n_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                               obj.cudaStream_));
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(obj.cudaStream_));
            this->n_ = obj.n_;
            this->cudaStream_ = obj.cudaStream_;
        }

        // copy assignment
        cuda_auto_ptr &operator=(const cuda_auto_ptr &obj)
        {
            if (this == &obj)
            {
                return *this;
            }

            reset();

            // see the copy constructor for why the source stream is synced
            PHANTOM_CHECK_CUDA(cudaMallocAsync(&this->ptr_, obj.n_ * sizeof(T), obj.cudaStream_));
            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(this->ptr_, obj.ptr_, obj.n_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                               obj.cudaStream_));
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(obj.cudaStream_));
            this->n_ = obj.n_;
            this->cudaStream_ = obj.cudaStream_;
            return *this;
        }

        // move constructor
        cuda_auto_ptr(cuda_auto_ptr &&dyingObj) noexcept
        {
            // share the underlying pointer
            this->ptr_ = dyingObj.ptr_;
            this->n_ = dyingObj.n_;
            this->cudaStream_ = dyingObj.cudaStream_;

            // reset the dying object
            dyingObj.ptr_ = nullptr;
            dyingObj.n_ = 0;
            dyingObj.cudaStream_ = nullptr;
        }

        // move assignment
        cuda_auto_ptr &operator=(cuda_auto_ptr &&dyingObj) noexcept
        {
            if (this == &dyingObj)
            {
                return *this;
            }

            reset();

            this->ptr_ = dyingObj.ptr_;
            this->n_ = dyingObj.n_;
            this->cudaStream_ = dyingObj.cudaStream_;

            // reset the dying object
            dyingObj.ptr_ = nullptr;
            dyingObj.n_ = 0;
            dyingObj.cudaStream_ = nullptr;

            return *this;
        }

        ~cuda_auto_ptr() // destructor
        {
            reset();
        }

        T *get() const
        {
            return this->ptr_;
        }

        T *operator->() const
        {
            return this->ptr_;
        }

        T &operator*() const
        {
            return this->ptr_;
        }

        [[nodiscard]] auto &get_n() const
        {
            return this->n_;
        }

        [[nodiscard]] auto &get_stream() const
        {
            return this->cudaStream_;
        }

        void reset()
        {
            if (ptr_ == nullptr)
            {
                return;
            }
            // [EXP quarantine] defer the free; the block is only returned to
            // the pool after a device-wide sync (see detail::quarantine_flush).
            detail::quarantined_free(ptr_, n_ * sizeof(T), cudaStream_);
            ptr_ = nullptr;
            n_ = 0;
            cudaStream_ = nullptr;
        }
    };

    template <class T>
    cuda_auto_ptr<T> make_cuda_auto_ptr(size_t n, const cudaStream_t &stream)
    {
        T *ptr;
        PHANTOM_CHECK_CUDA(cudaMallocAsync(&ptr, n * sizeof(T), stream));
        return cuda_auto_ptr<T>(ptr, n, stream);
    }

    class CUDATimer
    {
    public:
        explicit CUDATimer(std::string func_name, const phantom::util::cuda_stream_wrapper &stream_wrapper) : func_name_(std::move(func_name))
        {
            stream_ = stream_wrapper.get_stream();
            cudaEventCreate(&start_event_);
            cudaEventCreate(&stop_event_);
        }

        ~CUDATimer()
        {
            cudaEventDestroy(start_event_);
            cudaEventDestroy(stop_event_);
            auto n_trials = time_.size();
            auto mean_time = mean(time_);
            auto min_time = min(time_);
            auto median_time = median(time_);
            auto stddev = std_dev(time_);
            std::cout << func_name_ << ","
                      << n_trials << ","
                      << median_time << ","
                      << mean_time << std::endl;
        }

        inline void start() const
        {
            cudaEventRecord(start_event_, stream_);
        }

        inline void stop()
        {
            cudaEventRecord(stop_event_, stream_);
            cudaEventSynchronize(stop_event_);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start_event_, stop_event_);
            time_.push_back(milliseconds);
        }

    private:
        std::string func_name_;
        cudaStream_t stream_;
        cudaEvent_t start_event_{}, stop_event_{};
        std::vector<float> time_;

        static float mean(std::vector<float> const &v)
        {
            if (v.empty())
                return 0;

            auto const count = static_cast<float>(v.size());
            return std::reduce(v.begin(), v.end()) / count * 1000;
        }

        static float median(std::vector<float> v)
        {
            size_t size = v.size();

            if (size == 0)
                return 0;

            sort(v.begin(), v.end());
            return v[size / 2] * 1000;
        }

        static float min(std::vector<float> v)
        {
            size_t size = v.size();

            if (size == 0)
                return 0;

            sort(v.begin(), v.end());
            return v.front() * 1000;
        }

        static float max(std::vector<float> v)
        {
            size_t size = v.size();

            if (size == 0)
                return 0;

            sort(v.begin(), v.end());
            return v.back() * 1000;
        }

        static double std_dev(std::vector<float> const &v)
        {
            if (v.empty())
                return 0;

            auto const count = static_cast<float>(v.size());
            float mean = std::reduce(v.begin(), v.end()) / count;

            std::vector<double> diff(v.size());

            std::transform(v.begin(), v.end(), diff.begin(), [mean](double x)
                           { return x - mean; });
            double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            return std::sqrt(sq_sum / count);
        }
    };

}
