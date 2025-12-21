#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>

namespace forge_engine {

/**
 * Thread-safe circular buffer for zero-copy frame queuing.
 * Decouples network I/O thread from processing thread.
 */
template<typename T>
class CircularBuffer {
public:
    explicit CircularBuffer(size_t capacity)
        : buffer_(capacity)
        , capacity_(capacity)
        , size_(0)
        , read_pos_(0)
        , write_pos_(0)
    {}

    ~CircularBuffer() = default;

    // Non-copyable
    CircularBuffer(const CircularBuffer&) = delete;
    CircularBuffer& operator=(const CircularBuffer&) = delete;

    /**
     * Push data into buffer (non-blocking).
     * Returns true if successful, false if buffer is full.
     */
    bool try_push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size_ >= capacity_) {
            return false; // Buffer full
        }

        buffer_[write_pos_] = item;
        write_pos_ = (write_pos_ + 1) % capacity_;
        size_++;
        
        cv_.notify_one();
        return true;
    }

    /**
     * Pop data from buffer (blocking).
     * Blocks until data is available.
     */
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        cv_.wait(lock, [this] { return size_ > 0; });

        T item = buffer_[read_pos_];
        read_pos_ = (read_pos_ + 1) % capacity_;
        size_--;

        return item;
    }

    /**
     * Try to pop data from buffer (non-blocking).
     * Returns true if successful, false if buffer is empty.
     */
    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size_ == 0) {
            return false; // Buffer empty
        }

        item = buffer_[read_pos_];
        read_pos_ = (read_pos_ + 1) % capacity_;
        size_--;

        return true;
    }

    /**
     * Get current size (thread-safe).
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

    /**
     * Check if buffer is empty (thread-safe).
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ == 0;
    }

    /**
     * Check if buffer is full (thread-safe).
     */
    bool full() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ >= capacity_;
    }

    /**
     * Clear buffer (thread-safe).
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        size_ = 0;
        read_pos_ = 0;
        write_pos_ = 0;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<T> buffer_;
    size_t capacity_;
    size_t size_;
    size_t read_pos_;
    size_t write_pos_;
};

} // namespace forge_engine

