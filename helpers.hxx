#pragma once

struct cuda_timer_t {
  float time;

  cuda_timer_t() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }

  ~cuda_timer_t() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { cudaEventRecord(start_); }
  
  long long stop() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&time, start_, stop_);
    return microseconds();
  }
  
  long long microseconds() { 
    return (long long)(1000 * time);
  }

 private:
  cudaEvent_t start_, stop_;
};