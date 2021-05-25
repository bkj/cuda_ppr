#pragma GCC diagnostic ignored "-Wunused-result"

#include <cstdlib>  // EXIT_SUCCESS
#include "omp.h"
#include "nvToolsExt.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include <thrust/transform_scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <string.h>
#include <omp.h>
#include <queue>
#include <limits>
#include <vector>

#include "timer.hxx"

using namespace std;
using namespace std::chrono;

// --
// Global defs

typedef int Int;
typedef float Real;

// graph
Int n_nodes, n_edges;
Int* indptr;
Int* rindices;
Int* cindices;
Real* data;
Real* degrees;

Int n_seeds = 50;

// --
// IO

void load_data(std::string inpath) {
    FILE *ptr;
    ptr = fopen(inpath.c_str(), "rb");

    fread(&n_nodes,   sizeof(Int), 1, ptr);
    fread(&n_nodes,   sizeof(Int), 1, ptr);
    fread(&n_edges,   sizeof(Int), 1, ptr);

    indptr    = (Int*)  malloc(sizeof(Int)  * (n_nodes + 1)  );
    cindices  = (Int*)  malloc(sizeof(Int)  * n_edges        );
    rindices  = (Int*)  malloc(sizeof(Int)  * n_edges        );
    data      = (Real*) malloc(sizeof(Real) * n_edges        );
    degrees   = (Real*) malloc(sizeof(Real) * n_nodes        );
    
    fread(indptr,  sizeof(Int),   n_nodes + 1 , ptr);  // send directy to the memory since thats what the thing is.
    fread(cindices, sizeof(Int),  n_edges     , ptr);
    fread(data,    sizeof(Real),  n_edges     , ptr);
    
    for(Int src = 0; src < n_nodes; src++) {
        for(Int offset = indptr[src]; offset < indptr[src + 1]; offset++) {
            rindices[offset] = src;
        }
    }
    
    for(Int src = 0; src < n_nodes; src++) {
        degrees[src] = (Real)(indptr[src + 1] - indptr[src]);
    }
}

// --
// Run

long long cuda_ppr(Real* p, Int n_seeds, Int n_nodes, Int n_edges) {
    Real alpha   = 0.15;
    Real epsilon = 1e-6;

    // --
    // Copy graph from host to device
    
    Int* d_cindices;
    Int* d_rindices;
    Real* d_data;
    Real* d_degrees;

    cudaMalloc(&d_cindices,  n_edges       * sizeof(Int));
    cudaMalloc(&d_rindices,  n_edges       * sizeof(Int));
    cudaMalloc(&d_data,      n_edges       * sizeof(Real));
    cudaMalloc(&d_degrees,   n_nodes       * sizeof(Real));

    cudaMemcpy(d_cindices, cindices,  n_edges       * sizeof(Int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_rindices, rindices,  n_edges       * sizeof(Int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,     data,      n_edges       * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_degrees,  degrees,   n_nodes       * sizeof(Real), cudaMemcpyHostToDevice);
    
    // --
    // Setup problem on host
    
    Int k = n_nodes * n_seeds;
    
    char* frontier_in  = (char*)malloc(k * sizeof(char));
    char* frontier_out = (char*)malloc(k * sizeof(char));
    for(Int i = 0; i < k; i++) frontier_in[i]   = -1;
    for(Int i = 0; i < k; i++) frontier_out[i]  = -1;
    
    Real* r       = (Real*)malloc(k * sizeof(Real));
    Real* r_prime = (Real*)malloc(k * sizeof(Real));
    for(Int i = 0; i < k; i++) p[i]       = 0;
    for(Int i = 0; i < k; i++) r[i]       = 0;
    for(Int i = 0; i < k; i++) r_prime[i] = 0;
    
    for(Int seed = 0; seed < n_seeds; seed++) {
        Int idx = seed * n_nodes + seed;
        r[idx]           = 1;
        r_prime[idx]     = 1;
        frontier_in[idx] = 0;
    }
    
    int iteration = 0;
    
    // --
    // Copy data to device
    
    char* d_frontier_in;
    char* d_frontier_out;
    
    cudaMalloc(&d_frontier_in,  k * sizeof(char));
    cudaMalloc(&d_frontier_out, k * sizeof(char));

    cudaMemcpy(d_frontier_in,  frontier_in,  k * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier_out, frontier_out, k * sizeof(char), cudaMemcpyHostToDevice);
    
    Real* d_p;
    Real* d_r;    
    Real* d_r_prime;
    cudaMalloc(&d_p,       k * sizeof(Real));
    cudaMalloc(&d_r,       k * sizeof(Real));
    cudaMalloc(&d_r_prime, k * sizeof(Real));
    
    cudaMemcpy(d_p,       p,        k * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,       r,        k * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_prime, r_prime,  k * sizeof(Real), cudaMemcpyHostToDevice);

    // --
    // Run
    
    cudaDeviceSynchronize();
    auto t = high_resolution_clock::now();
    
    Real _2a1a = (2 * alpha) / (1 + alpha);
    Real _1a1a = ((1 - alpha) / (1 + alpha));
    
    // while(true) {
    while(iteration < 32) {
        
        int iteration1 = iteration + 1;
        
        auto node_op = [=] __device__(int const& offset) -> void {
            if(d_frontier_in[offset] != iteration) return;
            d_p[offset] += _2a1a * d_r[offset];
            d_r_prime[offset] = 0;
        };
        
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(n_seeds * n_nodes),
            node_op
        );

        auto edge_op = [=] __device__(int const& offset) -> void {
            Int s    = offset / n_edges;
            Int src  = d_rindices[offset % n_edges];
            Int _src = s * n_nodes + src;
            
            if(d_frontier_in[_src] != iteration) return;
            
            Int dst  = d_cindices[offset % n_edges];
            Int _dst = s * n_nodes + dst;
            
            Real update = _1a1a * d_r[_src] / d_degrees[src];
            Real oldval = atomicAdd(d_r_prime + _dst, update);
            Real newval = oldval + update;
            Real thresh = d_degrees[dst] * epsilon;
            
            if((oldval < thresh) && (newval >= thresh))
                d_frontier_out[_dst] = iteration1;
        };

        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(n_seeds * n_edges), // could probably use a zip iterator here
            edge_op
        );
        
        cudaMemcpy(d_r, d_r_prime, k * sizeof(Real), cudaMemcpyDeviceToDevice);
        
        char* tmp      = d_frontier_in;
        d_frontier_in  = d_frontier_out;
        d_frontier_out = tmp;
        
        iteration++;
    }
    
    cudaMemcpy(p, d_p, k * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    auto elapsed = high_resolution_clock::now() - t;
    return duration_cast<microseconds>(elapsed).count();
}


int main(int n_args, char** argument_array) {
    
    // ---------------- INPUT ------------------------------

    load_data(argument_array[1]);
        
    // ---------------- GPU IMPLEMENTATION -----------------
    
    Real* gpu_p = (Real*)malloc(n_seeds * n_nodes * sizeof(Real));
    auto gpu_time = cuda_ppr(gpu_p, n_seeds, n_nodes, n_edges);

    // ---------------- VALIDATION -------------------------
        
    for(Int seed = 0; seed < n_seeds; seed++) {
        for(Int i = 0; i < n_nodes; i++) {
            std::cout << gpu_p[seed * n_nodes + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cerr << "gpu_time=" << gpu_time << std::endl;
    
    return 0;
}
