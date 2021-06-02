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
#include "nvToolsExt.h"

#include "helpers.hxx"

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

Int* d_cindices;
Int* d_rindices;
Real* d_data;
Real* d_degrees;

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

    cudaMalloc(&d_cindices,  n_edges       * sizeof(Int));
    cudaMalloc(&d_rindices,  n_edges       * sizeof(Int));
    cudaMalloc(&d_data,      n_edges       * sizeof(Real));
    cudaMalloc(&d_degrees,   n_nodes       * sizeof(Real));

    cudaMemcpy(d_cindices, cindices,  n_edges       * sizeof(Int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_rindices, rindices,  n_edges       * sizeof(Int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,     data,      n_edges       * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_degrees,  degrees,   n_nodes       * sizeof(Real), cudaMemcpyHostToDevice);
}

// --
// Run

void cuda_ppr(
    Real* d_p, 
    Int n_seeds, 
    Int n_nodes, 
    Int n_edges,
    Int* d_cindices,
    Int* d_rindices,
    Real* d_data,
    Real* d_degrees
) {
    Real alpha   = 0.15;
    Real epsilon = 1e-6;

    // --
    // Setup problem on host
    
    Int k = n_nodes * n_seeds;
    
    char* frontier_in  = (char*)malloc(k * sizeof(char));
    char* frontier_out = (char*)malloc(k * sizeof(char));
    for(Int i = 0; i < k; i++) frontier_in[i]   = -1;
    for(Int i = 0; i < k; i++) frontier_out[i]  = -1;
    
    Real* r       = (Real*)malloc(k * sizeof(Real));
    Real* r_prime = (Real*)malloc(k * sizeof(Real));
    // for(Int i = 0; i < k; i++) p[i]       = 0;
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
    
    Real* d_r;    
    Real* d_r_prime;
    cudaMalloc(&d_r,       k * sizeof(Real));
    cudaMalloc(&d_r_prime, k * sizeof(Real));
    
    cudaMemcpy(d_r,       r,        k * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_prime, r_prime,  k * sizeof(Real), cudaMemcpyHostToDevice);

    // --
    // Run
    
    cudaDeviceSynchronize();
    
    Real _2a1a = (2 * alpha) / (1 + alpha);
    Real _1a1a = ((1 - alpha) / (1 + alpha));
    
    // while(true) {
    while(iteration < 32) { // need to be careful about how many iterations we run
        
        int iteration1 = iteration + 1;
        
        thrust::for_each_n(
            thrust::device,
            thrust::make_counting_iterator<int>(0),
            n_seeds * n_nodes,
            [=] __device__(int const& noffset) {
                d_r[noffset] = d_r_prime[noffset];
                if(d_frontier_in[noffset] != iteration) return;
                d_p[noffset] += _2a1a * d_r[noffset];
                d_r_prime[noffset] = 0;
            }
        );

        auto edge_op = [=] __device__(int const& eoffset) {
            Int s    = eoffset / n_edges;
            Int src  = d_rindices[eoffset % n_edges];
            Int _src = s * n_nodes + src;
            
            if(d_frontier_in[_src] != iteration) return;
            
            Int dst  = d_cindices[eoffset % n_edges];
            Int _dst = s * n_nodes + dst;
            
            Real update = _1a1a * d_r[_src] / d_degrees[src];
            Real oldval = atomicAdd(d_r_prime + _dst, update);
            Real newval = oldval + update;
            Real thresh = d_degrees[dst] * epsilon;
            
            if((oldval < thresh) && (newval >= thresh))
                d_frontier_out[_dst] = iteration1;
        };

        thrust::for_each_n(
            thrust::device,
            thrust::make_counting_iterator<int>(0),
            n_seeds * n_edges,
            edge_op
        );
        
        char* tmp      = d_frontier_in;
        d_frontier_in  = d_frontier_out;
        d_frontier_out = tmp;
        
        iteration++;
    }
}


int main(int n_args, char** argument_array) {
    
    // ---------------- INPUT ------------------------------

    load_data(argument_array[1]);
        
    // ---------------- GPU IMPLEMENTATION -----------------
    
    Real* d_p; cudaMalloc(&d_p, n_seeds * n_nodes * sizeof(Real));
    cudaDeviceSynchronize();
    
    cuda_timer_t timer;
    timer.start();
    
    nvtxRangePushA("cuda_ppr");
    cuda_ppr(
        d_p, 
        n_seeds, 
        n_nodes, 
        n_edges,
        d_cindices,
        d_rindices,
        d_data,
        d_degrees
    );
    cudaDeviceSynchronize();
    nvtxRangePop();
    
    long long gpu_time = timer.stop();
    
    Real* p = (Real*)malloc(n_seeds * n_nodes * sizeof(Real));
    cudaMemcpy(p, d_p, n_seeds * n_nodes * sizeof(Real), cudaMemcpyDeviceToHost);

    // ---------------- VALIDATION -------------------------
        
    for(Int i = 0; i < n_nodes; i++) {
        for(Int seed = 0; seed < n_seeds; seed++) {
            std::cout << std::setprecision(10) << p[seed * n_nodes + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cerr << "gpu_time=" << gpu_time << std::endl;
    
    return 0;
}
