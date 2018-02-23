
#ifndef PROJECT_GPU_H
#define PROJECT_GPU_H

#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>

// funkce pro osetreni chyb
static void HandleError( cudaError_t error, const char *file, int line ) {
    if (error != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( error ), file, line );
        scanf(" ");
        exit( EXIT_FAILURE );
    }
}

#define CHECK_ERROR( error ) ( HandleError( error, __FILE__, __LINE__ ) )

#endif //PROJECT_GPU_H