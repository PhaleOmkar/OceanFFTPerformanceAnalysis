#pragma once

#include "../header/commongl.h"

// pointers to device object
float *g_hptr = NULL;
float2 *g_sptr = NULL;

// simulation parameters
const float gravitationalConstant = 9.81f; // gravitational constant
const float waveScaleFactor = 1e-7f;       // wave scale factor
const float patchSize = 100;               // patch size
float windSpeed = 100.0f;
float windDir = CUDART_PI_F / 3.0f;
float dirDepend = 0.07f;

cufftHandle fftPlan;
float2 *d_h0 = 0; // heightfield at time 0
float2 *h_h0 = 0;
float2 *d_ht = 0; // heightfield at time t
float2 *d_slope = 0;
cudaError_t cuda_result;
struct cudaGraphicsResource *cuda_positionVB_resource = NULL;
struct cudaGraphicsResource *cuda_heightVB_resource = NULL;
struct cudaGraphicsResource *cuda_slopeVB_resource = NULL; // handles OpenGL-CUDA exchange

GLfloat animationDelayValue = 0.0f;

GLfloat rotateAlongX = 60.0f;
GLfloat rotateAlongY = 0.0f;
GLfloat rotateAlongZ = 0.0f;

GLfloat translateX = 0.0f;
GLfloat translateY = 0.0f;
GLfloat translateZ = -3.5f;

GLfloat translateOceanAlongY = 0.6f;

GLuint vbo_cpu_height;
GLuint vbo_cpu_slope;
GLuint heightVertexBuffer;
GLuint slopeVertexBuffer;

float *cpu_heightMap = NULL;
float2 *cpu_slopeOut = NULL;
float2 h_ht_1[MESH_SIZE * MESH_SIZE];
