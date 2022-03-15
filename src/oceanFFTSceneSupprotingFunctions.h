#pragma once

#include "./oceanFFTScene.h"

extern HWND ghwnd;
extern FILE *gpFile;

float2 array_1[MESH_SIZE * MESH_SIZE]; // initial spectrum data
float2 array_2[MESH_SIZE * MESH_SIZE]; // empty array

extern "C" void cudaGenerateSpectrumKernel(float2 *d_h0, float2 *d_ht, unsigned int in_width, unsigned int out_width, unsigned int out_height, float animationDelayValue, float patchSize);
extern "C" void cudaUpdateHeightmapKernel(float *d_heightMap, float2 *d_ht, unsigned int width, unsigned int height);
extern "C" void cudaCalculateSlopeKernel(float *h, float2 *slopeOut, unsigned int width, unsigned int height);

// Phillips spectrum
// (Kx, Ky) - normalized wave vector
// Vdir - wind angle in radians
// V - wind speed
// waveScaleFactor - constant
float phillips(float Kx, float Ky, float Vdir, float V, float waveScaleFactor, float dir_depend)
{
    float k_squared = Kx * Kx + Ky * Ky;

    if (k_squared == 0.0f)
    {
        return (0.0f);
    }

    // largest possible wave from constant wind of velocity v
    float L = V * V / gravitationalConstant;

    float k_x = Kx / sqrtf(k_squared);
    float k_y = Ky / sqrtf(k_squared);
    float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

    float phillips = waveScaleFactor * expf(-1.0f / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;

    // filter out waves moving opposite to wind
    if (w_dot_k < 0.0f)
    {
        phillips *= dir_depend;
    }

    // damp out waves with very small length w << l
    // float w = L / 10000;
    // phillips *= expf(-k_squared * w * w);

    return (phillips);
}

float urand(void)
{
    return (rand() / (float)RAND_MAX);
}

// Generates Gaussian random number with mean 0 and standard deviation 1.
float gauss()
{
    float u1 = urand();
    float u2 = urand();

    if (u1 < 1e-6f)
    {
        u1 = 1e-6f;
    }

    return (sqrtf(-2 * logf(u1)) * cosf(2 * CUDART_PI_F * u2));
}

// Generate base heightfield in frequency space
void generate_h0(float2 *h0)
{
    for (unsigned int y = 0; y <= meshSizeLimit; y++)
    {
        for (unsigned int x = 0; x <= meshSizeLimit; x++)
        {
            float kx = (-(int)meshSizeLimit / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
            float ky = (-(int)meshSizeLimit / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

            float P = sqrtf(phillips(kx, ky, windDir, windSpeed, waveScaleFactor, dirDepend));

            if (kx == 0.0f && ky == 0.0f)
            {
                P = 0.0f;
            }

            // float Er = urand()*2.0f-1.0f;
            // float Ei = urand()*2.0f-1.0f;
            float Er = gauss();
            float Ei = gauss();

            float h0_re = Er * P * CUDART_SQRT_HALF_F;
            float h0_im = Ei * P * CUDART_SQRT_HALF_F;

            int i = y * spectrumW + x;
            h0[i].x = h0_re;
            h0[i].y = h0_im;
        }
    }
}

void createVBO(GLuint *vbo, int size)
{
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// create fixed vertex buffer to store mesh vertices
void createMeshPositionVBO(GLuint *id, int w, int h)
{
    createVBO(id, w * h * 4 * sizeof(float));

    glBindBuffer(GL_ARRAY_BUFFER, *id);
    float *position = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

    if (!position)
    {
        return;
    }

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float u = x / (float)(w - 1);
            float v = y / (float)(h - 1);

            *position++ = u * 2.0f - 1.0f;
            *position++ = 0.0f;

            *position++ = v * 2.0f - 1.0f;
            *position++ = 1.0f;
        }
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// create index buffer for rendering quad mesh
void createMeshIndexBuffer(GLuint *id, int w, int h)
{
    int size = ((w * 2) + 2) * (h - 1) * sizeof(GLuint);

    // create index buffer
    glGenBuffers(1, id);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

    // fill with indices for rendering mesh as triangle strips
    GLuint *indices =
        (GLuint *)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

    if (!indices)
    {
        return;
    }

    for (int y = 0; y < h - 1; y++)
    {
        for (int x = 0; x < w; x++)
        {
            *indices++ = y * w + x;
            *indices++ = (y + 1) * w + x;
        }

        // start new strip with degenerate triangle
        *indices++ = (y + 1) * w + (w - 1);
        *indices++ = (y + 1) * w;
    }

    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// cpu_conjugate() Definition
float2 cpu_conjugate(float2 arg)
{
    return make_float2(arg.x, -arg.y);
}

// cpu_complex_exp() Definition
float2 cpu_complex_exp(float arg)
{
    return make_float2(cosf(arg), sinf(arg));
}

// cpu_complex_add() Definition
float2 cpu_complex_add(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

// cpu_complex_subtract() Definition
float2 cpu_complex_subtract(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

// cpu_complex_mult() Definition
float2 cpu_complex_mult(float2 ab, float2 cd)
{
    return make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x);
}

float2 MultiplyComplex(float2 a, float2 b)
{
    return (make_float2(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y));
}

float4 ButterflyOperation(float2 a, float2 b, float2 twiddle)
{
    float2 twiddle_b = MultiplyComplex(twiddle, b);
    float2 t1 = cpu_complex_add(a, twiddle_b);
    float2 t2 = cpu_complex_subtract(a, twiddle_b);

    float4 result = make_float4(t1.x, t1.y, t2.x, t2.y);

    return (result);
}

// cpu_generateSpectrumKernel() Definition
void cpu_generateSpectrumKernel(float2 *h0, float *heightMap, float2 *slopeOut, unsigned int in_width, unsigned int out_width, unsigned int out_height, float t, float patchSize)
{
    // fprintf(gpFile, "-[%s @%d] begin::cpu_generateSpectrumKernel()]\n", __FILE__, __LINE__);

    // local variable declaration
    unsigned int x;
    unsigned int y;
    int row, column;
    float2 ht;

    for (y = 0; y < out_height; y++)
    {
        for (x = 0; x < out_width; x++)
        {
            unsigned int in_index = y * in_width + x;
            unsigned int in_mindex = (out_height - y) * in_width + (out_width - x);
            unsigned int out_index = y * out_width + x;

            // calculate wave vector
            float2 k;
            k.x = (-(int)out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
            k.y = (-(int)out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

            // calculate dispersion w(k)
            float k_len = sqrtf(k.x * k.x + k.y * k.y);
            float w = sqrtf(9.81f * k_len);

            if ((x < out_width) && (y < out_height))
            {
                float2 h0_k = h0[in_index];
                float2 h0_mk = h0[in_mindex];

                // output frequency-space complex values
                h_ht_1[out_index] = cpu_complex_add(cpu_complex_mult(h0_k, cpu_complex_exp(w * t)), cpu_complex_mult(cpu_conjugate(h0_mk), cpu_complex_exp(-w * t)));
                // fprintf(gpFile, "\tinfo>> %3.6f %3.6f\n", ht.x, ht.y);

                // float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;
                // heightMap[out_index] = ht.y * sign_correction;
                // fprintf(gpFile, "\tinfo>> %3.6f\n", heightMap[out_index]);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // // cpu fft waves
    // int array_size = meshSize * meshSize;

    // Rearrange(h_ht_1, array_size);
    // Perform(h_ht_1, array_size);
    // Scale(h_ht_1, array_size);

    // fprintf(gpFile, "\n===================================================\n");
    // for (int index = 0; index < meshSize * meshSize; index++)
    // {
    // 	fprintf(gpFile, "[%d] %3.6f %3.6f\n", index, h_ht_1[index].x, h_ht_1[index].y);
    // }
    // fprintf(gpFile, "\n\n");

    /* NOT WORKING
    if (memcpy(array_1, h_ht_1, meshSizeLimit * meshSizeLimit * sizeof(float2)) == NULL)
    {
        fprintf(gpFile, "\terror>> [%d]memcpy() failed...\n", __LINE__);
    }

    float2 *ptr_1 = array_1;
    float2 *ptr_2 = array_2;

    for (int p = 1; p < meshSizeLimit; p = p << 1)
    {
        for (int y = 0; y < 16; y++)
        {
            for (int x = 0; x < 16; x++)
            {
                int thread_count = meshSizeLimit / 2;
                int id = x;

                int in_idx = id & (p - 1);
                int out_idx = ((id - in_idx) << 1) + in_idx;

                float angle = -M_PI * (float(in_idx) / float(p));
                float2 twiddle = make_float2(cos(angle), sin(angle));

                float2 a = ptr_1[y * meshSizeLimit + x];                  // vec4 a = imageLoad(u_input, pixel_coord);
                float2 b = ptr_1[y * meshSizeLimit + (x + thread_count)]; // vec4 b = imageLoad(u_input, ivec2(pixel_coord.x + thread_count, pixel_coord.y));

                float4 result0 = ButterflyOperation(a, b, twiddle);
                // vec4 result1 = ButterflyOperation(a.zw, b.zw, twiddle);

                ptr_2[y * meshSizeLimit + out_idx].x = result0.x;
                ptr_2[y * meshSizeLimit + out_idx].y = result0.y;
                // ptr_2[y * meshSizeLimit + out_idx] = vec4(result0.xy, result1.xy);
                // ptr_2[y * meshSizeLimit + out_idx + p] = vec4(result0.zw, result1.zw);
            }
        }

        float2 *temp = ptr_1;
        ptr_1 = ptr_2;
        ptr_2 = temp;
    }

    if (memcpy(h_ht_1, ptr_2, meshSizeLimit * meshSizeLimit * sizeof(float2)) == NULL)
    {
        fprintf(gpFile, "\terror>> [%d]memcpy() failed...\n", __LINE__);
    }*/

    cuda_result = cudaMemcpy(d_ht, h_ht_1, meshSizeLimit * meshSizeLimit * sizeof(float2), cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d] before :: cuda_result::cudaMemcpy(cudaMemcpyHostToDevice) failed...\n", __LINE__);
    }

    cufftResult result;
    result = cufftExecC2C(fftPlan, d_ht, d_ht, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cufftExecC2C() failed...\n", __LINE__);
    }

    cuda_result = cudaMemcpy(h_ht_1, d_ht, meshSizeLimit * meshSizeLimit * sizeof(float2), cudaMemcpyDeviceToHost);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]after :: cuda_result::cudaMemcpy(cudaMemcpyDeviceToHost) failed...\n", __LINE__);
    }

    // for (int index = 0; index < meshSize * meshSize; index++)
    // {
    // 	fprintf(gpFile, "[%d] %3.6f %3.6f\n", index, h_ht_1[index].x, h_ht_1[index].y);
    // }
    // fprintf(gpFile, "\n===================================================\n");

    for (y = 0; y < out_height; y++)
    {
        for (x = 0; x < out_width; x++)
        {
            unsigned int out_index = y * out_width + x;

            float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;
            heightMap[out_index] = h_ht_1[out_index].x * sign_correction;
            // fprintf(gpFile, "\tinfo>> %3.6f\n", heightMap[out_index]);
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////
    for (y = 0; y < (out_height - 1); y++)
    {
        for (x = 0; x < (out_width - 1); x++)
        {
            unsigned int out_index = y * out_width + x;

            if ((x > 0) && (y > 0) && (x < out_width - 1) && (y < out_height - 1))
            {
                slopeOut[out_index].x = heightMap[out_index + 1] - heightMap[out_index - 1];
                slopeOut[out_index].y = heightMap[out_index + out_width] - heightMap[out_index - out_width];
            }
        }
    }

    // fprintf(gpFile, "\tinfo>> successful...\n");
    // exit(EXIT_SUCCESS);

    // fprintf(gpFile, "-[%s @%d] end::cpu_generateSpectrumKernel()]\n", __FILE__, __LINE__);
}

// runOceanCPU() Definition
void runOceanCPU(void)
{
    // local variable declaration
    // code
    // (float2 * h0, float *heightMap, float2 *slopeOut, unsigned int in_width, unsigned int out_width, unsigned int out_height, float t, float patchSize);
    cpu_generateSpectrumKernel(h_h0, cpu_heightMap, cpu_slopeOut, spectrumW, meshSizeLimit, meshSizeLimit, animationDelayValue, patchSize);

    // glBindVertexArray(vao_cpu_fft);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_cpu_height);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * meshSizeLimit * meshSizeLimit, cpu_heightMap, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(OUP_HEIGHT_IN, 1, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_HEIGHT_IN);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_cpu_slope);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * meshSizeLimit * meshSizeLimit, cpu_slopeOut, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(OUP_SLOPE_IN, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_SLOPE_IN);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // glBindVertexArray(0);
}

// runCudaGPU() Definition
void runCudaGPU(void)
{
    // local variable declaration
    size_t num_bytes;
    cufftResult result;

    // generate wave spectrum in frequency domain
    cudaGenerateSpectrumKernel(d_h0, d_ht, spectrumW, meshSizeLimit, meshSizeLimit, animationDelayValue, patchSize);

    // execute inverse FFT to convert to spatial domain
    result = cufftExecC2C(fftPlan, d_ht, d_ht, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cufftExecC2C() failed...\n", __LINE__);
    }

    // update heightmap values in vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, heightVertexBuffer);
    cuda_result = cudaGraphicsMapResources(1, &cuda_heightVB_resource, 0);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaGraphicsMapResources() failed...\n", __LINE__);
    }
    glVertexAttribPointer(OUP_HEIGHT_IN, 1, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_HEIGHT_IN);

    cuda_result = cudaGraphicsResourceGetMappedPointer((void **)&g_hptr, &num_bytes, cuda_heightVB_resource);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaGraphicsResourceGetMappedPointer() failed...\n", __LINE__);
    }

    cudaUpdateHeightmapKernel(g_hptr, d_ht, meshSizeLimit, meshSizeLimit);

    // calculate slope for shading
    glBindBuffer(GL_ARRAY_BUFFER, slopeVertexBuffer);
    cuda_result = cudaGraphicsMapResources(1, &cuda_slopeVB_resource, 0);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaGraphicsMapResources() failed...\n", __LINE__);
    }
    glVertexAttribPointer(OUP_SLOPE_IN, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_SLOPE_IN);

    cuda_result = cudaGraphicsResourceGetMappedPointer((void **)&g_sptr, &num_bytes, cuda_slopeVB_resource);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaGraphicsResourceGetMappedPointer() failed...\n", __LINE__);
    }

    cudaCalculateSlopeKernel(g_hptr, g_sptr, meshSizeLimit, meshSizeLimit);

    cuda_result = cudaGraphicsUnmapResources(1, &cuda_heightVB_resource, 0);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaGraphicsUnmapResources() failed...\n", __LINE__);
    }

    cuda_result = cudaGraphicsUnmapResources(1, &cuda_slopeVB_resource, 0);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaGraphicsUnmapResources() failed...\n", __LINE__);
    }
}
