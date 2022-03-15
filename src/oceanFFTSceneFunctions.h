#pragma once

#include "./oceanFFTScene.h"
#include "./oceanFFTSceneSupprotingFunctions.h"

#define OCEAN_WAVES_SPEED 0.0125f
#define CUDART_PI_F 3.141592654f
#define CUDART_SQRT_HALF_F 0.707106781f

extern HWND ghwnd;
extern FILE *gpFile;

unsigned int currectRotationAxis = ALONG_NEGATIVE_Y_AXIS;
GLuint vertexShaderObjectOceanFFT;
GLuint fragmentShaderObjectFFT;
GLuint shaderProgramObjectOceanFFT;

GLuint u_modelMatrixUniform;
GLuint u_viewMatrixUniform;
GLuint u_projectionMatrixUniform;

GLuint heightScaleUniform;
GLuint chopinessUniform;
GLuint sizeUniform;
GLuint deepColorUniform;
GLuint shallowColorUniform;
GLuint skyColorUniform;
GLuint lightDirUniform;
GLuint indexBuffer;

// OpenGL vertex buffers
GLuint positionVertexBuffer;

GLuint cpu_positionVertexBuffer;
GLuint vao_cpu_fft;
GLuint indexBufferCPU;

GLuint vao_cuda_fft;
GLuint vbo_position;
GLuint mvpMatrixUniform;

void initializeOcean(void)
{
    fprintf(gpFile, "\n+[%s @%d] begin::initializeOcean()]\n", __FILE__, __LINE__);

    // Local Function Declaration
    void generate_h0(float2 * h0);
    void createVBO(GLuint * vbo, int size);

    void createMeshPositionVBO(GLuint * id, int w, int h);
    void createMeshIndexBuffer(GLuint * id, int w, int h);

    void runOceanCPU(void);
    void runCudaGPU(void);

    // Local Variable Declaration
    int index;
    GLint iInfoLogLength = 0;
    GLint iShaderCompiledStatus = 0;
    GLint iShaderLinkerStatus = 0;
    GLchar *szInfoLogBuffer = NULL;

    // Vertex Shader - Creating Shader
    vertexShaderObjectOceanFFT = glCreateShader(GL_VERTEX_SHADER);
    const GLchar *pglcVertexShaderSourceCode =
        "#version 430 core																														\n"
        "																																		\n"
        "in vec4 vPosition;																														\n"
        "in float height_in;																													\n"
        "in vec2 slope_in;																														\n"
        "																																		\n"
        "uniform mat4 u_viewMatrix;																												\n"
        "uniform mat4 u_modelMatrix;																											\n"
        "uniform mat4 u_projectionMatrix;																										\n"
        "																																		\n"
        "uniform float heightScale;																												\n"
        "uniform float chopiness;																												\n"
        "																																		\n"
        "uniform vec2 size;																														\n"
        "																																		\n"
        "out vec3 eyeSpacePos;																													\n"
        "out vec3 eyeSpaceNormal;																												\n"
        "out vec3 worldSpaceNormal;																												\n"
        "																																		\n"
        "void main(void)																														\n"
        "{																																		\n"
        "	float height = height_in;																											\n"
        "	vec2 slope = slope_in;																												\n"
        "																																		\n"
        "	vec3 normal = normalize(cross(vec3(0.0, slope.y * heightScale, 2.0 / size.x), vec3(2.0 / size.y, slope.x * heightScale, 0.0)));		\n"
        "	worldSpaceNormal = normal;																											\n"
        "																																		\n"
        "	vec4 pos = vec4(vPosition.x, height * heightScale, vPosition.z, 1.0);																\n"
        "	eyeSpacePos = (u_viewMatrix * u_modelMatrix * pos).xyz;																				\n"
        "																																		\n"
        "	eyeSpaceNormal = (mat3(transpose(inverse(u_viewMatrix * u_modelMatrix))) * normal).xyz;												\n"
        "																																		\n"
        "	gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * pos;																\n"
        "}																																		\n";

    glShaderSource(vertexShaderObjectOceanFFT, 1, (const GLchar **)&pglcVertexShaderSourceCode, NULL);

    // Compiling Shader
    glCompileShader(vertexShaderObjectOceanFFT);

    // Error Checking
    glGetShaderiv(vertexShaderObjectOceanFFT, GL_COMPILE_STATUS, &iShaderCompiledStatus);
    if (iShaderCompiledStatus == GL_FALSE)
    {
        glGetShaderiv(vertexShaderObjectOceanFFT, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(vertexShaderObjectOceanFFT, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Vertex Shader Compilation Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    iInfoLogLength = 0;
    iShaderCompiledStatus = 0;
    iShaderLinkerStatus = 0;
    szInfoLogBuffer = NULL;

    // Fragment Shader - Creating Shader
    fragmentShaderObjectFFT = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar *pglcFragmentShaderSourceCode =
        "#version 430 core															                                                            \n"
        "																			                                                            \n"
        "in vec3 eyeSpacePos;														                                                            \n"
        "in vec3 worldSpaceNormal;													                                                            \n"
        "in vec3 eyeSpaceNormal;													                                                            \n"
        "																			                                                            \n"
        "uniform vec4 deepColor;													                                                            \n"
        "uniform vec4 shallowColor;													                                                            \n"
        "uniform vec4 skyColor;														                                                            \n"
        "uniform vec3 lightDir;														                                                            \n"
        "																			                                                            \n"
        "out vec4 fragmentColor;													                                                            \n"
        "																			                                                            \n"
        "void main(void)															                                                            \n"
        "{																			                                                            \n"
        "	vec3 eyeVector = normalize(eyeSpacePos);								                                                            \n"
        "	vec3 eyeSpaceNormalVector = normalize(eyeSpaceNormal);					                                                            \n"
        "	vec3 worldSpaceNormalVector = normalize(worldSpaceNormal);				                                                            \n"
        "																			                                                            \n"
        "	float facing = max(0.0, dot(eyeSpaceNormalVector, -eyeVector));			                                                            \n"
        "	float fresnel = pow(1.0 - facing, 5.0);									                                                            \n"
        "	float diffuse = max(0.0, dot(worldSpaceNormalVector, lightDir));		                                                            \n"
        "																			                                                            \n"
        "	vec4 waterColor = deepColor;											                                                            \n"
        "																			                                                            \n"
        "	fragmentColor = waterColor * diffuse + skyColor * fresnel;				                                                            \n"
        "}																			                                                            \n";

    glShaderSource(fragmentShaderObjectFFT, 1, (const GLchar **)&pglcFragmentShaderSourceCode, NULL);

    // Compiling Shader
    glCompileShader(fragmentShaderObjectFFT);

    // Error Checking
    glGetShaderiv(fragmentShaderObjectFFT, GL_COMPILE_STATUS, &iShaderCompiledStatus);
    if (iShaderCompiledStatus == GL_FALSE)
    {
        glGetShaderiv(fragmentShaderObjectFFT, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(fragmentShaderObjectFFT, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Fragment Shader Compilation Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    // Shader Program - Create Shader Program
    shaderProgramObjectOceanFFT = glCreateProgram();

    glAttachShader(shaderProgramObjectOceanFFT, vertexShaderObjectOceanFFT); // Attach Vertex Shader To Shader Program
    glAttachShader(shaderProgramObjectOceanFFT, fragmentShaderObjectFFT);    // Attach Fragment Shader To Shader Program

    // Bind Vertex Shader Position Attribute
    glBindAttribLocation(shaderProgramObjectOceanFFT, OUP_ATTRIBUTE_POSITION, "vPosition");
    glBindAttribLocation(shaderProgramObjectOceanFFT, OUP_HEIGHT_IN, "height_in");
    glBindAttribLocation(shaderProgramObjectOceanFFT, OUP_SLOPE_IN, "slope_in");

    // Link Shader Program
    glLinkProgram(shaderProgramObjectOceanFFT);

    // Error Checking
    glGetProgramiv(shaderProgramObjectOceanFFT, GL_LINK_STATUS, &iShaderLinkerStatus);
    if (iShaderLinkerStatus == GL_FALSE)
    {
        glGetShaderiv(shaderProgramObjectOceanFFT, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(shaderProgramObjectOceanFFT, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Shader Program Linking Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    // Get Uniform Location
    // vs
    u_viewMatrixUniform = glGetUniformLocation(shaderProgramObjectOceanFFT, "u_viewMatrix");
    u_modelMatrixUniform = glGetUniformLocation(shaderProgramObjectOceanFFT, "u_modelMatrix");
    u_projectionMatrixUniform = glGetUniformLocation(shaderProgramObjectOceanFFT, "u_projectionMatrix");
    heightScaleUniform = glGetUniformLocation(shaderProgramObjectOceanFFT, "heightScale");
    chopinessUniform = glGetUniformLocation(shaderProgramObjectOceanFFT, "chopiness");
    sizeUniform = glGetUniformLocation(shaderProgramObjectOceanFFT, "size");

    // fs
    deepColorUniform = glGetUniformLocation(shaderProgramObjectOceanFFT, "deepColor");
    shallowColorUniform = glGetUniformLocation(shaderProgramObjectOceanFFT, "shallowColor");
    skyColorUniform = glGetUniformLocation(shaderProgramObjectOceanFFT, "skyColor");
    lightDirUniform = glGetUniformLocation(shaderProgramObjectOceanFFT, "lightDir");

    // create FFT plan
    cufftResult result;
    result = cufftPlan2d(&fftPlan, meshSizeLimit, meshSizeLimit, CUFFT_C2C);
    if (result != CUFFT_SUCCESS)
    {
        fprintf(gpFile, "\terror>> [%d]result::cufftPlan2d() failed...\n", __LINE__);
    }

    // allocate memory
    int spectrumSize = SPECTRUM_SIZE_W * SPECTRUM_SIZE_H * sizeof(float2);
    cuda_result = cudaMalloc((void **)&d_h0, spectrumSize);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaMalloc() failed...\n", __LINE__);
    }

    h_h0 = (float2 *)malloc(spectrumSize);
    generate_h0(h_h0);
    if (h_h0 == NULL)
    {
        fprintf(gpFile, "\terror>> [%d] malloc(h_h0) is failed...\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    cuda_result = cudaMemcpy(d_h0, h_h0, spectrumSize, cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaMemcpy() failed...\n", __LINE__);
    }

    int outputSize = MESH_SIZE * MESH_SIZE * sizeof(float2);
    cuda_result = cudaMalloc((void **)&d_ht, outputSize);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaMalloc() failed...\n", __LINE__);
    }

    cuda_result = cudaMalloc((void **)&d_slope, outputSize);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaMalloc() failed...\n", __LINE__);
    }

    //******************************************************************************************************************
    glGenVertexArrays(1, &vao_cuda_fft);
    glBindVertexArray(vao_cuda_fft);

    // create vertex buffer for mesh
    createMeshPositionVBO(&positionVertexBuffer, MESH_SIZE, MESH_SIZE);
    glBindBuffer(GL_ARRAY_BUFFER, positionVertexBuffer);
    glVertexAttribPointer(OUP_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_ATTRIBUTE_POSITION);

    // create vertex buffers and register with CUDA
    createVBO(&heightVertexBuffer, MESH_SIZE * MESH_SIZE * sizeof(float));
    glVertexAttribPointer(OUP_HEIGHT_IN, 1, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_HEIGHT_IN);
    cuda_result = cudaGraphicsGLRegisterBuffer(&cuda_heightVB_resource, heightVertexBuffer, cudaGraphicsMapFlagsWriteDiscard);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaGraphicsGLRegisterBuffer() failed...\n", __LINE__);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    createVBO(&slopeVertexBuffer, outputSize);
    glVertexAttribPointer(OUP_SLOPE_IN, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_SLOPE_IN);
    cuda_result = cudaGraphicsGLRegisterBuffer(&cuda_slopeVB_resource, slopeVertexBuffer, cudaGraphicsMapFlagsWriteDiscard);
    if (cuda_result != cudaSuccess)
    {
        fprintf(gpFile, "\terror>> [%d]cuda_result::cudaGraphicsGLRegisterBuffer() failed...\n", __LINE__);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // create index buffer for mesh
    createMeshIndexBuffer(&indexBuffer, MESH_SIZE, MESH_SIZE);

    glBindVertexArray(0);
    //********************************************************************************************************************

    cpu_heightMap = (float *)malloc(MESH_SIZE * MESH_SIZE * sizeof(float));
    if (cpu_heightMap == NULL)
    {
        fprintf(gpFile, "\terror>> [%d] malloc(cpu_heightMap) is failed...\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    cpu_slopeOut = (float2 *)malloc(MESH_SIZE * MESH_SIZE * sizeof(float2));
    if (cpu_slopeOut == NULL)
    {
        fprintf(gpFile, "\terror>> [%d] malloc(cpu_slopeOut) is failed...\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    //******************************************************************************************************************
    glGenVertexArrays(1, &vao_cpu_fft);
    glBindVertexArray(vao_cpu_fft);

    // create vertex buffer for mesh
    createMeshPositionVBO(&cpu_positionVertexBuffer, MESH_SIZE, MESH_SIZE);
    glBindBuffer(GL_ARRAY_BUFFER, cpu_positionVertexBuffer);
    glVertexAttribPointer(OUP_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_ATTRIBUTE_POSITION);

    // create vertex buffers and register with cpu
    createVBO(&vbo_cpu_height, MESH_SIZE * MESH_SIZE * sizeof(float));
    glVertexAttribPointer(OUP_HEIGHT_IN, 1, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_HEIGHT_IN);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    createVBO(&vbo_cpu_slope, outputSize);
    glVertexAttribPointer(OUP_SLOPE_IN, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_SLOPE_IN);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // // create index buffer for mesh
    createMeshIndexBuffer(&indexBufferCPU, MESH_SIZE, MESH_SIZE);

    glBindVertexArray(0);
    //******************************************************************************************************************

    runCudaGPU();
    runOceanCPU();

    fprintf(gpFile, "+[%s @%d] end::initializeOcean()]\n", __FILE__, __LINE__);
}

void displayOcean(void)
{
    // Local Function Declaration
    void runOceanCPU(void);
    void runCudaGPU(void);
    void renderTextOnScreen(std::string renderTextOnScreenMessage, float xPosition, float yPosition, float zPosition, float fontSize, float redColor, float greenColor, float blueColor);

    // Code
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLineWidth(5.0f);

    if (sceneCounter == OCEANFFT_SCENE)
    {
        currectRotationAxis = ALONG_NEGATIVE_Y_AXIS;

        static GLfloat colorValuesForData[3] = {0.0f};
        if (gbOnRunGPU == true)
        {
            colorValuesForData[0] = 0.0f;
            colorValuesForData[1] = 1.0f;
            colorValuesForData[2] = 0.0f;
        }
        else
        {
            colorValuesForData[0] = 1.0f;
            colorValuesForData[1] = 0.0f;
            colorValuesForData[2] = 0.0f;
        }

        //////////////////////////////////////////////////////////////////// TOP ///////////////////////////////////////////////////////////////////
        sprintf(fontRendering.messageString, "%s", gbOnRunGPU == true ? "NVIDIA GeForce GTX 1650" : "INTEL Core i7 10750H");
        renderTextOnScreen(fontRendering.messageString, -99.0f, 50.0f, -140.0f, 0.1f, colorValuesForData[0], colorValuesForData[1], colorValuesForData[2]);

        sprintf(fontRendering.messageString, "Device : %s", gbOnRunGPU == true ? "NVIDIA (GPU)" : "INTEL (CPU)");
        renderTextOnScreen(fontRendering.messageString, 25.0f, 50.0f, -140.0f, 0.125f, colorValuesForData[0], colorValuesForData[1], colorValuesForData[2]);

        static float framesPerSecond = 0.0f;
        static int fps;
        static float lastTime = 0.0f;
        float currentTime = GetTickCount() * 0.001f;
        ++framesPerSecond;

        if (currentTime - lastTime > 1.0f)
        {
            lastTime = currentTime;
            fps = (int)framesPerSecond;
            framesPerSecond = 0;
        }

        sprintf(fontRendering.messageString, "   FPS : %d", (int)fps);
        renderTextOnScreen(fontRendering.messageString, 30.5f, 44.0f, -140.0f, 0.1f, colorValuesForData[0], colorValuesForData[1], colorValuesForData[2]);

        ////////////////////////////////////////////////////////////////// BOTTOM //////////////////////////////////////////////////////////////////
        sprintf(fontRendering.messageString, "Polygon Mode : %s", gbWireFrame == true ? "LINE" : "FILL");
        renderTextOnScreen(fontRendering.messageString, -99.0f, -37.0f, -140.0f, 0.08f, colorValuesForData[0], colorValuesForData[1], colorValuesForData[2]);

        sprintf(fontRendering.messageString, "Mesh Size    : %d x %d x 4", (int)meshSizeLimit, (int)meshSizeLimit);
        renderTextOnScreen(fontRendering.messageString, -99.0f, -43.0f, -140.0f, 0.08f, colorValuesForData[0], colorValuesForData[1], colorValuesForData[2]);

        sprintf(fontRendering.messageString, "No. Of. Vertices  : %d", (meshSizeLimit * meshSizeLimit));
        renderTextOnScreen(fontRendering.messageString, -99.0f, -49.0f, -140.0f, 0.08f, colorValuesForData[0], colorValuesForData[1], colorValuesForData[2]);

        sprintf(fontRendering.messageString, "No. Of. Triangles : %d", ((meshSizeLimit - 1) * (meshSizeLimit - 1) * 2));
        renderTextOnScreen(fontRendering.messageString, -99.0f, -55.0f, -140.0f, 0.08f, colorValuesForData[0], colorValuesForData[1], colorValuesForData[2]);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        glUseProgram(shaderProgramObjectOceanFFT);

        // set view matrix
        mat4 viewMatrix = mat4::identity();
        mat4 modelMatrix = mat4::identity();

        modelMatrix = modelMatrix * vmath::translate(0.0f, 0.3f, -3.5f);
        modelMatrix = modelMatrix * vmath::rotate(rotateAlongX, 1.0f, 0.0f, 0.0f);
        modelMatrix = modelMatrix * vmath::rotate(rotateAlongY, 0.0f, 1.0f, 0.0f);
        // modelMatrix = modelMatrix * vmath::rotate(rotateAlongZ, 0.0f, 0.0f, 1.0f);

        glUniformMatrix4fv(u_viewMatrixUniform, 1, GL_FALSE, viewMatrix);
        glUniformMatrix4fv(u_modelMatrixUniform, 1, GL_FALSE, modelMatrix);
        glUniformMatrix4fv(u_projectionMatrixUniform, 1, GL_FALSE, perspectiveProjectionMatrix);

        glUniform1f(heightScaleUniform, 0.25f);
        glUniform1f(chopinessUniform, 1.0f); // original call
        glUniform2f(sizeUniform, (float)meshSizeLimit, (float)meshSizeLimit);
        glUniform4f(deepColorUniform, 0.0f, 0.1f, 0.5f, 1.0f);
        glUniform4f(shallowColorUniform, 0.1f, 0.3f, 0.3f, 1.0f);
        glUniform4f(skyColorUniform, 1.0f, 1.0f, 1.0f, 1.0f);
        glUniform3f(lightDirUniform, -0.45f, 2.1f, -3.5f); // glUniform3f(lightDirUniform, -0.84f, 1.5f, -0.8f);

        if (meshSizeLimit > 64)
        {
            gbWireFrame = false;
        }
        else
        {
            gbWireFrame = true;
        }

        if (gbOnRunGPU == true)
        {
            glBindVertexArray(vao_cuda_fft);
            glBindBuffer(GL_ARRAY_BUFFER, positionVertexBuffer);
            glVertexAttribPointer(OUP_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(OUP_ATTRIBUTE_POSITION);

            runCudaGPU();

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
            glPolygonMode(GL_FRONT_AND_BACK, gbWireFrame ? GL_LINE : GL_FILL);
            glDrawElements(GL_TRIANGLE_STRIP, ((meshSizeLimit * 2) + 2) * (meshSizeLimit - 1), GL_UNSIGNED_INT, 0);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
        }
        else
        {
            glBindVertexArray(vao_cpu_fft);
            glBindBuffer(GL_ARRAY_BUFFER, cpu_positionVertexBuffer);
            glEnableVertexAttribArray(OUP_ATTRIBUTE_POSITION);
            glVertexAttribPointer(OUP_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glBindBuffer(GL_ARRAY_BUFFER, vbo_cpu_height);
            glEnableVertexAttribArray(OUP_HEIGHT_IN);
            glVertexAttribPointer(OUP_HEIGHT_IN, 1, GL_FLOAT, GL_FALSE, 0, NULL);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glBindBuffer(GL_ARRAY_BUFFER, vbo_cpu_slope);
            glEnableVertexAttribArray(OUP_SLOPE_IN);
            glVertexAttribPointer(OUP_SLOPE_IN, 2, GL_FLOAT, GL_FALSE, 0, NULL);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            runOceanCPU();

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferCPU);
            glPolygonMode(GL_FRONT_AND_BACK, gbWireFrame ? GL_LINE : GL_FILL);
            glDrawElements(GL_TRIANGLE_STRIP, ((meshSizeLimit * 2) + 2) * (meshSizeLimit - 1), GL_UNSIGNED_INT, 0);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
        }

        glUseProgram(0);
    }

    else if (sceneCounter == OPENGL_CUDA_SCENE)
    {
        glUseProgram(shaderProgramObjectOceanFFT);

        currectRotationAxis = -1;

        mat4 viewMatrixMixedScene = mat4::identity();
        mat4 modelMatrixMixedScene = mat4::identity();

        viewMatrixMixedScene = viewMatrixMixedScene * vmath::translate(0.0f, translateOceanAlongY, 0.0f);
        modelMatrixMixedScene = modelMatrixMixedScene * vmath::translate(0.0f, -0.40f, -1.3f);
        modelMatrixMixedScene = modelMatrixMixedScene * vmath::rotate(-14.0f, 1.0f, 0.0f, 0.0f);
        modelMatrixMixedScene = modelMatrixMixedScene * vmath::rotate(45.0f, 0.0f, 1.0f, 0.0f);

        glUniformMatrix4fv(u_viewMatrixUniform, 1, GL_FALSE, viewMatrixMixedScene);
        glUniformMatrix4fv(u_modelMatrixUniform, 1, GL_FALSE, modelMatrixMixedScene);
        glUniformMatrix4fv(u_projectionMatrixUniform, 1, GL_FALSE, perspectiveProjectionMatrix);

        glUniform1f(heightScaleUniform, 0.125f);
        glUniform1f(chopinessUniform, 1.0f); // original call
        glUniform2f(sizeUniform, (float)meshSizeLimit, (float)meshSizeLimit);
        glUniform4f(deepColorUniform, 0.0f, 0.1f, 0.4f, 1.0f);
        glUniform4f(shallowColorUniform, 0.1f, 0.1f, 0.1f, 1.0f);
        glUniform4f(skyColorUniform, 0.1f, 0.1f, 0.1f, 1.0f);
        glUniform3f(lightDirUniform, -0.7f, 1.5f, -2.0f);

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glBindVertexArray(vao_cuda_fft);
        glBindBuffer(GL_ARRAY_BUFFER, positionVertexBuffer);
        glVertexAttribPointer(OUP_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(OUP_ATTRIBUTE_POSITION);

        runCudaGPU();

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
        glDrawElements(GL_TRIANGLE_STRIP, ((meshSizeLimit * 2) + 2) * (meshSizeLimit - 1), GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        glUseProgram(0);
    }

    glLineWidth(1.0f);
}

void updateOcean(void)
{
    // Code
    animationDelayValue += OCEAN_WAVES_SPEED;

    if (currectRotationAxis == ALONG_POSITIVE_X_AXIS)
    {
        rotateAlongX += 0.1f;
    }
    else if (currectRotationAxis == ALONG_POSITIVE_Y_AXIS)
    {
        rotateAlongY += 0.1f;
    }

    if (currectRotationAxis == ALONG_NEGATIVE_X_AXIS)
    {
        rotateAlongX -= 0.1f;
    }
    else if (currectRotationAxis == ALONG_NEGATIVE_Y_AXIS)
    {
        rotateAlongY -= 0.1f;
    }

    if (sceneCounter == OPENGL_CUDA_SCENE)
    {
        if (translateOceanAlongY >= 0.01f)
        {
            translateOceanAlongY = translateOceanAlongY - 0.00025f;
        }
    }

    if (gbNeedToUpdate == true)
    {
        gbNeedToUpdate = false;

        ///////////////////////////////////////////////////////////////////////////////////
        // CPU //
        ///////////////////////////////////////////////////////////////////////////////////
        glBindBuffer(GL_ARRAY_BUFFER, cpu_positionVertexBuffer);

        float *cpu_position = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

        if (!cpu_position)
        {
            return;
        }

        for (int y = 0; y < meshSizeLimit; y++)
        {
            for (int x = 0; x < meshSizeLimit; x++)
            {
                float u = x / (float)(meshSizeLimit - 1);
                float v = y / (float)(meshSizeLimit - 1);

                *cpu_position++ = u * 2.0f - 1.0f;
                *cpu_position++ = 0.0f;

                *cpu_position++ = v * 2.0f - 1.0f;
                *cpu_position++ = 1.0f;
            }
        }

        glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // ------------------------------------------------------------------------------ //

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferCPU);

        // fill with indices for rendering mesh as triangle strips
        GLuint *indicesCPU = (GLuint *)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

        if (!indicesCPU)
        {
            return;
        }

        for (int y = 0; y < meshSizeLimit - 1; y++)
        {
            for (int x = 0; x < meshSizeLimit; x++)
            {
                *indicesCPU++ = y * meshSizeLimit + x;
                *indicesCPU++ = (y + 1) * meshSizeLimit + x;
            }

            // start new strip with degenerate triangle
            *indicesCPU++ = (y + 1) * meshSizeLimit + (meshSizeLimit - 1);
            *indicesCPU++ = (y + 1) * meshSizeLimit;
        }

        glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        ///////////////////////////////////////////////////////////////////////////////////
        // GPU //
        ///////////////////////////////////////////////////////////////////////////////////
        glBindBuffer(GL_ARRAY_BUFFER, positionVertexBuffer);
        float *position = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

        if (!position)
        {
            return;
        }

        for (int y = 0; y < meshSizeLimit; y++)
        {
            for (int x = 0; x < meshSizeLimit; x++)
            {
                float u = x / (float)(meshSizeLimit - 1);
                float v = y / (float)(meshSizeLimit - 1);

                *position++ = u * 2.0f - 1.0f;
                *position++ = 0.0f;

                *position++ = v * 2.0f - 1.0f;
                *position++ = 1.0f;
            }
        }

        glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // ------------------------------------------------------------------------------ //

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);

        // fill with indices for rendering mesh as triangle strips
        GLuint *indices = (GLuint *)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

        if (!indices)
        {
            return;
        }

        for (int y = 0; y < meshSizeLimit - 1; y++)
        {
            for (int x = 0; x < meshSizeLimit; x++)
            {
                *indices++ = y * meshSizeLimit + x;
                *indices++ = (y + 1) * meshSizeLimit + x;
            }

            // start new strip with degenerate triangle
            *indices++ = (y + 1) * meshSizeLimit + (meshSizeLimit - 1);
            *indices++ = (y + 1) * meshSizeLimit;
        }

        glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        ///////////////////////////////////////////////////////////////////////////////////

        generate_h0(h_h0);
        if (h_h0 == NULL)
        {
            fprintf(gpFile, "\terror>> [%d] malloc(h_h0) is failed...\n", __LINE__);
            exit(EXIT_FAILURE);
        }

        cuda_result = cudaMemcpy(d_h0, h_h0, (spectrumW * spectrumH * sizeof(float2)), cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess)
        {
            fprintf(gpFile, "\terror>> [%d]cuda_result::cudaMemcpy() failed...\n", __LINE__);
        }

        ///////////////////////////////////////////////////////////////////////////////////

        static cufftResult result;
        if (fftPlan)
        {
            cufftDestroy(fftPlan);
            fftPlan = NULL;

            result = cufftPlan2d(&fftPlan, meshSizeLimit, meshSizeLimit, CUFFT_C2C);
            if (result != CUFFT_SUCCESS)
            {
                fprintf(gpFile, "\terror>> [%d]result::cufftPlan2d() failed...\n", __LINE__);
            }
        }

        ///////////////////////////////////////////////////////////////////////////////////
    }
}

void uninitializeOcean(void)
{
    fprintf(gpFile, "\n-[%s @%d] begin::uninitializeOcean()]\n", __FILE__, __LINE__);

    // Local Variable Declaration
    GLsizei ShaderCount;
    GLsizei index;
    GLuint *pShaders = NULL;

    // Code
    if (vao_cuda_fft)
    {
        glDeleteBuffers(1, &vao_cuda_fft);
        vao_cuda_fft = 0;

        fprintf(gpFile, "\tinfo>> vao_cuda_fft deleted successfully...\n");
    }

    if (vbo_position)
    {
        glDeleteBuffers(1, &vbo_position);
        vbo_position = 0;

        fprintf(gpFile, "\tinfo>> vbo_position deleted successfully...\n");
    }

    if (vao_cpu_fft)
    {
        glDeleteBuffers(1, &vao_cpu_fft);
        vao_cpu_fft = 0;

        fprintf(gpFile, "\tinfo>> vao_cpu_fft deleted successfully...\n");
    }

    if (cpu_positionVertexBuffer)
    {
        glDeleteBuffers(1, &cpu_positionVertexBuffer);
        cpu_positionVertexBuffer = 0;

        fprintf(gpFile, "\tinfo>> cpu_positionVertexBuffer deleted successfully...\n");
    }

    if (vbo_cpu_height)
    {
        glDeleteBuffers(1, &vbo_cpu_height);
        vbo_cpu_height = 0;

        fprintf(gpFile, "\tinfo>> vbo_cpu_height deleted successfully...\n");
    }

    if (vbo_cpu_slope)
    {
        glDeleteBuffers(1, &vbo_cpu_slope);
        vbo_cpu_slope = 0;

        fprintf(gpFile, "\tinfo>> vbo_cpu_slope deleted successfully...\n");
    }

    if (indexBufferCPU)
    {
        glDeleteBuffers(1, &indexBufferCPU);
        indexBufferCPU = 0;

        fprintf(gpFile, "\tinfo>> indexBufferCPU deleted successfully...\n");
    }

    if (indexBuffer)
    {
        glDeleteBuffers(1, &indexBuffer);
        indexBuffer = 0;

        fprintf(gpFile, "\tinfo>> indexBuffer deleted successfully...\n");
    }

    if (shaderProgramObjectOceanFFT)
    {
        // Safe Shader Cleaup
        glUseProgram(shaderProgramObjectOceanFFT);

        glGetProgramiv(shaderProgramObjectOceanFFT, GL_ATTACHED_SHADERS, &ShaderCount);
        pShaders = (GLuint *)malloc(sizeof(GLuint) * ShaderCount);
        if (pShaders == NULL)
        {
            fprintf(gpFile, "\terror>> [%s@%d] malloc() failed inside uninitializeOcean()...\n", __FILE__, __LINE__);
        }

        glGetAttachedShaders(shaderProgramObjectOceanFFT, ShaderCount, &ShaderCount, pShaders);
        for (index = 0; index < ShaderCount; index++)
        {
            glDetachShader(shaderProgramObjectOceanFFT, pShaders[index]);
            glDeleteShader(pShaders[index]);
            pShaders[index] = 0;
        }

        free(pShaders);

        glDeleteShader(shaderProgramObjectOceanFFT);
        shaderProgramObjectOceanFFT = 0;

        glUseProgram(0);

        fprintf(gpFile, "\tinfo>> shaderProgramObjectOceanFFT safe released successfully...\n");
    }

    fprintf(gpFile, "-[%s @%d] end::uninitializeOcean()]\n", __FILE__, __LINE__);
}
