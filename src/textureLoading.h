#pragma once

#include "../header/commongl.h"
#include "./moonSphere.h"

// LoadGLTexture() Definition
bool LoadGLTexture(GLuint *gluiTexture, TCHAR szResourceID[])
{
    // Local Variable Declaration
    bool bResult = false;
    HBITMAP hBitmap = NULL;
    BITMAP bmp;

    // Code
    hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), szResourceID, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
    if (hBitmap)
    {
        bResult = true;
        GetObject(hBitmap, sizeof(BITMAP), &bmp);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glGenTextures(1, gluiTexture);
        glBindTexture(GL_TEXTURE_2D, *gluiTexture);

        // Setting Texture Parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

        // Actually Pushing The Data To Memory With The Help Of Graphic Driver
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp.bmWidth, bmp.bmHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, bmp.bmBits);
        glGenerateMipmap(GL_TEXTURE_2D); // gluBuild2DMipmaps(GL_TEXTURE_2D, 3, bmp.bmWidth, bmp.bmHeight, GL_BGR_EXT, GL_UNSIGNED_BYTE, bmp.bmBits);

        DeleteObject(hBitmap);
    }

    return (bResult);
}

void initializeAstronautPicture(void)
{
    fprintf(gpFile, "\n+[%s @%d] begin::initializeAstronautPicture()]\n", __FILE__, __LINE__);

    // Local Function Declaration
    bool LoadGLTexture(GLuint * gluiTexture, TCHAR szResourceID[]);

    // Local Variable Declaration
    GLint iInfoLogLength = 0;
    GLint iShaderCompiledStatus = 0;
    GLint iShaderLinkerStatus = 0;
    GLchar *szInfoLogBuffer = NULL;

    // Code
    // Vertex Shader - Creating Shader
    textureLoading.vertexShaderObjectTexture = glCreateShader(GL_VERTEX_SHADER);
    const GLchar *pglcVertexShaderSourceCode =
        "#version 430 core                                                              \n"
        "                                                                               \n"
        "in vec4 vPosition;                                                             \n"
        "in vec2 vTexCoord;                                                             \n"
        "                                                                               \n"
        "uniform mat4 u_mvpMatrix;                                                      \n"
        "                                                                               \n"
        "out vec2 out_vTexCoord;                                                        \n"
        "                                                                               \n"
        "void main(void)                                                                \n"
        "{                                                                              \n"
        "   gl_Position = u_mvpMatrix * vPosition;                                      \n"
        "   out_vTexCoord = vTexCoord;                                                  \n"
        "}                                                                              \n";

    glShaderSource(textureLoading.vertexShaderObjectTexture, 1, (const GLchar **)&pglcVertexShaderSourceCode, NULL);

    // Compiling Shader
    glCompileShader(textureLoading.vertexShaderObjectTexture);

    // Error Checking
    glGetShaderiv(textureLoading.vertexShaderObjectTexture, GL_COMPILE_STATUS, &iShaderCompiledStatus);
    if (iShaderCompiledStatus == GL_FALSE)
    {
        glGetShaderiv(textureLoading.vertexShaderObjectTexture, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(textureLoading.vertexShaderObjectTexture, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Vertex Shader Compilation Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    // Fragment Shader - Creating Shader
    textureLoading.fragmentShaderObjectTexture = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar *pglcFragmentShaderSourceCode =
        "#version 430 core                                                                                               \n"
        "                                                                                                                \n"
        "in vec2 out_vTexCoord;                                                                                          \n"
        "                                                                                                                \n"
        "uniform sampler2D u_Texture_Sampler;                                                                            \n"
        "uniform float u_alpha_value;                                                                                    \n"
        "                                                                                                                \n"
        "out vec4 FragmentColor;                                                                                         \n"
        "                                                                                                                \n"
        "void main(void)                                                                                                 \n"
        "{                                                                                                               \n"
        "   FragmentColor.rgb = texture(u_Texture_Sampler, out_vTexCoord).rgb;                                           \n"
        "   FragmentColor.a = u_alpha_value;                                                                             \n"
        "   if((FragmentColor.r < 0.15) && (FragmentColor.g < 0.15) && (FragmentColor.r < 0.15))                         \n"
        "   {                                                                                                            \n"
        "       discard;                                                                                                 \n"
        "   }                                                                                                            \n"
        "}                                                                                                               \n";

    glShaderSource(textureLoading.fragmentShaderObjectTexture, 1, (const GLchar **)&pglcFragmentShaderSourceCode, NULL);

    // Compiling Shader
    glCompileShader(textureLoading.fragmentShaderObjectTexture);

    // Error Checking
    glGetShaderiv(textureLoading.fragmentShaderObjectTexture, GL_COMPILE_STATUS, &iShaderCompiledStatus);
    if (iShaderCompiledStatus == GL_FALSE)
    {
        glGetShaderiv(textureLoading.fragmentShaderObjectTexture, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(textureLoading.fragmentShaderObjectTexture, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Fragment Shader Compilation Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    // Shader Program
    // Create Shader Program
    textureLoading.shaderProgramObjectTexture = glCreateProgram();

    glAttachShader(textureLoading.shaderProgramObjectTexture, textureLoading.vertexShaderObjectTexture);   // Attach Vertex Shader To Shader Program
    glAttachShader(textureLoading.shaderProgramObjectTexture, textureLoading.fragmentShaderObjectTexture); // Attach Fragment Shader To Shader Program

    // Bind Vertex Shader Position Attribute
    glBindAttribLocation(textureLoading.shaderProgramObjectTexture, OUP_ATTRIBUTE_POSITION, "vPosition");
    glBindAttribLocation(textureLoading.shaderProgramObjectTexture, OUP_ATTRIBUTE_TEXCOORD, "vTexCoord");

    // Link Shader Program
    glLinkProgram(textureLoading.shaderProgramObjectTexture);

    // Error Checking
    glGetProgramiv(textureLoading.shaderProgramObjectTexture, GL_LINK_STATUS, &iShaderLinkerStatus);
    if (iShaderLinkerStatus == GL_FALSE)
    {
        glGetShaderiv(textureLoading.shaderProgramObjectTexture, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(textureLoading.shaderProgramObjectTexture, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Shader Program Linking Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    // Get Uniform Location
    textureLoading.mvpMatrixUniform = glGetUniformLocation(textureLoading.shaderProgramObjectTexture, "u_mvpMatrix");
    textureLoading.textureSamplerUniform = glGetUniformLocation(textureLoading.shaderProgramObjectTexture, "u_Texture_Sampler");
    textureLoading.alphaValueUniform = glGetUniformLocation(textureLoading.shaderProgramObjectTexture, "u_alpha_value");

    // Code
    const GLfloat cubeTexCoords[] = {1.0f, 1.0f,
                                     0.0f, 1.0f,
                                     0.0f, 0.0f,
                                     1.0f, 0.0f};

    glGenVertexArrays(1, &textureLoading.vao_cube);
    glBindVertexArray(textureLoading.vao_cube);
    glGenBuffers(1, &textureLoading.vbo_position_cube);
    glBindBuffer(GL_ARRAY_BUFFER, textureLoading.vbo_position_cube);
    glBufferData(GL_ARRAY_BUFFER, (4 * 3 * sizeof(GLfloat)), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(OUP_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_ATTRIBUTE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &textureLoading.vbo_texture_cube);
    glBindBuffer(GL_ARRAY_BUFFER, textureLoading.vbo_texture_cube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeTexCoords), cubeTexCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(OUP_ATTRIBUTE_TEXCOORD, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_ATTRIBUTE_TEXCOORD);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    if (LoadGLTexture(&textureLoading.gluiTextureImage, MAKEINTRESOURCE(ASTRONAUT_IMAGE_TEXTURE)))
    {
        fprintf(gpFile, "\tinfo>> LoadGLTexture(ASTRONAUT_IMAGE_TEXTURE) successful...\n");
    }

    fprintf(gpFile, "+[%s @%d] end::initializeAstronautPicture()]\n", __FILE__, __LINE__);
}

void displayAstronautPicture(void)
{
    // Code
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glUseProgram(textureLoading.shaderProgramObjectTexture);

    // Textured Cube
    mat4 modelViewMatrix = mat4::identity();
    // translateX = 0.420000, translateY = -0.230000, translateZ = -3.500000
    // info>> 5] rotateX = 60.000000, rotateY = -24.500057, translateX = 0.370000, translateY = -0.200000, translateZ = -3.500000
    // modelViewMatrix = modelViewMatrix * vmath::translate(4.12f, -2.23f, -6.0f);

    // translateX = 0.190000, translateY = -0.100000, translateZ = -3.500000
    modelViewMatrix = modelViewMatrix * vmath::translate(1.85f, -1.0f, -2.75f);

    glUniformMatrix4fv(textureLoading.mvpMatrixUniform, 1, GL_FALSE, (perspectiveProjectionMatrix * modelViewMatrix));
    glUniform1f(textureLoading.alphaValueUniform, 0.4f);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureLoading.gluiTextureImage);
    glUniform1i(textureLoading.textureSamplerUniform, 0);

    glBindVertexArray(textureLoading.vao_cube);

    GLfloat cubeVertices[12];

    // +0.1f, +0.1f, 0.0f,
    // -0.1f, +0.1f, 0.0f,
    // -0.1f, -0.1f, 0.0f,
    // +0.1f, -0.1f, 0.0f

    cubeVertices[0] = 0.1f;
    cubeVertices[1] = 0.1f;
    cubeVertices[2] = 0.0f;

    cubeVertices[3] = -0.1f;
    cubeVertices[4] = 0.1f;
    cubeVertices[5] = 0.0f;

    cubeVertices[6] = -0.1f;
    cubeVertices[7] = -0.1f;
    cubeVertices[8] = 0.0f;

    cubeVertices[9] = 0.1f;
    cubeVertices[10] = -0.1f;
    cubeVertices[11] = 0.0f;

    glBindBuffer(GL_ARRAY_BUFFER, textureLoading.vbo_position_cube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_DYNAMIC_DRAW);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glUseProgram(0);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
}

void updateAstronautPicture(void)
{
    // Code
}

void uninitializeAstronautPicture(void)
{
    fprintf(gpFile, "\n-[%s @%d] begin::uninitializeAstronautPicture()]\n", __FILE__, __LINE__);

    // Local Variable Declaration
    GLsizei ShaderCount;
    GLsizei index;
    GLuint *pShaders = NULL;

    // Code
    if (textureLoading.vao_cube)
    {
        glDeleteBuffers(1, &textureLoading.vao_cube);
        textureLoading.vao_cube = 0;

        fprintf(gpFile, "\tinfo>> textureLoading.vao_cube deleted successfully...\n");
    }

    if (textureLoading.vbo_position_cube)
    {
        glDeleteBuffers(1, &textureLoading.vbo_position_cube);
        textureLoading.vbo_position_cube = 0;

        fprintf(gpFile, "\tinfo>> textureLoading.vbo_position_cube deleted successfully...\n");
    }

    if (textureLoading.vbo_texture_cube)
    {
        glDeleteBuffers(1, &textureLoading.vbo_texture_cube);
        textureLoading.vbo_texture_cube = 0;

        fprintf(gpFile, "\tinfo>> textureLoading.vbo_texture_cube deleted successfully...\n");
    }

    if (textureLoading.gluiTextureImage)
    {
        glDeleteTextures(1, &textureLoading.gluiTextureImage);
        textureLoading.gluiTextureImage = 0;

        fprintf(gpFile, "\tinfo>> textureLoading.gluiTextureImage deleted successfully...\n");
    }

    if (textureLoading.shaderProgramObjectTexture)
    {
        // Safe Shader Cleaup
        glUseProgram(textureLoading.shaderProgramObjectTexture);

        glGetProgramiv(textureLoading.shaderProgramObjectTexture, GL_ATTACHED_SHADERS, &ShaderCount);
        pShaders = (GLuint *)malloc(sizeof(GLuint) * ShaderCount);
        if (pShaders == NULL)
        {
            fprintf(gpFile, "\terror>> [%s@%d] malloc() failed inside uninitializeFontRendering()...\n", __FILE__, __LINE__);
        }

        glGetAttachedShaders(textureLoading.shaderProgramObjectTexture, ShaderCount, &ShaderCount, pShaders);
        for (index = 0; index < ShaderCount; index++)
        {
            glDetachShader(textureLoading.shaderProgramObjectTexture, pShaders[index]);
            glDeleteShader(pShaders[index]);
            pShaders[index] = 0;
        }

        free(pShaders);

        glDeleteShader(textureLoading.shaderProgramObjectTexture);
        textureLoading.shaderProgramObjectTexture = 0;

        glUseProgram(0);

        fprintf(gpFile, "\tinfo>> textureLoading.shaderProgramObjectTexture safe released successfully...\n");
    }

    fprintf(gpFile, "-[%s @%d] end::uninitializeAstronautPicture()]\n", __FILE__, __LINE__);
}
