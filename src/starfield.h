#pragma once

#include "../header/commongl.h"

struct header
{
    unsigned char identifier[12];
    unsigned int endianness;
    unsigned int gltype;
    unsigned int gltypesize;
    unsigned int glformat;
    unsigned int glinternalformat;
    unsigned int glbaseinternalformat;
    unsigned int pixelwidth;
    unsigned int pixelheight;
    unsigned int pixeldepth;
    unsigned int arrayelements;
    unsigned int faces;
    unsigned int miplevels;
    unsigned int keypairbytes;
};

enum
{
    NUM_STARS = 8000
};

struct
{
    GLint time;
    GLint projection_matrix;
    GLint starColorAlpha;
} uniforms_t;

GLuint vertexShaderObjectStarfield;
GLuint fragmentShaderObjectStarfield;
GLuint shaderProgramObjectStarfield;

GLuint starfield_vao;
GLuint starfield_vbo;
GLuint star_texture;
GLuint textureSamplerUniform;
GLfloat timeValue = 1.0;

// GLfloat TIME_INCREMENT_VALUE = 0.0025f;
GLfloat TIME_INCREMENT_VALUE = 0.000025f;

int initializeStarfield(void)
{
    fprintf(gpFile, "\n+[%s @%d] begin::initializeStarfield()]\n", __FILE__, __LINE__);

    // Local Function Declaration
    unsigned int loadKTXImage(const char *filename, unsigned int tex = 0);
    GLfloat random_float(void);

    // Local Variable Declaration
    GLint iInfoLogLength = 0;
    GLint iShaderCompiledStatus = 0;
    GLint iShaderLinkerStatus = 0;
    GLchar *szInfoLogBuffer = NULL;

    // Vertex Shader
    // Creating Shader
    vertexShaderObjectStarfield = glCreateShader(GL_VERTEX_SHADER);
    const GLchar *pglcVertexShaderSourceCode =
        "#version 430 core									  			                        \n"
        "													  			                        \n"
        "in vec4 vPosition;									  			                        \n"
        "in vec4 vColor;                                      			                        \n"
        "uniform float u_time;                                			                        \n"
        "uniform mat4 u_projectionMatrix;					  			                        \n"
        "uniform float starColorAlpha;                             			                    \n"
        "													  			                        \n"
        "flat out vec4 starColor;                             			                        \n"
        "													  			                        \n"
        "void main(void)									  			                        \n"
        "{													  			                        \n"
        "    vec4 newVertex = vPosition;                      			                        \n"
        "                                                     			                        \n"
        "    newVertex.z += u_time;                           			                        \n"
        "    newVertex.z = fract(newVertex.z);                			                        \n"
        "                                                     			                        \n"
        "    float size = (10.0 * newVertex.z * newVertex.z); 			                        \n"
        "                                                     			                        \n"
        "    starColor = vec4(vColor.r, vColor.g, vColor.b, starColorAlpha);					\n"
        "                                                     			                        \n"
        "    newVertex.z = (999.9 * newVertex.z) - 950.0;    			                        \n"
        "    gl_Position = u_projectionMatrix * newVertex;    			                        \n"
        "    gl_PointSize = size;                             			                        \n"
        "}                                                    			                        \n";

    glShaderSource(vertexShaderObjectStarfield, 1, (const GLchar **)&pglcVertexShaderSourceCode, NULL);

    // Compiling Shader
    glCompileShader(vertexShaderObjectStarfield);

    // Error Checking
    glGetShaderiv(vertexShaderObjectStarfield, GL_COMPILE_STATUS, &iShaderCompiledStatus);
    if (iShaderCompiledStatus == GL_FALSE)
    {
        glGetShaderiv(vertexShaderObjectStarfield, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(vertexShaderObjectStarfield, iInfoLogLength, &written, szInfoLogBuffer);

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

    // Fragment Shader
    // Creating Shader
    fragmentShaderObjectStarfield = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar *pglcFragmentShaderSourceCode =
        "#version 430 core															            \n"
        "																			            \n"
        "flat in vec4 starColor;                                        			            \n"
        "uniform sampler2D starTexSampler;                                    		            \n"
        "out vec4 FragmentColor;													            \n"
        "																			            \n"
        "void main(void)															            \n"
        "{																			            \n"
        "    FragmentColor = starColor * texture(starTexSampler, gl_PointCoord);  	            \n"
        "}																			            \n";

    glShaderSource(fragmentShaderObjectStarfield, 1, (const GLchar **)&pglcFragmentShaderSourceCode, NULL);

    // Compiling Shader
    glCompileShader(fragmentShaderObjectStarfield);

    // Error Checking
    glGetShaderiv(fragmentShaderObjectStarfield, GL_COMPILE_STATUS, &iShaderCompiledStatus);
    if (iShaderCompiledStatus == GL_FALSE)
    {
        glGetShaderiv(fragmentShaderObjectStarfield, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(fragmentShaderObjectStarfield, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Fragment Shader Compilation Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    // Shader Program
    // Create Shader Program
    shaderProgramObjectStarfield = glCreateProgram();

    // Attach Vertex Shader To Shader Program
    glAttachShader(shaderProgramObjectStarfield, vertexShaderObjectStarfield);

    // Attach Fragment Shader To Shader Program
    glAttachShader(shaderProgramObjectStarfield, fragmentShaderObjectStarfield);

    // Bind Vertex Shader Position Attribute
    glBindAttribLocation(shaderProgramObjectStarfield, OUP_ATTRIBUTE_POSITION, "vPosition");
    glBindAttribLocation(shaderProgramObjectStarfield, OUP_ATTRIBUTE_COLOR, "vColor");

    // Link Shader Program
    glLinkProgram(shaderProgramObjectStarfield);

    // Error Checking
    glGetProgramiv(shaderProgramObjectStarfield, GL_LINK_STATUS, &iShaderLinkerStatus);
    if (iShaderLinkerStatus == GL_FALSE)
    {
        glGetShaderiv(shaderProgramObjectStarfield, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(shaderProgramObjectStarfield, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Shader Program Linking Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    // Get Uniform Location
    // vs
    uniforms_t.projection_matrix = glGetUniformLocation(shaderProgramObjectStarfield, "u_projectionMatrix");
    uniforms_t.time = glGetUniformLocation(shaderProgramObjectStarfield, "u_time");
    uniforms_t.starColorAlpha = glGetUniformLocation(shaderProgramObjectStarfield, "starColorAlpha");

    // fs
    textureSamplerUniform = glGetUniformLocation(shaderProgramObjectStarfield, "starTexSampler");

    struct star_t
    {
        vec3 position;
        vec3 color;
    };

    glGenVertexArrays(1, &starfield_vao);
    glBindVertexArray(starfield_vao);
    glGenBuffers(1, &starfield_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, starfield_vbo);
    glBufferData(GL_ARRAY_BUFFER, NUM_STARS * sizeof(star_t), NULL, GL_STATIC_DRAW);

    star_t *star = (star_t *)glMapBufferRange(GL_ARRAY_BUFFER, 0, NUM_STARS * sizeof(star_t), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

    for (int i = 0; i < 1000; i++)
    {
        star[i].position[0] = (random_float() * 3.0f - 1.0f) * 200.0f;
        star[i].position[1] = (random_float() * 3.0f - 1.0f) * 100.0f;
        star[i].position[2] = (random_float() * 3.0f - 1.0f) * 85.0f;

        star[i].color[0] = 0.8f + random_float() * 0.75f;
        star[i].color[1] = 0.8f + random_float() * 0.75f;
        star[i].color[2] = 0.8f + random_float() * 0.75f;
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);

    glVertexAttribPointer(OUP_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, sizeof(star_t), NULL);
    glEnableVertexAttribArray(OUP_ATTRIBUTE_POSITION);

    glVertexAttribPointer(OUP_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, sizeof(star_t), (void *)sizeof(vec3));
    glEnableVertexAttribArray(OUP_ATTRIBUTE_COLOR);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Loading Texture
    star_texture = loadKTXImage("./res/star.ktx");
    if (star_texture == 0)
    {
        fprintf(gpFile, "\terror>> loadKTXImage(star.ktx) failed...\n");
    }
    else
    {
        fprintf(gpFile, "\tinfo>> loadKTXImage(star.ktx)::%d successful...\n", star_texture);
    }

    fprintf(gpFile, "+[%s @%d] end::initializeStarfield()]\n", __FILE__, __LINE__);

    return (0);
}

GLfloat starColorAlphaValue = 0.0f;

void displayStarfield(void)
{
    glUseProgram(shaderProgramObjectStarfield);

    mat4 projection = mat4::identity();
    projection = perspectiveProjectionMatrix * vmath::translate(-100.0f, -30.0f, 0.0f);

    glUniform1f(uniforms_t.time, timeValue);
    glUniform1f(uniforms_t.starColorAlpha, starColorAlphaValue);
    glUniformMatrix4fv(uniforms_t.projection_matrix, 1, GL_FALSE, projection);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, star_texture);
    glUniform1i(textureSamplerUniform, 0);

    glBindVertexArray(starfield_vao);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, NUM_STARS);
    glBindVertexArray(0);

    glUseProgram(0);
}

void updateStarfield(void)
{
    // Code
    timeValue += TIME_INCREMENT_VALUE;
    timeValue -= floor(timeValue);

    starColorAlphaValue = starColorAlphaValue + 0.001f;
}

void uninitializeStarfield(void)
{
    fprintf(gpFile, "\n-[%s @%d] begin::uninitializeStarfield()]\n", __FILE__, __LINE__);

    // Local Variable Declaration
    GLsizei ShaderCount;
    GLsizei index;
    GLuint *pShaders = NULL;

    // Code
    if (starfield_vao)
    {
        glDeleteBuffers(1, &starfield_vao);
        starfield_vao = 0;

        fprintf(gpFile, "\tinfo>> starfield_vao deleted successfully...\n");
    }

    if (starfield_vbo)
    {
        glDeleteBuffers(1, &starfield_vbo);
        starfield_vbo = 0;

        fprintf(gpFile, "\tinfo>> starfield_vbo deleted successfully...\n");
    }

    if (shaderProgramObjectStarfield)
    {
        // Safe Shader Cleaup
        glUseProgram(shaderProgramObjectStarfield);

        glGetProgramiv(shaderProgramObjectStarfield, GL_ATTACHED_SHADERS, &ShaderCount);
        pShaders = (GLuint *)malloc(sizeof(GLuint) * ShaderCount);
        if (pShaders == NULL)
        {
            fprintf(gpFile, "\terror>> [%s@%d] malloc() failed inside uninitializeFontRendering()...\n", __FILE__, __LINE__);
        }

        glGetAttachedShaders(shaderProgramObjectStarfield, ShaderCount, &ShaderCount, pShaders);
        for (index = 0; index < ShaderCount; index++)
        {
            glDetachShader(shaderProgramObjectStarfield, pShaders[index]);
            glDeleteShader(pShaders[index]);
            pShaders[index] = 0;
        }

        free(pShaders);

        glDeleteShader(shaderProgramObjectStarfield);
        shaderProgramObjectStarfield = 0;

        glUseProgram(0);

        fprintf(gpFile, "\tinfo>> shaderProgramObjectStarfield safe released successfully...\n");
    }

    fprintf(gpFile, "-[%s @%d] end::uninitializeStarfield()]\n", __FILE__, __LINE__);
}

//##########################################################################################################################//
const unsigned char identifier_array[] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

// loadKTXImage() Definition
unsigned int loadKTXImage(const char *filename, unsigned int tex)
{
    // Local Function Declaration
    const unsigned int swap32(const unsigned int u32);
    unsigned int calculate_face_size(const header &h);
    unsigned int calculate_stride(const header &h, unsigned int width, unsigned int pad = 4);

    // Local Variable Declaration
    FILE *fp;
    GLuint temp = 0;
    GLuint retval = 0;
    header h;
    size_t data_start, data_end;
    unsigned char *data;
    GLenum target = GL_NONE;

    // Code
    fp = fopen(filename, "rb");

    if (!fp)
    {
        return (0);
    }

    if (fread(&h, sizeof(h), 1, fp) != 1)
    {
        goto fail_read;
    }

    if (memcmp(h.identifier, identifier_array, sizeof(identifier_array)) != 0)
    {
        goto fail_header;
    }

    if (h.endianness == 0x04030201)
    {
        // No swap needed
    }
    else if (h.endianness == 0x01020304)
    {
        // Swap needed
        h.endianness = swap32(h.endianness);
        h.gltype = swap32(h.gltype);
        h.gltypesize = swap32(h.gltypesize);
        h.glformat = swap32(h.glformat);
        h.glinternalformat = swap32(h.glinternalformat);
        h.glbaseinternalformat = swap32(h.glbaseinternalformat);
        h.pixelwidth = swap32(h.pixelwidth);
        h.pixelheight = swap32(h.pixelheight);
        h.pixeldepth = swap32(h.pixeldepth);
        h.arrayelements = swap32(h.arrayelements);
        h.faces = swap32(h.faces);
        h.miplevels = swap32(h.miplevels);
        h.keypairbytes = swap32(h.keypairbytes);
    }
    else
    {
        goto fail_header;
    }

    // Guess target (texture type)
    if (h.pixelheight == 0)
    {
        if (h.arrayelements == 0)
        {
            target = GL_TEXTURE_1D;
        }
        else
        {
            target = GL_TEXTURE_1D_ARRAY;
        }
    }
    else if (h.pixeldepth == 0)
    {
        if (h.arrayelements == 0)
        {
            if (h.faces == 0)
            {
                target = GL_TEXTURE_2D;
            }
            else
            {
                target = GL_TEXTURE_CUBE_MAP;
            }
        }
        else
        {
            if (h.faces == 0)
            {
                target = GL_TEXTURE_2D_ARRAY;
            }
            else
            {
                target = GL_TEXTURE_CUBE_MAP_ARRAY;
            }
        }
    }
    else
    {
        target = GL_TEXTURE_3D;
    }

    // Check for insanity...
    if (target == GL_NONE ||                       // Couldn't figure out target
        (h.pixelwidth == 0) ||                     // Texture has no width???
        (h.pixelheight == 0 && h.pixeldepth != 0)) // Texture has depth but no height???
    {
        goto fail_header;
    }

    temp = tex;
    if (tex == 0)
    {
        glGenTextures(1, &tex);
    }

    glBindTexture(target, tex);

    data_start = ftell(fp) + h.keypairbytes;
    fseek(fp, 0, SEEK_END);
    data_end = ftell(fp);
    fseek(fp, data_start, SEEK_SET);

    data = new unsigned char[data_end - data_start];
    memset(data, 0, data_end - data_start);

    fread(data, 1, data_end - data_start, fp);

    if (h.miplevels == 0)
    {
        h.miplevels = 1;
    }

    switch (target)
    {
    case GL_TEXTURE_1D:
        glTexStorage1D(GL_TEXTURE_1D, h.miplevels, h.glinternalformat, h.pixelwidth);
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, h.pixelwidth, h.glformat, h.glinternalformat, data);
        break;

    case GL_TEXTURE_2D:
        // glTexImage2D(GL_TEXTURE_2D, 0, h.glinternalformat, h.pixelwidth, h.pixelheight, 0, h.glformat, h.gltype, data);
        if (h.gltype == GL_NONE)
        {
            glCompressedTexImage2D(GL_TEXTURE_2D, 0, h.glinternalformat, h.pixelwidth, h.pixelheight, 0, 420 * 380 / 2, data);
        }
        else
        {
            glTexStorage2D(GL_TEXTURE_2D, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight);
            {
                unsigned char *ptr = data;
                unsigned int height = h.pixelheight;
                unsigned int width = h.pixelwidth;
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                for (unsigned int i = 0; i < h.miplevels; i++)
                {
                    glTexSubImage2D(GL_TEXTURE_2D, i, 0, 0, width, height, h.glformat, h.gltype, ptr);
                    ptr += height * calculate_stride(h, width, 1);
                    height >>= 1;
                    width >>= 1;

                    if (!height)
                    {
                        height = 1;
                    }

                    if (!width)
                    {
                        width = 1;
                    }
                }
            }
        }
        break;

    case GL_TEXTURE_3D:
        glTexStorage3D(GL_TEXTURE_3D, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight, h.pixeldepth);
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, h.pixelwidth, h.pixelheight, h.pixeldepth, h.glformat, h.gltype, data);
        break;

    case GL_TEXTURE_1D_ARRAY:
        glTexStorage2D(GL_TEXTURE_1D_ARRAY, h.miplevels, h.glinternalformat, h.pixelwidth, h.arrayelements);
        glTexSubImage2D(GL_TEXTURE_1D_ARRAY, 0, 0, 0, h.pixelwidth, h.arrayelements, h.glformat, h.gltype, data);
        break;

    case GL_TEXTURE_2D_ARRAY:
        glTexStorage3D(GL_TEXTURE_2D_ARRAY, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight, h.arrayelements);
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, h.pixelwidth, h.pixelheight, h.arrayelements, h.glformat, h.gltype, data);
        break;

    case GL_TEXTURE_CUBE_MAP:
        glTexStorage2D(GL_TEXTURE_CUBE_MAP, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight);
        // glTexSubImage3D(GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0, h.pixelwidth, h.pixelheight, h.faces, h.glformat, h.gltype, data);
        {
            unsigned int face_size = calculate_face_size(h);
            for (unsigned int i = 0; i < h.faces; i++)
            {
                glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, 0, 0, h.pixelwidth, h.pixelheight, h.glformat, h.gltype, data + face_size * i);
            }
        }
        break;

    case GL_TEXTURE_CUBE_MAP_ARRAY:
        glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight, h.arrayelements);
        glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 0, 0, 0, 0, h.pixelwidth, h.pixelheight, h.faces * h.arrayelements, h.glformat, h.gltype, data);
        break;

    default: // Should never happen
        goto fail_target;
    }

    if (h.miplevels == 1)
    {
        glGenerateMipmap(target);
    }

    retval = tex;

fail_target:
    delete[] data;

fail_header:;
fail_read:;
    fclose(fp);

    return (retval);
}

// swap32() Definition
const unsigned int swap32(const unsigned int u32)
{
    // Code
    union
    {
        unsigned int u32;
        unsigned char u8[4];
    } a, b;

    a.u32 = u32;
    b.u8[0] = a.u8[3];
    b.u8[1] = a.u8[2];
    b.u8[2] = a.u8[1];
    b.u8[3] = a.u8[0];

    return (b.u32);
}

// calculate_face_size() Definition
unsigned int calculate_face_size(const header &h)
{
    // Local Function Declaration
    unsigned int calculate_stride(const header &h, unsigned int width, unsigned int pad = 4);

    // Code
    unsigned int stride = calculate_stride(h, h.pixelwidth);

    return (stride * h.pixelheight);
}

// calculate_stride() Definition
unsigned int calculate_stride(const header &h, unsigned int width, unsigned int pad = 4)
{
    // Local Variable Declaration
    unsigned int channels = 0;

    // Code
    switch (h.glbaseinternalformat)
    {
    case GL_RED:
        channels = 1;
        break;

    case GL_RG:
        channels = 2;
        break;

    case GL_BGR:
    case GL_RGB:
        channels = 3;
        break;

    case GL_BGRA:
    case GL_RGBA:
        channels = 4;
        break;
    }

    unsigned int stride = h.gltypesize * channels * width;

    stride = (stride + (pad - 1)) & ~(pad - 1);

    return (stride);
}

GLuint seed = 0x13371337;

// random_float() Definition
GLfloat random_float(void)
{
    // Local Variable Declaration
    GLfloat res;
    GLuint tmp;

    // Code
    seed *= 16807;
    tmp = seed ^ (seed >> 4) ^ (seed << 15);
    *((unsigned int *)&res) = (tmp >> 9) | 0x3F800000;

    return (res - 1.0f);
}
