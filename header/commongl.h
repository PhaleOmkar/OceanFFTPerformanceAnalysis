#pragma once

#include <windows.h>
#include <stdio.h>

#include <GL/glew.h>
#include <gl/GL.h>

#include <al.h>
#include <alc.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <assert.h>
#include <string>
#include <map>
#include <string>

#include "../header/ft2build.h"
#include FT_FREETYPE_H

#include "../header/vmath.h"
#include "../header/camera.h"
#include "../header/seminarProject.h"
#include "../header/wavhelper.h"

///////////////////////////////////////////////////////////////////////////
// Global Macro Definitions
#define MESH_SIZE 1024
#define SPECTRUM_SIZE_W (MESH_SIZE + 4)
#define SPECTRUM_SIZE_H (MESH_SIZE + 1)

#define CUDART_PI_F 3.141592654f
#define CUDART_SQRT_HALF_F 0.707106781f

// Pragma Declaration
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cufft.lib")

#pragma comment(lib, "freetype.lib")

#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "openal32.lib")

using namespace vmath;

///////////////////////////////////////////////////////////////////////////
enum
{
    OUP_ATTRIBUTE_POSITION,
    OUP_ATTRIBUTE_COLOR,
    OUP_ATTRIBUTE_NORMAL,
    OUP_ATTRIBUTE_TEXCOORD,
    OUP_HEIGHT_IN,
    OUP_SLOPE_IN
};

enum
{
    ALONG_POSITIVE_X_AXIS,
    ALONG_POSITIVE_Y_AXIS,
    ALONG_NEGATIVE_X_AXIS,
    ALONG_NEGATIVE_Y_AXIS
};

enum
{
    INTRO_SCENE,
    DETAILS_SCENE,
    OCEANFFT_SCENE,
    OPENGL_CUDA_SCENE,
    END_CREDITS_SCENE
};
GLuint sceneCounter = INTRO_SCENE;

///////////////////////////////////////////////////////////////////////////
struct windowInfo
{
    int WindowWidth;
    int WindowHeight;
} Info;

bool gbWireFrame = true;
bool gbNeedToUpdate = true;

unsigned int meshSizeLimit = 2;
unsigned int spectrumW = MESH_SIZE + 4;
unsigned int spectrumH = MESH_SIZE + 1;

mat4 perspectiveProjectionMatrix;

///////////////////////////////////////////////////////////////////////////
// OpenAL Variable Declaration
ALCdevice *device = NULL;
ALCcontext *context = NULL;

unsigned int bufferAlways;
unsigned int sourceAlways;

unsigned int bufferShantiMantra;
unsigned int sourceShantiMantra;

ALvoid *alDataAlways;
ALvoid *alDataShantiMantra;

ALsizei alSize, alFrequency;
ALenum alFormat;
ALboolean alLoop = AL_FALSE;

///////////////////////////////////////////////////////////////////////////
struct TextureLoading
{
    GLuint vertexShaderObjectTexture;
    GLuint fragmentShaderObjectTexture;
    GLuint shaderProgramObjectTexture;

    GLuint vao_cube;
    GLuint vbo_position_cube;
    GLuint vbo_texture_cube;

    GLuint mvpMatrixUniform;
    GLuint textureSamplerUniform;
    GLuint alphaValueUniform;

    GLuint gluiTextureImage;
} textureLoadingDetails, textureLoading;
