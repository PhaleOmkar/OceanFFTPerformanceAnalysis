#pragma once

#include "./textureLoading.h"

GLfloat alphaValue = 0.0f;

void initializeDetailsTexture(void)
{
    fprintf(gpFile, "+[%s @%d] begin::initializeDetailsTexture()]\n", __FILE__, __LINE__);

    // Code
    // Get Uniform Location
    textureLoadingDetails.mvpMatrixUniform = glGetUniformLocation(textureLoading.shaderProgramObjectTexture, "u_mvpMatrix");
    textureLoadingDetails.textureSamplerUniform = glGetUniformLocation(textureLoading.shaderProgramObjectTexture, "u_Texture_Sampler");
    textureLoadingDetails.alphaValueUniform = glGetUniformLocation(textureLoading.shaderProgramObjectTexture, "u_alpha_value");

    const GLfloat cubeTexCoords[] = {1.0f, 1.0f,
                                     0.0f, 1.0f,
                                     0.0f, 0.0f,
                                     1.0f, 0.0f};

    glGenVertexArrays(1, &textureLoadingDetails.vao_cube);
    glBindVertexArray(textureLoadingDetails.vao_cube);
    glGenBuffers(1, &textureLoadingDetails.vbo_position_cube);
    glBindBuffer(GL_ARRAY_BUFFER, textureLoadingDetails.vbo_position_cube);
    glBufferData(GL_ARRAY_BUFFER, (4 * 3 * sizeof(GLfloat)), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(OUP_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_ATTRIBUTE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &textureLoadingDetails.vbo_texture_cube);
    glBindBuffer(GL_ARRAY_BUFFER, textureLoadingDetails.vbo_texture_cube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeTexCoords), cubeTexCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(OUP_ATTRIBUTE_TEXCOORD, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_ATTRIBUTE_TEXCOORD);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    if (LoadGLTexture(&textureLoadingDetails.gluiTextureImage, MAKEINTRESOURCE(DETAILS_TEXTURE)))
    {
        fprintf(gpFile, "\tinfo>> LoadGLTexture(DETAILS_TEXTURE) successful...\n");
    }

    fprintf(gpFile, "+[%s @%d] end::initializeDetailsTexture()]\n", __FILE__, __LINE__);
}

void displayDetailsTexture(void)
{
    // Code
    glEnable(GL_TEXTURE_2D);
    glUseProgram(textureLoading.shaderProgramObjectTexture);

    // Textured Cube
    mat4 modelViewMatrixDetails = mat4::identity();
    // translateX = 0.090000, translateY = -1.109999, translateZ = -2.870001
    // modelViewMatrix = modelViewMatrix * vmath::translate(0.09f, -1.109999f, -2.870001f);
    // info>> 3] rotateX = 60.000000, rotateY = 0.000000, translateX = 0.000000, translateY = -1.009999, translateZ = -2.390001 : modified
    modelViewMatrixDetails = modelViewMatrixDetails * vmath::translate(0.0f, -1.009999f, -2.39f);
    modelViewMatrixDetails = modelViewMatrixDetails * vmath::rotate(-90.0f, 0.0f, 0.0f, 1.0f);

    glUniformMatrix4fv(textureLoadingDetails.mvpMatrixUniform, 1, GL_FALSE, (perspectiveProjectionMatrix * modelViewMatrixDetails));
    glUniform1f(textureLoadingDetails.alphaValueUniform, alphaValue);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, textureLoadingDetails.gluiTextureImage);
    glUniform1i(textureLoadingDetails.textureSamplerUniform, 1);

    glBindVertexArray(textureLoadingDetails.vao_cube);

    GLfloat cubeVerticesDetails[12];
    cubeVerticesDetails[0] = -2.0f;
    cubeVerticesDetails[1] = 2.15f;
    cubeVerticesDetails[2] = 0.0f;

    cubeVerticesDetails[3] = -2.0f;
    cubeVerticesDetails[4] = -2.15f;
    cubeVerticesDetails[5] = 0.0f;

    cubeVerticesDetails[6] = 0.0f;
    cubeVerticesDetails[7] = -2.15f;
    cubeVerticesDetails[8] = 0.0f;

    cubeVerticesDetails[9] = 0.0f;
    cubeVerticesDetails[10] = 2.15f;
    cubeVerticesDetails[11] = 0.0f;

    glBindBuffer(GL_ARRAY_BUFFER, textureLoadingDetails.vbo_position_cube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVerticesDetails), cubeVerticesDetails, GL_DYNAMIC_DRAW);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glUseProgram(0);
    glDisable(GL_TEXTURE_2D);
}

void updateDetailsTexture(void)
{
    // Code
    static bool bColorValueSet = false;
    if (bColorValueSet == false)
    {
        alphaValue= alphaValue + 0.0015f;
        if (alphaValue > 1.0f)
        {
            bColorValueSet = true;
        }
    }
    else
    {
        alphaValue = alphaValue - 0.0015f;
        if (alphaValue <= 0.0f)
        {
            bColorValueSet = false;
            sceneCounter = OCEANFFT_SCENE;

            alphaValue = 0.0f;
        }
    }

    // // intiailZ = -2.390001f;
    // // finalZ = -0.020003f;
    // if (delayLoopParameter < -0.5f)
    // {
    //     delayLoopParameter = delayLoopParameter + 0.003f;
    // }
    // else
    // {
    //     sceneCounter = OCEANFFT_SCENE;
    // }
}

void uninitializeDetailsTexture(void)
{
    // Code
    fprintf(gpFile, "-[%s @%d] begin::uninitializeDetailsTexture()]\n", __FILE__, __LINE__);

    fprintf(gpFile, "-[%s @%d] end::uninitializeDetailsTexture()]\n", __FILE__, __LINE__);
}