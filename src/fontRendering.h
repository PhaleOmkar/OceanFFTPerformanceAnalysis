#pragma once

#include "../header/commongl.h"
#include "./oceanFFTSceneSupprotingFunctions.h"

HWND ghwnd;
FILE *gpFile;
bool gbOnRunGPU = false;

struct Character
{
    unsigned int TextureID; // ID handle of the glyph texture
    vec2 Size;              // Size of glyph
    vec2 Bearing;           // Offset from baseline to left/top of glyph
    unsigned int Advance;   // Horizontal offset to advance to next glyph
};
std::map<GLchar, Character> Characters;

struct FontRendering
{
    GLuint vertexShaderObject;
    GLuint fragmentShaderObject;
    GLuint shaderProgramObject;

    GLuint modelViewMatrixUniform;
    GLuint projectionMatrixUniform;
    GLuint textureSamplerUniform;
    GLuint textColorUniform;

    GLuint vao_font_rendering;
    GLuint vbo_font_rendering;

    const char *fontName = "./fonts/OCRAEXT.TTF";
    char messageString[1024];
    GLfloat textPositionY = -85.0f; //-20.0f;
    GLfloat colorValue = 0.0f;

    GLfloat TEXT_ANIMATION_SPEED = 0.07f;
} fontRendering;

void initializeFontRendering(void)
{
    fprintf(gpFile, "\n+[%s @%d] begin::initializeFontRendering()]\n", __FILE__, __LINE__);

    // Local Function Declaration
    void uninitializeFontRendering(void);

    // Local Variable Declaration
    GLint iInfoLogLength = 0;
    GLint iShaderCompiledStatus = 0;
    GLint iShaderLinkerStatus = 0;
    GLchar *szInfoLogBuffer = NULL;

    // Vertex Shader
    // Creating Shader
    fontRendering.vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
    const GLchar *pglcVertexShaderSourceCode =
        "#version 430 core                                                          				\n"
        "                                                                           				\n"
        "in vec4 vPosition;																			\n"
        "                                                                           				\n"
        "uniform mat4 u_projection_matrix;															\n"
        "                                                                           				\n"
        "out vec2 vTexCoords;																		\n"
        "                                                                           				\n"
        "void main(void)																			\n"
        "{																							\n"
        "	gl_Position = u_projection_matrix * vec4(vPosition.xy, 0.0, 1.0);						\n"
        "   vTexCoords = vPosition.zw;																\n"
        "}																							\n";

    glShaderSource(fontRendering.vertexShaderObject, 1, (const GLchar **)&pglcVertexShaderSourceCode, NULL);

    // Compiling Shader
    glCompileShader(fontRendering.vertexShaderObject);

    // Error Checking
    glGetShaderiv(fontRendering.vertexShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
    if (iShaderCompiledStatus == GL_FALSE)
    {
        glGetShaderiv(fontRendering.vertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(fontRendering.vertexShaderObject, iInfoLogLength, &written, szInfoLogBuffer);

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
    fontRendering.fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar *pglcFragmentShaderSourceCode =
        "#version 430 core                                                              			\n"
        "                                                                               			\n"
        "in vec2 vTexCoords;																		\n"
        "                                                                               			\n"
        "out vec4 FragmentColor;																	\n"
        "                                                                               			\n"
        "uniform sampler2D u_texture_sampler;														\n"
        "uniform vec3 textColor;																	\n"
        "                                                                               			\n"
        "void main(void)																			\n"
        "{																							\n"
        "	vec4 sampled = vec4(1.0, 1.0, 1.0, texture(u_texture_sampler, vTexCoords).r);			\n"
        "	FragmentColor = vec4(textColor, 1.0) * sampled;											\n"
        "}																							\n";

    glShaderSource(fontRendering.fragmentShaderObject, 1, (const GLchar **)&pglcFragmentShaderSourceCode, NULL);

    // Compiling Shader
    glCompileShader(fontRendering.fragmentShaderObject);

    // Error Checking
    glGetShaderiv(fontRendering.fragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
    if (iShaderCompiledStatus == GL_FALSE)
    {
        glGetShaderiv(fontRendering.fragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(fontRendering.fragmentShaderObject, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Fragment Shader Compilation Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    // Shader Program
    // Create Shader Program
    fontRendering.shaderProgramObject = glCreateProgram();

    glAttachShader(fontRendering.shaderProgramObject, fontRendering.vertexShaderObject);   // Attach Vertex Shader To Shader Program
    glAttachShader(fontRendering.shaderProgramObject, fontRendering.fragmentShaderObject); // Attach Fragment Shader To Shader Program

    // Bind Vertex Shader Position Attribute
    glBindAttribLocation(fontRendering.shaderProgramObject, OUP_ATTRIBUTE_POSITION, "vPosition");

    // Link Shader Program
    glLinkProgram(fontRendering.shaderProgramObject);

    // Error Checking
    glGetProgramiv(fontRendering.shaderProgramObject, GL_LINK_STATUS, &iShaderLinkerStatus);
    if (iShaderLinkerStatus == GL_FALSE)
    {
        glGetShaderiv(fontRendering.shaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(fontRendering.shaderProgramObject, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Shader Program Linking Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    // Get Uniform Location
    fontRendering.projectionMatrixUniform = glGetUniformLocation(fontRendering.shaderProgramObject, "u_projection_matrix");
    fontRendering.textureSamplerUniform = glGetUniformLocation(fontRendering.shaderProgramObject, "u_texture_sampler");
    fontRendering.textColorUniform = glGetUniformLocation(fontRendering.shaderProgramObject, "textColor");

    glGenVertexArrays(1, &fontRendering.vao_font_rendering);
    glBindVertexArray(fontRendering.vao_font_rendering);

    glGenBuffers(1, &fontRendering.vbo_font_rendering);
    glBindBuffer(GL_ARRAY_BUFFER, fontRendering.vbo_font_rendering);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(OUP_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
    glEnableVertexAttribArray(OUP_ATTRIBUTE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    FT_Library ft;
    if (FT_Init_FreeType(&ft))
    {
        fprintf(gpFile, "\terror>> [LINE - %d] Could not init freeType library\n", __LINE__);
        uninitializeFontRendering();
    }
    else
    {
        fprintf(gpFile, "\tinfo>> Initialization of freeType library successful...\n");
    }

    FT_Face face;
    if (FT_New_Face(ft, fontRendering.fontName, 0, &face))
    {
        fprintf(gpFile, "\terror>> [LINE - %d] Failed to load font\n", __LINE__);
        uninitializeFontRendering();
    }
    else
    {
        fprintf(gpFile, "\tinfo>> Font loading successful...\n");
    }

    FT_Set_Pixel_Sizes(face, 0, 48);

    if (FT_Load_Char(face, 'X', FT_LOAD_RENDER))
    {
        fprintf(gpFile, "\terror>> [LINE - %d] Failed to load Glyph\n", __LINE__);
        uninitializeFontRendering();
    }
    else
    {
        fprintf(gpFile, "\tinfo>> Glyph loading successful...\n");
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // disable byte-alignment restriction

    for (unsigned char c = 0; c < 128; c++)
    {
        // load character glyph
        if (FT_Load_Char(face, c, FT_LOAD_RENDER))
        {
            fprintf(gpFile, "\terror>> [LINE - %d] Failed to load Glyph\n", __LINE__);
            continue;
        }
        // else
        // {
        // 	fprintf(gpFile, "load Glyph successful...\n");
        // }

        // generate texture
        unsigned int texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width, face->glyph->bitmap.rows, 0, GL_RED, GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer);

        // set texture options
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // now store character for later use
        Character character = {texture,
                               vec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                               vec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                               (unsigned int)face->glyph->advance.x};
        Characters.insert(std::pair<char, Character>(c, character));
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    fprintf(gpFile, "+[%s @%d] end::initializeFontRendering()]\n", __FILE__, __LINE__);
}

void displayIntroCredits(void)
{
    // Local Function Declaration
    void renderTextOnScreen(std::string renderTextOnScreenMessage, float xPosition, float yPosition, float zPosition, float fontSize, float redColor, float greenColor, float blueColor);

    // Code
    sprintf(fontRendering.messageString, "Ocean FFT");
    renderTextOnScreen(fontRendering.messageString, -15.5f, 5.0f, -100.0f, 0.125f, 0.0f, 0.0f, fontRendering.colorValue);
    // renderTextOnScreen(fontRendering.messageString, -53.0f, 0.0f, 0.125f, colorValue, colorValue, colorValue);

    sprintf(fontRendering.messageString, "Performance Analysis");
    renderTextOnScreen(fontRendering.messageString, -35.5f, -5.0f, -100.0f, 0.125f, 0.0f, 0.0f, fontRendering.colorValue);
}

void displayOceanFFTScene(void)
{
    // Local Function Declaration
    void renderTextOnScreen(std::string renderTextOnScreenMessage, float xPosition, float yPosition, float zPosition, float fontSize, float redColor, float greenColor, float blueColor);

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
    renderTextOnScreen(fontRendering.messageString, 25.0f, 44.0f, -140.0f, 0.125f, colorValuesForData[0], colorValuesForData[1], colorValuesForData[2]);

    sprintf(fontRendering.messageString, "Polygon Mode : %s", gbWireFrame == true ? "LINE" : "FILL");
    renderTextOnScreen(fontRendering.messageString, -99.0f, -49.0f, -140.0f, 0.125f, colorValuesForData[0], colorValuesForData[1], colorValuesForData[2]);

    sprintf(fontRendering.messageString, "Mesh Size    : %d x %d x 4", (int)meshSizeLimit, (int)meshSizeLimit);
    renderTextOnScreen(fontRendering.messageString, -99.0f, -55.0f, -140.0f, 0.125f, colorValuesForData[0], colorValuesForData[1], colorValuesForData[2]);
}

void displayEndCredits(void)
{
    // Local Function Declaration
    void renderTextOnScreen(std::string renderTextOnScreenMessage, float xPosition, float yPosition, float zPosition, float fontSize, float redColor, float greenColor, float blueColor);

    // Code
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (fontRendering.textPositionY <= 70.0f)
    {
        sprintf(fontRendering.messageString, "******************************************");
        renderTextOnScreen(fontRendering.messageString, -30.0f, 23.0f + fontRendering.textPositionY, -140.0f, 0.05f, 1.0f, 1.0f, 1.0f);

        sprintf(fontRendering.messageString, "     Ignited By :");
        // sprintf(fontRendering.messageString, "Impossible Without :");
        renderTextOnScreen(fontRendering.messageString, -30.0f, 20.5f + fontRendering.textPositionY, -140.0f, 0.05f, 1.0f, 1.0f, 1.0f);

        sprintf(fontRendering.messageString, "******************************************");
        renderTextOnScreen(fontRendering.messageString, -30.0f, 18.0f + fontRendering.textPositionY, -140.0f, 0.05f, 1.0f, 1.0f, 1.0f);

        sprintf(fontRendering.messageString, " Dr. Vijay D. Gokhale");
        renderTextOnScreen(fontRendering.messageString, -5.0f, 20.5f + fontRendering.textPositionY, -140.0f, 0.05f, fontRendering.colorValue, fontRendering.colorValue, fontRendering.colorValue);

        sprintf(fontRendering.messageString, "+ Platform          -");
        renderTextOnScreen(fontRendering.messageString, -39.5f, 10.5f + fontRendering.textPositionY, -140.0f, 0.05f, 1.0f, 1.0f, 1.0f);

        sprintf(fontRendering.messageString, "Windows 10 (Win32 SDK)");
        renderTextOnScreen(fontRendering.messageString, -7.0f, 10.5f + fontRendering.textPositionY, -140.0f, 0.05f, fontRendering.colorValue, fontRendering.colorValue, fontRendering.colorValue);

        sprintf(fontRendering.messageString, "+ References        -");
        renderTextOnScreen(fontRendering.messageString, -39.5f, 7.5f + fontRendering.textPositionY, -140.0f, 0.05f, 1.0f, 1.0f, 1.0f);

        sprintf(fontRendering.messageString, "RTR Assignments, HPP 2020 & CUDA Samples");
        renderTextOnScreen(fontRendering.messageString, -7.0f, 7.5f + fontRendering.textPositionY, -140.0f, 0.05f, fontRendering.colorValue, fontRendering.colorValue, fontRendering.colorValue);

        sprintf(fontRendering.messageString, "+ Technologies      -");
        renderTextOnScreen(fontRendering.messageString, -39.5f, 4.5f + fontRendering.textPositionY, -140.0f, 0.05f, 1.0f, 1.0f, 1.0f);

        sprintf(fontRendering.messageString, "OpenGL, OpenAL & CUDA");
        renderTextOnScreen(fontRendering.messageString, -7.0f, 4.5f + fontRendering.textPositionY, -140.0f, 0.05f, fontRendering.colorValue, fontRendering.colorValue, fontRendering.colorValue);

        sprintf(fontRendering.messageString, "+ Background Music  -");
        renderTextOnScreen(fontRendering.messageString, -39.5f, 1.5f + fontRendering.textPositionY, -140.0f, 0.05f, 1.0f, 1.0f, 1.0f);

        sprintf(fontRendering.messageString, "\"Always\" by Peder B. Helland");
        renderTextOnScreen(fontRendering.messageString, -7.0f, 1.5f + fontRendering.textPositionY, -140.0f, 0.05f, fontRendering.colorValue, fontRendering.colorValue, fontRendering.colorValue);

        // Om Purnamadah Purnamidam | Shanti Mantra from Ishavasya Upanishad
        sprintf(fontRendering.messageString, "\"Om Purnamadah Purnamidam\" Shanti Mantra");
        renderTextOnScreen(fontRendering.messageString, -7.0f, -1.5f + fontRendering.textPositionY, -140.0f, 0.05f, fontRendering.colorValue, fontRendering.colorValue, fontRendering.colorValue);

        sprintf(fontRendering.messageString, "+ Presented By      -");
        renderTextOnScreen(fontRendering.messageString, -39.5f, -4.5f + fontRendering.textPositionY, -140.0f, 0.05f, 1.0f, 1.0f, 1.0f);

        sprintf(fontRendering.messageString, "Omkar U. Phale (RTR2021 Vertex Group Leader)");
        renderTextOnScreen(fontRendering.messageString, -7.0f, -4.5f + fontRendering.textPositionY, -140.0f, 0.05f, fontRendering.colorValue, fontRendering.colorValue, fontRendering.colorValue);
    }
    else
    {
        sprintf(fontRendering.messageString, "==========================");
        renderTextOnScreen(fontRendering.messageString, -35.0f, 4.0f, -140.0f, 0.1f, 1.0f, 1.0f, 1.0f);

        sprintf(fontRendering.messageString, "The End!");
        renderTextOnScreen(fontRendering.messageString, -7.5f, 0.0f, -140.0f, 0.1f, 1.0f, 1.0f, 1.0f);

        sprintf(fontRendering.messageString, "==========================");
        renderTextOnScreen(fontRendering.messageString, -35.0f, -4.0f, -140.0f, 0.1f, 1.0f, 1.0f, 1.0f);
    }
}

// renderTextOnScreen() Definition
void renderTextOnScreen(std::string renderTextOnScreenMessage, float xPosition, float yPosition, float zPosition, float fontSize, float redColor, float greenColor, float blueColor)
{
    glUseProgram(fontRendering.shaderProgramObject);

    mat4 modelViewMatrix = mat4::identity();
    modelViewMatrix = modelViewMatrix * vmath::translate(0.0f, 0.0f, zPosition);

    glUniform3f(fontRendering.textColorUniform, redColor, greenColor, blueColor);
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(fontRendering.textureSamplerUniform, 0);
    glUniformMatrix4fv(fontRendering.projectionMatrixUniform, 1, GL_FALSE, perspectiveProjectionMatrix * modelViewMatrix);
    glBindVertexArray(fontRendering.vao_font_rendering);

    // iterate through all characters
    std::string::const_iterator c;
    for (c = renderTextOnScreenMessage.begin(); c != renderTextOnScreenMessage.end(); c++)
    {
        Character ch = Characters[*c];

        float xpos = xPosition + ch.Bearing[0] * fontSize;
        float ypos = yPosition - (ch.Size[1] - ch.Bearing[1]) * fontSize;

        float w = ch.Size[0] * fontSize;
        float h = ch.Size[1] * fontSize;

        // update VBO for each character
        float vertices[6][4] = {{xpos, ypos + h, 0.0f, 0.0f},
                                {xpos, ypos, 0.0f, 1.0f},
                                {xpos + w, ypos, 1.0f, 1.0f},

                                {xpos, ypos + h, 0.0f, 0.0f},
                                {xpos + w, ypos, 1.0f, 1.0f},
                                {xpos + w, ypos + h, 1.0f, 0.0f}};

        // render glyph texture over quad
        glBindTexture(GL_TEXTURE_2D, ch.TextureID);

        // update content of VBO memory
        glBindBuffer(GL_ARRAY_BUFFER, fontRendering.vbo_font_rendering);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // now advance cursors for next glyph (note that advance is number of 1/64 pixels)
        xPosition += (ch.Advance >> 6) * fontSize; // bitshift by 6 to get value in pixels (2^6 = 64)
    }

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(0);
}

void updateFontRenderingIntroCredits(void)
{
    // Code
    static bool bColorValueSet = false;
    if (bColorValueSet == false)
    {
        fontRendering.colorValue = fontRendering.colorValue + 0.0025f;
        if (fontRendering.colorValue > 1.0f)
        {
            bColorValueSet = true;
        }
    }
    else
    {
        fontRendering.colorValue = fontRendering.colorValue - 0.0025f;
        if (fontRendering.colorValue <= 0.0f)
        {
            bColorValueSet = false;
            sceneCounter = DETAILS_SCENE;

            fontRendering.colorValue = 0.0f;
        }
    }
}

void updateFontRenderingEndCredits(void)
{
    // Code
    if (fontRendering.textPositionY <= 70.0f)
    {
        fontRendering.textPositionY = fontRendering.textPositionY + fontRendering.TEXT_ANIMATION_SPEED;
    }

    fontRendering.colorValue = fontRendering.colorValue + 0.0008f;
    if (fontRendering.colorValue > 1.0f)
    {
        fontRendering.colorValue = 1.0f;
    }
}

void uninitializeFontRendering(void)
{
    fprintf(gpFile, "\n-[%s @%d] begin::uninitializeFontRendering()]\n", __FILE__, __LINE__);

    // Local Variable Declaration
    GLsizei ShaderCount;
    GLsizei index;
    GLuint *pShaders = NULL;

    // Code
    if (fontRendering.vao_font_rendering)
    {
        glDeleteBuffers(1, &fontRendering.vao_font_rendering);
        fontRendering.vao_font_rendering = 0;

        fprintf(gpFile, "\tinfo>> fontRendering.vao_font_rendering deleted successfully...\n");
    }

    if (fontRendering.vbo_font_rendering)
    {
        glDeleteBuffers(1, &fontRendering.vbo_font_rendering);
        fontRendering.vbo_font_rendering = 0;

        fprintf(gpFile, "\tinfo>> fontRendering.vbo_font_rendering deleted successfully...\n");
    }

    if (fontRendering.shaderProgramObject)
    {
        // Safe Shader Cleaup
        glUseProgram(fontRendering.shaderProgramObject);

        glGetProgramiv(fontRendering.shaderProgramObject, GL_ATTACHED_SHADERS, &ShaderCount);
        pShaders = (GLuint *)malloc(sizeof(GLuint) * ShaderCount);
        if (pShaders == NULL)
        {
            fprintf(gpFile, "\terror>> [%s@%d] malloc() failed inside uninitializeFontRendering()...\n", __FILE__, __LINE__);
        }

        glGetAttachedShaders(fontRendering.shaderProgramObject, ShaderCount, &ShaderCount, pShaders);
        for (index = 0; index < ShaderCount; index++)
        {
            glDetachShader(fontRendering.shaderProgramObject, pShaders[index]);
            glDeleteShader(pShaders[index]);
            pShaders[index] = 0;
        }

        free(pShaders);

        glDeleteShader(fontRendering.shaderProgramObject);
        fontRendering.shaderProgramObject = 0;

        glUseProgram(0);

        fprintf(gpFile, "\tinfo>> fontRendering.shaderProgramObject safe released successfully...\n");
    }

    fprintf(gpFile, "-[%s @%d] end::uninitializeFontRendering()]\n", __FILE__, __LINE__);
}
