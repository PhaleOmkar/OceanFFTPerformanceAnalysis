#pragma once

#include <vector>

#include "../header/commongl.h"
#include "./textureLoading.h"

GLuint vertexShaderObjectSphere;
GLuint fragmentShaderObjectSphere;
GLuint shaderProgramObjectSphere;

GLuint vao_sphere;
GLuint vbo_element_sphere;
GLuint vbo_position_sphere;
GLuint vbo_normal_sphere;
GLuint vbo_texture_sphere;

GLuint La_uniform_sphere;
GLuint Ld_uniform_sphere;
GLuint Ls_uniform_sphere;
GLuint gLightPositionUniform;
GLuint LightKeyPressedUniform;

GLuint Ka_uniform_sphere;
GLuint Kd_uniform_sphere;
GLuint Ks_uniform_sphere;
GLuint materialShininessUniformSphere;

GLuint textureSamplerUniformSphere;

GLuint modelMatrixUniformSphere;
GLuint viewMatrixUniformSphere;
GLuint projectionMatrixUniformSphere;

struct SphereData
{
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> texCoords;
    std::vector<unsigned int> indices;
    std::vector<unsigned int> lineIndices;
} sphereDataValues;

GLfloat moonRotationAngle = 0.0f;
GLuint gluiTextureImage = 0;

vec3 lightAmbientDiffuseVector = {0.0f, 0.0f, 0.0f};

void initializeMoonSphere(void)
{
    fprintf(gpFile, "\n+[%s @%d] begin::initializeMoonSphere()]\n", __FILE__, __LINE__);

    // Local Function Declaration
    bool LoadGLTexture(GLuint * gluiTexture, TCHAR szResourceID[]);
    void SphereRendering(void);

    // Local Variable Declaration
    GLint iInfoLogLength = 0;
    GLint iShaderCompiledStatus = 0;
    GLchar *szInfoLogBuffer = NULL;

    // Vertex Shader -  For Sphere - Vertices + Texture + Lights (Per Fragment)
    // Creating Shader
    vertexShaderObjectSphere = glCreateShader(GL_VERTEX_SHADER);
    const GLchar *pglcVertexShaderSourceCode =
        "#version 430 core																						        \n"
        "																										        \n"
        "in vec4 vPosition;																						        \n"
        "in vec3 vNormal;																						        \n"
        "in vec2 vTexCoord;																						        \n"
        "																										        \n"
        "uniform mat4 u_view_matrix;																			        \n"
        "uniform mat4 u_model_matrix;																			        \n"
        "uniform mat4 u_projection_matrix;																		        \n"
        "uniform vec4 u_light_position;																			        \n"
        "																										        \n"
        "out vec3 transformed_normals;																			        \n"
        "out vec3 light_direction;																				        \n"
        "out vec3 viewer_vector;																				        \n"
        "out vec2 out_vTexCoord;																				        \n"
        "																										        \n"
        "void main(void)																						        \n"
        "{																										        \n"
        "	int row;																							        \n"
        "	int column;																							        \n"
        "																										        \n"
        "	vec4 eye_coordinates = u_view_matrix* u_model_matrix * vPosition;								        	\n"
        "	transformed_normals = mat3(u_view_matrix * u_model_matrix) * vNormal;							        	\n"
        "	light_direction = vec3(u_light_position - eye_coordinates);										        	\n"
        "	viewer_vector = -eye_coordinates.xyz;	/* Swizzling - Using Individual Vector Components */	        	\n"
        "																										        \n"
        "	gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;						        \n"
        "	out_vTexCoord = vTexCoord;																			        \n"
        "}																										        \n";

    glShaderSource(vertexShaderObjectSphere, 1, (const GLchar **)&pglcVertexShaderSourceCode, NULL);

    // Compiling Shader
    glCompileShader(vertexShaderObjectSphere);

    // Error Checking
    glGetShaderiv(vertexShaderObjectSphere, GL_COMPILE_STATUS, &iShaderCompiledStatus);
    if (iShaderCompiledStatus == GL_FALSE)
    {
        glGetShaderiv(vertexShaderObjectSphere, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(vertexShaderObjectSphere, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "%s : \n\terror>> [Vertex Shader Compilation Error Log : %s]\n", __FILE__, szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    iInfoLogLength = 0;
    iShaderCompiledStatus = 0;
    szInfoLogBuffer = NULL;

    // Fragment Shader -  For Sphere - Vertices + Texture + Lights (Per Fragment)
    // Creating Shader
    fragmentShaderObjectSphere = glCreateShader(GL_FRAGMENT_SHADER);

    const GLchar *pglcFragmentShaderSourceCode =
        "#version 430 core																												\n"
        "																																\n"
        "in vec3 transformed_normals;																									\n"
        "in vec3 light_direction;																										\n"
        "in vec3 viewer_vector;																											\n"
        "in vec2 out_vTexCoord;																											\n"
        "																																\n"
        "uniform vec3 u_La;																												\n"
        "uniform vec3 u_Ld;																												\n"
        "uniform vec3 u_Ls;																												\n"
        "uniform vec3 u_Ka;																												\n"
        "uniform vec3 u_Kd;																												\n"
        "uniform sampler2D u_Texture_Sampler;																							\n"
        "																																\n"
        "out vec4 FragmentColor;																										\n"
        "																																\n"
        "void main(void)																												\n"
        "{																																\n"
        "	vec3 phong_ads_light;																										\n"
        "   vec4 color;                                                                                                                 \n"
        "																																\n"
        "	vec3 normalized_transformed_normals = normalize(transformed_normals);														\n"
        "	vec3 normalized_light_direction = normalize(light_direction);																\n"
        "	vec3 normalized_viewer_vector = normalize(viewer_vector);																	\n"
        "																																\n"
        "	vec3 v_ambient = u_La * u_Ka;																								\n"
        "	vec3 v_diffuse = u_Ld * u_Kd * max(dot(normalized_light_direction, normalized_transformed_normals), 0.0f);					\n"
        "	vec3 reflection_vector = reflect(-normalized_light_direction, normalized_transformed_normals);								\n"
        "																																\n"
        "	phong_ads_light = phong_ads_light + v_ambient + v_diffuse;																	\n"
        "   color = texture(u_Texture_Sampler, out_vTexCoord) * vec4(phong_ads_light, 1.0f);                                        	\n"
        "																																\n"
        "	FragmentColor = color;	                        								                                            \n"
        "}																																\n";

    glShaderSource(fragmentShaderObjectSphere, 1, (const GLchar **)&pglcFragmentShaderSourceCode, NULL);

    // Compiling Shader
    glCompileShader(fragmentShaderObjectSphere);

    // Error Checking
    glGetShaderiv(fragmentShaderObjectSphere, GL_COMPILE_STATUS, &iShaderCompiledStatus);
    if (iShaderCompiledStatus == GL_FALSE)
    {
        glGetShaderiv(fragmentShaderObjectSphere, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(fragmentShaderObjectSphere, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Fragment Shader Compilation Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    iInfoLogLength = 0;
    GLint iShaderLinkerStatus = 0;
    szInfoLogBuffer = NULL;

    // Shader Program -  For Sphere - Vertices + Texture + Lights (Per Fragment)
    // Create Shader Program
    shaderProgramObjectSphere = glCreateProgram();

    glAttachShader(shaderProgramObjectSphere, vertexShaderObjectSphere);   // Attach Vertex Shader To Shader Program
    glAttachShader(shaderProgramObjectSphere, fragmentShaderObjectSphere); // Attach Fragment Shader To Shader Program

    // Bind Vertex Shader Position Attribute
    glBindAttribLocation(shaderProgramObjectSphere, OUP_ATTRIBUTE_POSITION, "vPosition");
    glBindAttribLocation(shaderProgramObjectSphere, OUP_ATTRIBUTE_NORMAL, "vNormal");
    glBindAttribLocation(shaderProgramObjectSphere, OUP_ATTRIBUTE_TEXCOORD, "vTexCoord");

    // Link Shader Program
    glLinkProgram(shaderProgramObjectSphere);

    // Error Checking
    glGetProgramiv(shaderProgramObjectSphere, GL_LINK_STATUS, &iShaderLinkerStatus);
    if (iShaderLinkerStatus == GL_FALSE)
    {
        glGetShaderiv(shaderProgramObjectSphere, GL_INFO_LOG_LENGTH, &iInfoLogLength);
        if (iInfoLogLength > 0)
        {
            szInfoLogBuffer = (GLchar *)malloc(iInfoLogLength * sizeof(GLchar));
            if (szInfoLogBuffer != NULL)
            {
                GLsizei written;

                glGetShaderInfoLog(shaderProgramObjectSphere, iInfoLogLength, &written, szInfoLogBuffer);

                fprintf(gpFile, "\n\terror>> [Shader Program Linking Error Log : %s]\n", szInfoLogBuffer);
                free(szInfoLogBuffer);
                DestroyWindow(ghwnd);
            }
        }
    }

    // Get Uniform Location
    viewMatrixUniformSphere = glGetUniformLocation(shaderProgramObjectSphere, "u_view_matrix");
    modelMatrixUniformSphere = glGetUniformLocation(shaderProgramObjectSphere, "u_model_matrix");
    projectionMatrixUniformSphere = glGetUniformLocation(shaderProgramObjectSphere, "u_projection_matrix");

    La_uniform_sphere = glGetUniformLocation(shaderProgramObjectSphere, "u_La");
    Ld_uniform_sphere = glGetUniformLocation(shaderProgramObjectSphere, "u_Ld");
    Ls_uniform_sphere = glGetUniformLocation(shaderProgramObjectSphere, "u_Ls");
    gLightPositionUniform = glGetUniformLocation(shaderProgramObjectSphere, "u_light_position");

    Ka_uniform_sphere = glGetUniformLocation(shaderProgramObjectSphere, "u_Ka");
    Kd_uniform_sphere = glGetUniformLocation(shaderProgramObjectSphere, "u_Kd");

    textureSamplerUniformSphere = glGetUniformLocation(shaderProgramObjectSphere, "u_Texture_Sampler");

    SphereRendering();

    // vao_sphere
    glGenVertexArrays(1, &vao_sphere);
    glBindVertexArray(vao_sphere);
    // vbo_position_sphere
    glGenBuffers(1, &vbo_position_sphere);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position_sphere);
    glBufferData(GL_ARRAY_BUFFER, sphereDataValues.vertices.size() * sizeof(float), sphereDataValues.vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(OUP_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_ATTRIBUTE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // vbo_normal_sphere
    glGenBuffers(1, &vbo_normal_sphere);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal_sphere);
    glBufferData(GL_ARRAY_BUFFER, sphereDataValues.normals.size() * sizeof(float), sphereDataValues.normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(OUP_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_ATTRIBUTE_NORMAL);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // vbo_texture_sphere
    glGenBuffers(1, &vbo_texture_sphere);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_texture_sphere);
    glBufferData(GL_ARRAY_BUFFER, sphereDataValues.texCoords.size() * sizeof(float), sphereDataValues.texCoords.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(OUP_ATTRIBUTE_TEXCOORD, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(OUP_ATTRIBUTE_TEXCOORD);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // vbo_element_sphere
    glGenBuffers(1, &vbo_element_sphere);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_element_sphere);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereDataValues.indices.size() * sizeof(unsigned int), sphereDataValues.indices.data(), GL_STATIC_DRAW);
    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Loading Texture
    LoadGLTexture(&gluiTextureImage, MAKEINTRESOURCE(MOON_IMAGE_TEXTURE));

    fprintf(gpFile, "+[%s @%d] end::initializeMoonSphere()]\n", __FILE__, __LINE__);
}

// SphereRendering() Definition
void SphereRendering(void)
{
    fprintf(gpFile, "\tinfo>> SphereRendering() started\n");

    // Local Variable Declaration
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> texCoords;
    std::vector<unsigned int> indices;
    std::vector<unsigned int> lineIndices;

    int stackCount = 100;
    int sectorCount = 100;
    float radius = 1.0f;

    // Code
    vertices.clear();
    normals.clear();
    texCoords.clear();

    std::vector<float>().swap(vertices);
    std::vector<float>().swap(normals);
    std::vector<float>().swap(texCoords);

    float x, y, z, xy;                           // vertex position
    float nx, ny, nz, lengthInv = 1.0f / radius; // vertex normal
    float s, t;                                  // vertex texCoord

    float sectorStep = 2 * M_PI / sectorCount;
    float stackStep = M_PI / stackCount;
    float sectorAngle, stackAngle;

    for (int i = 0; i <= stackCount; ++i)
    {
        stackAngle = M_PI / 2 - i * stackStep; // starting from M_PI/2 to -pi/2
        xy = radius * cosf(stackAngle);        // r * cos(u)
        z = radius * sinf(stackAngle);         // r * sin(u)

        // add (sectorCount + 1) vertices per stack
        // the first and last vertices have same position and normal, but different tex coords
        for (int j = 0; j <= sectorCount; ++j)
        {
            sectorAngle = j * sectorStep; // starting from 0 to 2pi

            // vertex position (x, y, z)
            x = xy * cosf(sectorAngle); // r * cos(u) * cos(v)
            y = xy * sinf(sectorAngle); // r * cos(u) * sin(v)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            // normalized vertex normal (nx, ny, nz)
            nx = x * lengthInv;
            ny = y * lengthInv;
            nz = z * lengthInv;
            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);

            // vertex tex coord (s, t) range between [0, 1]
            s = (float)j / sectorCount;
            t = (float)i / stackCount;

            texCoords.push_back(s);
            texCoords.push_back(t);
        }
    }

    int k1, k2;
    for (int i = 0; i < stackCount; ++i)
    {
        k1 = i * (sectorCount + 1); // beginning of current stack
        k2 = k1 + sectorCount + 1;  // beginning of next stack

        for (int j = 0; j < sectorCount; ++j, ++k1, ++k2)
        {
            // 2 triangles per sector excluding first and last stacks
            // k1 => k2 => k1+1
            if (i != 0)
            {
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);
            }

            // k1+1 => k2 => k2+1
            if (i != (stackCount - 1))
            {
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }

            // store indices for lines
            // vertical lines for all stacks, k1 => k2
            lineIndices.push_back(k1);
            lineIndices.push_back(k2);
            if (i != 0) // horizontal lines except 1st stack, k1 => k+1
            {
                lineIndices.push_back(k1);
                lineIndices.push_back(k1 + 1);
            }
        }
    }

    sphereDataValues.vertices = vertices;
    sphereDataValues.normals = normals;
    sphereDataValues.texCoords = texCoords;
    sphereDataValues.indices = indices;
    sphereDataValues.lineIndices = lineIndices;

    fprintf(gpFile, "\tinfo>> SphereRendering() finished\n");
}

void displayMoonSphere(void)
{
    glUseProgram(shaderProgramObjectSphere);

    // Setting Light Properties
    glUniform3fv(La_uniform_sphere, 1, vec3(0.0f, 0.0f, 0.0f));
    glUniform3fv(Ld_uniform_sphere, 1, lightAmbientDiffuseVector);
    glUniform3fv(Ls_uniform_sphere, 1, lightAmbientDiffuseVector);
    // glUniform4fv(gLightPositionUniform, 1, vec3(-25.0f, -25.0f, 0.0f));
    glUniform4fv(gLightPositionUniform, 1, vec3(-50.0f, -100.0f, 0.0f));

    // Setting Material Properties
    glUniform3fv(Ka_uniform_sphere, 1, vec3(0.0f, 0.0f, 0.0f));
    glUniform3fv(Kd_uniform_sphere, 1, vec3(1.0f, 1.0f, 1.0f));

    mat4 modelMatrix = mat4::identity();
    mat4 viewMatrix = mat4::identity();

    // translateX = 2.539998, translateY = 1.379999
    // modelMatrix = modelMatrix * vmath::translate(translateX * 10.0f, translateY * 10.0f, -10.0f);
    modelMatrix = modelMatrix * vmath::translate(25.4f, 13.8f, -10.0f);
    modelMatrix = modelMatrix * vmath::scale(1.0f, 1.0f, 1.0f);
    modelMatrix = modelMatrix * vmath::rotate(moonRotationAngle, 0.0f, 1.0f, 0.0f);
    modelMatrix = modelMatrix * vmath::rotate(90.0f, 1.0f, 0.0f, 0.0f);

    glUniformMatrix4fv(viewMatrixUniformSphere, 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(modelMatrixUniformSphere, 1, GL_FALSE, modelMatrix);

    static mat4 orthographicProjectionMatrix;
    if (Info.WindowWidth <= Info.WindowHeight)
    {
        orthographicProjectionMatrix = vmath::ortho(0.0f, 15.5f, 0.0f, 15.5f * ((GLfloat)Info.WindowHeight / (GLfloat)Info.WindowWidth), -10.0f, 10.0f);
    }
    else
    {
        orthographicProjectionMatrix = vmath::ortho(0.0f, 15.5f * ((GLfloat)Info.WindowWidth / (GLfloat)Info.WindowHeight), 0.0f, 15.5f, -10.0f, 10.0f);
    }
    glUniformMatrix4fv(projectionMatrixUniformSphere, 1, GL_FALSE, orthographicProjectionMatrix);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gluiTextureImage);
    glUniform1i(textureSamplerUniformSphere, 1);

    glBindVertexArray(vao_sphere);
    glDrawElements(GL_TRIANGLES, sphereDataValues.indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

void updateMoonSphere(void)
{
    // Code
    moonRotationAngle = moonRotationAngle + 0.025f;
    if (moonRotationAngle >= 360.0f)
    {
        moonRotationAngle = 0.0f;
    }

    if (lightAmbientDiffuseVector[0] <= 1.0f)
    {
        lightAmbientDiffuseVector[0] += 0.001f;
    }

    if (lightAmbientDiffuseVector[1] <= 1.0f)
    {
        lightAmbientDiffuseVector[1] += 0.001f;
    }

    if (lightAmbientDiffuseVector[2] <= 1.0f)
    {
        lightAmbientDiffuseVector[2] += 0.001f;
    }
}

void uninitializeMoonSphere(void)
{
    // Code
    fprintf(gpFile, "\n-[%s @%d] begin::uninitializeMoonSphere()]\n", __FILE__, __LINE__);

    // Local Variable Declaration
    GLsizei ShaderCount;
    GLsizei index;
    GLuint *pShaders = NULL;

    // Code
    if (sphereDataValues.vertices.data())
    {
        sphereDataValues.vertices.clear();

        fprintf(gpFile, "\tinfo>> sphereDataValues.vertices deleted successfully...\n");
    }

    if (sphereDataValues.normals.data())
    {
        sphereDataValues.normals.clear();

        fprintf(gpFile, "\tinfo>> sphereDataValues.normals deleted successfully...\n");
    }

    if (sphereDataValues.texCoords.data())
    {
        sphereDataValues.texCoords.clear();

        fprintf(gpFile, "\tinfo>> sphereDataValues.texCoords deleted successfully...\n");
    }

    if (sphereDataValues.indices.data())
    {
        sphereDataValues.indices.clear();

        fprintf(gpFile, "\tinfo>> sphereDataValues.indices deleted successfully...\n");
    }

    if (sphereDataValues.lineIndices.data())
    {
        sphereDataValues.lineIndices.clear();

        fprintf(gpFile, "\tinfo>> sphereDataValues.lineIndices deleted successfully...\n");
    }

    if (vao_sphere)
    {
        glDeleteBuffers(1, &vao_sphere);
        vao_sphere = 0;

        fprintf(gpFile, "\tinfo>> vao_sphere deleted successfully...\n");
    }

    if (vbo_element_sphere)
    {
        glDeleteBuffers(1, &vbo_element_sphere);
        vbo_element_sphere = 0;

        fprintf(gpFile, "\tinfo>> vbo_element_sphere deleted successfully...\n");
    }

    if (vbo_position_sphere)
    {
        glDeleteBuffers(1, &vbo_position_sphere);
        vbo_position_sphere = 0;

        fprintf(gpFile, "\tinfo>> vbo_position_sphere deleted successfully...\n");
    }

    if (vbo_normal_sphere)
    {
        glDeleteBuffers(1, &vbo_normal_sphere);
        vbo_normal_sphere = 0;

        fprintf(gpFile, "\tinfo>> vbo_normal_sphere deleted successfully...\n");
    }

    if (vbo_texture_sphere)
    {
        glDeleteBuffers(1, &vbo_texture_sphere);
        vbo_texture_sphere = 0;

        fprintf(gpFile, "\tinfo>> vbo_texture_sphere deleted successfully...\n");
    }

    if (gluiTextureImage)
    {
        glDeleteTextures(1, &gluiTextureImage);
        gluiTextureImage = 0;

        fprintf(gpFile, "\tinfo>> gluiTextureImage deleted successfully...\n");
    }

    if (shaderProgramObjectSphere)
    {
        // Safe Shader Cleaup
        glUseProgram(shaderProgramObjectSphere);

        glGetProgramiv(shaderProgramObjectSphere, GL_ATTACHED_SHADERS, &ShaderCount);
        pShaders = (GLuint *)malloc(sizeof(GLuint) * ShaderCount);
        if (pShaders == NULL)
        {
            fprintf(gpFile, "\terror>> [%s@%d] malloc() failed inside uninitializeFontRendering()...\n", __FILE__, __LINE__);
        }

        glGetAttachedShaders(shaderProgramObjectSphere, ShaderCount, &ShaderCount, pShaders);
        for (index = 0; index < ShaderCount; index++)
        {
            glDetachShader(shaderProgramObjectSphere, pShaders[index]);
            glDeleteShader(pShaders[index]);
            pShaders[index] = 0;
        }

        free(pShaders);

        glDeleteShader(shaderProgramObjectSphere);
        shaderProgramObjectSphere = 0;

        glUseProgram(0);

        fprintf(gpFile, "\tinfo>> shaderProgramObjectSphere safe released successfully...\n");
    }

    fprintf(gpFile, "-[%s @%d] end::uninitializeMoonSphere()]\n", __FILE__, __LINE__);
}
