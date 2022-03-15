#pragma once

#include "../header/commongl.h"

// initializeOpenAL() Definition
void initializeOpenAL(const char *sourceFileName, unsigned int *buffer, unsigned int *sourceFile, ALvoid *alData)
{
    fprintf(gpFile, "\n+[%s @%d] begin::initializeOpenAL()]\n", __FILE__, __LINE__);

    // Local Variable Declaration
    int channel, sampleRate, beatsPerSecond, size;
    device = alcOpenDevice(NULL);

    // Code
    if (device)
    {
        fprintf(gpFile, "\tinfo>> Device created!\n");
        context = alcCreateContext(device, NULL);

        if (context != NULL)
        {
            fprintf(gpFile, "\tinfo>> Context created!\n");
        }

        alcMakeContextCurrent(context);

        alGenBuffers(1, buffer);
        alGenSources(1, sourceFile);

        if (sourceFile != NULL)
        {
            fprintf(gpFile, "\tinfo>> sourceFile Generated!\n");
        }

        alData = loadWav(sourceFileName, &channel, &sampleRate, &beatsPerSecond, &size);
        if (alData)
        {
            fprintf(gpFile, "\tinfo>> Wave File Loaded!\n");
        }

        if (channel == 1)
        {
            if (beatsPerSecond == 8)
            {
                alFormat = AL_FORMAT_MONO8;
                fprintf(gpFile, "\tinfo>> mono8\n");
            }
            else
            {
                alFormat = AL_FORMAT_MONO16;
                fprintf(gpFile, "\tinfo>> mono16\n");
            }
        }
        else
        {
            if (beatsPerSecond == 8)
            {
                alFormat = AL_FORMAT_STEREO8;
                fprintf(gpFile, "\tinfo>> stereo8\n");
            }
            else
            {
                alFormat = AL_FORMAT_STEREO16;
                fprintf(gpFile, "\tinfo>> stereo16\n");
            }
        }

        alBufferData(*buffer, alFormat, alData, size, sampleRate);

        alSourcei(*sourceFile, AL_BUFFER, *buffer);
    }

    fprintf(gpFile, "+[%s @%d] end::initializeOpenAL()]\n", __FILE__, __LINE__);
}

// uninitializeOpenAL() Definition
void uninitializeOpenAL(void)
{
    // Code
    fprintf(gpFile, "+[%s @%d] begin::uninitializeOpenAL()]\n", __FILE__, __LINE__);
    fprintf(gpFile, "+[%s @%d] end::uninitializeOpenAL()]\n", __FILE__, __LINE__);
}
