// Header File Declaration
#include "../header/commongl.h"

#include "./fontRendering.h"
#include "./detailsTexture.h"
#include "./oceanFFTSceneFunctions.h"
#include "./starfield.h"
#include "./moonSphere.h"
#include "./textureLoading.h"
#include "./backgroundMusicFunctions.h"

// Global Function Declaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// Global Macro Definitions
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

// Global Variable Declaration
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = {sizeof(WINDOWPLACEMENT)};

extern HWND ghwnd = NULL;
extern FILE *gpFile;
HDC ghdc = NULL;
HGLRC ghrc = NULL;

bool gbFullScreen = false;
bool gbActiveWindow = false;
bool gbUpdateRendering = false;

// WinMain() Defintion
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	// Local Funtion Declaration
	void Initialize(void);
	void ToggleFullScreen(void);
	void Display(void);
	void Update(void);

	// Local Variable Declaration
	WNDCLASSEX wndclassex;
	HWND hwnd;
	MSG message;
	TCHAR szAppName[] = TEXT("SeminarProject");
	RECT rect;
	BOOL bResult;
	int iCenterX = 0;
	int iCenterY = 0;
	bool bDone = false;

	// Code
	// Error Checking Of 'fopen_s()'
	if (fopen_s(&gpFile, "CodeExecutionLog.txt", "w") != 0)
	{
		MessageBox(NULL,
				   TEXT("Cannot open desired file"),
				   TEXT("Message::fopen_s() failed"),
				   MB_ICONERROR);
		exit(0);
	}
	else
	{
		fprintf(gpFile, "====================================================================\n");
		fprintf(gpFile, "Project Is Started Successfully\n");
	}

	// Getting System Parameters Info
	bResult = SystemParametersInfo(
		SPI_GETWORKAREA,
		0,
		&rect,
		0);

	// Error Checking :: bResult
	if (bResult == TRUE)
	{
		iCenterX = ((int)(rect.left + rect.right) - (int)(WINDOW_WIDTH)) / 2;
		iCenterY = ((int)(rect.top + rect.bottom) - (int)(WINDOW_HEIGHT)) / 2;
	}

	// Initialization Of WNDCLASSEX
	wndclassex.cbSize = sizeof(WNDCLASSEX);
	wndclassex.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclassex.cbClsExtra = 0;
	wndclassex.cbWndExtra = 0;
	wndclassex.lpfnWndProc = WndProc;
	wndclassex.hInstance = hInstance;
	wndclassex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(DEFAULT_LARGE_ICON));
	wndclassex.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclassex.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclassex.lpszClassName = szAppName;
	wndclassex.lpszMenuName = NULL;
	wndclassex.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(DEFAULT_LARGE_ICON));

	// Register Above Class
	RegisterClassEx(&wndclassex);

	// Create Window
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
						  szAppName,
						  TEXT("HPP 2022 : OpenGL & CUDA Interop"),
						  WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
						  iCenterX,
						  iCenterY,
						  (int)WINDOW_WIDTH,
						  (int)WINDOW_HEIGHT,
						  NULL,
						  NULL,
						  hInstance,
						  NULL);

	// Copying Window Handle To Global Handle
	ghwnd = hwnd;

	// Initialize() Called
	Initialize();

	// // ToggleFullScreen() Called
	ToggleFullScreen();

	// Show Window
	ShowWindow(hwnd, iCmdShow);

	// Setting Fore Ground Window and Focus
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	// Game Loop
	while (bDone == false)
	{
		if (PeekMessage(&message, NULL, 0, 0, PM_REMOVE))
		{
			if (message.message == WM_QUIT)
			{
				bDone = true;
			}

			else
			{
				TranslateMessage(&message);
				DispatchMessage(&message);
			}
		}

		else
		{
			if (gbActiveWindow == true)
			{
				if (gbUpdateRendering == true)
				{
					// Update()
					Update();
				}

				// Display()
				Display();
			}
		}
	}

	return ((int)message.wParam);
}

// WndProc() Definition
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMessage, WPARAM wParam, LPARAM lParam)
{
	// Function Declaration
	void ToggleFullScreen(void);
	void Resize(int, int);
	void Uninitialize(void);

	// Local Variable Declaration
	GLfloat movement_value = 0.01f;
	// char stringMessage[256];
	static int count = 0;

	// Code
	switch (iMessage)
	{
	case WM_SETFOCUS:
		gbActiveWindow = true;
		break;

	case WM_KILLFOCUS:
		gbActiveWindow = false;
		break;

	case WM_ERASEBKGND:
		return (0);

	case WM_SIZE:
		Info.WindowWidth = LOWORD(lParam);
		Info.WindowHeight = HIWORD(lParam);
		Resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		case 0x46:
		case 0x66:
			ToggleFullScreen();
			break;

		case VK_NUMPAD0:
			sceneCounter = INTRO_SCENE;
			break;

		case VK_NUMPAD1:
			sceneCounter = DETAILS_SCENE;
			break;

		case VK_NUMPAD2:
			sceneCounter = OCEANFFT_SCENE;
			break;

		case VK_NUMPAD3:
			sceneCounter = OPENGL_CUDA_SCENE;
			break;

		case VK_NUMPAD4:
			sceneCounter = END_CREDITS_SCENE;
			break;

		case VK_SPACE:
			sceneCounter = END_CREDITS_SCENE;
			break;

		case VK_UP:
			translateY = translateY + movement_value;
			break;

		case VK_DOWN:
			translateY = translateY - movement_value;
			break;

		case VK_LEFT:
			translateX = translateX + movement_value;
			break;

		case VK_RIGHT:
			translateX = translateX - movement_value;
			break;

		case VK_ADD:
			meshSizeLimit = meshSizeLimit * 2;
			if (meshSizeLimit >= MESH_SIZE)
			{
				meshSizeLimit = MESH_SIZE;
			}
			spectrumW = meshSizeLimit + 4;
			spectrumH = meshSizeLimit + 1;
			gbNeedToUpdate = true;
			break;

		case VK_SUBTRACT:
			meshSizeLimit = meshSizeLimit / 2;
			if (meshSizeLimit <= 2)
			{
				meshSizeLimit = 2;
			}
			spectrumW = meshSizeLimit + 4;
			spectrumH = meshSizeLimit + 1;
			gbNeedToUpdate = true;
			break;

		default:
			break;
		}

		break;

	case WM_CHAR:
		switch (wParam)
		{
		case 'p':
		case 'P':
			gbUpdateRendering ^= 1;
			break;

		case 'g':
		case 'G':
			gbOnRunGPU ^= 1;
			break;

		case 'x':
		case 'X':
			currectRotationAxis = ALONG_POSITIVE_X_AXIS;
			break;

		case 'y':
		case 'Y':
			currectRotationAxis = ALONG_POSITIVE_Y_AXIS;
			break;

		case 'z':
		case 'Z':
			currectRotationAxis = 3;
			fprintf(gpFile, "\tinfo>> %d] rotateX = %lf, rotateY = %lf, translateX = %lf, translateY = %lf, translateZ = %lf\n", ++count, rotateAlongX, rotateAlongY, translateX, translateY, translateZ);
			break;

		case 'e':
		case 'E':
			currectRotationAxis = ALONG_NEGATIVE_X_AXIS;
			break;

		case 'r':
		case 'R':
			currectRotationAxis = ALONG_NEGATIVE_Y_AXIS;
			break;

		case 'u':
		case 'U':
			translateZ = translateZ - movement_value;
			break;

		case 'i':
		case 'I':
			translateZ = translateZ + movement_value;
			break;

		case 'w':
		case 'W':
			gbWireFrame = !gbWireFrame;
			break;

		default:
			break;
		}

		// sprintf(stringMessage, "%s %d * %d", (gbOnRunGPU == true ? "GPU" : "CPU"), meshSizeLimit, meshSizeLimit);
		// SetWindowTextA(hwnd, stringMessage);
		break;

	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_DESTROY:
		Uninitialize();
		PostQuitMessage(0);
		break;
	}

	return (DefWindowProc(hwnd, iMessage, wParam, lParam));
}

// ToggleFullScreen() Definition
void ToggleFullScreen(void)
{
	// Local Variable Declarations
	MONITORINFO monitorinfo = {sizeof(MONITORINFO)};

	// Code
	if (gbFullScreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);

		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &monitorinfo))
			{
				SetWindowLong(ghwnd, GWL_STYLE, (dwStyle & ~WS_OVERLAPPEDWINDOW));
				SetWindowPos(ghwnd,
							 HWND_TOP,
							 monitorinfo.rcMonitor.left,
							 monitorinfo.rcMonitor.top,
							 (monitorinfo.rcMonitor.right - monitorinfo.rcMonitor.left),
							 (monitorinfo.rcMonitor.bottom - monitorinfo.rcMonitor.top),
							 SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}

		ShowCursor(FALSE);
		gbFullScreen = true;
	}

	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, (dwStyle | WS_OVERLAPPEDWINDOW));
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
					 HWND_TOP,
					 0,
					 0,
					 0,
					 0,
					 SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		gbFullScreen = false;
	}
}

// Initialize() Definition
void Initialize(void)
{
	// Local Function Declaration
	void Resize(int, int);

	// Local Variable Declaration
	PIXELFORMATDESCRIPTOR pixelformatdescriptor;
	int iPixelFormatIndex;
	int iIndex;

	GLenum glew_error;
	GLint numberOfExtensions;

	// Code
	ghdc = GetDC(ghwnd);
	ZeroMemory(&pixelformatdescriptor, sizeof(PIXELFORMATDESCRIPTOR));

	// Iniitialization of PIXELFORMATDESCRIPTOR
	pixelformatdescriptor.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pixelformatdescriptor.nVersion = 1;
	pixelformatdescriptor.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pixelformatdescriptor.iPixelType = PFD_TYPE_RGBA;
	pixelformatdescriptor.cColorBits = 32;
	pixelformatdescriptor.cRedBits = 8;
	pixelformatdescriptor.cGreenBits = 8;
	pixelformatdescriptor.cBlueBits = 8;
	pixelformatdescriptor.cAlphaBits = 8;
	pixelformatdescriptor.cDepthBits = 32;

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pixelformatdescriptor);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "\terror>> ChoosePixelFormat() failed...\n");
		DestroyWindow(ghwnd);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pixelformatdescriptor) == FALSE)
	{
		fprintf(gpFile, "\terror>> SetPixelFormat() failed...\n");
		DestroyWindow(ghwnd);
	}

#pragma region CONTEXT_SETUP
	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "\terror>> wglCreateContext() failed...\n");
		DestroyWindow(ghwnd);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "\terror>> wglMakeCurrent() failed...\n");
		DestroyWindow(ghwnd);
	}

	glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		DestroyWindow(ghwnd);
	}
#pragma endregion

#pragma region LOG_PRINTING
	// Log Generation Time
	char *pcDayOfTheWeek = NULL;
	SYSTEMTIME stSystemTime;

	ZeroMemory(&stSystemTime, sizeof(stSystemTime));
	GetLocalTime(&stSystemTime);
	switch (stSystemTime.wDayOfWeek)
	{
	case 0:
		pcDayOfTheWeek = "Sunday";
		break;
	case 1:
		pcDayOfTheWeek = "Monday";
		break;
	case 2:
		pcDayOfTheWeek = "Tuesday";
		break;
	case 3:
		pcDayOfTheWeek = "Wednesday";
		break;
	case 4:
		pcDayOfTheWeek = "Thursday";
		break;
	case 5:
		pcDayOfTheWeek = "Friday";
		break;
	case 6:
		pcDayOfTheWeek = "Saturday";
		break;
	default:
		break;
	}

	// OpenGL Related Specification Log
	fprintf(gpFile, "%s - %2.2d/%2.2d/%4d, %2.2d:%2.2d:%2.2d:%2.3d\n",
			pcDayOfTheWeek, stSystemTime.wDay, stSystemTime.wMonth, stSystemTime.wYear, stSystemTime.wHour, stSystemTime.wMinute, stSystemTime.wSecond, stSystemTime.wMilliseconds);

	fprintf(gpFile, "====================================================================\n");
	fprintf(gpFile, "===              OpenGL Related Specification Log                ===\n");
	fprintf(gpFile, "====================================================================\n");
	fprintf(gpFile, "|## OpenGL Vendor   : %s\n", glGetString(GL_VENDOR));
	fprintf(gpFile, "|## OpenGL Renderer : %s\n", glGetString(GL_RENDERER));
	fprintf(gpFile, "|## OpenGL Version  : %s\n", glGetString(GL_VERSION));
	fprintf(gpFile, "|## GLSL Version    : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	fprintf(gpFile, "====================================================================\n");

	// OpenGL Enable Extensions
	fprintf(gpFile, "\n====================================================================\n");
	fprintf(gpFile, "===                 OpenGL Supported Extensions                  ===\n");
	fprintf(gpFile, "====================================================================\n");
	glGetIntegerv(GL_NUM_EXTENSIONS, &numberOfExtensions);
	for (iIndex = 0; iIndex < numberOfExtensions; iIndex++)
	{
		fprintf(gpFile, "%3.2d] %s\n", iIndex, glGetStringi(GL_EXTENSIONS, iIndex));
	}
	fprintf(gpFile, "====================================================================");
#pragma endregion

	initializeAstronautPicture();
	initializeOpenAL("./res/always.wav", &bufferAlways, &sourceAlways, alDataAlways);
	initializeFontRendering();
	initializeDetailsTexture();
	initializeOcean();
	initializeStarfield();
	initializeMoonSphere();
	// openALInitialize("./res/shanti_mantra.wav", &bufferShantiMantra, &sourceShantiMantra, alDataShantiMantra);

	// Depth Settings
	glClearDepth(1.0f);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_CULL_FACE);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Set Clear Color
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	perspectiveProjectionMatrix = mat4::identity();

	Resize(WINDOW_WIDTH, WINDOW_HEIGHT); // Warm Up Call To Resize()
}

// Display() Definition
void Display(void)
{
	// Code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (sceneCounter == INTRO_SCENE)
	{
		displayIntroCredits();
	}

	if (sceneCounter == DETAILS_SCENE)
	{
		displayDetailsTexture();
	}

	if (sceneCounter == OCEANFFT_SCENE || sceneCounter == OPENGL_CUDA_SCENE)
	{
		displayOcean();
	}

	if (sceneCounter == OPENGL_CUDA_SCENE)
	{
		displayStarfield();

		displayMoonSphere();
	}

	if (sceneCounter == END_CREDITS_SCENE)
	{
		displayEndCredits();
	}

	displayAstronautPicture();

	SwapBuffers(ghdc);
}

// Update() Definition
void Update(void)
{
	// Code
	static bool bPlayAlways = false;
	static bool bPlayShantiMantra = false;

	if ((sceneCounter == OCEANFFT_SCENE) && (bPlayAlways == false))
	{
		alSourcePlay(sourceAlways);

		bPlayAlways = true;
	}

	if ((sceneCounter == OPENGL_CUDA_SCENE) && (bPlayShantiMantra == false))
	{
		alDeleteSources(1, &sourceAlways);
		alDeleteBuffers(1, &bufferAlways);

		PlaySound(MAKEINTRESOURCE(BACKGROUND_SOUND), NULL, SND_ASYNC | SND_RESOURCE);
		bPlayShantiMantra = true;
	}

	if (sceneCounter == INTRO_SCENE)
	{
		updateFontRenderingIntroCredits();
	}

	if (sceneCounter == DETAILS_SCENE)
	{
		updateDetailsTexture();
	}

	if (sceneCounter == OCEANFFT_SCENE || sceneCounter == OPENGL_CUDA_SCENE)
	{
		updateOcean();
	}

	if (sceneCounter == OPENGL_CUDA_SCENE)
	{
		updateStarfield();

		updateMoonSphere();
	}

	if (sceneCounter == END_CREDITS_SCENE)
	{
		updateFontRenderingEndCredits();
	}
}

// Resize() Definition
void Resize(int iWidth, int iHeight)
{
	// Code
	if (iHeight == 0)
	{
		iHeight = 1;
	}

	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);

	perspectiveProjectionMatrix = vmath::perspective(45.0f,
													 (GLfloat)iWidth / (GLfloat)iHeight,
													 0.1f,
													 1000.0f);
}

// Unintialize() Definition
void Uninitialize(void)
{
	// Local Function Declaration
	void ToggleFullScreen(void);

	// Code
	if (gbFullScreen == true)
	{
		ToggleFullScreen();
	}

	if (wglGetCurrentContext() == ghrc)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	uninitializeMoonSphere();
	uninitializeStarfield();
	uninitializeOcean();
	uninitializeDetailsTexture();
	uninitializeFontRendering();
	uninitializeOpenAL();
	uninitializeAstronautPicture();

	// Closing 'gpFile'
	if (gpFile)
	{
		fprintf(gpFile, "====================================================================\n");

		// Log Generation Time
		char *pcDayOfTheWeek = NULL;
		SYSTEMTIME stSystemTime;

		ZeroMemory(&stSystemTime, sizeof(stSystemTime));
		GetLocalTime(&stSystemTime);
		switch (stSystemTime.wDayOfWeek)
		{
		case 0:
			pcDayOfTheWeek = "Sunday";
			break;
		case 1:
			pcDayOfTheWeek = "Monday";
			break;
		case 2:
			pcDayOfTheWeek = "Tuesday";
			break;
		case 3:
			pcDayOfTheWeek = "Wednesday";
			break;
		case 4:
			pcDayOfTheWeek = "Thursday";
			break;
		case 5:
			pcDayOfTheWeek = "Friday";
			break;
		case 6:
			pcDayOfTheWeek = "Saturday";
			break;
		default:
			break;
		}

		fprintf(gpFile, "Project Is Terminated Successfully\n");
		fprintf(gpFile, "%s - %2.2d/%2.2d/%4d, %2.2d:%2.2d:%2.2d:%2.3d\n",
				pcDayOfTheWeek, stSystemTime.wDay, stSystemTime.wMonth, stSystemTime.wYear, stSystemTime.wHour, stSystemTime.wMinute, stSystemTime.wSecond, stSystemTime.wMilliseconds);

		fprintf(gpFile, "====================================================================\n");
		fprintf(gpFile, "Project By : Omkar Phale\n");

		fclose(gpFile);
		gpFile = NULL;
	}
}
