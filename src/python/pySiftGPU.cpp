#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <optional>
#include <vector>


///////////////////////////////////////////////////////////////////////////////
// C API : similar to cv2::SIFT
///////////////////////////////////////////////////////////////////////////////
void create(int nfeatures, int nOctaveLayers, 
			float contrastThreshold, float edgeThreshold, bool existGL);
py::array detectAndCompute(const py::array& image);
void empty(); // destructor?


///////////////////////////////////////////////////////////////////////////////
// unnamed namespace for data sealing
///////////////////////////////////////////////////////////////////////////////

//define this to get dll import definition for win32
#define SIFTGPU_DLL_RUNTIME

#ifdef _WIN32
    #ifdef SIFTGPU_DLL_RUNTIME
        #define WIN32_LEAN_AND_MEAN
        #include <windows.h>
        #define FREE_MYLIB FreeLibrary
        #define GET_MYPROC GetProcAddress
    #else
        //define this to get dll import definition for win32
        #define SIFTGPU_DLL
        #ifdef _DEBUG 
            #pragma comment(lib, "../../lib/siftgpu_d.lib")
        #else
            #pragma comment(lib, "../../lib/siftgpu.lib")
        #endif
    #endif
#else
    #ifdef SIFTGPU_DLL_RUNTIME
        #include <dlfcn.h>
        #define FREE_MYLIB dlclose
        #define GET_MYPROC dlsym
    #endif
#endif


///////////////////////////////////////////////////////////////////////////////
// unnamed namespace for data sealing
///////////////////////////////////////////////////////////////////////////////
#include "../SiftGPU/SiftGPU.h"
#include <GL/GL.h>

namespace {
HMODULE  hsiftgpu;
SiftGPU* sift;

int   _nfeatures;
bool  _init_sift = false;
bool  _existGL;

HDC   g_hdc;
HGLRC g_hglrc;
HGLRC m_hglrc;
}


///////////////////////////////////////////////////////////////////////////////
// C API
///////////////////////////////////////////////////////////////////////////////
void create(int nfeatures, int nOctaveLayers,
			float contrastThreshold, float edgeThreshold, bool existGL)
{
	empty();

	// append values
	_nfeatures = nfeatures;
	_existGL   = existGL;

	// load SiftGPU.dll
	hsiftgpu = LoadLibrary("SiftGPU.dll");
	if (hsiftgpu == NULL) {
		py::print("abort: cannot find SiftGPU.dll");
		return;
	}

	SiftGPU* (*pCreateNewSiftGPU)(int) = NULL;
	SiftMatchGPU* (*pCreateNewSiftMatchGPU)(int) = NULL;
	pCreateNewSiftGPU = (SiftGPU* (*) (int)) GET_MYPROC(hsiftgpu, "CreateNewSiftGPU");

	sift = pCreateNewSiftGPU(1);

	// set arguments
	std::vector<char*> argv;

	argv.push_back((char*)"-fo"); // feature octave
	argv.push_back((char*)"-1");
	
	argv.push_back((char*)"-v"); // verbose
	argv.push_back((char*)"0");

	int argc = (int)argv.size();
	sift->ParseParam(argc, &argv[0]);

	_init_sift = true;

	// set rendering context for GPUSift
	if (_existGL) {
		// preserve & create GL rendering context
		g_hdc   = wglGetCurrentDC();
		g_hglrc = wglGetCurrentContext();
		m_hglrc = wglCreateContext(g_hdc);

		wglMakeCurrent(g_hdc, m_hglrc); // set GL context
		if (sift->VerifyContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
		{
			_init_sift = false;
			py::print("abort: SIFTGPU Initialization failed..\n");
		}
		wglMakeCurrent(g_hdc, g_hglrc); // reset GL context
	}
	else {
		if (sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
		{
			_init_sift = false;
			py::print("abort: SIFTGPU Initialization failed..\n");
		}
	}
}

py::array detectAndCompute(const py::array& image)
{
	if (!_init_sift)
	{
		py::print("abort: GPUSift is not initialized.");
		return make_tuple(py::none(), py::none());
	}

	// check the image size
	size_t image_dim = image.ndim();
	if (image_dim < 2 || 3 < image_dim) {
		py::print("abort: unsupported image format");
		return make_tuple(py::none(), py::none());
	}

	size_t H = image.shape(0);
	size_t W = image.shape(1);

	unsigned int gl_format = GL_LUMINANCE;
	if (image_dim == 3) {
		gl_format = GL_BGR_EXT;
		py::print("abort: color image is not supported.");
		return make_tuple(py::none(), py::none());
	}

	// check the data type
	unsigned int gl_type;

	auto uint8   = py::dtype::of<unsigned char>();
	auto uint16  = py::dtype::of<unsigned short>();
	auto float32 = py::dtype::of<float>();

	if (uint8.is(image.dtype())) {
		gl_type = GL_UNSIGNED_BYTE;
	}
	else if (uint16.is(image.dtype())) {
		gl_type = GL_UNSIGNED_SHORT;
	}
	else if (float32.is(image.dtype())) {
		gl_type = GL_FLOAT;
	}
	else {
		py::print("abort: unknown image type.");
		return make_tuple(py::none(), py::none());
	}

	if (_existGL) wglMakeCurrent(g_hdc, m_hglrc); // set GL context for SiftGPU
	sift->VerifyContextGL();

	// copy image & run SiftGPU
	bool succeeded = sift->RunSIFT(W, H, (void*)image.request().ptr, gl_format, gl_type);

	if (succeeded)
	{
		int num = min(sift->GetFeatureNum(), _nfeatures);

		if (num <= 0) {
			py::print("abort: no feature is extracted.");
			return make_tuple(py::none(), py::none());
		}

		py::print(num);

		//sift->GetFeatureVector(kp_ptr, desc_ptr); // [TODO] copy data
	}

	if (_existGL) wglMakeCurrent(g_hdc, g_hglrc); // reset GL context

	if (!succeeded) {
		py::print("error: cannot run GPUSift.");
		return make_tuple(py::none(), py::none());
	}
	
	return make_tuple(py::none(), py::none()); // [TODO] output conversion
}

void empty()
{
	if (_init_sift) {
		_init_sift = false;

		delete sift; sift = NULL;
		FREE_MYLIB(hsiftgpu);
	}
}


///////////////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(pySiftGPU, m) {
    m.doc() = R"pbdoc(
    SiftGPU Plugin with pybind11
    ----------------------------
    )pbdoc";

    m.def("create", &create, 
		py::arg("nfeatures")=4096, py::arg("nOctaveLayers")=3,
		py::arg("contrastThreshold")=0.04, py::arg("edgeThreshold")=10.0,
		py::arg("existGL")=false,
		R"pbdoc(
		  initialize SiftGPU module.
          )pbdoc");

    m.def("detectAndCompute", &detectAndCompute, 
		py::arg("image"),
		R"pbdoc(
		  detect Sift keypoints and compute their descriptors.
          )pbdoc");

	m.def("empty", &empty,
		R"pbdoc(
          remove SiftGPU module.
          )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
