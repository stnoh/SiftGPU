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
py::object detectAndCompute(const py::array& image);
void empty(); // destructor?

py::array match(const py::array& desc1, const py::array& desc2, 
	float distmax, float ratiomax);


///////////////////////////////////////////////////////////////////////////////
// unnamed namespace for data sealing
///////////////////////////////////////////////////////////////////////////////

#define FREE(obj) { if (NULL != obj) { delete obj; obj = NULL; } }

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
HMODULE hsiftgpu = NULL;
SiftGPU      *sift    = NULL;
SiftMatchGPU *matcher = NULL;

int (*match_buf)[2];

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
	pCreateNewSiftMatchGPU = (SiftMatchGPU* (*)(int)) GET_MYPROC(hsiftgpu, "CreateNewSiftMatchGPU");

	sift    = pCreateNewSiftGPU(1);
	matcher = pCreateNewSiftMatchGPU(_nfeatures);
	match_buf = new int[_nfeatures][2];

	// parse and set arguments
	std::vector<char*> argv;

	argv.push_back((char*)"-fo"); // feature octave
	argv.push_back((char*)"-1");

	argv.push_back((char*)"-tc");
	char buf_tc[128];
	sprintf_s(buf_tc, 128, "%d", _nfeatures);
	argv.push_back(buf_tc);

	argv.push_back((char*)"-d");
	char buf_d[128];
	sprintf_s(buf_d, 128, "%d", nOctaveLayers);
	argv.push_back(buf_d);

	argv.push_back((char*)"-t"); // contrast threshold
	char buf_t[128];
	sprintf_s(buf_t, 128, "%f", contrastThreshold);
	argv.push_back(buf_t);

	argv.push_back((char*)"-e");
	char buf_e[128];
	sprintf_s(buf_e, 128, "%f", edgeThreshold);
	argv.push_back(buf_e);

	argv.push_back((char*)"-v"); // verbose
	argv.push_back((char*)"0");
	argv.push_back((char*)"-noprep");
	//argv.push_back((char*)"-unn"); // un-normalized descriptor

	//py::print(argv); // check arguments

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

py::object detectAndCompute(const py::array& image)
{
	if (!_init_sift)
	{
		py::print("abort: SiftGPU is not initialized.");
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

	py::array keypoints;
	py::array descriptors;

	if (_existGL) wglMakeCurrent(g_hdc, m_hglrc); // set GL context for SiftGPU
	sift->VerifyContextGL();

	// copy image & run SiftGPU
	bool succeeded = sift->RunSIFT(W, H, (void*)image.request().ptr, gl_format, gl_type);

	if (succeeded)
	{
		int num = sift->GetFeatureNum();

		if (num <= 0) {
			py::print("abort: no feature is extracted.");
			return make_tuple(py::none(), py::none());
		}

		keypoints = py::array_t<SiftKeypoint>(num);
		py::array _descriptors = py::array_t<float>(num << 7);
		
		sift->GetFeatureVector((SiftKeypoint*)keypoints.request().ptr, (float*)_descriptors.request().ptr); // copy data

		// [FIX ME LATER] conversion from SiftKeypoint

		descriptors = _descriptors.reshape({num, 128}); // convert 1D [num*128] -> 2D [num,128]
	}

	if (_existGL) wglMakeCurrent(g_hdc, g_hglrc); // reset GL context

	if (!succeeded) {
		py::print("error: cannot run GPUSift.");
		return make_tuple(py::none(), py::none());
	}

	return make_tuple(keypoints, descriptors);
}

void empty()
{
	if (_init_sift) {
		FREE(sift);
		FREE(matcher);
		FREE(match_buf);

		_init_sift = false;
	}

	if (NULL != hsiftgpu) {
		FREE_MYLIB(hsiftgpu);
		hsiftgpu = NULL;
	}
}

py::array match(const py::array& desc0, const py::array& desc1,
				float distmax, float ratiomax)
{
	if (NULL == matcher)
	{
		py::print("abort: SiftMatchGPU is not initialized.");
		return py::none();
	}

	// check the descriptor shape
	size_t desc0_dim = desc0.ndim();
	size_t desc1_dim = desc1.ndim();
	if (desc0_dim != 2 || desc1_dim != 2) {
		py::print("abort: invalid descriptor");
		return py::none();
	}
	if (desc0.shape(1) != 128 || desc1.shape(1) != 128) {
		py::print("abort: invalid descriptor");
		return py::none();
	}

	int num_desc0 = desc0.shape(0);
	int num_desc1 = desc1.shape(0);

	if (num_desc0 <= 0 || num_desc1 <= 0) {
		py::print("abort: invalid size of descriptor");
		return py::none();
	}

	if (_existGL) wglMakeCurrent(g_hdc, m_hglrc); // set GL context for SiftGPU
	matcher->VerifyContextGL();

	// run matcher
	matcher->SetDescriptors(0, num_desc0, (const float *)desc0.request().ptr);
	matcher->SetDescriptors(1, num_desc1, (const float *)desc1.request().ptr);

	int num_min = min(num_desc0, num_desc1);
	int num_match = matcher->GetSiftMatch(num_min, match_buf, distmax, ratiomax);

	if (_existGL) wglMakeCurrent(g_hdc, g_hglrc); // reset GL context

	// conversion for python
	py::array_t<int> match0 = py::array_t<int>(num_match);
	py::array_t<int> match1 = py::array_t<int>(num_match);
	int *match0_ptr = static_cast<int*>(match0.request().ptr);
	int *match1_ptr = static_cast<int*>(match1.request().ptr);

	for (int n = 0; n < num_match; n++)
	{
		//py::print(match_buf[n][0], " - ", match_buf[n][1]);

		match0_ptr[n] = match_buf[n][0];
		match1_ptr[n] = match_buf[n][1];
	}

	return make_tuple(match0, match1);
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
		py::arg("contrastThreshold")=0.01472, py::arg("edgeThreshold")=10.0,
		py::arg("existGL")=false,
		R"pbdoc(
		  (Re)initialize SiftGPU module.
          arg0: maximum number of feature points handled by this module (int).
                if set to zero, this module handles all points. default is 4096 (maximum).
          arg1: the number of octave inside of this module (int). default is 3.
          arg2: parameter for contrast threshold (float). default is 0.01472 .
          arg3: parameter for edge threshold (float). default is 10.0 .
          arg4: if there is GL context, use that GL for GLSL. default is False.
          )pbdoc");

    m.def("detectAndCompute", &detectAndCompute,
		py::arg("image"),
		R"pbdoc(
          arg1: grayscale image. Should be 2-dimensional numpy array (H, W).
		  detect Sift keypoints and compute their descriptors.
          )pbdoc");

	m.def("empty", &empty,
		R"pbdoc(
          remove SiftGPU module.
          )pbdoc");

	m.def("match", &match,
		py::arg("desc0"),
		py::arg("desc1"),
		py::arg("distmax") = 0.70f,
		py::arg("ratiomax") = 0.80f,
		R"pbdoc(
          get brute-force matching with GPU
          arg1: descriptor #0. Should be 2-dimensional numpy array (N, 128).
          arg2: descriptor #1. Should be 2-dimensional numpy array (M, 128).
          arg2: max distance for matching (float). default is 0.70 .
          arg3: max ratio for matching (float). default is 0.80 .
          )pbdoc");

	PYBIND11_NUMPY_DTYPE(SiftKeypoint, x, y, s, o); // [FIX ME LATER]

	// [FIX ME LATER]
	/*
	py::class_<SiftKeypoint>(m, "SiftKeypoint")
		.def(py::init())
		.def_readwrite("x", &SiftKeypoint::x)
		.def_readwrite("y", &SiftKeypoint::y)
		.def_readwrite("s", &SiftKeypoint::s)
		.def_readwrite("o", &SiftKeypoint::o);
	//*/

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
