##############################################################################
# CMake-based build system for pySiftGPU
# Author: Seung-Tak Noh (seungtak.noh@gmail.com)
##############################################################################
cmake_minimum_required(VERSION 3.1)
project("pySiftGPU" CXX)

set(CMAKE_CXX_STANDARD 17) ## pybind11 (> v2.9) needs >= C++17

add_compile_definitions(SIFTGPU_NO_DEVIL)
add_compile_definitions(DLL_EXPORT)
add_compile_definitions(SIFTGPU_DLL)
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)


###############################################
# minimum external libraries
###############################################
find_package(OpenGL REQUIRED)

# pybind11
set(PYBIND11_INSTALL OFF CACHE INTERNAL "" FORCE)
set(PYBIND11_TEST    OFF CACHE INTERNAL "" FORCE)
set(PYBIND11_PYTHONLIBS_OVERWRITE OFF CACHE INTERNAL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)


###############################################
# SiftGPU (main library)
###############################################

file(GLOB
	headers
	${CMAKE_CURRENT_SOURCE_DIR}/src/SiftGPU/*.h
)
file(GLOB
	sources
	${CMAKE_CURRENT_SOURCE_DIR}/src/SiftGPU/*.cpp
)
add_library(SiftGPU SHARED ${headers} ${sources})

include_directories(${CMAKE_SOURCE_DIR}/include)

if(NOT WIN32)
	message("WARNING: This CMake script only supports Windows x64. You need to set library files by yourself." )
endif()

if (WIN32)
	target_link_libraries(SiftGPU
		opengl32.lib glu32.lib ${CMAKE_SOURCE_DIR}/lib/glew32.lib winmm.lib
	)

	## copy glew32.dll as post-build event
	macro(copy_3rdparty_dlls APP)
	add_custom_command(TARGET ${APP} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/bin/glew32.dll ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}
	)
	endmacro()

	copy_3rdparty_dlls(SiftGPU)
endif()

###############################################
# pySiftGPU (python binding)
###############################################

pybind11_add_module(pySiftGPU src/python/pySiftGPU.cpp)
target_link_libraries(pySiftGPU PRIVATE SiftGPU pybind11::module)
