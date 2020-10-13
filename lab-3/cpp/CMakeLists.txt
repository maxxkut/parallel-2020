cmake_minimum_required(VERSION 3.15)
project(lab-3)

set(CMAKE_CXX_STANDARD 17)

IF (WIN32)
    # Require MPI for this project:
    #This is for finding MS-MPI.
    set(MPIEXEC "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe")
    # For building MPI programs the selected Visual Studio compiler is used,namely cl.exe.
    # So there is no need to set a specific MPI compiler.
    set(MPI_CXX_INCLUDE_PATH "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
    # Make sure the correct libraries (64-bit or 32-bit) are selected.
    # Decide between 32-bit and 64-bit libraries for Microsoft's MPI
    if ("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
        set(MS_MPI_ARCH_DIR x64)
    else ()
        set(MS_MPI_ARCH_DIR x86)
    endif ()
    set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/${MS_MPI_ARCH_DIR}/msmpi.lib")
    set(MPI_C_INCLUDE_PATH "${MPI_CXX_INCLUDE_PATH}")
    set(MPI_C_LIBRARIES "${MPI_CXX_LIBRARIES}")
    set(MPIEXEC_NUMPROC_FLAG "-np" CACHE STRING "Flag used by MPI to specify the number of processes for MPIEXEC; the next option will be the number of processes.")
else ()
    find_package(MPI REQUIRED)
endif ()

include_directories(${MPI_CXX_INCLUDE_PATH})
set(SOURCES main.cpp )

add_executable(lab-3 ${SOURCES})
target_link_libraries(lab-3 ${MPI_CXX_LIBRARIES})
