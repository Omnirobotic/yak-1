project(yak-lib LANGUAGES CXX CUDA)

add_library(${PROJECT_NAME}
    yak/src/yak_server.cpp
    yak/src/mc/marching_cubes_tables.cpp
    yak/src/kfusion/core.cpp
    yak/src/kfusion/device_memory.cpp
    yak/src/kfusion/imgproc.cpp
    yak/src/kfusion/kinfu.cpp
    yak/src/kfusion/precomp.cpp
    yak/src/kfusion/projective_icp.cpp
    yak/src/kfusion/tsdf_volume.cpp
    yak/src/cuda/imgproc.cu
    yak/src/cuda/proj_icp.cu
    yak/src/cuda/tsdf_volume.cu
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    yak/include
)

