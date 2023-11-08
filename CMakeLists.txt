cmake_minimum_required(VERSION 3.0)
project("DeepSDF")

find_package(CLI11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nanoflann REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Threads REQUIRED)


add_subdirectory(third-party/cnpy)

include_directories(third-party)

set(SOURCES
src/ShaderProgram.cpp
src/Utils.cpp)

# add_executable(PreprocessMeshToNormal src/PreprocessMeshToNormal.cpp ${SOURCES})
# target_include_directories(PreprocessMeshToNormal PRIVATE third-party/cnpy)
# target_link_libraries(PreprocessMeshToNormal PRIVATE CLI11::CLI11 pangolin cnpy Eigen3::Eigen nanoflann::nanoflann Threads::Threads)
# target_compile_features(PreprocessMeshToNormal PRIVATE cxx_std_14)
# set_target_properties(PreprocessMeshToNormal PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(PreprocessMesh src/PreprocessMesh_raw.cpp ${SOURCES})
target_include_directories(PreprocessMesh PRIVATE third-party/cnpy)
target_link_libraries(PreprocessMesh PRIVATE CLI11::CLI11 pangolin cnpy Eigen3::Eigen nanoflann::nanoflann Threads::Threads)
target_compile_features(PreprocessMesh PRIVATE cxx_std_14)
set_target_properties(PreprocessMesh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(SampleVisibleMeshSurface src/SampleVisibleMeshSurface.cpp ${SOURCES})
target_include_directories(SampleVisibleMeshSurface PRIVATE third-party/cnpy)
target_link_libraries(SampleVisibleMeshSurface PRIVATE CLI11::CLI11 pangolin cnpy Eigen3::Eigen nanoflann::nanoflann Threads::Threads)
target_compile_features(SampleVisibleMeshSurface PRIVATE cxx_std_14)
set_target_properties(SampleVisibleMeshSurface PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")