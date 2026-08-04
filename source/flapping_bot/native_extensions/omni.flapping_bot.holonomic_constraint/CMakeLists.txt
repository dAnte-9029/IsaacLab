cmake_minimum_required(VERSION 3.22)
project(flapping_bot_holonomic_constraint LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_RPATH ON)

if(NOT DEFINED PHYSX_OMNI_ROOT)
    message(FATAL_ERROR "PHYSX_OMNI_ROOT must point to the matching PhysX omni directory.")
endif()

find_package(Python3 3.11 EXACT REQUIRED COMPONENTS Development.Module)

set(PHYSX_REPO_ROOT "${PHYSX_OMNI_ROOT}/..")
set(PHYSX_BUILD_ROOT "${PHYSX_OMNI_ROOT}/_build")
set(PHYSX_TARGET_DEPS "${PHYSX_BUILD_ROOT}/target-deps")
set(PHYSX_RELEASE_ROOT "${PHYSX_BUILD_ROOT}/linux-x86_64/release")

add_library(_native MODULE plugins/FlappingWingTrajectoryConstraint.cpp)
set_target_properties(
    _native
    PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}/omni/flapping_bot/holonomic_constraint"
)

target_include_directories(
    _native
    SYSTEM PRIVATE
        "${PHYSX_REPO_ROOT}/physx/include"
        "${PHYSX_OMNI_ROOT}/include"
        "${PHYSX_RELEASE_ROOT}/kit/dev/include"
        "${PHYSX_TARGET_DEPS}/usd/release/include"
)

target_link_directories(
    _native
    PRIVATE
        "${PHYSX_RELEASE_ROOT}/kit"
        "${PHYSX_TARGET_DEPS}/usd/release/lib"
)

target_link_libraries(
    _native
    PRIVATE
        Python3::Module
        carb
        usd_tf
        usd_sdf
        usd_usd
        usd_usdPhysics
)

target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic)
