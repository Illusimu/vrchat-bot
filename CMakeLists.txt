cmake_minimum_required(VERSION 3.24)
project(DesktopBody LANGUAGES CXX)

set(OPENVR_SDK "" CACHE PATH "Path to the extracted Valve OpenVR SDK")
cmake_path(ABSOLUTE_PATH OPENVR_SDK BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    NORMALIZE OUTPUT_VARIABLE OPENVR_ROOT)
set(OPENVR_SAMPLES "${OPENVR_ROOT}/samples/drivers")
set(HAND_SAMPLE "${OPENVR_SAMPLES}/drivers/handskeletonsimulation/src")
if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "The Desktop Body driver requires 64-bit Windows.")
endif()
if(NOT EXISTS "${HAND_SAMPLE}/hand_simulation.cpp" OR
   NOT EXISTS "${OPENVR_ROOT}/lib/win64/openvr_api.lib")
    message(FATAL_ERROR "Set OPENVR_SDK to the extracted Valve OpenVR SDK v2.12.14.")
endif()

add_library(driver_desktopbody SHARED driver/desktopbody.cpp "${HAND_SAMPLE}/hand_simulation.cpp")
target_compile_features(driver_desktopbody PRIVATE cxx_std_17)
target_compile_definitions(driver_desktopbody PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_include_directories(driver_desktopbody PRIVATE
    "${OPENVR_ROOT}/headers" "${HAND_SAMPLE}" "${OPENVR_SAMPLES}/utils/vrmath")
target_link_libraries(driver_desktopbody PRIVATE ws2_32 "${OPENVR_ROOT}/lib/win64/openvr_api.lib")

install(TARGETS driver_desktopbody RUNTIME DESTINATION desktopbody/bin/win64)
add_executable(vr_eye driver/capture.cpp)
target_include_directories(vr_eye PRIVATE "${OPENVR_ROOT}/headers")
target_link_libraries(vr_eye PRIVATE d3d11 dxgi "${OPENVR_ROOT}/lib/win64/openvr_api.lib")
install(TARGETS vr_eye RUNTIME DESTINATION tools)
install(FILES "${OPENVR_ROOT}/bin/win64/openvr_api.dll" DESTINATION tools)
install(DIRECTORY driver/package/ DESTINATION desktopbody)
install(FILES "${OPENVR_ROOT}/LICENSE" DESTINATION desktopbody RENAME OPENVR-LICENSE.txt)
