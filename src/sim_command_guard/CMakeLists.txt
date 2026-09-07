cmake_minimum_required(VERSION 3.16)
project(sim_command_guard)

find_package(ament_cmake REQUIRED)
find_package(ignition-gazebo6 REQUIRED)
find_package(ignition-msgs8 REQUIRED)
find_package(ignition-transport11 REQUIRED)

add_library(sim_command_guard SHARED src/command_guard_system.cc)
target_compile_features(sim_command_guard PUBLIC cxx_std_17)
target_include_directories(sim_command_guard PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(sim_command_guard
  ignition-gazebo6::core
  ignition-msgs8::core
  ignition-transport11::core)

install(TARGETS sim_command_guard DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
ament_environment_hooks(env-hooks/sim_command_guard.dsv.in)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_command_guard_core test/test_command_guard_core.cc)
  target_include_directories(test_command_guard_core PRIVATE include)
endif()

ament_package()
