cmake_minimum_required(VERSION 3.10)

# set the project name
project(Tutorial)

# add the executable
add_executable(Tutorial tutorial.cpp)
target_include_directories(Tutorial PUBLIC "${PROJECT_BINARY_DIR}")

