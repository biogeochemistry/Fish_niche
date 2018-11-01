cmake_minimum_required (VERSION 2.8)
project (Reaction_parsing_module)

include_directories(/Users/MarkelovIgor/eigen)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

set(CMAKE_CXX_COMPILER "g++") #compiler g++
# add_definitions("-std=c++11 -g -O0 -fopenmp -fPIC -Wall -Wextra -Wno-missing-field-initializers  -fdiagnostics-color=always -pthread")
add_definitions("-std=c++11 -O3 -ffast-math -ffinite-math-only -fopenmp -fPIC -Wall -Wextra -pthread -Wno-missing-field-initializers  -fdiagnostics-color=always")
add_executable(ph pH.cpp)

target_link_libraries(ph ${Boost_LIBRARIES})
