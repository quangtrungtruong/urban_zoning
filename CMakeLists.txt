cmake_minimum_required(VERSION 2.8)
project( urban_zoning )
# where to look for .cmake for module search
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
find_package( OpenCV REQUIRED )
add_executable(urban_zoning
    main.cpp
    urban_object.cpp
    stats.cpp
    util.cpp
    proposal.cpp
    crf/pdf.cpp
    crf/denseho/densecrf.cpp
    crf/denseho/potential.cpp
    crf/denseho/permutohedral.cpp
    )
target_link_libraries( urban_zoning ${OpenCV_LIBS} )
