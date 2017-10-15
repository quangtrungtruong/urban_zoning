project(denseho)

# -------
# Library
# -------
# 
set(LIB_SOURCES densecrf.cpp
                filter.cpp
                # image.cpp
                permutohedral.cpp
                potential.cpp
                # probimage.cpp
                # std.cpp
                util.cpp
                evaluate.cpp
                higherorder.cpp)

add_library(libdenseho ${LIB_SOURCES})
set_target_properties(libdenseho PROPERTIES
                                 OUTPUT_NAME "denseho")

# ----------
# Executable
# ----------
# 
add_executable(denseho denseho.cpp)
target_link_libraries(denseho libdenseho libale libprobimage)
