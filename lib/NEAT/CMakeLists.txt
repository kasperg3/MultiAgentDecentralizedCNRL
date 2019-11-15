cmake_minimum_required(VERSION 2.4)

PROJECT( neat )

FILE(GLOB neat_sources src/*.cpp include/*.h)

ADD_EXECUTABLE( neat ${neat_sources} )
