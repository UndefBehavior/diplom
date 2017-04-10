TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt


LIBS+= -fopenmp -lgomp -lpthread
QMAKE_CXXFLAGS += -fopenmp
SOURCES += main.cpp

