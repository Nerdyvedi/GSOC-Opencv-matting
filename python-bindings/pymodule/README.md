Following files are here:

1. gs.cpp, pycompat.hpp : gs.cpp is a slightly modified version of the wrapper file (cv2.cpp) that comes with OpenCV. It uses pycompat.hpp for Python 2 / 3 compatibility checks.

2. gen2.py, hdr_parser.py : The Python bindings generator script (gen2.py) calls the header parser script (hdr_parser.py). These files are provided as part of the OpenCV source files. According to the OpenCV tutorial, “this header parser splits the complete header file into small Python lists. So these lists contain all details about a particular function, class etc.” In other words, these scripts automatically parse the header files and register the functions, classes, methods etc. with the module.

3. headers.txt: A text file containing all the header files to be compiled into the module. In our example, it contains just one line src/globalmatting.h

4. src/globalmatting.cpp: This cpp file contains the functions and class definitions

5. src/globalmatting.h:  This header file explictly mentions the classes and functions we want to export



