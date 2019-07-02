## Files ##

1. gs.cpp, pycompat.hpp : gs.cpp is a slightly modified version of the wrapper file (cv2.cpp) that comes with OpenCV. It uses pycompat.hpp for Python 2 / 3 compatibility checks.

2. gen2.py, hdr_parser.py : The Python bindings generator script (gen2.py) calls the header parser script (hdr_parser.py). These files are provided as part of the OpenCV source files. According to the OpenCV tutorial, “this header parser splits the complete header file into small Python lists. So these lists contain all details about a particular function, class etc.” In other words, these scripts automatically parse the header files and register the functions, classes, methods etc. with the module.

3. headers.txt: A text file containing all the header files to be compiled into the module. In our example, it contains just one line src/globalmatting.h

4. src/globalmatting.cpp: This cpp file contains the functions and class definitions

5. src/globalmatting.h:  This header file explictly mentions the classes and functions we want to export

## Steps ##


Step 1: Put your c++ source code and header files inside the src directory.

Step 2: Include your header file in headers.txt

Step 3: Make a build directory.

```
mkdir build
```
Step 4: Use gen2.py to generate the Python binding files. You need to specify the prefix (pybv), the location of the temporary files (build) and the location of the header files (headers.txt).
```	
python3 gen2.py pygs build headers.txt
```
This should generate a whole bunch of header files with prefix pygs_*.h. 

Step 5: Compile the module

```
g++ -shared -rdynamic -g -O3 -Wall -fPIC \
gv.cpp src/bvmodule.cpp \
-DMODULE_STR=bv -DMODULE_PREFIX=pygs \
-DNDEBUG -DPY_MAJOR_VERSION=3 \
`pkg-config --cflags --libs opencv`  \
`python3-config --includes --ldflags` \
-I . -I/usr/local/lib/python3.5/dist-packages/numpy/core/include \
-o build/gsmat.so  
```
