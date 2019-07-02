static PyObject* pygs_gs_alphamatte(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace gs;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_trimap = NULL;
    Mat trimap;
    PyObject* pyobj_foreground = NULL;
    Mat foreground;
    PyObject* pyobj_alpha = NULL;
    Mat alpha;

    const char* keywords[] = { "image", "trimap", "foreground", "alpha", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:alphamatte", (char**)keywords, &pyobj_image, &pyobj_trimap, &pyobj_foreground, &pyobj_alpha) &&
        pygs_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pygs_to(pyobj_trimap, trimap, ArgInfo("trimap", 0)) &&
        pygs_to(pyobj_foreground, foreground, ArgInfo("foreground", 1)) &&
        pygs_to(pyobj_alpha, alpha, ArgInfo("alpha", 1)) )
    {
        ERRWRAP2(gs::alphamatte(image, trimap, foreground, alpha));
        return Py_BuildValue("(NN)", pygs_from(foreground), pygs_from(alpha));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_trimap = NULL;
    UMat trimap;
    PyObject* pyobj_foreground = NULL;
    UMat foreground;
    PyObject* pyobj_alpha = NULL;
    UMat alpha;

    const char* keywords[] = { "image", "trimap", "foreground", "alpha", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:alphamatte", (char**)keywords, &pyobj_image, &pyobj_trimap, &pyobj_foreground, &pyobj_alpha) &&
        pygs_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pygs_to(pyobj_trimap, trimap, ArgInfo("trimap", 0)) &&
        pygs_to(pyobj_foreground, foreground, ArgInfo("foreground", 1)) &&
        pygs_to(pyobj_alpha, alpha, ArgInfo("alpha", 1)) )
    {
        ERRWRAP2(gs::alphamatte(image, trimap, foreground, alpha));
        return Py_BuildValue("(NN)", pygs_from(foreground), pygs_from(alpha));
    }
    }

    return NULL;
}

