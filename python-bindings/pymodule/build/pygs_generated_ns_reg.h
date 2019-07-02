static PyMethodDef methods_gs[] = {
    {"alphamatte", (PyCFunction)pygs_gs_alphamatte, METH_VARARGS | METH_KEYWORDS, "alphamatte(image, trimap[, foreground[, alpha]]) -> foreground, alpha\n."},
    {NULL, NULL}
};

static ConstDef consts_gs[] = {
    {NULL, 0}
};

static void init_submodules(PyObject * root) 
{
  init_submodule(root, MODULESTR"", methods_gs, consts_gs);
};
