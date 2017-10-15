#define LIBRARY

#ifdef LIBRARY
extern "C"
#ifdef WIN32
__declspec(dllexport)
#endif
#endif
void initCuda(int argc, char **argv);

#ifdef LIBRARY
extern "C"
#ifdef WIN32
__declspec(dllexport)
#endif
#endif
void filter(float *im, float *ref, int pd, int vd, int w, int h, bool accurate);

