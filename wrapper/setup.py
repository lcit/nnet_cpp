from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# --> to compile the extendion type:
# python setup.py build_ext --inplace

setup(ext_modules=[Extension(
        "nnet_ext", # name of the extension (to import in python)
        ["nnet_ext.pyx", "../src/nnet.cpp", "../wrapper/nnet_interface.cpp", "../utils/Eigen_plus.cpp", "../utils/auc.cpp"], # source file and wrapper
        language="c++", # specify the c++ compiler
        extra_compile_args=['-g', '-std=c++14', '-D_hypot=hypot', '-I../utils/'], # extra flags , '-Wreorder', '-lstdc++',
        extra_link_args=['-g', '-pthread'], # extra flags
        #include_dirs=["../utils/", "../src/"]
    )], 
    cmdclass = {'build_ext': build_ext}
)     