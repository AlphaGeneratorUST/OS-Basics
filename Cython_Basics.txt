Cython的基本知识

1）基本概念：
常规Python函数，运行时间559 ns
Cython def函数，声明一个Python函数，既可以在模块内调用，也可以在模块外调用。模块内运行时间524.2 ns，模块外运行时间512 ns：
%%cython
def f1(x):
    return x ** 2 - x
    
Cython cpdef函数，声明一个C函数和一个Python wrapper，在模块内被当做C函数调用，在模块外被.py文件当做Python函数调用。模块内运行时间43.7 ns，模块外运行时间81.7 ns:
%%cython
cpdef long f1(long x):
    return x ** 2 - x
    
Cython cdef函数，声明一个C函数，不可以在模块外被Python调用。模块内运行时间34.8 ns
%%cython
cdef long f1(long x):
    return x ** 2 - x

2) 使用C标准库替代Python的math模块：
cdef extern from "math.h":
    float cosf(float theta)
    float sinf(float theta)
    float acosf(float theta)
    
3) Cython编译：
对于Windows系统，.pyx -> .pyd
转换动态类型的Python版本为静态类型的Cython版本


# Cython type inference

To enable type inference for a function, we can use the decorator form of infer_types:
cimport cython
@cython.infer_types(True)
def more_inference():
    i = 1
    d = 2.0
    c = 3+4j
    r = i * d + c
    return r
    
from cython cimport operator
print operator.dereference(p_double)
# => 1.618

cdef st_t *p_st = make_struct()
cdef int a_doubled = p_st.a + p_st.a

def integrate(a, b, f):
    cdef int i
    cdef int N=2000
    cdef float dx, s=0.0
    dx = (b-a)/N
    for i in range(N):
    s += f(a+i*dx)
    return s * dx

def integrate(a, b, f):
    cdef:
    int i
    int N=2000
    float dx, s=0.0
    # ...

cdef list particles, modified_particles
cdef dict names_from_particles
cdef str pname
cdef set unique_particles

cdef long c_fact(long n):
    """Computes n!"""
    if n <= 1:
    return 1
    return n * c_fact(n - 1)

cpdef long cp_fact(long n):
    """Computes n!"""
    if n <= 1:
    return 1
    return n * cp_fact(n - 1)

cdef int *ptr_i = <int*>v

def print_address(a):
    cdef void *v = <void*>a
    cdef long addr = <long>v
    print "Cython address:", addr
    print "Python id :", id(a)

cdef struct mycpx:
    float real
    float imag
    
cdef union uu:
    int a
    short b, c

ctypedef struct mycpx:
    float real
    float imag

ctypedef union uu:
    int a
    short b, c

ctypedef double real
ctypedef long integral

def displacement(real d0, real v0, real a, real t):
    """Calculates displacement under constant acceleration."""
    cdef real d = d0 + (v0 * t) + (0.5 * a * t**2)
    return d

cdef unsigned int i, n = 100
    for i in range(n):
    # ...

cdef int i, N
for i in range(N):
    a[i] = i + 1

cdef unsigned int i, n = len(a) - 1
for i in range(1, n):
    a[i] = (a[i-1] + a[i] + a[i+1]) / 3.0

from distutils.core import setup
from Cython.Build import cythonize

setup(name="nbody", ext_modules=cythonize("nbody.pyx"))

cdef class Particle:
    """Simple Particle extension type."""
    cdef public double mass
    cdef readonly double position
    cdef double velocity
    # ...

cdef class Matrix:
    cdef:
        unsigned int nrows, ncols
        double *_matrix
    def __cinit__(self, nr, nc):
        self.nrows = nr
        self.ncols = nc
        self._matrix = <double*>malloc(nr * nc * sizeof(double))
        if self._matrix == NULL:
            raise MemoryError()

cdef class Particle:
    """Simple Particle extension type."""
    cdef double mass, position, velocity
    # ...
    cpdef double get_momentum(self):
        return self.mass * self.velocity

def add_momentums_typed(list particles):
    """Returns the sum of the particle momentums."""
    cdef:
        double total_mom = 0.0
        Particle particle
    for particle in particles:
        total_mom += particle.get_momentum()
    return total_mom

cdef class CParticle(Particle):
    cdef double momentum
    def __init__(self, m, p, v):
        super(CParticle, self).__init__(m, p, v)
        self.momentum = self.mass * self.velocity
    cpdef double get_momentum(self):
        return self.momentum


from libc.stdlib cimport rand, srand, qsort, malloc, free
cdef int *a = <int*>malloc(10 * sizeof(int))

from libc.string cimport memcpy as c_memcpy

from libcpp.vector cimport vector
cdef vector[int] *vi = new vector[int](10)

from libc.math cimport sin as csin
from math import sin as pysin

cdef extern from "header.h":
    double M_PI
    float MAX(float a, float b)
    double hypot(double x, double y)
    ctypedef int integral
    ctypedef double real
    void func(integral a, integral b, real c)
    real *func_arrays(integral[] i, integral[][10] j, real **k)
    
from distutils.core import setup, Extension
from Cython.Build import cythonize
ext = Extension("mt_random", sources=["mt_random.pyx", "mt19937ar.c"])
setup(name="mersenne_random", ext_modules = cythonize([ext]))

from distutils.core import setup, Extension
from Cython.Build import cythonize
ext = Extension("RNG", sources=["RNG.pyx", "mt19937.cpp"])

from cython cimport boundscheck, wraparound
@boundscheck(False)
@wraparound(False)
def calc_julia(...):
    # ...

from distutils.core import setup
from Cython.Build import cythonize
setup(name="julia", ext_modules=cythonize("julia.pyx"))


## parallel
from cython.parallel cimport prange

def calc_julia(...):
    # ...
    with nogil:
        for i in prange(resolution + 1):
            real = -bound + i * step
            for j in range(resolution + 1):
                # ...
    # ...

def calc_julia(...):
    # ...
    for i in prange(resolution + 1, nogil=True):
        real = -bound + i * step
        for j in range(resolution + 1):
            # ...

import pyximport
pyximport.install()

from fib import fib
print(fib(10))

## Cython + IPython
%load_ext cythonmagic

%%cython
def cyfib(int n):
    cdef int a, b, i
    a, b = 1, 1
    for i in range(n):
        a, b = a+b, a
    return a
    
python set_up.py build_ext --inplace --force --compiler=mingw32
python set_up.py build_ext -if -c mingw32

cython -a hamming_cython_solution.pyx


## import externel c
from distutils.core import setup, Extension
from Cython.Build import cythonize

ld = Extension(name = "test_cython", sources = ["test_cython.pyx", "test.c"])
lds = Extension(name = "test_solution", sources=["test_solution.pyx", "test.c"])

setup(ext_modules = cythonize([ld, lds]))

## Cython help
cython -h

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)


## IPython run.py
%load_ext Cython
%load_ext cythonmagic

%%cython
def first_cython_func(int i):
    return i * 2
    
%%cython -a
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX


def first_cython_func(int i):
    return i * 2

%%cython -n first_cython_func

%run -p test.py

%%cython?

!ls ~/.ipython/cython
!head ~/test/...

from dis import dis
dis(foo)

%timeit foo(1, 2)

import datetime
cimport cpython.datetime

import array
cimport cpython.array

cdef:
    list lst = [1]
    dict dd = {'a':'b'}
    set ss = set([1])
    cpython.datetime.datetime dt = datetime.datetime.now()
    cpython.array.array aa = array.array('i', [1,2,3])
    

cpdef int func(int i) except *:
    pass
    
cpdef int func(int i) except -1:
    # ...
    raise ValueError("...")


%%prun
reverse(list(range(1000)))

cython: profile = True


import pyximport; pyximport.install()
1) Use pyximport;
2) Extension mudule;
3) Define types


import timeit

cy = timeit.timeit('example_cy.test(5)', setup='import example_cy', number=100)

cython -a example_cy.pyx


## Cython优化numpy和pandas实例
# 纯Python版本
%load_ext Cython
import numpy as np

def shannon_entropy_py(p_x):
    return - np.sum(p_x * np.log(p_x))

# 把python中涉及到变量的部分 用cython定义一遍，这里就要用到 cimport numpy as cnp了，numpy给cython留了调用的c-level的接口，
# 使用cimport可以在cython中导入c模块从而通过调用c模块来加快程序运行的速度
%%cython 

import numpy as np
cimport numpy as cnp

def shannon_entropy_cy(cnp.ndarray p_x):
    return - np.sum(p_x * np.log(p_x))


# 接下来在上一部的cython定义里面，我们把np.log改为从c库中导入
%%cython 

cimport numpy as cnp
from libc.math cimport log as clog

def shannon_entropy_v1(cnp.ndarray p_x):
    cdef double res = 0.0
    cdef int n = p_x.shape[0]
    cdef int i
    for i in range(n):
        res += p_x[i] * clog(p_x[i])
    return -res
    

# 这里实际上涉及到矢量化的概念，矢量化的大概的含义就是将一些比较复杂的运算转化为张量（比如二维条件下的矩阵运算）的运算，最典型的例子就是通过矩阵乘的方法代替了for循环，
# 比如，针对两个5*5的数组进行求和操作，如果使用for循环我们需要循环一大圈，但是从矩阵的角度出发就方便多了，对应位置的元素相加即可。
def shannon_entropy_v1(p_x):
    res=0.0
    n = p_x.shape[0]
    for i in range(n):
        res += p_x[i] * np.log(p_x[i])
    return -res
    
%%timeit
shannon_entropy_v1(pmf)


# 下面我们来使用神器的numpy数据类型的指定。相对于上面再上面的那个cython实现，下面的代码仅仅改变了cnp.ndarray[double]，从原来的cnp.ndarray改成了cnp.ndarray[double]
%%cython 

cimport numpy as cnp
from libc.math cimport log as clog

def shannon_entropy_v2(cnp.ndarray[double] p_x):
    cdef double res = 0.0
    cdef int n = p_x.shape[0]
    cdef int i
    for i in range(n):
        res += p_x[i] * clog(p_x[i])
    return -res
    

# 对函数加入decorator取消boundscheck和wraparound
%%cython -a

cimport cython
cimport numpy as cnp
from libc.math cimport clog

@cython.boundscheck(False)
@cython.wraparound(False)
def shannon_entropy_v3(cnp.ndarray[double] p_x):
    cdef double res = 0.0
    cdef int n = p_x.shape[0]
    cdef int i
    for i in range(n):
        res += p_x[i] * clog(p_x[i])
    return -res

%%timeit
shannon_entropy_v3(pmf)


# 针对循环体做了一些改变: https://stackoverflow.com/questions/21382180/cython-pure-c-loop-optimization
%%cython -a

cimport cython
cimport numpy as cnp
from libc.math cimport log

@cython.boundscheck(False)
@cython.wraparound(False)
def shannon_entropy_v3(cnp.ndarray[double] p_x):
    cdef double res = 0.0
    cdef int n = p_x.shape[0]
    cdef int i
    for i from 0<=i<n:
        res += p_x[i] * log(p_x[i])
    return -res
    
# 针对for loop using the for-from Pyrex notation
 def loop1(int start, int stop, int step):
    cdef int x, t = 0
    for x in range(start, stop, step):
        t += x
    return t

def loop2(int start, int stop, int step):
    cdef int x, t = 0
    for x from start <= x < stop by step:
        t += x
    return t
    

# 在使用pandas的场合都替换为numpy，一方面使用numpy在大部分场景下都可以直接加速，一方面也便于扩展，julia、cython、numba都是可以直接和numpy进行交互的


## Cython中类的定义
# 纯Python的类定义
class PyLCG(object):
    def __init__(self, a=1664525, c=1013904223, m=2**3, seed=0):
        self.a = a
        self.c = c
        if m <= 0:
            raise ValueError("m must be > 0, given {}".format(m))
        self.m = m
        # The RNG state.
        self.x = seed
        
    def _advance(self):
        r = self.x
        self.x = (self.a * self.x + self.c) % self.m
        return r
        
    def randint(self, size=None):
        if size is None:
            return self._advance()
        return np.asarray([self._advance() for _ in range(size)])
        
        
# 然后是cython下的类定义：可以看到比较特别的地方在于：
（1）cython下的类在初始化__init__的时候用的是__cinit__
（2）__cinit__中的参数要提前在_cinit_之间进行声明否则类无法直接识别
（3）list、dict、tuple这些python的数据类型要在cython中通过纯c或c++的方式表达出来也是比较难

%%cython -a

import numpy as np
cimport numpy as cnp
cimport cython

# Creates a new extension type: https://docs.python.org/3/extending/newtypes.html
cdef class CyLCG:
    # We declare the compile-time types of our *instance* attributes here.
    # This is similar to C++ class declaration syntax.
    cdef long a, c, m, x
    
    # Special Cython-defined initializer.
    # Called before __init__ to initialize all C-level attributes.
    def __cinit__(self, long a=1664525, long c=1013904223, long m=2**3, long seed=0):
        self.a = a
        self.c = c
        if m <= 0:
            raise ValueError("m must be > 0, given {}".format(m))
        self.m = m
        self.x = seed
    
    # cdef / cpdef methods are supported
    @cython.cdivision(True)
    cdef long _advance(self):
        cdef long r = self.x
        self.x = (self.a * self.x + self.c) % self.m
        return r
    
    # Regular def method
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def randint(self, size=None):
        cdef long r
        if size is None:
            # Call to self._advance() here is efficient and at the C level.
            r = self._advance()
            return r
        cdef long[:] a = np.empty((size,), dtype='i4')
        cdef int i
        cdef int n = int(size)
        for i in range(n):
            a[i] = self._advance()
        return np.asarray(a)
        
 
 
 
 
 
 








































