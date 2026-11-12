# annie
Search for any prime number

## About

**annie** is a an [OpenCL™](https://www.khronos.org/opencl/) application on GPU.  

It performs a fast probable primality test for any number using an [Euler test](https://en.wikipedia.org/wiki/Euler%E2%80%93Jacobi_pseudoprime).  
*a* is chosen such that Euler's criterion yields -1: it is the condition of [Proth's theorem](https://en.wikipedia.org/wiki/Proth%27s_theorem).
If the input is of the form *k*&nbsp;&middot;&nbsp;2<sup>*n*</sup>&nbsp;+&nbsp;1, where *k* is odd and *k*&nbsp;<&nbsp;2<sup>*n*</sup>, *annie* performs a primality test.  

The input is a [Pari/GP](https://pari.math.u-bordeaux.fr/) math expression. For example: "208003!-1", "3267113#-1", "fibonacci(4740217)", "(sqrtnint(10^999999,1024)+407852)^1024+1".

[Philip B. McLaughlin](https://pubs.ams.org/MCOM/2004-73-246/S0025-5718-03-01543-6) extended [Montgomery modular multiplication](https://en.wikipedia.org/wiki/Montgomery_modular_multiplication) to modular polynomial arithmetic. This is suited for use with Number Theoretic Transform (NTT) methods for multiplication.  
Yves Gallot and George Woltman improved this algorithm by using the polynomials *R*&nbsp;=&nbsp;2<sup>*n*</sup>&nbsp;-&nbsp;1 and *Q*&nbsp;=&nbsp;2<sup>*n*</sup>&nbsp;+&nbsp;1 and the form *x*&nbsp;'&nbsp;=&nbsp;2<sup>-1</sup>&nbsp;R&nbsp;x&nbsp;(mod&nbsp;*N*).  
The proof of the algorithm can be found [here](https://github.com/galloty/annie/tree/main/doc).  

*annie* implements an [Efficient Modular Exponentiation Proof Scheme](https://arxiv.org/abs/2209.15623).
The test is validated with [Gerbicz - Li](https://www.mersenneforum.org/showthread.php?t=22510) error checking.  

Any number with up to approximately 48 million digits can be tested.  

## Operating system

 - Linux x64  
 - Windows x64  
 - MacOS for Arm or Intel has not been tested, but *annie* should work.

## Build

Select the [makefile](https://github.com/galloty/annie/tree/main/build) of your target. On Windows, [MSYS2](https://www.msys2.org/) distribution and building platform is recommended.  
gcc or clang are supported.  

## Licence

**annie** is free source code, under the MIT license (see [LICENSE](https://github.com/galloty/annie/blob/main/LICENSE)). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.

## TODO

 - Transforms of length 3&nbsp;&middot;&nbsp;2<sup>*n*</sup> and 5&nbsp;&middot;&nbsp;2<sup>*n*</sup>.  
 - Main and shared memory 128-bit access width.  
