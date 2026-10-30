/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_kernel = \
"/*\n" \
"Copyright 2026, Yves Gallot\n" \
"\n" \
"annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"#if __OPENCL_VERSION__ >= 120\n" \
"	#define INLINE	static inline\n" \
"#else\n" \
"	#define INLINE\n" \
"#endif\n" \
"\n" \
"#if defined(__NV_CL_C_VERSION)\n" \
"	#define PTX_ASM	1\n" \
"#endif\n" \
"\n" \
"#if !defined(N_SZ)\n" \
"#define N_SZ			65536u\n" \
"#define LBASE			22\n" \
"#define P1				4194304001u\n" \
"#define Q1				100663297u\n" \
"#define RSQ1			232465106u\n" \
"#define P2				4076863489u\n" \
"#define Q2				218103809u\n" \
"#define RSQ2			3444438393u\n" \
"#define INVP2_P1		1797558821u\n" \
"#define P1P2			17099604843443519489ul\n" \
"#define NORM1			4194176001u\n" \
"#define NORM2			4076739073u\n" \
"#define CARRY_LENGTH	4u\n" \
"#endif\n" \
"#define BASE			(1u << LBASE)\n" \
"\n" \
"typedef uint	sz_t;\n" \
"typedef uint	uint_32;\n" \
"typedef int		int_32;\n" \
"typedef ulong	uint_64;\n" \
"typedef long	int_64;\n" \
"typedef uint2	uint2_32;\n" \
"typedef uint4	uint4_32;\n" \
"typedef uint8	uint8_32;\n" \
"\n" \
"// --- modular arithmetic\n" \
"\n" \
"#define RNS		uint2_32\n" \
"#define RNS2	uint4_32\n" \
"#define RNS4	uint8_32\n" \
"\n" \
"INLINE uint_32 _addmod(const uint_32 lhs, const uint_32 rhs, const uint_32 p)\n" \
"{\n" \
"	return lhs + rhs - ((lhs >= p - rhs) ? p : 0);\n" \
"}\n" \
"\n" \
"INLINE uint_32 _submod(const uint_32 lhs, const uint_32 rhs, const uint_32 p)\n" \
"{\n" \
"	return lhs - rhs + ((lhs < rhs) ? p : 0);\n" \
"}\n" \
"\n" \
"// 2 mul + 2 mul_hi\n" \
"INLINE uint_32 _mulmod(const uint_32 lhs, const uint_32 rhs, const uint_32 p, const uint_32 q)\n" \
"{\n" \
"	const uint_64 t = lhs * (uint_64)(rhs);\n" \
"	const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);\n" \
"	const uint_32 mp = mul_hi(lo * q, p);\n" \
"	return _submod(hi, mp, p);\n" \
"}\n" \
"\n" \
"INLINE int_32 _get_int(const uint_32 n, const uint_32 p) { return (int_32)(n - ((n >= p / 2) ? p : 0)); }\n" \
"INLINE uint_32 _set_int(const int_32 i, const uint_32 p) { return (uint_32)(i) + ((i < 0) ? p : 0); }\n" \
"\n" \
"INLINE RNS addmod(const RNS lhs, const RNS rhs)\n" \
"{\n" \
"	return (RNS)(_addmod(lhs.s0, rhs.s0, P1), _addmod(lhs.s1, rhs.s1, P2));\n" \
"}\n" \
"\n" \
"INLINE RNS submod(const RNS lhs, const RNS rhs)\n" \
"{\n" \
"	return (RNS)(_submod(lhs.s0, rhs.s0, P1), _submod(lhs.s1, rhs.s1, P2));\n" \
"}\n" \
"\n" \
"INLINE RNS mulmod(const RNS lhs, const RNS rhs)\n" \
"{\n" \
"	return (RNS)(_mulmod(lhs.s0, rhs.s0, P1, Q1), _mulmod(lhs.s1, rhs.s1, P2, Q2));\n" \
"}\n" \
"\n" \
"INLINE RNS set_int(const int_32 i)\n" \
"{\n" \
"	return (RNS)(_set_int(i, P1), _set_int(i, P2));\n" \
"}\n" \
"\n" \
"INLINE int_32 get_int(const RNS lhs)\n" \
"{\n" \
"	return _get_int(lhs.s0, P1);\n" \
"}\n" \
"\n" \
"// --- I/O ---\n" \
"\n" \
"INLINE void _loadg(const sz_t n, RNS * const xl, __global const RNS * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }\n" \
"INLINE void _storeg(const sz_t n, __global RNS * restrict const x, const sz_t s, const RNS * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }\n" \
"\n" \
"INLINE RNS2 _load2g(__global const RNS * restrict const w, const sz_t j) { return ((__global const RNS2 *)w)[j]; }\n" \
"INLINE RNS4 _load4g(__global const RNS * restrict const w, const sz_t j) { return ((__global const RNS4 *)w)[j]; }\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define FWD2(x0, x1, w) \\\n" \
"{ \\\n" \
"	const RNS t = mulmod(x1, w); \\\n" \
"	x1 = submod(x0, t); x0 = addmod(x0, t); \\\n" \
"}\n" \
"\n" \
"#define BCK2(x0, x1, w_i) \\\n" \
"{ \\\n" \
"	const RNS t = submod(x0, x1); x0 = addmod(x0, x1); \\\n" \
"	x1 = mulmod(t, w_i); \\\n" \
"}\n" \
"\n" \
"#define SQR2(x0, x1, w) \\\n" \
"{ \\\n" \
"	const RNS t = mulmod(mulmod(x1, x1), w); \\\n" \
"	x1 = mulmod(addmod(x0, x0), x1); \\\n" \
"	x0 = addmod(mulmod(x0, x0), t); \\\n" \
"}\n" \
"\n" \
"#define SQR2N(x0, x1, w) \\\n" \
"{ \\\n" \
"	const RNS t = mulmod(mulmod(x1, x1), w); \\\n" \
"	x1 = mulmod(addmod(x0, x0), x1); \\\n" \
"	x0 = submod(mulmod(x0, x0), t); \\\n" \
"}\n" \
"\n" \
"#define MUL2(x0, x1, xp0, xp1, w) \\\n" \
"{ \\\n" \
"	const RNS t = mulmod(mulmod(x1, xp1), w); \\\n" \
"	x1 = addmod(mulmod(x0, xp1), mulmod(xp0, x1)); \\\n" \
"	x0 = addmod(mulmod(x0, xp0), t); \\\n" \
"}\n" \
"\n" \
"#define MUL2N(x0, x1, xp0, xp1, w) \\\n" \
"{ \\\n" \
"	const RNS t = mulmod(mulmod(x1, xp1), w); \\\n" \
"	x1 = addmod(mulmod(x0, xp1), mulmod(xp0, x1)); \\\n" \
"	x0 = submod(mulmod(x0, xp0), t); \\\n" \
"}\n" \
"\n" \
"// --- transform/inline ---\n" \
"\n" \
"INLINE void _forward8(RNS x[8], const RNS w1, const RNS2 w2, const RNS4 w4)\n" \
"{\n" \
"	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);\n" \
"	FWD2(x[0], x[2], w2.s01); FWD2(x[1], x[3], w2.s01); FWD2(x[4], x[6], w2.s23); FWD2(x[5], x[7], w2.s23);\n" \
"	FWD2(x[0], x[1], w4.s01); FWD2(x[2], x[3], w4.s23); FWD2(x[4], x[5], w4.s45); FWD2(x[6], x[7], w4.s67);\n" \
"}\n" \
"\n" \
"INLINE void _backward8(RNS x[8], const RNS w1_i, const RNS2 w2_i, const RNS4 w4_i)\n" \
"{\n" \
"	BCK2(x[0], x[1], w4_i.s01); BCK2(x[2], x[3], w4_i.s23); BCK2(x[4], x[5], w4_i.s45); BCK2(x[6], x[7], w4_i.s67);\n" \
"	BCK2(x[0], x[2], w2_i.s01); BCK2(x[1], x[3], w2_i.s01); BCK2(x[4], x[6], w2_i.s23); BCK2(x[5], x[7], w2_i.s23);\n" \
"	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);\n" \
"}\n" \
"\n" \
"INLINE void _forward8_0(RNS x[8], const RNS w1, const RNS2 w2, const RNS4 w4)\n" \
"{\n" \
"	for (sz_t i = 0; i < 4; ++i) x[i] = mulmod(x[i], (RNS)(RSQ1, RSQ2));\n" \
"	_forward8(x, w1, w2, w4);\n" \
"}\n" \
"\n" \
"INLINE void _square2x4(RNS x[8], const RNS2 w2)\n" \
"{\n" \
"	SQR2(x[0], x[1], w2.s01); SQR2N(x[2], x[3], w2.s01); SQR2(x[4], x[5], w2.s23); SQR2N(x[6], x[7], w2.s23);\n" \
"}\n" \
"\n" \
"INLINE void _square4x2(RNS x[8], const RNS2 w2, const RNS2 w2_i)\n" \
"{\n" \
"	FWD2(x[0], x[2], w2.s01); FWD2(x[1], x[3], w2.s01); FWD2(x[4], x[6], w2.s23); FWD2(x[5], x[7], w2.s23);\n" \
"	_square2x4(x, w2);\n" \
"	BCK2(x[0], x[2], w2_i.s01); BCK2(x[1], x[3], w2_i.s01); BCK2(x[4], x[6], w2_i.s23); BCK2(x[5], x[7], w2_i.s23);\n" \
"}\n" \
"\n" \
"INLINE void _square8(RNS x[8], const RNS w1, const RNS w1_i, const RNS2 w2, const RNS2 w2_i)\n" \
"{\n" \
"	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);\n" \
"	_square4x2(x, w2, w2_i);\n" \
"	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);\n" \
"}\n" \
"\n" \
"INLINE void _mul2x4(RNS x[8], const RNS xp[8], const RNS2 w2)\n" \
"{\n" \
"	MUL2(x[0], x[1], xp[0], xp[1], w2.s01); MUL2N(x[2], x[3], xp[2], xp[3], w2.s01); MUL2(x[4], x[5], xp[4], xp[5], w2.s23); MUL2N(x[6], x[7], xp[6], xp[7], w2.s23);\n" \
"}\n" \
"\n" \
"INLINE void _mul4x2(RNS x[8], const RNS xp[8], const RNS2 w2, const RNS2 w2_i)\n" \
"{\n" \
"	FWD2(x[0], x[2], w2.s01); FWD2(x[1], x[3], w2.s01); FWD2(x[4], x[6], w2.s23); FWD2(x[5], x[7], w2.s23);\n" \
"	_mul2x4(x, xp, w2);\n" \
"	BCK2(x[0], x[2], w2_i.s01); BCK2(x[1], x[3], w2_i.s01); BCK2(x[4], x[6], w2_i.s23); BCK2(x[5], x[7], w2_i.s23);\n" \
"}\n" \
"\n" \
"INLINE void _mul8(RNS x[8], const RNS xp[8], const RNS w1, const RNS w1_i, const RNS2 w2, const RNS2 w2_i)\n" \
"{\n" \
"	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);\n" \
"	_mul4x2(x, xp, w2, w2_i);\n" \
"	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);\n" \
"}\n" \
"\n" \
"INLINE void _fwd4x2(RNS x[8], const RNS2 w2)\n" \
"{\n" \
"	FWD2(x[0], x[2], w2.s01); FWD2(x[1], x[3], w2.s01); FWD2(x[4], x[6], w2.s23); FWD2(x[5], x[7], w2.s23);\n" \
"}\n" \
"\n" \
"INLINE void _fwd8(RNS x[8], const RNS w1, const RNS2 w2)\n" \
"{\n" \
"	FWD2(x[0], x[4], w1); FWD2(x[2], x[6], w1); FWD2(x[1], x[5], w1); FWD2(x[3], x[7], w1);\n" \
"	_fwd4x2(x, w2);\n" \
"}\n" \
"\n" \
"// ---\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"// --- transform without local mem ---\n" \
"\n" \
"__kernel\n" \
"void forward8(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w, const int_32 lm, const uint_32 s)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const RNS * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, sj = s + (id >> lm), k = 7 * (id & ~(m - 1)) + id;\n" \
"\n" \
"	const RNS w1 = w[sj];\n" \
"	const RNS2 w2 = _load2g(w, sj);\n" \
"	const RNS4 w4 = _load4g(w, sj);\n" \
"\n" \
"	RNS xl[8]; _loadg(8, xl, &x[k], m);\n" \
"	_forward8(xl, w1, w2, w4);\n" \
"	_storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void backward8(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w, const int_32 lm, const uint_32 s)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const RNS * restrict const w_i = &reg[reg_w * N_SZ + N_SZ / 2];\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, sj = s + (id >> lm), k = 7 * (id & ~(m - 1)) + id;\n" \
"\n" \
"	const RNS w1_i = w_i[sj];\n" \
"	const RNS2 w2_i = _load2g(w_i, sj);\n" \
"	const RNS4 w4_i = _load4g(w_i, sj);\n" \
"\n" \
"	RNS xl[8]; _loadg(8, xl, &x[k], m);\n" \
"	_backward8(xl, w1_i, w2_i, w4_i);\n" \
"	_storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void forward8_0(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const RNS * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = N_SZ / 8, k = id;\n" \
"\n" \
"	// Montgomery form of MF of I to convert input into MF\n" \
"	const RNS w1 = w[0];\n" \
"	const RNS2 w2 = _load2g(w, 1);\n" \
"	const RNS4 w4 = _load4g(w, 1);\n" \
"\n" \
"	RNS xl[8]; _loadg(8, xl, &x[k], m);\n" \
"	_forward8_0(xl, w1, w2, w4);\n" \
"	_storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square2x4(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const RNS * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const RNS2 w2 = _load2g(w, sj);\n" \
"\n" \
"	RNS xl[8]; _loadg(8, xl, &x[k], m);\n" \
"	_square2x4(xl, w2);\n" \
"	_storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square4x2(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const RNS * restrict const w = &reg[reg_w * N_SZ];\n" \
"	__global const RNS * restrict const w_i = &w[N_SZ / 2];\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const RNS2 w2 = _load2g(w, sj), w2_i = _load2g(w_i, sj);\n" \
"\n" \
"	RNS xl[8]; _loadg(8, xl, &x[k], m);\n" \
"	_square4x2(xl, w2, w2_i);\n" \
"	_storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square8(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const RNS * restrict const w = &reg[reg_w * N_SZ];\n" \
"	__global const RNS * restrict const w_i = &w[N_SZ / 2];\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const RNS w1 = w[sj], w1_i = w_i[sj];\n" \
"	const RNS2 w2 = _load2g(w, sj), w2_i = _load2g(w_i, sj);\n" \
"\n" \
"	RNS xl[8]; _loadg(8, xl, &x[k], m);\n" \
"	_square8(xl, w1, w1_i, w2, w2_i);\n" \
"	_storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2x4(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const RNS * restrict const xp = &reg[reg_xp * N_SZ];\n" \
"	__global const RNS * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const RNS2 w2 = _load2g(w, sj);\n" \
"\n" \
"	RNS xl[8]; _loadg(8, xl, &x[k], m);\n" \
"	RNS xpl[8]; _loadg(8, xpl, &xp[k], m);\n" \
"	_mul2x4(xl, xpl, w2);\n" \
"	_storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4x2(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const RNS * restrict const xp = &reg[reg_xp * N_SZ];\n" \
"	__global const RNS * restrict const w = &reg[reg_w * N_SZ];\n" \
"	__global const RNS * restrict const w_i = &w[N_SZ / 2];\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const RNS2 w2 = _load2g(w, sj), w2_i = _load2g(w_i, sj);\n" \
"\n" \
"	RNS xl[8]; _loadg(8, xl, &x[k], m);\n" \
"	RNS xpl[8]; _loadg(8, xpl, &xp[k], m);\n" \
"	_mul4x2(xl, xpl, w2, w2_i);\n" \
"	_storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul8(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const RNS * restrict const xp = &reg[reg_xp * N_SZ];\n" \
"	__global const RNS * restrict const w = &reg[reg_w * N_SZ];\n" \
"	__global const RNS * restrict const w_i = &w[N_SZ / 2];\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const RNS w1 = w[sj], w1_i = w_i[sj];\n" \
"	const RNS2 w2 = _load2g(w, sj), w2_i = _load2g(w_i, sj);\n" \
"\n" \
"	RNS xl[8]; _loadg(8, xl, &x[k], m);\n" \
"	RNS xpl[8]; _loadg(8, xpl, &xp[k], m);\n" \
"	_mul8(xl, xpl, w1, w1_i, w2, w2_i);\n" \
"	_storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void fwd4x2(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const RNS * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const RNS2 w2 = _load2g(w, sj);\n" \
"\n" \
"	RNS xl[8]; _loadg(8, xl, &x[k], m);\n" \
"	_fwd4x2(xl, w2);\n" \
"	_storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void fwd8(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const RNS * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const RNS w1 = w[sj];\n" \
"	const RNS2 w2 = _load2g(w, sj);\n" \
"\n" \
"	RNS xl[8]; _loadg(8, xl, &x[k], m);\n" \
"	_fwd8(xl, w1, w2);\n" \
"	_storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"// --- transform using local mem ---\n" \
"\n" \
"// --- carry ---\n" \
"\n" \
"// Chinese remainder theorem\n" \
"INLINE int_64 garner2(const RNS r)\n" \
"{\n" \
"	const uint_32 r1 = r.s0, r2 = r.s1;\n" \
"	const uint_32 u = _mulmod(_submod(r1, r2, P1), INVP2_P1, P1, Q1);	// P2 < P1\n" \
"	const uint_64 n = r2 + u * (uint_64)(P2);\n" \
"	const bool b = (n > P1P2 / 2);\n" \
"	return (int_64)(n - (b ? P1P2 : 0));\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void carry_mul(__global RNS * restrict const reg, __global int_64 * restrict const carry, const uint_32 reg_x, const int_32 a)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global RNS * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];\n" \
"\n" \
"	int_64 f = 0;\n" \
"	for (sz_t j = 0; j < CARRY_LENGTH; ++j)\n" \
"	{\n" \
"		// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF\n" \
"		const RNS u = mulmod(xi[j], (RNS)(NORM1, NORM2));\n" \
"		f += a * garner2(u);\n" \
"		const uint_32 r = (uint_32)(f) & (BASE - 1);\n" \
"		xi[j] = (RNS)(r, r);\n" \
"		f >>= LBASE;\n" \
"	}\n" \
"	carry[id] = f;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void carry_sub(__global RNS * restrict const reg, __global int_64 * restrict const carry, const uint_32 reg_x, const uint_32 reg_y)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global RNS * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];\n" \
"	__global const RNS * restrict const yi = &reg[reg_y * N_SZ + CARRY_LENGTH * id];\n" \
"\n" \
"	int_64 f = 0;\n" \
"	for (sz_t j = 0; j < CARRY_LENGTH; ++j)\n" \
"	{\n" \
"		// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF\n" \
"		const RNS u = mulmod(xi[j], (RNS)(NORM1, NORM2));\n" \
"		f += garner2(u);\n" \
"		f -= get_int(yi[j]);\n" \
"		const uint_32 r = (uint_32)(f) & (BASE - 1);\n" \
"		xi[j] = (RNS)(r, r);\n" \
"		f >>= LBASE;\n" \
"	}\n" \
"	carry[id] = f;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void carry_complete(__global RNS * restrict const reg, __global const int_64 * restrict const carry, const uint_32 reg_x, const int_32 is_nega)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global RNS * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];\n" \
"\n" \
"	int_64 f = carry[((id == 0) ? (N_SZ / CARRY_LENGTH) : id) - 1];\n" \
"	if ((id == 0) && (is_nega != 0)) f = -f;\n" \
"\n" \
"	for (sz_t j = 0; j < CARRY_LENGTH - 1; ++j)\n" \
"	{\n" \
"		f += (int_64)(xi[j].s0);\n" \
"		const uint_32 r = (uint_32)(f) & (BASE - 1);\n" \
"		xi[j] = (RNS)(r, r);\n" \
"		f >>= LBASE;\n" \
"	}\n" \
"	f += (int_64)(xi[CARRY_LENGTH - 1].s0);\n" \
"	xi[CARRY_LENGTH - 1] = set_int((int_32)f);\n" \
"}\n" \
"\n" \
"// --- strict modulo ---\n" \
"\n" \
"// b-1, b-1, ..., b-1, b + a, ... => b, 0, ..., 0, a, ...\n" \
"INLINE void fix_base(__global RNS * restrict const x)\n" \
"{\n" \
"	if (x[N_SZ - 1].s0 == BASE - 1)\n" \
"	{\n" \
"		sz_t i = 2; for (; i < N_SZ; ++i) { if (x[N_SZ - i].s0 != BASE - 1) break; }\n" \
"		if (get_int(x[N_SZ - i]) >= (int_32)(BASE))\n" \
"		{\n" \
"			x[N_SZ - i] -= (RNS)(BASE, BASE);\n" \
"			for (sz_t j = i - 1; j >= 2; --j) x[N_SZ - j] = (RNS)(0, 0);\n" \
"			x[N_SZ - 1] = (RNS)(BASE, BASE);\n" \
"		}\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mod_Q(__global RNS * restrict const reg, const uint_32 reg_x)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"\n" \
"	fix_base(x);\n" \
"\n" \
"	int_32 hi = get_int(x[N_SZ - 1]);\n" \
"	if (hi >= (int_32)(BASE))\n" \
"	{\n" \
"		int_32 lo = get_int(x[0]);\n" \
"		while (hi >= (int_32)(BASE)) { hi -= (int_32)(BASE); lo -= 1; }	// - Q\n" \
"		x[N_SZ - 1] = set_int(hi); x[0] = set_int(lo);\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mod_R_add2(__global RNS * restrict const reg, const uint_32 reg_x)\n" \
"{\n" \
"	__global RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"\n" \
"	fix_base(x);\n" \
"\n" \
"	int_32 hi = get_int(x[N_SZ - 1]), lo = get_int(x[0]);\n" \
"	if (hi >= (int_32)(BASE))\n" \
"	{\n" \
"		while (hi >= (int_32)(BASE)) { hi -= (int_32)(BASE); lo += 1; }	// - R\n" \
"		x[N_SZ - 1] = set_int(hi);\n" \
"	}\n" \
"	x[0] = set_int(lo + 2);\n" \
"}\n" \
"\n" \
"// --- misc ---\n" \
"\n" \
"__kernel\n" \
"void set(__global RNS * restrict const reg, const uint_32 reg_y, const uint_32 reg_x)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global RNS * restrict const y = &reg[reg_y * N_SZ];\n" \
"	__global const RNS * restrict const x = &reg[reg_x * N_SZ];\n" \
"	y[id] = x[id];\n" \
"}\n" \
"";
