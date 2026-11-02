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
"#define CARRY_WG_SZ     256u\n" \
"#define Z_OFFSET		655360u\n" \
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
"\n" \
"// --- modular arithmetic\n" \
"\n" \
"typedef uint_32		ZP;\n" \
"typedef uint2_32	ZP2;\n" \
"typedef uint4_32	ZP4;\n" \
"\n" \
"#define	PQ1		(ZP2)(P1, Q1)\n" \
"#define	PQ2		(ZP2)(P2, Q2)\n" \
"\n" \
"__constant ZP2 g_pq[2] = { PQ1, PQ2 };\n" \
"__constant ZP g_rsq[2] = { RSQ1, RSQ2 };\n" \
"\n" \
"INLINE ZP addmod(const ZP lhs, const ZP rhs, const ZP p)\n" \
"{\n" \
"	return lhs + rhs - ((lhs >= p - rhs) ? p : 0);\n" \
"}\n" \
"\n" \
"INLINE ZP submod(const ZP lhs, const ZP rhs, const ZP p)\n" \
"{\n" \
"	return lhs - rhs + ((lhs < rhs) ? p : 0);\n" \
"}\n" \
"\n" \
"// 2 mul + 2 mul_hi\n" \
"INLINE ZP mulmod(const ZP lhs, const ZP rhs, const ZP2 pq)\n" \
"{\n" \
"	const uint_64 t = lhs * (uint_64)(rhs);\n" \
"	const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);\n" \
"	const uint_32 mp = mul_hi(lo * pq.s1, pq.s0);\n" \
"	return submod(hi, mp, pq.s0);\n" \
"}\n" \
"\n" \
"INLINE int_32 get_int(const uint_32 n, const uint_32 p) { return (int_32)(n - ((n >= p / 2) ? p : 0)); }\n" \
"INLINE uint_32 set_int(const int_32 i, const uint_32 p) { return (uint_32)(i) + ((i < 0) ? p : 0); }\n" \
"\n" \
"// --- I/O ---\n" \
"\n" \
"INLINE void loadg(const sz_t n, ZP * const xl, __global const ZP * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }\n" \
"INLINE void storeg(const sz_t n, __global ZP * restrict const x, const sz_t s, const ZP * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }\n" \
"\n" \
"INLINE ZP load1g(__global const ZP * restrict const x, const sz_t j) { return x[j]; }\n" \
"INLINE ZP2 load2g(__global const ZP * restrict const x, const sz_t j) { return ((__global const ZP2 *)x)[j]; }\n" \
"INLINE ZP4 load4g(__global const ZP * restrict const x, const sz_t j) { return ((__global const ZP4 *)x)[j]; }\n" \
"\n" \
"INLINE void store4g(__global ZP * restrict const w, const sz_t j, const ZP4 x4) { ((__global ZP4 *)w)[j] = x4; }\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define FWD2(x0, x1, w) \\\n" \
"{ \\\n" \
"	const ZP t = mulmod(x1, w, pq); \\\n" \
"	x1 = submod(x0, t, pq.s0); x0 = addmod(x0, t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define BCK2(x0, x1, w_i) \\\n" \
"{ \\\n" \
"	const ZP t = submod(x0, x1, pq.s0); x0 = addmod(x0, x1, pq.s0); \\\n" \
"	x1 = mulmod(t, w_i, pq); \\\n" \
"}\n" \
"\n" \
"#define SQR2(x0, x1, w) \\\n" \
"{ \\\n" \
"	const ZP t = mulmod(mulmod(x1, x1, pq), w, pq); \\\n" \
"	x1 = mulmod(addmod(x0, x0, pq.s0), x1, pq); \\\n" \
"	x0 = addmod(mulmod(x0, x0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define SQR2N(x0, x1, w) \\\n" \
"{ \\\n" \
"	const ZP t = mulmod(mulmod(x1, x1, pq), w, pq); \\\n" \
"	x1 = mulmod(addmod(x0, x0, pq.s0), x1, pq); \\\n" \
"	x0 = submod(mulmod(x0, x0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define MUL2(x0, x1, xp0, xp1, w) \\\n" \
"{ \\\n" \
"	const ZP t = mulmod(mulmod(x1, xp1, pq), w, pq); \\\n" \
"	x1 = addmod(mulmod(x0, xp1, pq), mulmod(xp0, x1, pq), pq.s0); \\\n" \
"	x0 = addmod(mulmod(x0, xp0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define MUL2N(x0, x1, xp0, xp1, w) \\\n" \
"{ \\\n" \
"	const ZP t = mulmod(mulmod(x1, xp1, pq), w, pq); \\\n" \
"	x1 = addmod(mulmod(x0, xp1, pq), mulmod(xp0, x1, pq), pq.s0); \\\n" \
"	x0 = submod(mulmod(x0, xp0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"// --- transform/inline ---\n" \
"\n" \
"INLINE void _forward8(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2, const ZP4 w4)\n" \
"{\n" \
"	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);\n" \
"	FWD2(x[0], x[2], w2.s0); FWD2(x[1], x[3], w2.s0); FWD2(x[4], x[6], w2.s1); FWD2(x[5], x[7], w2.s1);\n" \
"	FWD2(x[0], x[1], w4.s0); FWD2(x[2], x[3], w4.s1); FWD2(x[4], x[5], w4.s2); FWD2(x[6], x[7], w4.s3);\n" \
"}\n" \
"\n" \
"INLINE void _backward8(const ZP2 pq, ZP x[8], const ZP w1_i, const ZP2 w2_i, const ZP4 w4_i)\n" \
"{\n" \
"	BCK2(x[0], x[1], w4_i.s0); BCK2(x[2], x[3], w4_i.s1); BCK2(x[4], x[5], w4_i.s2); BCK2(x[6], x[7], w4_i.s3);\n" \
"	BCK2(x[0], x[2], w2_i.s0); BCK2(x[1], x[3], w2_i.s0); BCK2(x[4], x[6], w2_i.s1); BCK2(x[5], x[7], w2_i.s1);\n" \
"	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);\n" \
"}\n" \
"\n" \
"INLINE void _forward8_0(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2, const ZP4 w4, const ZP rsq)\n" \
"{\n" \
"	for (sz_t i = 0; i < 4; ++i) x[i] = mulmod(x[i], rsq, pq);\n" \
"	_forward8(pq, x, w1, w2, w4);\n" \
"}\n" \
"\n" \
"INLINE void _square2x4(const ZP2 pq, ZP x[8], const ZP2 w2)\n" \
"{\n" \
"	SQR2(x[0], x[1], w2.s0); SQR2N(x[2], x[3], w2.s0); SQR2(x[4], x[5], w2.s1); SQR2N(x[6], x[7], w2.s1);\n" \
"}\n" \
"\n" \
"INLINE void _square4x2(const ZP2 pq, ZP x[8], const ZP2 w2, const ZP2 w2_i)\n" \
"{\n" \
"	FWD2(x[0], x[2], w2.s0); FWD2(x[1], x[3], w2.s0); FWD2(x[4], x[6], w2.s1); FWD2(x[5], x[7], w2.s1);\n" \
"	_square2x4(pq, x, w2);\n" \
"	BCK2(x[0], x[2], w2_i.s0); BCK2(x[1], x[3], w2_i.s0); BCK2(x[4], x[6], w2_i.s1); BCK2(x[5], x[7], w2_i.s1);\n" \
"}\n" \
"\n" \
"INLINE void _square8(const ZP2 pq, ZP x[8], const ZP w1, const ZP w1_i, const ZP2 w2, const ZP2 w2_i)\n" \
"{\n" \
"	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);\n" \
"	_square4x2(pq, x, w2, w2_i);\n" \
"	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);\n" \
"}\n" \
"\n" \
"INLINE void _mul2x4(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP2 w2)\n" \
"{\n" \
"	MUL2(x[0], x[1], xp[0], xp[1], w2.s0); MUL2N(x[2], x[3], xp[2], xp[3], w2.s0); MUL2(x[4], x[5], xp[4], xp[5], w2.s1); MUL2N(x[6], x[7], xp[6], xp[7], w2.s1);\n" \
"}\n" \
"\n" \
"INLINE void _mul4x2(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP2 w2, const ZP2 w2_i)\n" \
"{\n" \
"	FWD2(x[0], x[2], w2.s0); FWD2(x[1], x[3], w2.s0); FWD2(x[4], x[6], w2.s1); FWD2(x[5], x[7], w2.s1);\n" \
"	_mul2x4(pq, x, xp, w2);\n" \
"	BCK2(x[0], x[2], w2_i.s0); BCK2(x[1], x[3], w2_i.s0); BCK2(x[4], x[6], w2_i.s1); BCK2(x[5], x[7], w2_i.s1);\n" \
"}\n" \
"\n" \
"INLINE void _mul8(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP w1, const ZP w1_i, const ZP2 w2, const ZP2 w2_i)\n" \
"{\n" \
"	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);\n" \
"	_mul4x2(pq, x, xp, w2, w2_i);\n" \
"	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);\n" \
"}\n" \
"\n" \
"INLINE void _fwd4x2(const ZP2 pq, ZP x[8], const ZP2 w2)\n" \
"{\n" \
"	FWD2(x[0], x[2], w2.s0); FWD2(x[1], x[3], w2.s0); FWD2(x[4], x[6], w2.s1); FWD2(x[5], x[7], w2.s1);\n" \
"}\n" \
"\n" \
"INLINE void _fwd8(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2)\n" \
"{\n" \
"	FWD2(x[0], x[4], w1); FWD2(x[2], x[6], w1); FWD2(x[1], x[5], w1); FWD2(x[3], x[7], w1);\n" \
"	_fwd4x2(pq, x, w2);\n" \
"}\n" \
"\n" \
"// ---\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define DECLARE_VAR_REG() \\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid / (N_SZ / 8), id = gid % (N_SZ / 8); \\\n" \
"	const ZP2 pq = g_pq[lid]; \\\n" \
"	__global ZP * restrict const reg = &regm[lid * Z_OFFSET];\n" \
"\n" \
"// --- transform without local mem ---\n" \
"\n" \
"__kernel\n" \
"void forward8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const int_32 lm, const uint_32 s)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const ZP * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t m = 1u << lm, sj = s + (id >> lm), k = 7 * (id & ~(m - 1)) + id;\n" \
"\n" \
"	const ZP w1 = load1g(w, sj); const ZP2 w2 = load2g(w, sj); const ZP4 w4 = load4g(w, sj);\n" \
"\n" \
"	ZP xl[8]; loadg(8, xl, &x[k], m);\n" \
"	_forward8(pq, xl, w1, w2, w4);\n" \
"	storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void backward8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const int_32 lm, const uint_32 s)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const ZP * restrict const w_i = &reg[reg_w * N_SZ + N_SZ / 2];\n" \
"	const sz_t m = 1u << lm, sj = s + (id >> lm), k = 7 * (id & ~(m - 1)) + id;\n" \
"\n" \
"	const ZP w1_i = load1g(w_i, sj); const ZP2 w2_i = load2g(w_i, sj); const ZP4 w4_i = load4g(w_i, sj);\n" \
"\n" \
"	ZP xl[8]; loadg(8, xl, &x[k], m);\n" \
"	_backward8(pq, xl, w1_i, w2_i, w4_i);\n" \
"	storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void forward8_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const ZP * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t m = N_SZ / 8, k = id;\n" \
"\n" \
"	// Montgomery form of MF of I to convert input into MF\n" \
"	const ZP w1 = load1g(w, 0); const ZP2 w2 = load2g(w, 1); const ZP4 w4 = load4g(w, 1);\n" \
"\n" \
"	ZP xl[8]; loadg(8, xl, &x[k], m);\n" \
"	_forward8_0(pq, xl, w1, w2, w4, g_rsq[lid]);\n" \
"	storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square2x4(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const ZP * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const ZP2 w2 = load2g(w, sj);\n" \
"\n" \
"	ZP xl[8]; loadg(8, xl, &x[k], m);\n" \
"	_square2x4(pq, xl, w2);\n" \
"	storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const ZP * restrict const w = &reg[reg_w * N_SZ];\n" \
"	__global const ZP * restrict const w_i = &w[N_SZ / 2];\n" \
"	const sz_t m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const ZP2 w2 = load2g(w, sj), w2_i = load2g(w_i, sj);\n" \
"\n" \
"	ZP xl[8]; loadg(8, xl, &x[k], m);\n" \
"	_square4x2(pq, xl, w2, w2_i);\n" \
"	storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const ZP * restrict const w = &reg[reg_w * N_SZ];\n" \
"	__global const ZP * restrict const w_i = &w[N_SZ / 2];\n" \
"	const sz_t m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const ZP w1 = load1g(w, sj), w1_i = load1g(w_i, sj);\n" \
"	const ZP2 w2 = load2g(w, sj), w2_i = load2g(w_i, sj);\n" \
"\n" \
"	ZP xl[8]; loadg(8, xl, &x[k], m);\n" \
"	_square8(pq, xl, w1, w1_i, w2, w2_i);\n" \
"	storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2x4(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const ZP * restrict const xp = &reg[reg_xp * N_SZ];\n" \
"	__global const ZP * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const ZP2 w2 = load2g(w, sj);\n" \
"\n" \
"	ZP xl[8]; loadg(8, xl, &x[k], m);\n" \
"	ZP xpl[8]; loadg(8, xpl, &xp[k], m);\n" \
"	_mul2x4(pq, xl, xpl, w2);\n" \
"	storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const ZP * restrict const xp = &reg[reg_xp * N_SZ];\n" \
"	__global const ZP * restrict const w = &reg[reg_w * N_SZ];\n" \
"	__global const ZP * restrict const w_i = &w[N_SZ / 2];\n" \
"	const sz_t m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const ZP2 w2 = load2g(w, sj), w2_i = load2g(w_i, sj);\n" \
"\n" \
"	ZP xl[8]; loadg(8, xl, &x[k], m);\n" \
"	ZP xpl[8]; loadg(8, xpl, &xp[k], m);\n" \
"	_mul4x2(pq, xl, xpl, w2, w2_i);\n" \
"	storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const ZP * restrict const xp = &reg[reg_xp * N_SZ];\n" \
"	__global const ZP * restrict const w = &reg[reg_w * N_SZ];\n" \
"	__global const ZP * restrict const w_i = &w[N_SZ / 2];\n" \
"	const sz_t m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const ZP w1 = load1g(w, sj), w1_i = load1g(w_i, sj);\n" \
"	const ZP2 w2 = load2g(w, sj), w2_i = load2g(w_i, sj);\n" \
"\n" \
"	ZP xl[8]; loadg(8, xl, &x[k], m);\n" \
"	ZP xpl[8]; loadg(8, xpl, &xp[k], m);\n" \
"	_mul8(pq, xl, xpl, w1, w1_i, w2, w2_i);\n" \
"	storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void fwd4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const ZP * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const ZP2 w2 = load2g(w, sj);\n" \
"\n" \
"	ZP xl[8]; loadg(8, xl, &x[k], m);\n" \
"	_fwd4x2(pq, xl, w2);\n" \
"	storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void fwd8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"	__global const ZP * restrict const w = &reg[reg_w * N_SZ];\n" \
"	const sz_t m = 1, sj = N_SZ / 8 + id, k = 8 * id;\n" \
"\n" \
"	const ZP w1 = load1g(w, sj);\n" \
"	const ZP2 w2 = load2g(w, sj);\n" \
"\n" \
"	ZP xl[8]; loadg(8, xl, &x[k], m);\n" \
"	_fwd8(pq, xl, w1, w2);\n" \
"	storeg(8, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"// --- transform using local mem ---\n" \
"\n" \
"// --- carry ---\n" \
"\n" \
"// Chinese remainder theorem\n" \
"INLINE int_64 garner2(const ZP r1, const ZP r2)\n" \
"{\n" \
"	const ZP u = mulmod(submod(r1, r2, P1), INVP2_P1, PQ1);	// P2 < P1\n" \
"	const uint_64 n = r2 + u * (uint_64)(P2);\n" \
"	const bool b = (n > P1P2 / 2);\n" \
"	return (int_64)(n - (b ? P1P2 : 0));\n" \
"}\n" \
"\n" \
"#define NULL_PTR	((__global const ZP *)0)\n" \
"\n" \
"INLINE void carry_mul_sub_1(const __global ZP * restrict const xi, __global const ZP * restrict const yi,\n" \
"	__global int_64 * restrict const carry, __local int_64 * const cl, uint_32 r[CARRY_LENGTH], const sz_t id, const int_32 a)\n" \
"{\n" \
"	// assert CARRY_LENGTH = 4\n" \
"	const ZP4 x1_4 = load4g(xi, 0), x2_4 = load4g(xi, Z_OFFSET / 4);\n" \
"	ZP x1[4]; x1[0] = x1_4.s0; x1[1] = x1_4.s1; x1[2] = x1_4.s2; x1[3] = x1_4.s3;\n" \
"	ZP x2[4]; x2[0] = x2_4.s0; x2[1] = x2_4.s1; x2[2] = x2_4.s2; x2[3] = x2_4.s3;\n" \
"\n" \
"	int_64 f = 0;\n" \
"	for (sz_t j = 0; j < CARRY_LENGTH; ++j)\n" \
"	{\n" \
"		// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF\n" \
"		const ZP u1 = mulmod(x1[j], NORM1, PQ1), u2 = mulmod(x2[j], NORM2, PQ2);\n" \
"		f += a * garner2(u1, u2);\n" \
"		if (yi != NULL_PTR) f -= get_int(yi[j], P1);\n" \
"		r[j] = (uint_32)(f) & (BASE - 1);\n" \
"		f >>= LBASE;\n" \
"	}\n" \
"\n" \
"	const sz_t lid = id % CARRY_WG_SZ;\n" \
"	cl[lid] = f;\n" \
"\n" \
"	if (lid == CARRY_WG_SZ - 1) carry[id / CARRY_WG_SZ] = f;\n" \
"}\n" \
"\n" \
"INLINE void carry_2(__global ZP * restrict const xi, const __local int_64 * const cl,\n" \
"	uint_32 r[CARRY_LENGTH], const sz_t id)\n" \
"{\n" \
"	const sz_t lid = id % CARRY_WG_SZ;\n" \
"	int_64 f = (lid != 0) ? cl[lid - 1] : (int_64)(0);\n" \
"\n" \
"	for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)\n" \
"	{\n" \
"		f += (int_64)(r[j]);\n" \
"		r[j] = (uint_32)(f) & (BASE - 1);\n" \
"		f >>= LBASE;\n" \
"		if (f == 0) break;\n" \
"	}\n" \
"	f += (int_64)(r[CARRY_LENGTH - 1]);\n" \
"\n" \
"	// assert CARRY_LENGTH = 4\n" \
"	ZP4 x_4; x_4.s0 = r[0]; x_4.s1 = r[1]; x_4.s2 = r[2];\n" \
"	x_4.s3 = set_int((int_32)f, P1);\n" \
"	store4g(xi, 0, x_4);\n" \
"	x_4.s3 = set_int((int_32)f, P2);\n" \
"	store4g(xi, Z_OFFSET / 4, x_4);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))\n" \
"void carry_mul(__global ZP * restrict const reg, __global int_64 * restrict const carry, const uint_32 reg_x, const int_32 a)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];\n" \
"	__local int_64 cl[CARRY_WG_SZ];\n" \
"	uint_32 r[CARRY_LENGTH];\n" \
"\n" \
"	carry_mul_sub_1(xi, NULL_PTR, carry, cl, r, id, a);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	carry_2(xi, cl, r, id);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))\n" \
"void carry_sub(__global ZP * restrict const reg, __global int_64 * restrict const carry, const uint_32 reg_x, const uint_32 reg_y)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];\n" \
"	__global const ZP * restrict const yi = &reg[reg_y * N_SZ + CARRY_LENGTH * id];\n" \
"	__local int_64 cl[CARRY_WG_SZ];\n" \
"	uint_32 r[CARRY_LENGTH];\n" \
"\n" \
"	carry_mul_sub_1(xi, yi, carry, cl, r, id, 1);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	carry_2(xi, cl, r, id);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void carry_complete(__global ZP * restrict const reg, __global const int_64 * restrict const carry, const uint_32 reg_x, const int_32 is_nega)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * CARRY_WG_SZ * id];\n" \
"	uint_32 r[CARRY_LENGTH];\n" \
"\n" \
"	int_64 f = carry[((id == 0) ? (N_SZ / CARRY_LENGTH / CARRY_WG_SZ) : id) - 1];\n" \
"	if ((id == 0) && (is_nega != 0)) f = -f;\n" \
"\n" \
"	// assert CARRY_LENGTH = 4\n" \
"	const ZP4 x1_4 = load4g(xi, 0);\n" \
"	r[0] = x1_4.s0; r[1] = x1_4.s1; r[2] = x1_4.s2; r[3] = x1_4.s3;\n" \
"\n" \
"	for (sz_t j = 0; j < CARRY_LENGTH - 1; ++j)\n" \
"	{\n" \
"		f += (int_64)(r[j]);\n" \
"		r[j] = (uint_32)(f) & (BASE - 1);\n" \
"		f >>= LBASE;\n" \
"	}\n" \
"	f += (int_64)(r[CARRY_LENGTH - 1]);\n" \
"\n" \
"	// assert CARRY_LENGTH = 4\n" \
"	ZP4 x_4; x_4.s0 = r[0]; x_4.s1 = r[1]; x_4.s2 = r[2];\n" \
"	x_4.s3 = set_int((int_32)f, P1);\n" \
"	store4g(xi, 0, x_4);\n" \
"	x_4.s3 = set_int((int_32)f, P2);\n" \
"	store4g(xi, Z_OFFSET / 4, x_4);\n" \
"}\n" \
"\n" \
"// --- strict modulo ---\n" \
"\n" \
"// b-1, b-1, ..., b-1, b + a, ... => b, 0, ..., 0, a, ...\n" \
"INLINE void fix_base(__global ZP * restrict const x)\n" \
"{\n" \
"	if (x[N_SZ - 1] == BASE - 1)\n" \
"	{\n" \
"		sz_t i = 2; for (; i < N_SZ; ++i) { if (x[N_SZ - i] != BASE - 1) break; }\n" \
"		if (get_int(x[N_SZ - i], P1) >= (int_32)(BASE))\n" \
"		{\n" \
"			x[N_SZ - i] -= BASE; x[N_SZ - i + Z_OFFSET] -= BASE;\n" \
"			for (sz_t j = i - 1; j >= 2; --j)\n" \
"			{\n" \
"				x[N_SZ - j] = 0; x[N_SZ - j + Z_OFFSET] = 0;\n" \
"			}\n" \
"			x[N_SZ - 1] = BASE; x[N_SZ - 1 + Z_OFFSET] = BASE;\n" \
"		}\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mod_Q(__global ZP * restrict const reg, const uint_32 reg_x)\n" \
"{\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"\n" \
"	fix_base(x);\n" \
"\n" \
"	int_32 hi = get_int(x[N_SZ - 1], P1);\n" \
"	if (hi >= (int_32)(BASE))\n" \
"	{\n" \
"		int_32 lo = get_int(x[0], P1);\n" \
"		while (hi >= (int_32)(BASE)) { hi -= (int_32)(BASE); lo -= 1; }	// - Q\n" \
"		x[N_SZ - 1] = set_int(hi, P1); x[N_SZ - 1 + Z_OFFSET] = set_int(hi, P2);\n" \
"		x[0] = set_int(lo, P1); x[0 + Z_OFFSET] = set_int(lo, P2);\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mod_R_add2(__global ZP * restrict const reg, const uint_32 reg_x)\n" \
"{\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"\n" \
"	fix_base(x);\n" \
"\n" \
"	int_32 hi = get_int(x[N_SZ - 1], P1), lo = get_int(x[0], P1);\n" \
"	if (hi >= (int_32)(BASE))\n" \
"	{\n" \
"		while (hi >= (int_32)(BASE)) { hi -= (int_32)(BASE); lo += 1; }	// - R\n" \
"		x[N_SZ - 1] = set_int(hi, P1); x[N_SZ - 1 + Z_OFFSET] = set_int(hi, P2);\n" \
"	}\n" \
"	lo += 2;\n" \
"	x[0] = set_int(lo, P1); x[0 + Z_OFFSET] = set_int(lo, P2);\n" \
"}\n" \
"\n" \
"// --- misc ---\n" \
"\n" \
"__kernel\n" \
"void set(__global ZP4 * restrict const reg, const uint_32 reg_y, const uint_32 reg_x)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global ZP4 * restrict const y = &reg[reg_y * N_SZ / 4];\n" \
"	__global const ZP4 * restrict const x = &reg[reg_x * N_SZ / 4];\n" \
"\n" \
"	y[id] = x[id]; y[id + Z_OFFSET / 4] = x[id + Z_OFFSET / 4];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void extend(__global ZP4 * restrict const reg)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global ZP4 * restrict const x = &reg[0];\n" \
"\n" \
"	const ZP4 u = x[id]; ZP4 r;\n" \
"	r.s0 = set_int(get_int(u.s0, P1), P2);\n" \
"	r.s1 = set_int(get_int(u.s1, P1), P2);\n" \
"	r.s2 = set_int(get_int(u.s2, P1), P2);\n" \
"	r.s3 = set_int(get_int(u.s3, P1), P2);\n" \
"	x[id + Z_OFFSET / 4] = r;\n" \
"}\n" \
"";
