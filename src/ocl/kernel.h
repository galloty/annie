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
"#define LN_SZ			10\n" \
"#define N_SZ			1024u\n" \
"#define LBASE			28\n" \
"#define PRM_COUNT		4\n" \
"#define RHO				2\n" \
"#define P1				4194304001u\n" \
"#define Q1				100663297u\n" \
"#define RSQ1			232465106u\n" \
"#define P2				4076863489u\n" \
"#define Q2				218103809u\n" \
"#define RSQ2			3444438393u\n" \
"#define INVP2_P1		1797558821u\n" \
"#define NORM1			4186112001u\n" \
"#define NORM2			4068900865u\n" \
"#define P3				3942645761u\n" \
"#define Q3				352321537u\n" \
"#define RSQ3			3810498414u\n" \
"#define INVP3_P1		3075822917u\n" \
"#define INVP3_P2		4076863457u\n" \
"#define NORM3			3934945281u\n" \
"#define P4				3892314113u\n" \
"#define Q4				402653185u\n" \
"#define RSQ4			1703176688u\n" \
"#define INVP4_P1		3262236431u\n" \
"#define INVP4_P2		2223743698u\n" \
"#define INVP4_P3		2628430422u\n" \
"#define NORM4			3884711937u\n" \
"#define RBASE1			3684545070u\n" \
"#define RBASE2			1998905176u\n" \
"#define RBASE3			730986871u\n" \
"#define RBASE4			106448543u\n" \
"#define RBASESQ1		754572067u\n" \
"#define RBASESQ2		2163363318u\n" \
"#define RBASESQ3		2263424920u\n" \
"#define RBASESQ4		249922666u\n" \
"#define CARRY_LENGTH	4u\n" \
"#define CARRY_WG_SZ		256u\n" \
"#define Z_OFFSET		10240u\n" \
"#define BLK16			8\n" \
"#define BLK32			8\n" \
"#define BLK64			8\n" \
"#define BLK128			4\n" \
"#define BLK256			2\n" \
"#define BLK512			1\n" \
"#define BLK1024 		1\n" \
"#define BLK2048 		1\n" \
"#define CHUNK64 		8\n" \
"#define CHUNK512		4\n" \
"#endif\n" \
"\n" \
"typedef uint	sz_t;\n" \
"typedef uint	uint_32;\n" \
"typedef int		int_32;\n" \
"typedef ulong	uint_64;\n" \
"typedef long	int_64;\n" \
"typedef uint2	uint2_32;\n" \
"typedef uint4	uint4_32;\n" \
"\n" \
"// --- 128-bit arithmetic\n" \
"\n" \
"#if (PRM_COUNT == 4)\n" \
"\n" \
"typedef struct { uint_64 s0; int_64 s1; } int_128;\n" \
"\n" \
"INLINE int_128 int128_set(const uint_64 l, const int_64 h) { int_128 r; r.s0 = l; r.s1 = h; return r; }\n" \
"\n" \
"INLINE int_128 int128_neg(const int_128 x)\n" \
"{\n" \
"	int_128 r;\n" \
"	r.s0 = -x.s0;\n" \
"	r.s1 = -x.s1 - ((x.s0 != 0) ? 1 : 0);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE int_128 int128_add(const int_128 x, const int_128 y)\n" \
"{\n" \
"	int_128 r;\n" \
"// #if defined(PTX_ASM)\n" \
"// 	asm volatile (\"add.cc.u64 %0, %1, %2;\" : \"=r\" (r.s0) : \"r\" (x.s0), \"r\" (y.s0));\n" \
"// 	asm volatile (\"addc.s64 %0, %1, %2;\" : \"=r\" (r.s1) : \"r\" (x.s1), \"r\" (y.s1));\n" \
"// #else\n" \
"	r.s0 = x.s0 + y.s0;\n" \
"	r.s1 = x.s1 + y.s1 + ((r.s0 < x.s0) ? 1 : 0);\n" \
"// #endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE int_128 int128_add_u64(const int_128 x, const uint_64 y)\n" \
"{\n" \
"	int_128 r;\n" \
"// #if defined(PTX_ASM)\n" \
"// 	asm volatile (\"add.cc.u64 %0, %1, %2;\" : \"=r\" (r.s0) : \"r\" (x.s0), \"r\" (y));\n" \
"// 	asm volatile (\"addc.s64 %0, %1, 0;\" : \"=r\" (r.s1) : \"r\" (x.s1));\n" \
"// #else\n" \
"	r.s0 = x.s0 + y;\n" \
"	r.s1 = x.s1 + ((r.s0 < x.s0) ? 1 : 0);\n" \
"// #endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE int_128 int128_add_i64(const int_128 x, const int_64 y)\n" \
"{\n" \
"	int_128 r;\n" \
"	r.s0 = x.s0 + (uint_64)(y);\n" \
"	r.s1 = x.s1 + ((y < 0) ? -1 : 0) + ((r.s0 < x.s0) ? 1 : 0);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE int_128 int128_sub_i64(const int_128 x, const int_64 y)\n" \
"{\n" \
"	int_128 r;\n" \
"// #if defined(PTX_ASM)	// TODO\n" \
"// 	asm volatile (\"add.cc.u64 %0, %1, %2;\" : \"=r\" (r.s0) : \"r\" (x.s0), \"r\" (y));\n" \
"// 	asm volatile (\"addc.s64 %0, %1, 0;\" : \"=r\" (r.s1) : \"r\" (x.s1));\n" \
"// #else\n" \
"	r.s0 = x.s0 - (uint_64)(y);\n" \
"	r.s1 = x.s1 - ((y < 0) ? -1 : 0) - ((x.s0 < (uint_64)(y)) ? 1 : 0);\n" \
"// #endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE int_128 int128_rshift(const int_128 x, const int s)	// s != 0\n" \
"{\n" \
"	int_128 r;\n" \
"	r.s0 = (x.s0 >> s) | (uint_64)(x.s1 << (64 - s));\n" \
"	r.s1 = x.s1 >> s;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE int_128 int128_mul_u32(const int_128 x, const uint_32 y)\n" \
"{\n" \
"	int_128 r;\n" \
"	r.s0 = x.s0 * y;\n" \
"	r.s1 = x.s1 * (int_64)(y) + (int_64)(mul_hi(x.s0, (uint_64)(y)));\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE int_128 int128_mul_u64i64(const uint_64 x, const int_64 y)\n" \
"{\n" \
"	int_128 r;\n" \
"	r.s0 = x * (uint_64)(y);\n" \
"	r.s1 = (int_64)(mul_hi(x, (uint_64)(y)) - ((y < 0) ? x : 0));\n" \
"	return r;\n" \
"}\n" \
"\n" \
"#endif\n" \
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
"#if (PRM_COUNT == 2)\n" \
"__constant ZP2 g_pq[2] = { PQ1, PQ2 };\n" \
"__constant ZP g_rsq[2] = { RSQ1, RSQ2 };\n" \
"#elif (PRM_COUNT == 4)\n" \
"#define	PQ3		(ZP2)(P3, Q3)\n" \
"#define	PQ4		(ZP2)(P4, Q4)\n" \
"\n" \
"__constant ZP2 g_pq[4] = { PQ1, PQ2, PQ3, PQ4 };\n" \
"__constant ZP g_rsq[4] = { RSQ1, RSQ2, RSQ3, RSQ4 };\n" \
"#endif\n" \
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
"INLINE ZP mulmod(const ZP lhs, const ZP rhs, const ZP2 pq)	// TODO other formula => const\n" \
"{\n" \
"	const uint_64 t = lhs * (uint_64)(rhs);\n" \
"	const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);\n" \
"	const uint_32 mp = mul_hi(lo * pq.s1, pq.s0);\n" \
"	return submod(hi, mp, pq.s0);\n" \
"}\n" \
"\n" \
"INLINE ZP submulmod(const ZP lhs, const ZP rhs, const ZP m, const ZP2 pq)\n" \
"{\n" \
"	return mulmod(submod(lhs, rhs, pq.s0), m, pq);\n" \
"}\n" \
"\n" \
"INLINE uint_32 set_i32(const int_32 i, const uint_32 p) { return (uint_32)(i) + ((i < 0) ? p : 0); }\n" \
"INLINE int_32 get_i32(const uint_32 u, const uint_32 p) { return (int_32)(u - ((u >= p / 2) ? p : 0)); }\n" \
"\n" \
"typedef struct { ZP r[2]; } RNS2;\n" \
"\n" \
"#if (PRM_COUNT == 2)\n" \
"\n" \
"INLINE RNS2 set2_u32(const uint_32 n) { RNS2 r; for (sz_t l = 0; l < 2; ++l) r.r[l] = n; return r; }\n" \
"INLINE RNS2 set2_i32(const int_32 i) { RNS2 r; r.r[0] = set_i32(i, P1); r.r[1] = set_i32(i, P2); return r; }\n" \
"\n" \
"INLINE RNS2 addmod2(const RNS2 lhs, const RNS2 rhs)\n" \
"{\n" \
"	RNS2 r; r.r[0] = addmod(lhs.r[0], rhs.r[0], P1); r.r[1] = addmod(lhs.r[1], rhs.r[1], P2); return r;\n" \
"}\n" \
"\n" \
"INLINE RNS2 mulmod2(const RNS2 lhs, const RNS2 rhs)\n" \
"{\n" \
"	RNS2 r; r.r[0] = mulmod(lhs.r[0], rhs.r[0], PQ1); r.r[1] = mulmod(lhs.r[1], rhs.r[1], PQ2); return r;\n" \
"}\n" \
"\n" \
"INLINE RNS2 NORM_2() { RNS2 r; r.r[0] = NORM1; r.r[1] = NORM2; return r; }\n" \
"\n" \
"#elif (PRM_COUNT == 4)\n" \
"\n" \
"typedef struct { ZP r[4]; } RNS4;\n" \
"\n" \
"INLINE RNS4 set4_u32(const uint_32 n) { RNS4 r; for (sz_t l = 0; l < 4; ++l) r.r[l] = n; return r; }\n" \
"INLINE RNS4 set4_i32(const int_32 i) { RNS4 r; r.r[0] = set_i32(i, P1); r.r[1] = set_i32(i, P2); r.r[2] = set_i32(i, P3); r.r[3] = set_i32(i, P4); return r; }\n" \
"\n" \
"INLINE RNS4 addmod4(const RNS4 lhs, const RNS4 rhs)\n" \
"{\n" \
"	RNS4 r;\n" \
"	r.r[0] = addmod(lhs.r[0], rhs.r[0], P1); r.r[1] = addmod(lhs.r[1], rhs.r[1], P2);\n" \
"	r.r[2] = addmod(lhs.r[2], rhs.r[2], P3); r.r[3] = addmod(lhs.r[3], rhs.r[3], P4);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE RNS4 submod4(const RNS4 lhs, const RNS4 rhs)\n" \
"{\n" \
"	RNS4 r;\n" \
"	r.r[0] = submod(lhs.r[0], rhs.r[0], P1); r.r[1] = submod(lhs.r[1], rhs.r[1], P2);\n" \
"	r.r[2] = submod(lhs.r[2], rhs.r[2], P3); r.r[3] = submod(lhs.r[3], rhs.r[3], P4);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE RNS4 mulmod4(const RNS4 lhs, const RNS4 rhs)\n" \
"{\n" \
"	RNS4 r;\n" \
"	r.r[0] = mulmod(lhs.r[0], rhs.r[0], PQ1); r.r[1] = mulmod(lhs.r[1], rhs.r[1], PQ2);\n" \
"	r.r[2] = mulmod(lhs.r[2], rhs.r[2], PQ3); r.r[3] = mulmod(lhs.r[3], rhs.r[3], PQ4);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE RNS4 NORM_4() { RNS4 r; r.r[0] = NORM1; r.r[1] = NORM2; r.r[2] = NORM3; r.r[3] = NORM4; return r; }\n" \
"INLINE RNS4 RBASE_4() { RNS4 r; r.r[0] = RBASE1; r.r[1] = RBASE2; r.r[2] = RBASE3; r.r[3] = RBASE4; return r; }\n" \
"INLINE RNS4 RBASESQ_4() { RNS4 r; r.r[0] = RBASESQ1; r.r[1] = RBASESQ2; r.r[2] = RBASESQ3; r.r[3] = RBASESQ4; return r; }\n" \
"\n" \
"#endif\n" \
"\n" \
"// --- I/O ---\n" \
"\n" \
"INLINE void loadg(const sz_t n, ZP * const xl, __global const ZP * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }\n" \
"INLINE void storeg(const sz_t n, __global ZP * restrict const x, const sz_t s, const ZP * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }\n" \
"\n" \
"INLINE void loadl(const sz_t n, ZP * const xl, __local const ZP * restrict const X, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = X[l * s]; }\n" \
"INLINE void storel(const sz_t n, __local ZP * restrict const X, const sz_t s, const ZP * const xl) { for (sz_t l = 0; l < n; ++l) X[l * s] = xl[l]; }\n" \
"\n" \
"INLINE ZP load1g(__global const ZP * restrict const x, const sz_t j) { return x[j]; }\n" \
"INLINE ZP2 load2g(__global const ZP * restrict const x, const sz_t j) { return ((__global const ZP2 *)x)[j]; }\n" \
"INLINE ZP4 load4g(__global const ZP * restrict const x, const sz_t j) { return ((__global const ZP4 *)x)[j]; }\n" \
"\n" \
"INLINE RNS2 loadRNS2(__global const ZP * restrict const x) { RNS2 r; for (sz_t l = 0; l < 2; ++l) r.r[l] = x[l * Z_OFFSET]; return r; }\n" \
"INLINE void storeRNS2(__global ZP * restrict const x, const RNS2 r2) { for (sz_t l = 0; l < 2; ++l) x[l * Z_OFFSET] = r2.r[l]; }\n" \
"\n" \
"#if (PRM_COUNT == 4)\n" \
"\n" \
"INLINE RNS4 loadRNS4(__global const ZP * restrict const x) { RNS4 r; for (sz_t l = 0; l < 4; ++l) r.r[l] = x[l * Z_OFFSET]; return r; }\n" \
"INLINE void storeRNS4(__global ZP * restrict const x, const RNS4 r4) { for (sz_t l = 0; l < 4; ++l) x[l * Z_OFFSET] = r4.r[l]; }\n" \
"\n" \
"#endif\n" \
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
"#define LOADG(m)	ZP xl[8]; loadg(8, xl, x, m);\n" \
"#define LOADPG(m)	ZP xpl[8]; loadg(8, xpl, xp, m);\n" \
"#define STOREG(m)	storeg(8, x, m, xl);\n" \
"\n" \
"#define LOADL(m) \\\n" \
"	barrier(CLK_LOCAL_MEM_FENCE); \\\n" \
"	ZP xl[8]; loadl(8, xl, X, m);\n" \
"\n" \
"#define STOREL(m)	storel(8, X, m, xl);\n" \
"\n" \
"#define DECLARE_W124() \\\n" \
"	const ZP w1 = load1g(w, sj); const ZP2 w2 = load2g(w, sj); const ZP4 w4 = load4g(w, sj);\n" \
"\n" \
"#define DECLARE_WI124() \\\n" \
"	__global const ZP * restrict const w_i = &w[N_SZ / 2]; \\\n" \
"	const ZP w1_i = load1g(w_i, sj); const ZP2 w2_i = load2g(w_i, sj); const ZP4 w4_i = load4g(w_i, sj);\n" \
"\n" \
"#define DECLARE_W124_0() \\\n" \
"	const ZP w1 = load1g(w, 0); const ZP2 w2 = load2g(w, 1); const ZP4 w4 = load4g(w, 1);\n" \
"\n" \
"#define DECLARE_W2()	const ZP2 w2 = load2g(w, sj);\n" \
"#define DECLARE_W12()	const ZP w1 = load1g(w, sj); const ZP2 w2 = load2g(w, sj);\n" \
"\n" \
"#define DECLARE_W2I2() \\\n" \
"	__global const ZP * restrict const w_i = &w[N_SZ / 2]; \\\n" \
"	const ZP2 w2 = load2g(w, sj), w2_i = load2g(w_i, sj);\n" \
"\n" \
"#define DECLARE_W12I12() \\\n" \
"	DECLARE_W2I2(); \\\n" \
"	const ZP w1 = load1g(w, sj), w1_i = load1g(w_i, sj);\n" \
"\n" \
"// ---\n" \
"\n" \
"INLINE void forward8g(const ZP2 pq, const sz_t m, __global ZP * restrict const x,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W124();\n" \
"	LOADG(m);\n" \
"	_forward8(pq, xl, w1, w2, w4);\n" \
"	STOREG(m);\n" \
"}\n" \
"\n" \
"INLINE void forward8i(const ZP2 pq, const sz_t ml, __local ZP * restrict const X, const sz_t mg,\n" \
"	__global ZP * restrict const x, __global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W124();\n" \
"	LOADG(mg);\n" \
"	_forward8(pq, xl, w1, w2, w4);\n" \
"	STOREL(ml);\n" \
"}\n" \
"\n" \
"INLINE void forward8l(const ZP2 pq, const sz_t m, __local ZP * restrict const X,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W124();\n" \
"	LOADL(m);\n" \
"	_forward8(pq, xl, w1, w2, w4);\n" \
"	STOREL(m);\n" \
"}\n" \
"\n" \
"INLINE void forward8o(const ZP2 pq, const sz_t mg, __global ZP * restrict const x, const sz_t ml,\n" \
"	__local ZP * restrict const X, __global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W124();\n" \
"	LOADL(ml);\n" \
"	_forward8(pq, xl, w1, w2, w4);\n" \
"	STOREG(mg);\n" \
"}\n" \
"\n" \
"INLINE void backward8g(const ZP2 pq, const sz_t m, __global ZP * restrict const x,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_WI124();\n" \
"	LOADG(m);\n" \
"	_backward8(pq, xl, w1_i, w2_i, w4_i);\n" \
"	STOREG(m);\n" \
"}\n" \
"\n" \
"INLINE void backward8i(const ZP2 pq, const sz_t ml, __local ZP * restrict const X, const sz_t mg,\n" \
"	__global ZP * restrict const x, __global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_WI124();\n" \
"	LOADG(mg);\n" \
"	_backward8(pq, xl, w1_i, w2_i, w4_i);\n" \
"	STOREL(ml);\n" \
"}\n" \
"\n" \
"INLINE void backward8l(const ZP2 pq, const sz_t m, __local ZP * restrict const X,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_WI124();\n" \
"	LOADL(m);\n" \
"	_backward8(pq, xl, w1_i, w2_i, w4_i);\n" \
"	STOREL(m);\n" \
"}\n" \
"\n" \
"INLINE void backward8o(const ZP2 pq, const sz_t mg, __global ZP * restrict const x, const sz_t ml,\n" \
"	__local ZP * restrict const X, __global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_WI124();\n" \
"	LOADL(ml);\n" \
"	_backward8(pq, xl, w1_i, w2_i, w4_i);\n" \
"	STOREG(mg);\n" \
"}\n" \
"\n" \
"INLINE void forward8_0g(const ZP2 pq, const sz_t m, __global ZP * restrict const x,\n" \
"	__global const ZP * restrict const w, const ZP rsq)\n" \
"{\n" \
"	// Montgomery form of MF of I to convert input into MF\n" \
"	DECLARE_W124_0();\n" \
"	LOADG(m);\n" \
"	_forward8_0(pq, xl, w1, w2, w4, rsq);\n" \
"	STOREG(m);\n" \
"}\n" \
"\n" \
"INLINE void forward8_0i(const ZP2 pq,const sz_t ml, __local ZP * restrict const X, const sz_t mg,\n" \
"	__global ZP * restrict const x, __global const ZP * restrict const w, const ZP rsq)\n" \
"{\n" \
"	DECLARE_W124_0();\n" \
"	LOADG(mg);\n" \
"	_forward8_0(pq, xl, w1, w2, w4, rsq);\n" \
"	STOREL(ml);\n" \
"}\n" \
"\n" \
"INLINE void square2x4g(const ZP2 pq, __global ZP * restrict const x,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2();\n" \
"	LOADG(1);\n" \
"	_square2x4(pq, xl, w2);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"INLINE void square2x4l(const ZP2 pq, __local ZP * restrict const X,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2();\n" \
"	LOADL(1);\n" \
"	_square2x4(pq, xl, w2);\n" \
"	STOREL(1);\n" \
"}\n" \
"\n" \
"INLINE void square4x2g(const ZP2 pq, __global ZP * restrict const x,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2I2();\n" \
"	LOADG(1);\n" \
"	_square4x2(pq, xl, w2, w2_i);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"INLINE void square4x2l(const ZP2 pq, __local ZP * restrict const X,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2I2();\n" \
"	LOADL(1);\n" \
"	_square4x2(pq, xl, w2, w2_i);\n" \
"	STOREL(1);\n" \
"}\n" \
"\n" \
"INLINE void square8g(const ZP2 pq, __global ZP * restrict const x,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12I12();\n" \
"	LOADG(1);\n" \
"	_square8(pq, xl, w1, w1_i, w2, w2_i);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"INLINE void square8l(const ZP2 pq, __local ZP * restrict const X,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12I12();\n" \
"	LOADL(1);\n" \
"	_square8(pq, xl, w1, w1_i, w2, w2_i);\n" \
"	STOREL(1);\n" \
"}\n" \
"\n" \
"INLINE void mul2x4g(const ZP2 pq, __global ZP * restrict const x, __global const ZP * restrict const xp,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2();\n" \
"	LOADG(1);\n" \
"	LOADPG(1);\n" \
"	_mul2x4(pq, xl, xpl, w2);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"INLINE void mul2x4l(const ZP2 pq, __local ZP * restrict const X, __global const ZP * restrict const xp,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2();\n" \
"	LOADL(1);\n" \
"	LOADPG(1);\n" \
"	_mul2x4(pq, xl, xpl, w2);\n" \
"	STOREL(1);\n" \
"}\n" \
"\n" \
"INLINE void mul4x2g(const ZP2 pq, __global ZP * restrict const x, __global const ZP * restrict const xp,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2I2();\n" \
"	LOADG(1);\n" \
"	LOADPG(1);\n" \
"	_mul4x2(pq, xl, xpl, w2, w2_i);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"INLINE void mul4x2l(const ZP2 pq, __local ZP * restrict const X, __global const ZP * restrict const xp,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2I2();\n" \
"	LOADL(1);\n" \
"	LOADPG(1);\n" \
"	_mul4x2(pq, xl, xpl, w2, w2_i);\n" \
"	STOREL(1);\n" \
"}\n" \
"\n" \
"INLINE void mul8g(const ZP2 pq, __global ZP * restrict const x, __global const ZP * restrict const xp,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12I12();\n" \
"	LOADG(1);\n" \
"	LOADPG(1);\n" \
"	_mul8(pq, xl, xpl, w1, w1_i, w2, w2_i);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"INLINE void mul8l(const ZP2 pq, __local ZP * restrict const X, __global const ZP * restrict const xp,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12I12();\n" \
"	LOADL(1);\n" \
"	LOADPG(1);\n" \
"	_mul8(pq, xl, xpl, w1, w1_i, w2, w2_i);\n" \
"	STOREL(1);\n" \
"}\n" \
"\n" \
"INLINE void fwd4x2g(const ZP2 pq, __global ZP * restrict const x,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2();\n" \
"	LOADG(1);\n" \
"	_fwd4x2(pq, xl, w2);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"INLINE void fwd4x2o(const ZP2 pq, __global ZP * restrict const x, __local ZP * restrict const X,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2();\n" \
"	LOADL(1);\n" \
"	_fwd4x2(pq, xl, w2);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"INLINE void fwd8g(const ZP2 pq, __global ZP * restrict const x,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12();\n" \
"	LOADG(1);\n" \
"	_fwd8(pq, xl, w1, w2);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"INLINE void fwd8o(const ZP2 pq, __global ZP * restrict const x, __local ZP * restrict const X,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12();\n" \
"	LOADL(1);\n" \
"	_fwd8(pq, xl, w1, w2);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define DECLARE_VAR_REG() \\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid / (N_SZ / 8), id = gid % (N_SZ / 8); \\\n" \
"	const ZP2 pq = g_pq[lid]; \\\n" \
"	__global ZP * restrict const reg = &regm[lid * Z_OFFSET]; \\\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ]; \\\n" \
"	__global const ZP * restrict const w = &reg[reg_w * N_SZ];\n" \
"\n" \
"#define DECLARE_VARP_REG()	__global const ZP * restrict const xp = &reg[reg_xp * N_SZ];\n" \
"\n" \
"// --- transform without local mem ---\n" \
"\n" \
"#if (LN_SZ >= 21)\n" \
"__kernel\n" \
"void forward8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const int_32 lm, const uint_32 s)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t m = 1u << lm, sj = s + (id >> lm), k = 7 * (id & ~(m - 1)) + id;\n" \
"	forward8g(pq, m, &x[k], w, sj);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void backward8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const int_32 lm, const uint_32 s)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t m = 1u << lm, sj = s + (id >> lm), k = 7 * (id & ~(m - 1)) + id;\n" \
"	backward8g(pq, m, &x[k], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ <= 12)\n" \
"__kernel\n" \
"void forward8_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t m = N_SZ / 8, k = id;\n" \
"	forward8_0g(pq, m, &x[k], w, g_rsq[lid]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void backward8_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t m = N_SZ / 8, sj = 1, k = id;\n" \
"	backward8g(pq, m, &x[k], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"// ---\n" \
"\n" \
"#if (LN_SZ == 4)\n" \
"__kernel\n" \
"void square2x4(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t sj = N_SZ / 8 + id, k = 8 * id;\n" \
"	square2x4g(pq, &x[k], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 5)\n" \
"__kernel\n" \
"void square4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t sj = N_SZ / 8 + id, k = 8 * id;\n" \
"	square4x2g(pq, &x[k], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 6)\n" \
"__kernel\n" \
"void square8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t sj = N_SZ / 8 + id, k = 8 * id;\n" \
"	square8g(pq, &x[k], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"// ---\n" \
"\n" \
"#if (LN_SZ == 4)\n" \
"__kernel\n" \
"void mul2x4(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t sj = N_SZ / 8 + id, k = 8 * id;\n" \
"	mul2x4g(pq, &x[k], &xp[k], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 5)\n" \
"__kernel\n" \
"void mul4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t sj = N_SZ / 8 + id, k = 8 * id;\n" \
"	mul4x2g(pq, &x[k], &xp[k], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 6)\n" \
"__kernel\n" \
"void mul8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t sj = N_SZ / 8 + id, k = 8 * id;\n" \
"	mul8g(pq, &x[k], &xp[k], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"// ---\n" \
"\n" \
"#if (LN_SZ == 5)\n" \
"__kernel\n" \
"void fwd4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t sj = N_SZ / 8 + id, k = 8 * id;\n" \
"	fwd4x2g(pq, &x[k], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 6)\n" \
"__kernel\n" \
"void fwd8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t sj = N_SZ / 8 + id, k = 8 * id;\n" \
"	fwd8g(pq, &x[k], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 7)\n" \
"__kernel\n" \
"void fwd16(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t sj = N_SZ / 8 + id, k2 = 7 * (id & ~(2 - 1)) + id;\n" \
"	forward8g(pq, 2, &x[k2], w, sj / 2);\n" \
"}\n" \
"#endif\n" \
"\n" \
"// --- transform using local mem ---\n" \
"\n" \
"#define DECLARE_VAR_FB(N, CHUNK_N) \\\n" \
"	__local ZP X[N * CHUNK_N]; \\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const size_t chunk_id = id % CHUNK_N, local_id = (id / CHUNK_N) % (N / 8), block_id = id & ~(N / 8 * CHUNK_N - 1); \\\n" \
"	__global ZP * restrict const xt = &x[block_id / (N / 8) + chunk_id]; \\\n" \
"	__local ZP * const Xt = &X[N * chunk_id];\n" \
"\n" \
"#if (LN_SZ > 12) && (LN_SZ <= 16)\n" \
"\n" \
"#define DECLARE_VAR_FB64() \\\n" \
"	const sz_t kl0 = local_id, k0 = kl0 * (N_SZ / 64), ml0 = 8, m0 = ml0 * (N_SZ / 64); \\\n" \
"	const sz_t kl8 = local_id * 8, k8 = kl8 * (N_SZ / 64), ml8 = 1, m8 = ml8 * (N_SZ / 64); \\\n" \
"	const sz_t sj8 = 8 + local_id;\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64 / 8 * CHUNK64, 1, 1)))\n" \
"void forward64_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_FB(64, CHUNK64);\n" \
"	DECLARE_VAR_FB64();\n" \
"\n" \
"	forward8_0i(pq, ml0, &Xt[kl0], m0, &xt[k0], w, g_rsq[lid]);\n" \
"	forward8o(pq, m8, &xt[k8], ml8, &Xt[kl8], w, sj8);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64 / 8 * CHUNK64, 1, 1)))\n" \
"void backward64_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_FB(64, CHUNK64);\n" \
"	DECLARE_VAR_FB64();\n" \
"\n" \
"	backward8i(pq, ml8, &Xt[kl8], m8, &xt[k8], w, sj8);\n" \
"	backward8o(pq, m0, &xt[k0], ml0, &Xt[kl0], w, 1);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ > 16)\n" \
"\n" \
"#define DECLARE_VAR_FB512() \\\n" \
"	const sz_t kl0 = local_id, k0 = kl0 * (N_SZ / 512), ml0 = 64, m0 = ml0 * (N_SZ / 512); \\\n" \
"	const sz_t kl8 = (local_id % 8) + (local_id / 8) * 64, ml8 = 8; \\\n" \
"	const sz_t kl64 = local_id * 8, k64 = kl64 * (N_SZ / 512), ml64 = 1, m64 = ml64 * (N_SZ / 512); \\\n" \
"	const sz_t sj64 = 64 + local_id, sj8 = 8 + (local_id / 8) % 8;\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(512 / 8 * CHUNK512, 1, 1)))\n" \
"void forward512_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_FB(512, CHUNK512);\n" \
"	DECLARE_VAR_FB512();\n" \
"\n" \
"	forward8_0i(pq, ml0, &Xt[kl0], m0, &xt[k0], w, g_rsq[lid]);\n" \
"	forward8l(pq, ml8, &Xt[kl8], w, sj8);\n" \
"	forward8o(pq, m64, &xt[k64], ml64, &Xt[kl64], w, sj64);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(512 / 8 * CHUNK512, 1, 1)))\n" \
"void backward512_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_FB(512, CHUNK512);\n" \
"	DECLARE_VAR_FB512();\n" \
"\n" \
"	backward8i(pq, ml64, &Xt[kl64], m64, &xt[k64], w, sj64);\n" \
"	backward8l(pq, ml8, &Xt[kl8], w, sj8);\n" \
"	backward8o(pq, m0, &xt[k0], ml0, &Xt[kl0], w, 1);\n" \
"}\n" \
"#endif\n" \
"\n" \
"// ---\n" \
"\n" \
"#define DECLARE_VAR_SQRMUL(N, BLK_N) \\\n" \
"	__local ZP X[N * BLK_N]; \\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t block_id = (id / (N / 8)) % BLK_N; \\\n" \
"	__local ZP * const Xb = &X[N * block_id]; \\\n" \
"	const sz_t s = N_SZ / 8, sj = s + id, k = 8 * id;\n" \
"\n" \
"#if (LN_SZ == 7)\n" \
"\n" \
"#define DECLARE_VAR_SQRMUL16() \\\n" \
"	DECLARE_VAR_SQRMUL(16, BLK16); \\\n" \
"	const sz_t k2 = 7 * (id & ~(2 - 1)) + id;\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(16 / 8 * BLK16, 1, 1)))\n" \
"void square16(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL16();\n" \
"\n" \
"	forward8i(pq, 2, &Xb[k2 % 16], 2, &x[k2], w, sj / 2);\n" \
"	square2x4l(pq, &Xb[k % 16], w, sj);\n" \
"	backward8o(pq, 2, &x[k2], 2, &Xb[k2 % 16], w, sj / 2);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(16 / 8 * BLK16, 1, 1)))\n" \
"void mul16(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL16();\n" \
"	DECLARE_VARP_REG();\n" \
"\n" \
"	forward8i(pq, 2, &Xb[k2 % 16], 2, &x[k2], w, sj / 2);\n" \
"	mul2x4l(pq, &Xb[k % 16], &xp[k], w, sj);\n" \
"	backward8o(pq, 2, &x[k2], 2, &Xb[k2 % 16], w, sj / 2);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 8)\n" \
"\n" \
"#define DECLARE_VAR_SQRMUL32() \\\n" \
"	DECLARE_VAR_SQRMUL(32, BLK32); \\\n" \
"	const sz_t k4 = 7 * (id & ~(4 - 1)) + id;\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(32 / 8 * BLK32, 1, 1)))\n" \
"void square32(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL32();\n" \
"\n" \
"	forward8i(pq, 4, &Xb[k4 % 32], 4, &x[k4], w, sj / 4);\n" \
"	square4x2l(pq, &Xb[k % 32], w, sj);\n" \
"	backward8o(pq, 4, &x[k4], 4, &Xb[k4 % 32], w, sj / 4);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(32 / 8 * BLK32, 1, 1)))\n" \
"void mul32(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL32();\n" \
"	DECLARE_VARP_REG();\n" \
"\n" \
"	forward8i(pq, 4, &Xb[k4 % 32], 4, &x[k4], w, sj / 4);\n" \
"	mul4x2l(pq, &Xb[k % 32], &xp[k], w, sj);\n" \
"	backward8o(pq, 4, &x[k4], 4, &Xb[k4 % 32], w, sj / 4);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(32 / 8 * BLK32, 1, 1)))\n" \
"void fwd32(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL32();\n" \
"\n" \
"	forward8i(pq, 4, &Xb[k4 % 32], 4, &x[k4], w, sj / 4);\n" \
"	fwd4x2o(pq, &x[k], &Xb[k % 32], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 9)\n" \
"\n" \
"#define DECLARE_VAR_SQRMUL64() \\\n" \
"	DECLARE_VAR_SQRMUL(64, BLK64); \\\n" \
"	const sz_t k8 = 7 * (id & ~(8 - 1)) + id;\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))\n" \
"void square64(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL64();\n" \
"\n" \
"	forward8i(pq, 8, &Xb[k8 % 64], 8, &x[k8], w, sj / 8);\n" \
"	square8l(pq, &Xb[k % 64], w, sj);\n" \
"	backward8o(pq, 8, &x[k8], 8, &Xb[k8 % 64], w, sj / 8);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))\n" \
"void mul64(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL64();\n" \
"	DECLARE_VARP_REG();\n" \
"\n" \
"	forward8i(pq, 8, &Xb[k8 % 64], 8, &x[k8], w, sj / 8);\n" \
"	mul8l(pq, &Xb[k % 64], &xp[k], w, sj);\n" \
"	backward8o(pq, 8, &x[k8], 8, &Xb[k8 % 64], w, sj / 8);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))\n" \
"void fwd64(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL64();\n" \
"\n" \
"	forward8i(pq, 8, &Xb[k8 % 64], 8, &x[k8], w, sj / 8);\n" \
"	fwd8o(pq, &x[k], &Xb[k % 64], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 10) || (LN_SZ == 13)\n" \
"\n" \
"#define DECLARE_VAR_SQRMUL128() \\\n" \
"	DECLARE_VAR_SQRMUL(128, BLK128); \\\n" \
"	const sz_t k2 = 7 * (id & ~(2 - 1)) + id, k16 = 7 * (id & ~(16 - 1)) + id;\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(128 / 8 * BLK128, 1, 1)))\n" \
"void square128(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL128();\n" \
"\n" \
"	forward8i(pq, 16, &Xb[k16 % 128], 16, &x[k16], w, sj / 16);\n" \
"	forward8l(pq, 2, &Xb[k2 % 128], w, sj / 2);\n" \
"	square2x4l(pq, &Xb[k % 128], w, sj);\n" \
"	backward8l(pq, 2, &Xb[k2 % 128], w, sj / 2);\n" \
"	backward8o(pq, 16, &x[k16], 16, &Xb[k16 % 128], w, sj / 16);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(128 / 8 * BLK128, 1, 1)))\n" \
"void mul128(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL128();\n" \
"	DECLARE_VARP_REG();\n" \
"\n" \
"	forward8i(pq, 16, &Xb[k16 % 128], 16, &x[k16], w, sj / 16);\n" \
"	forward8l(pq, 2, &Xb[k2 % 128], w, sj / 2);\n" \
"	mul2x4l(pq, &Xb[k % 128], &xp[k], w, sj);\n" \
"	backward8l(pq, 2, &Xb[k2 % 128], w, sj / 2);\n" \
"	backward8o(pq, 16, &x[k16], 16, &Xb[k16 % 128], w, sj / 16);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(128 / 8 * BLK128, 1, 1)))\n" \
"void fwd128(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL128();\n" \
"\n" \
"	forward8i(pq, 16, &Xb[k16 % 128], 16, &x[k16], w, sj / 16);\n" \
"	forward8o(pq, 2, &x[k2], 2, &Xb[k2 % 128], w, sj / 2);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 11) || (LN_SZ == 14) || (LN_SZ == 17)\n" \
"\n" \
"#define DECLARE_VAR_SQRMUL256() \\\n" \
"	DECLARE_VAR_SQRMUL(256, BLK256); \\\n" \
"	const sz_t k4 = 7 * (id & ~(4 - 1)) + id, k32 = 7 * (id & ~(32 - 1)) + id;\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 8 * BLK256, 1, 1)))\n" \
"void square256(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL256();\n" \
"\n" \
"	forward8i(pq, 32, &Xb[k32 % 256], 32, &x[k32], w, sj / 32);\n" \
"	forward8l(pq, 4, &Xb[k4 % 256], w, sj / 4);\n" \
"	square4x2l(pq, &Xb[k % 256], w, sj);\n" \
"	backward8l(pq, 4, &Xb[k4 % 256], w, sj / 4);\n" \
"	backward8o(pq, 32, &x[k32], 32, &Xb[k32 % 256], w, sj / 32);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 8 * BLK256, 1, 1)))\n" \
"void mul256(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL256();\n" \
"	DECLARE_VARP_REG();\n" \
"\n" \
"	forward8i(pq, 32, &Xb[k32 % 256], 32, &x[k32], w, sj / 32);\n" \
"	forward8l(pq, 4, &Xb[k4 % 256], w, sj / 4);\n" \
"	mul4x2l(pq, &Xb[k % 256], &xp[k], w, sj);\n" \
"	backward8l(pq, 4, &Xb[k4 % 256], w, sj / 4);\n" \
"	backward8o(pq, 32, &x[k32], 32, &Xb[k32 % 256], w, sj / 32);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 8 * BLK256, 1, 1)))\n" \
"void fwd256(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL256();\n" \
"\n" \
"	forward8i(pq, 32, &Xb[k32 % 256], 32, &x[k32], w, sj / 32);\n" \
"	forward8l(pq, 4, &Xb[k4 % 256], w, sj / 4);\n" \
"	fwd4x2o(pq, &x[k], &Xb[k % 256], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 12) || (LN_SZ == 15) || (LN_SZ == 18) || (LN_SZ == 21)\n" \
"\n" \
"#define DECLARE_VAR_SQRMUL512() \\\n" \
"	DECLARE_VAR_SQRMUL(512, BLK512); \\\n" \
"	const sz_t k8 = 7 * (id & ~(8 - 1)) + id, k64 = 7 * (id & ~(64 - 1)) + id;\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(512 / 8 * BLK512, 1, 1)))\n" \
"void square512(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL512();\n" \
"\n" \
"	forward8i(pq, 64, &Xb[k64 % 512], 64, &x[k64], w, sj / 64);\n" \
"	forward8l(pq, 8, &Xb[k8 % 512], w, sj / 8);\n" \
"	square8l(pq, &Xb[k % 512], w, sj);\n" \
"	backward8l(pq, 8, &Xb[k8 % 512], w, sj / 8);\n" \
"	backward8o(pq, 64, &x[k64], 64, &Xb[k64 % 512], w, sj / 64);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(512 / 8 * BLK512, 1, 1)))\n" \
"void mul512(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL512();\n" \
"	DECLARE_VARP_REG();\n" \
"\n" \
"	forward8i(pq, 64, &Xb[k64 % 512], 64, &x[k64], w, sj / 64);\n" \
"	forward8l(pq, 8, &Xb[k8 % 512], w, sj / 8);\n" \
"	mul8l(pq, &Xb[k % 512], &xp[k], w, sj);\n" \
"	backward8l(pq, 8, &Xb[k8 % 512], w, sj / 8);\n" \
"	backward8o(pq, 64, &x[k64], 64, &Xb[k64 % 512], w, sj / 64);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(512 / 8 * BLK512, 1, 1)))\n" \
"void fwd512(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL512();\n" \
"\n" \
"	forward8i(pq, 64, &Xb[k64 % 512], 64, &x[k64], w, sj / 64);\n" \
"	forward8l(pq, 8, &Xb[k8 % 512], w, sj / 8);\n" \
"	fwd8o(pq, &x[k], &Xb[k % 512], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 16) || (LN_SZ == 19) || (LN_SZ == 22)\n" \
"\n" \
"#define DECLARE_VAR_SQRMUL1024() \\\n" \
"	DECLARE_VAR_SQRMUL(1024, BLK1024); \\\n" \
"	const sz_t k2 = 7 * (id & ~(2 - 1)) + id, k16 = 7 * (id & ~(16 - 1)) + id, k128 = 7 * (id & ~(128 - 1)) + id;\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 8 * BLK1024, 1, 1)))\n" \
"void square1024(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL1024();\n" \
"\n" \
"	forward8i(pq, 128, &Xb[k128 % 1024], 128, &x[k128], w, sj / 128);\n" \
"	forward8l(pq, 16, &Xb[k16 % 1024], w, sj / 16);\n" \
"	forward8l(pq, 2, &Xb[k2 % 1024], w, sj / 2);\n" \
"	square2x4l(pq, &Xb[k % 1024], w, sj);\n" \
"	backward8l(pq, 2, &Xb[k2 % 1024], w, sj / 2);\n" \
"	backward8l(pq, 16, &Xb[k16 % 1024], w, sj / 16);\n" \
"	backward8o(pq, 128, &x[k128], 128, &Xb[k128 % 1024], w, sj / 128);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 8 * BLK1024, 1, 1)))\n" \
"void mul1024(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL1024();\n" \
"	DECLARE_VARP_REG();\n" \
"\n" \
"	forward8i(pq, 128, &Xb[k128 % 1024], 128, &x[k128], w, sj / 128);\n" \
"	forward8l(pq, 16, &Xb[k16 % 1024], w, sj / 16);\n" \
"	forward8l(pq, 2, &Xb[k2 % 1024], w, sj / 2);\n" \
"	mul2x4l(pq, &Xb[k % 1024], &xp[k], w, sj);\n" \
"	backward8l(pq, 2, &Xb[k2 % 1024], w, sj / 2);\n" \
"	backward8l(pq, 16, &Xb[k16 % 1024], w, sj / 16);\n" \
"	backward8o(pq, 128, &x[k128], 128, &Xb[k128 % 1024], w, sj / 128);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 8 * BLK1024, 1, 1)))\n" \
"void fwd1024(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL1024();\n" \
"\n" \
"	forward8i(pq, 128, &Xb[k128 % 1024], 128, &x[k128], w, sj / 128);\n" \
"	forward8l(pq, 16, &Xb[k16 % 1024], w, sj / 16);\n" \
"	forward8o(pq, 2, &x[k2], 2, &Xb[k2 % 1024], w, sj / 2);\n" \
"}\n" \
"#endif\n" \
"\n" \
"#if (LN_SZ == 20) || (LN_SZ == 23)\n" \
"\n" \
"#define DECLARE_VAR_SQRMUL2048() \\\n" \
"	DECLARE_VAR_SQRMUL(2048, BLK2048); \\\n" \
"	const sz_t k4 = 7 * (id & ~(4 - 1)) + id, k32 = 7 * (id & ~(32 - 1)) + id, k256 = 7 * (id & ~(256 - 1)) + id;\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(2048 / 8 * BLK2048, 1, 1)))\n" \
"void square2048(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL2048();\n" \
"\n" \
"	forward8i(pq, 256, &Xb[k256 % 2048], 256, &x[k256], w, sj / 256);\n" \
"	forward8l(pq, 32, &Xb[k32 % 2048], w, sj / 32);\n" \
"	forward8l(pq, 4, &Xb[k4 % 2048], w, sj / 4);\n" \
"	square4x2l(pq, &Xb[k % 2048], w, sj);\n" \
"	backward8l(pq, 4, &Xb[k4 % 2048], w, sj / 4);\n" \
"	backward8l(pq, 32, &Xb[k32 % 2048], w, sj / 32);\n" \
"	backward8o(pq, 256, &x[k256], 256, &Xb[k256 % 2048], w, sj / 256);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(2048 / 8 * BLK2048, 1, 1)))\n" \
"void mul2048(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL2048();\n" \
"	DECLARE_VARP_REG();\n" \
"\n" \
"	forward8i(pq, 256, &Xb[k256 % 2048], 256, &x[k256], w, sj / 256);\n" \
"	forward8l(pq, 32, &Xb[k32 % 2048], w, sj / 32);\n" \
"	forward8l(pq, 4, &Xb[k4 % 2048], w, sj / 4);\n" \
"	mul4x2l(pq, &Xb[k % 2048], &xp[k], w, sj);\n" \
"	backward8l(pq, 4, &Xb[k4 % 2048], w, sj / 4);\n" \
"	backward8l(pq, 32, &Xb[k32 % 2048], w, sj / 32);\n" \
"	backward8o(pq, 256, &x[k256], 256, &Xb[k256 % 2048], w, sj / 256);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(2048 / 8 * BLK2048, 1, 1)))\n" \
"void fwd2048(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)\n" \
"{\n" \
"	DECLARE_VAR_SQRMUL2048();\n" \
"\n" \
"	forward8i(pq, 256, &Xb[k256 % 2048], 256, &x[k256], w, sj / 256);\n" \
"	forward8l(pq, 32, &Xb[k32 % 2048], w, sj / 32);\n" \
"	forward8l(pq, 4, &Xb[k4 % 2048], w, sj / 4);\n" \
"	fwd4x2o(pq, &x[k], &Xb[k % 2048], w, sj);\n" \
"}\n" \
"#endif\n" \
"\n" \
"// --- carry ---\n" \
"\n" \
"#define BASE		(1u << LBASE)\n" \
"#define NULL_PTR	((__global const ZP *)0)\n" \
"\n" \
"// Chinese remainder theorem\n" \
"INLINE int_64 garner2(const ZP r1, const ZP r2)\n" \
"{\n" \
"	const uint_32 u2 = r2;\n" \
"	const uint_32 u1 = submulmod(r1, u2, INVP2_P1, PQ1);	// P2 < P1\n" \
"	const int_32 i1 = (u1 > P1 / 2) ? (int_32)(u1 - P1) : (int_32)(u1);\n" \
"\n" \
"	return (int_64)(u2) + (int_64)(P2) * i1;\n" \
"}\n" \
"\n" \
"INLINE int_64 get_i64(const RNS2 r) { return garner2(r.r[0], r.r[1]); }\n" \
"\n" \
"#if (PRM_COUNT == 4)\n" \
"\n" \
"INLINE int_128 garner4(const ZP r1, const ZP r2, const ZP r3, const ZP r4)\n" \
"{\n" \
"	// convert to mixed-radix representation, we must have P4 < P3 < P2 < P1\n" \
"	const uint_32 u4 = r4;\n" \
"	const uint_32 u3 = submulmod(r3, u4, INVP4_P3, PQ3);\n" \
"	const uint_32 u2 = submulmod(submulmod(r2, u4, INVP4_P2, PQ2), u3, INVP3_P2, PQ2);\n" \
"	const uint_32 u1 = submulmod(submulmod(submulmod(r1, u4, INVP4_P1, PQ1), u3, INVP3_P1, PQ1), u2, INVP2_P1, PQ1);\n" \
"	const int_32 i1 = (u1 > P1 / 2) ? (int_32)(u1 - P1) : (int_32)(u1);\n" \
"\n" \
"	const uint_64 u34 = u4 + (uint_64)(P4) * u3;\n" \
"	const int_64 i12 = (int_64)(u2) + (int_64)(P2) * i1;\n" \
"	return int128_add_u64(int128_mul_u64i64(P3 * (uint_64)(P4), i12), u34);\n" \
"}\n" \
"\n" \
"INLINE int_128 get_i128(const RNS4 r) { return garner4(r.r[0], r.r[1], r.r[2], r.r[3]); }\n" \
"\n" \
"#endif\n" \
"\n" \
"#if (PRM_COUNT == 4)\n" \
"\n" \
"INLINE void carry_mul_sub_1(const __global ZP * restrict const xi, __global const ZP * restrict const yi,\n" \
"	__global int_128 * restrict const carry, __local int_128 * const cl, uint2_32 r[CARRY_LENGTH], const sz_t id, const uint_32 a)\n" \
"{\n" \
"	// assert CARRY_LENGTH = 4\n" \
"	ZP4 x4[4]; for (sz_t l = 0; l < 4; ++l) x4[l] = load4g(xi, l * Z_OFFSET / 4);\n" \
"	RNS4 y[4]; for (sz_t l = 0; l < 4; ++l) { y[0].r[l] = x4[l].s0; y[1].r[l] = x4[l].s1; y[2].r[l] = x4[l].s2; y[3].r[l] = x4[l].s3; }\n" \
"\n" \
"	int_128 f = int128_set(0u, 0);\n" \
"	for (sz_t j = 0; j < CARRY_LENGTH; ++j)\n" \
"	{\n" \
"		// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF\n" \
"		f = int128_add(f, int128_mul_u32(get_i128(mulmod4(y[j], NORM_4())), a));\n" \
"		if (yi != NULL_PTR) { const RNS2 r2 = loadRNS2(&yi[j]); f = int128_sub_i64(f, get_i64(r2)); }\n" \
"\n" \
"		r[j].s0 = (uint_32)(f.s0) & (BASE - 1);\n" \
"		f = int128_rshift(f, LBASE);\n" \
"		r[j].s1 = (uint_32)(f.s0) & (BASE - 1);\n" \
"		f = int128_rshift(f, LBASE);\n" \
"	}\n" \
"\n" \
"	const sz_t lid = id % CARRY_WG_SZ;\n" \
"	cl[lid] = f;\n" \
"\n" \
"	if (lid == CARRY_WG_SZ - 1) carry[id / CARRY_WG_SZ] = f;\n" \
"}\n" \
"\n" \
"INLINE void carry_2(__global ZP * restrict const xi, const __local int_128 * const cl, const uint2_32 r[CARRY_LENGTH], const sz_t id)\n" \
"{\n" \
"	const sz_t lid = id % CARRY_WG_SZ;\n" \
"	int_128 f = (lid != 0) ? cl[lid - 1] : int128_set(0u, 0);\n" \
"\n" \
"	for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)\n" \
"	{\n" \
"		f = int128_add_u64(f, r[j].s0 | ((uint_64)(r[j].s1) << LBASE));\n" \
"		const uint_32 r1 = (uint_32)(f.s0) & (BASE - 1);\n" \
"		f = int128_rshift(f, LBASE);\n" \
"		int_32 r2 = (int_32)((uint_32)(f.s0) & (BASE - 1));\n" \
"		f = int128_rshift(f, LBASE);\n" \
"\n" \
"		if (r2 >= (int_32)(BASE / 2)) { r2 -= (int_32)(BASE); f = int128_add_u64(f, 1); }	// balanced\n" \
"\n" \
"		const RNS4 u = addmod4(set4_u32(r1), mulmod4(set4_i32(r2), RBASE_4()));\n" \
"		storeRNS4(&xi[j], u);\n" \
"	}\n" \
"\n" \
"	f = int128_add_u64(f, r[CARRY_LENGTH - 1].s0 | ((uint_64)(r[CARRY_LENGTH - 1].s1) << LBASE));\n" \
"	const uint_32 r1 = (uint_32)(f.s0) & (BASE - 1);\n" \
"	f = int128_rshift(f, LBASE);\n" \
"	const int_32 r2 = (int_32)(f.s0);\n" \
"\n" \
"	const RNS4 u = addmod4(set4_u32(r1), mulmod4(set4_i32(r2), RBASE_4()));\n" \
"	storeRNS4(&xi[CARRY_LENGTH - 1], u);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))\n" \
"void carry_mul(__global ZP * restrict const reg, __global int_128 * restrict const carry, const uint_32 reg_x, const uint_32 a)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];\n" \
"	__local int_128 cl[CARRY_WG_SZ];\n" \
"	uint2_32 r[CARRY_LENGTH];\n" \
"\n" \
"	carry_mul_sub_1(xi, NULL_PTR, carry, cl, r, id, a);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	carry_2(xi, cl, r, id);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))\n" \
"void carry_sub(__global ZP * restrict const reg, __global int_128 * restrict const carry, const uint_32 reg_x, const uint_32 reg_y)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];\n" \
"	__global const ZP * restrict const yi = &reg[reg_y * N_SZ + CARRY_LENGTH * id];\n" \
"	__local int_128 cl[CARRY_WG_SZ];\n" \
"	uint2_32 r[CARRY_LENGTH];\n" \
"\n" \
"	carry_mul_sub_1(xi, yi, carry, cl, r, id, 1);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	carry_2(xi, cl, r, id);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void carry_complete(__global ZP * restrict const reg, __global const int_128 * restrict const carry, const uint_32 reg_x, const int_32 is_nega)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * CARRY_WG_SZ * id];\n" \
"\n" \
"	int_128 f = carry[((id == 0) ? (N_SZ / CARRY_LENGTH / CARRY_WG_SZ) : id) - 1];\n" \
"	if ((id == 0) && (is_nega != 0)) f = int128_neg(f);\n" \
"\n" \
"	// assert CARRY_LENGTH = 4\n" \
"	ZP4 x4[2]; for (sz_t l = 0; l < 2; ++l) x4[l] = load4g(xi, l * Z_OFFSET / 4);\n" \
"	RNS2 y[4]; for (sz_t l = 0; l < 2; ++l) { y[0].r[l] = x4[l].s0; y[1].r[l] = x4[l].s1; y[2].r[l] = x4[l].s2; y[3].r[l] = x4[l].s3; }\n" \
"\n" \
"	for (sz_t j = 0; j < CARRY_LENGTH - 1; ++j)\n" \
"	{\n" \
"		f = int128_add_i64(f, get_i64(y[j]));\n" \
"		const uint_32 r1 = (uint_32)(f.s0) & (BASE - 1);\n" \
"		f = int128_rshift(f, LBASE);\n" \
"		int_32 r2 = (int_32)((uint_32)(f.s0) & (BASE - 1));\n" \
"		f = int128_rshift(f, LBASE);\n" \
"\n" \
"		if (r2 >= (int_32)(BASE / 2)) { r2 -= (int_32)(BASE); f = int128_add_u64(f, 1); }	// balanced\n" \
"\n" \
"		const RNS4 u = addmod4(set4_u32(r1), mulmod4(set4_i32(r2), RBASE_4()));\n" \
"		storeRNS4(&xi[j], u);\n" \
"	}\n" \
"\n" \
"	f = int128_add_i64(f, get_i64(y[CARRY_LENGTH - 1]));\n" \
"	const uint_32 r1 = (uint_32)(f.s0) & (BASE - 1);\n" \
"	f = int128_rshift(f, LBASE);\n" \
"	const int_32 r2 = (int_32)(f.s0);\n" \
"\n" \
"	const RNS4 u = addmod4(set4_u32(r1), mulmod4(set4_i32(r2), RBASE_4()));\n" \
"	storeRNS4(&xi[CARRY_LENGTH - 1], u);\n" \
"}\n" \
"\n" \
"#elif (PRM_COUNT == 2)\n" \
"\n" \
"INLINE void carry_mul_sub_1(const __global ZP * restrict const xi, __global const ZP * restrict const yi,\n" \
"	__global int_64 * restrict const carry, __local int_64 * const cl, uint_32 r[CARRY_LENGTH], const sz_t id, const uint_32 a)\n" \
"{\n" \
"	// assert CARRY_LENGTH = 4\n" \
"	ZP4 x4[2]; for (sz_t l = 0; l < 2; ++l) x4[l] = load4g(xi, l * Z_OFFSET / 4);\n" \
"	RNS2 y[4]; for (sz_t l = 0; l < 2; ++l) { y[0].r[l] = x4[l].s0; y[1].r[l] = x4[l].s1; y[2].r[l] = x4[l].s2; y[3].r[l] = x4[l].s3; }\n" \
"\n" \
"	int_64 f = 0;\n" \
"	for (sz_t j = 0; j < CARRY_LENGTH; ++j)\n" \
"	{\n" \
"		// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF\n" \
"		f += get_i64(mulmod2(y[j], NORM_2())) * a;\n" \
"		if (yi != NULL_PTR) f -= get_i32(yi[j], P1);\n" \
"\n" \
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
"INLINE void carry_2(__global ZP * restrict const xi, const __local int_64 * const cl, const uint_32 r[CARRY_LENGTH], const sz_t id)\n" \
"{\n" \
"	const sz_t lid = id % CARRY_WG_SZ;\n" \
"	int_64 f = (lid != 0) ? cl[lid - 1] : 0;\n" \
"\n" \
"	for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)\n" \
"	{\n" \
"		f += r[j];\n" \
"		int_32 r1 = (int_32)((uint_32)(f) & (BASE - 1));\n" \
"		f >>= LBASE;\n" \
"\n" \
"		if (r1 >= (int_32)(BASE / 2)) { r1 -= (int_32)(BASE); f += 1; }	// balanced\n" \
"\n" \
"		const RNS2 u = set2_i32(r1);\n" \
"		storeRNS2(&xi[j], u);\n" \
"	}\n" \
"\n" \
"	f += r[CARRY_LENGTH - 1];\n" \
"	const int_32 r1 = (int_32)(f);\n" \
"\n" \
"	const RNS2 u = set2_i32(r1);\n" \
"	storeRNS2(&xi[CARRY_LENGTH - 1], u);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))\n" \
"void carry_mul(__global ZP * restrict const reg, __global int_64 * restrict const carry, const uint_32 reg_x, const uint_32 a)\n" \
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
"\n" \
"	int_64 f = carry[((id == 0) ? (N_SZ / CARRY_LENGTH / CARRY_WG_SZ) : id) - 1];\n" \
"	if ((id == 0) && (is_nega != 0)) f = -f;\n" \
"\n" \
"	// assert CARRY_LENGTH = 4\n" \
"	const ZP4 x4 = load4g(xi, 0 * Z_OFFSET / 4);\n" \
"	ZP y[4]; y[0] = x4.s0; y[1] = x4.s1; y[2] = x4.s2; y[3] = x4.s3;\n" \
"\n" \
"	for (sz_t j = 0; j < CARRY_LENGTH - 1; ++j)\n" \
"	{\n" \
"		f += get_i32(y[j], P1);\n" \
"		int_32 r1 = (int_32)((uint_32)(f) & (BASE - 1));\n" \
"		f >>= LBASE;\n" \
"\n" \
"		if (r1 >= (int_32)(BASE / 2)) { r1 -= (int_32)(BASE); f += 1; }	// balanced\n" \
"\n" \
"		const RNS2 u = set2_i32(r1);\n" \
"		storeRNS2(&xi[j], u);\n" \
"	}\n" \
"\n" \
"	f += get_i32(y[CARRY_LENGTH - 1], P1);\n" \
"	const int_32 r1 = (int_32)(f);\n" \
"\n" \
"	const RNS2 u = set2_i32(r1);\n" \
"	storeRNS2(&xi[CARRY_LENGTH - 1], u);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"// --- strict modulo ---\n" \
"\n" \
"INLINE void lower(__global ZP * restrict const x, const int_32 l0)\n" \
"{\n" \
"#if (PRM_COUNT == 4)\n" \
"\n" \
"	const int_64 base2 = (int_64)(1) << (2 * LBASE);\n" \
"\n" \
"	const RNS2 r2 = loadRNS2(&x[N_SZ - 1]);\n" \
"	if (get_i64(r2) >= base2)	// >= base^n\n" \
"	{\n" \
"		// - R or Q\n" \
"		const RNS4 hi = submod4(loadRNS4(&x[N_SZ - 1]), RBASESQ_4());\n" \
"		storeRNS4(&x[N_SZ - 1], hi);\n" \
"\n" \
"		const RNS4 lo = submod4(loadRNS4(&x[0]), set4_i32(l0));\n" \
"		storeRNS4(&x[0], lo);\n" \
"	}\n" \
"\n" \
"#elif (PRM_COUNT == 2)\n" \
"\n" \
"	int_32 hi = get_i32(x[N_SZ - 1], P1);\n" \
"	if (hi >= (int_32)(BASE))	// >= base^n\n" \
"	{\n" \
"		// - R or Q\n" \
"		int_32 lo = get_i32(x[0], P1);\n" \
"		hi -= (int_32)(BASE); lo -= l0;	// - R or Q\n" \
"		storeRNS2(&x[N_SZ - 1], set2_i32(hi));\n" \
"		storeRNS2(&x[0], set2_i32(lo));\n" \
"	}\n" \
"\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mod_Q(__global ZP * restrict const reg, const uint_32 reg_x)\n" \
"{\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"\n" \
"	lower(x, 1);	// < Q - 1\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mod_R_add2(__global ZP * restrict const reg, const uint_32 reg_x)\n" \
"{\n" \
"	__global ZP * restrict const x = &reg[reg_x * N_SZ];\n" \
"\n" \
"	lower(x, -1);		// <= R\n" \
"\n" \
"#if (PRM_COUNT == 4)\n" \
"\n" \
"	const RNS4 x0 = addmod4(loadRNS4(&x[0]), set4_u32(2));\n" \
"	storeRNS4(&x[0], x0);\n" \
"\n" \
"#elif (PRM_COUNT == 2)\n" \
"\n" \
"	const RNS2 x0 = addmod2(loadRNS2(&x[0]), set2_u32(2));\n" \
"	storeRNS2(&x[0], x0);\n" \
"\n" \
"#endif\n" \
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
"	for (sz_t l = 0; l < 4; ++l) y[l * Z_OFFSET / 4 + id] = x[l * Z_OFFSET / 4 + id];\n" \
"}\n" \
"\n" \
"";
