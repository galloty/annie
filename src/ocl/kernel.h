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
"#define PCK_FACTOR		2\n" \
"#define P0				4194304001u\n" \
"#define Q0				100663297u\n" \
"#define RSQ0			232465106u\n" \
"#define P1				4076863489u\n" \
"#define Q1				218103809u\n" \
"#define RSQ1			3444438393u\n" \
"#define P2				3942645761u\n" \
"#define Q2				352321537u\n" \
"#define RSQ2			3810498414u\n" \
"#define P3				3892314113u\n" \
"#define Q3				402653185u\n" \
"#define RSQ3			1703176688u\n" \
"#define P4				3489660929u\n" \
"#define Q4				805306369u\n" \
"#define RSQ4			1961643719u\n" \
"#define P5				3238002689u\n" \
"#define Q5				1056964609u\n" \
"#define RSQ5			1601137523u\n" \
"#define P6				3221225473u\n" \
"#define Q6				1073741825u\n" \
"#define RSQ6			1789569709u\n" \
"#define P7				2885681153u\n" \
"#define Q7				1409286145u\n" \
"#define RSQ7			1727663083u\n" \
"#define P8				2868903937u\n" \
"#define Q8				1426063361u\n" \
"#define RSQ8			2445254707u\n" \
"#define P9				2717908993u\n" \
"#define Q9				1577058305u\n" \
"#define RSQ9			2541023286u\n" \
"#define P10				2634022913u\n" \
"#define Q10				1660944385u\n" \
"#define RSQ10			2593308780u\n" \
"#define P11				2533359617u\n" \
"#define Q11				1761607681u\n" \
"#define RSQ11			2298256381u\n" \
"#define P12				2483027969u\n" \
"#define Q12				1811939329u\n" \
"#define RSQ12			288386743u\n" \
"#define P13				2281701377u\n" \
"#define Q13				2013265921u\n" \
"#define RSQ13			773725730u\n" \
"#define CNORM			const uint c_norm[4] = { 3670016001u, 3567255553u, 3449815041u, 3405774849u };\n" \
"#define CBASE			const uint c_base[4] = { 3174786139u, 3997810352u, 1461973742u, 212897086u };\n" \
"#define CBASE_PCK		const uint c_base_pck[4] = { 3018288268u, 499726294u, 1168408158u, 999690664u };\n" \
"#define INVP1_P0		1797558821u\n" \
"#define INVP2_P0		3075822917u\n" \
"#define INVP2_P1		4076863457u\n" \
"#define INVP3_P0		3262236431u\n" \
"#define INVP3_P1		2223743698u\n" \
"#define INVP3_P2		2628430422u\n" \
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
"// --- modular arithmetic\n" \
"\n" \
"typedef uint_32		ZP;\n" \
"typedef uint2_32	ZP2;\n" \
"typedef uint4_32	ZP4;\n" \
"\n" \
"#define	PQ0		(ZP2)(P0, Q0)\n" \
"#define	PQ1		(ZP2)(P1, Q1)\n" \
"#define	PQ2		(ZP2)(P2, Q2)\n" \
"#define	PQ3		(ZP2)(P3, Q3)\n" \
"#define	PQ4		(ZP2)(P4, Q4)\n" \
"#define	PQ5		(ZP2)(P5, Q5)\n" \
"#define	PQ6		(ZP2)(P6, Q6)\n" \
"#define	PQ7		(ZP2)(P7, Q7)\n" \
"#define	PQ8		(ZP2)(P8, Q8)\n" \
"#define	PQ9		(ZP2)(P9, Q9)\n" \
"#define	PQ10	(ZP2)(P10, Q10)\n" \
"#define	PQ11	(ZP2)(P11, Q11)\n" \
"#define	PQ12	(ZP2)(P12, Q12)\n" \
"#define	PQ13	(ZP2)(P13, Q13)\n" \
"\n" \
"__constant ZP2 g_pq[14] = { PQ0, PQ1, PQ2, PQ3, PQ4, PQ5, PQ6, PQ7, PQ8, PQ9, PQ10, PQ11, PQ12, PQ13 };\n" \
"__constant ZP g_rsq[14] = { RSQ0, RSQ1, RSQ2, RSQ3, RSQ4, RSQ5, RSQ6, RSQ7, RSQ8, RSQ9, RSQ10, RSQ11, RSQ12, RSQ13 };\n" \
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
"// Montgomery multiplication by a constant: 1 mul + 2 mul_hi\n" \
"INLINE ZP mulmod_const(const ZP lhs, const ZP rhs, const ZP rhs_q, const uint_32 p)\n" \
"{\n" \
"	return submod(mul_hi(lhs, rhs), mul_hi(lhs * rhs_q, p), p);\n" \
"}\n" \
"\n" \
"// Montgomery multiplication: 2 mul + 2 mul_hi\n" \
"INLINE ZP mulmod(const ZP lhs, const ZP rhs, const ZP2 pq)\n" \
"{\n" \
"	const uint_64 t = lhs * (uint_64)(rhs);\n" \
"	const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);\n" \
"	return submod(hi, mul_hi(lo * pq.s1, pq.s0), pq.s0);\n" \
"}\n" \
"\n" \
"INLINE ZP submulmod(const ZP lhs, const ZP rhs, const ZP m, const ZP2 pq)\n" \
"{\n" \
"	return mulmod_const(submod(lhs, rhs, pq.s0), m, m * pq.s1, pq.s0);\n" \
"}\n" \
"\n" \
"INLINE uint_32 set_i32(const int_32 i, const uint_32 p) { return (uint_32)(i) + ((i < 0) ? p : 0); }\n" \
"INLINE int_32 get_i32(const uint_32 u, const uint_32 p) { return (int_32)(u - ((u >= p / 2) ? p : 0)); }\n" \
"\n" \
"typedef struct { ZP r[PRM_COUNT]; } RNS_K;\n" \
"\n" \
"INLINE RNS_K setK_u32(const uint_32 n) { RNS_K r; for (sz_t k = 0; k < PRM_COUNT; ++k) r.r[k] = n; return r; }\n" \
"INLINE RNS_K setK_i32(const int_32 i) { RNS_K r; for (sz_t k = 0; k < PRM_COUNT; ++k) r.r[k] = set_i32(i, g_pq[k].s0); return r; }\n" \
"\n" \
"INLINE RNS_K addmodK(const RNS_K lhs, const RNS_K rhs)\n" \
"{\n" \
"	RNS_K r; for (sz_t k = 0; k < PRM_COUNT; ++k) r.r[k] = addmod(lhs.r[k], rhs.r[k], g_pq[k].s0); return r;\n" \
"}\n" \
"\n" \
"INLINE RNS_K submodK(const RNS_K lhs, const RNS_K rhs)\n" \
"{\n" \
"	RNS_K r; for (sz_t k = 0; k < PRM_COUNT; ++k) r.r[k] = submod(lhs.r[k], rhs.r[k], g_pq[k].s0); return r;\n" \
"}\n" \
"\n" \
"INLINE RNS_K mulmodK(const RNS_K lhs, const RNS_K rhs)\n" \
"{\n" \
"	RNS_K r; for (sz_t k = 0; k < PRM_COUNT; ++k) r.r[k] = mulmod(lhs.r[k], rhs.r[k], g_pq[k]); return r;\n" \
"}\n" \
"\n" \
"INLINE RNS_K NORM_K() { CNORM; RNS_K r; for (sz_t k = 0; k < PRM_COUNT; ++k) r.r[k] = c_norm[k]; return r; }\n" \
"INLINE RNS_K RBASE_K() { CBASE; RNS_K r; for (sz_t k = 0; k < PRM_COUNT; ++k) r.r[k] = c_base[k]; return r; }\n" \
"INLINE RNS_K RBASE_PCK_K() { CBASE_PCK; RNS_K r; for (sz_t k = 0; k < PRM_COUNT; ++k) r.r[k] = c_base_pck[k]; return r; }\n" \
"\n" \
"typedef struct { uint_32 r[PCK_FACTOR]; } pck_type;\n" \
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
"INLINE RNS_K loadRNS_K(__global const ZP * restrict const x) { RNS_K r; for (sz_t k = 0; k < PRM_COUNT; ++k) r.r[k] = x[k * Z_OFFSET]; return r; }\n" \
"INLINE void storeRNS_K(__global ZP * restrict const x, const RNS_K r) { for (sz_t k = 0; k < PRM_COUNT; ++k) x[k * Z_OFFSET] = r.r[k]; }\n" \
"\n" \
"// INLINE pck_type loadRNS_K_pck(__global const ZP * restrict const x) { pck_type r; for (sz_t l = 0; l < PCK_FACTOR; ++l) r.r[l] = x[l * Z_OFFSET]; return r; }\n" \
"INLINE void storeRNS_K_pck(__global ZP * restrict const x, const pck_type r) { for (sz_t l = 0; l < PCK_FACTOR; ++l) x[l * Z_OFFSET] = r.r[l]; }\n" \
"\n" \
"INLINE void loadRNS_K_pck4(pck_type y[4], __global const ZP * restrict const x)\n" \
"{\n" \
"	ZP4 x4[PCK_FACTOR]; for (sz_t l = 0; l < PCK_FACTOR; ++l) x4[l] = load4g(x, l * Z_OFFSET / 4);\n" \
"	for (sz_t l = 0; l < PCK_FACTOR; ++l) { y[0].r[l] = x4[l].s0; y[1].r[l] = x4[l].s1; y[2].r[l] = x4[l].s2; y[3].r[l] = x4[l].s3; }\n" \
"}\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define FWD2(x0, x1, w) \\\n" \
"{ \\\n" \
"	const ZP t = mulmod(x1, w, pq); \\\n" \
"	x1 = submod(x0, t, pq.s0); x0 = addmod(x0, t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define FWD2_CONST(x0, x1, w, w_q) \\\n" \
"{ \\\n" \
"	const ZP t = mulmod_const(x1, w, w_q, pq.s0); \\\n" \
"	x1 = submod(x0, t, pq.s0); x0 = addmod(x0, t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define BCK2(x0, x1, w_i) \\\n" \
"{ \\\n" \
"	const ZP t = submod(x0, x1, pq.s0); x0 = addmod(x0, x1, pq.s0); \\\n" \
"	x1 = mulmod(t, w_i, pq); \\\n" \
"}\n" \
"\n" \
"#define BCK2_CONST(x0, x1, w_i, w_i_q) \\\n" \
"{ \\\n" \
"	const ZP t = submod(x0, x1, pq.s0); x0 = addmod(x0, x1, pq.s0); \\\n" \
"	x1 = mulmod_const(t, w_i, w_i_q, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define SQR2(x0, x1, w, w_q) \\\n" \
"{ \\\n" \
"	const ZP t = mulmod_const(mulmod(x1, x1, pq), w, w_q, pq.s0); \\\n" \
"	x1 = mulmod(addmod(x0, x0, pq.s0), x1, pq); \\\n" \
"	x0 = addmod(mulmod(x0, x0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define SQR2N(x0, x1, w, w_q) \\\n" \
"{ \\\n" \
"	const ZP t = mulmod_const(mulmod(x1, x1, pq), w, w_q, pq.s0); \\\n" \
"	x1 = mulmod(addmod(x0, x0, pq.s0), x1, pq); \\\n" \
"	x0 = submod(mulmod(x0, x0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define MUL2(x0, x1, xp0, xp1, w, w_q) \\\n" \
"{ \\\n" \
"	const ZP t = mulmod_const(mulmod(x1, xp1, pq), w, w_q, pq.s0); \\\n" \
"	x1 = addmod(mulmod(x0, xp1, pq), mulmod(xp0, x1, pq), pq.s0); \\\n" \
"	x0 = addmod(mulmod(x0, xp0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define MUL2N(x0, x1, xp0, xp1, w, w_q) \\\n" \
"{ \\\n" \
"	const ZP t = mulmod_const(mulmod(x1, xp1, pq), w, w_q, pq.s0); \\\n" \
"	x1 = addmod(mulmod(x0, xp1, pq), mulmod(xp0, x1, pq), pq.s0); \\\n" \
"	x0 = submod(mulmod(x0, xp0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"// --- transform/inline ---\n" \
"\n" \
"INLINE void _forward8(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2, const ZP4 w4)\n" \
"{\n" \
"	const ZP w1_q = w1 * pq.s1;\n" \
"	FWD2_CONST(x[0], x[4], w1, w1_q); FWD2_CONST(x[1], x[5], w1, w1_q);\n" \
"	FWD2_CONST(x[2], x[6], w1, w1_q); FWD2_CONST(x[3], x[7], w1, w1_q);\n" \
"	const ZP w20_q = w2.s0 * pq.s1;\n" \
"	FWD2_CONST(x[0], x[2], w2.s0, w20_q); FWD2_CONST(x[1], x[3], w2.s0, w20_q);\n" \
"	const ZP w21_q = w2.s1 * pq.s1;\n" \
"	FWD2_CONST(x[4], x[6], w2.s1, w21_q); FWD2_CONST(x[5], x[7], w2.s1, w21_q);\n" \
"	FWD2(x[0], x[1], w4.s0); FWD2(x[2], x[3], w4.s1);\n" \
"	FWD2(x[4], x[5], w4.s2); FWD2(x[6], x[7], w4.s3);\n" \
"}\n" \
"\n" \
"INLINE void _backward8(const ZP2 pq, ZP x[8], const ZP w1_i, const ZP2 w2_i, const ZP4 w4_i)\n" \
"{\n" \
"	BCK2(x[0], x[1], w4_i.s0); BCK2(x[2], x[3], w4_i.s1);\n" \
"	BCK2(x[4], x[5], w4_i.s2); BCK2(x[6], x[7], w4_i.s3);\n" \
"	const ZP w20_i_q = w2_i.s0 * pq.s1;\n" \
"	BCK2_CONST(x[0], x[2], w2_i.s0, w20_i_q); BCK2_CONST(x[1], x[3], w2_i.s0, w20_i_q);\n" \
"	const ZP w21_i_q = w2_i.s1 * pq.s1;\n" \
"	BCK2_CONST(x[4], x[6], w2_i.s1, w21_i_q); BCK2_CONST(x[5], x[7], w2_i.s1, w21_i_q);\n" \
"	const ZP w1_i_q = w1_i * pq.s1;\n" \
"	BCK2_CONST(x[0], x[4], w1_i, w1_i_q); BCK2_CONST(x[1], x[5], w1_i, w1_i_q);\n" \
"	BCK2_CONST(x[2], x[6], w1_i, w1_i_q); BCK2_CONST(x[3], x[7], w1_i, w1_i_q);\n" \
"}\n" \
"\n" \
"INLINE void _forward8_0(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2, const ZP4 w4, const ZP rsq)\n" \
"{\n" \
"	for (sz_t i = 0; i < 4; ++i) x[i] = mulmod(x[i], rsq, pq);\n" \
"	_forward8(pq, x, w1, w2, w4);\n" \
"}\n" \
"\n" \
"INLINE void _square2x4(const ZP2 pq, ZP x[8], const ZP2 w2, const ZP2 w2_q)\n" \
"{\n" \
"	SQR2(x[0], x[1], w2.s0, w2_q.s0); SQR2N(x[2], x[3], w2.s0, w2_q.s0);\n" \
"	SQR2(x[4], x[5], w2.s1, w2_q.s1); SQR2N(x[6], x[7], w2.s1, w2_q.s1);\n" \
"}\n" \
"\n" \
"INLINE void _square4x2(const ZP2 pq, ZP x[8], const ZP2 w2, const ZP2 w2_i)\n" \
"{\n" \
"	const ZP2 w2_q = w2 * pq.s1;\n" \
"	FWD2_CONST(x[0], x[2], w2.s0, w2_q.s0); FWD2_CONST(x[1], x[3], w2.s0, w2_q.s0);\n" \
"	FWD2_CONST(x[4], x[6], w2.s1, w2_q.s1); FWD2_CONST(x[5], x[7], w2.s1, w2_q.s1);\n" \
"	_square2x4(pq, x, w2, w2_q);\n" \
"	const ZP2 w2_i_q = w2_i * pq.s1;\n" \
"	BCK2_CONST(x[0], x[2], w2_i.s0, w2_i_q.s0); BCK2_CONST(x[1], x[3], w2_i.s0, w2_i_q.s0);\n" \
"	BCK2_CONST(x[4], x[6], w2_i.s1, w2_i_q.s1); BCK2_CONST(x[5], x[7], w2_i.s1, w2_i_q.s1);\n" \
"}\n" \
"\n" \
"INLINE void _square8(const ZP2 pq, ZP x[8], const ZP w1, const ZP w1_i, const ZP2 w2, const ZP2 w2_i)\n" \
"{\n" \
"	const ZP w1_q = w1 * pq.s1;\n" \
"	FWD2_CONST(x[0], x[4], w1, w1_q); FWD2_CONST(x[1], x[5], w1, w1_q);\n" \
"	FWD2_CONST(x[2], x[6], w1, w1_q); FWD2_CONST(x[3], x[7], w1, w1_q);\n" \
"	_square4x2(pq, x, w2, w2_i);\n" \
"	const ZP w1_i_q = w1_i * pq.s1;\n" \
"	BCK2_CONST(x[0], x[4], w1_i, w1_i_q); BCK2_CONST(x[1], x[5], w1_i, w1_i_q);\n" \
"	BCK2_CONST(x[2], x[6], w1_i, w1_i_q); BCK2_CONST(x[3], x[7], w1_i, w1_i_q);\n" \
"}\n" \
"\n" \
"INLINE void _mul2x4(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP2 w2, const ZP2 w2_q)\n" \
"{\n" \
"	MUL2(x[0], x[1], xp[0], xp[1], w2.s0, w2_q.s0); MUL2N(x[2], x[3], xp[2], xp[3], w2.s0, w2_q.s0);\n" \
"	MUL2(x[4], x[5], xp[4], xp[5], w2.s1, w2_q.s1); MUL2N(x[6], x[7], xp[6], xp[7], w2.s1, w2_q.s1);\n" \
"}\n" \
"\n" \
"INLINE void _mul4x2(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP2 w2, const ZP2 w2_i)\n" \
"{\n" \
"	const ZP2 w2_q = w2 * pq.s1;\n" \
"	FWD2_CONST(x[0], x[2], w2.s0, w2_q.s0); FWD2_CONST(x[1], x[3], w2.s0, w2_q.s0);\n" \
"	FWD2_CONST(x[4], x[6], w2.s1, w2_q.s1); FWD2_CONST(x[5], x[7], w2.s1, w2_q.s1);\n" \
"	_mul2x4(pq, x, xp, w2, w2_q);\n" \
"	const ZP2 w2_i_q = w2_i * pq.s1;\n" \
"	BCK2_CONST(x[0], x[2], w2_i.s0, w2_i_q.s0); BCK2_CONST(x[1], x[3], w2_i.s0, w2_i_q.s0);\n" \
"	BCK2_CONST(x[4], x[6], w2_i.s1, w2_i_q.s1); BCK2_CONST(x[5], x[7], w2_i.s1, w2_i_q.s1);\n" \
"}\n" \
"\n" \
"INLINE void _mul8(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP w1, const ZP w1_i, const ZP2 w2, const ZP2 w2_i)\n" \
"{\n" \
"	const ZP w1_q = w1 * pq.s1;\n" \
"	FWD2_CONST(x[0], x[4], w1, w1_q); FWD2_CONST(x[1], x[5], w1, w1_q);\n" \
"	FWD2_CONST(x[2], x[6], w1, w1_q); FWD2_CONST(x[3], x[7], w1, w1_q);\n" \
"	_mul4x2(pq, x, xp, w2, w2_i);\n" \
"	const ZP w1_i_q = w1_i * pq.s1;\n" \
"	BCK2_CONST(x[0], x[4], w1_i, w1_i_q); BCK2_CONST(x[1], x[5], w1_i, w1_i_q);\n" \
"	BCK2_CONST(x[2], x[6], w1_i, w1_i_q); BCK2_CONST(x[3], x[7], w1_i, w1_i_q);\n" \
"}\n" \
"\n" \
"INLINE void _fwd4x2(const ZP2 pq, ZP x[8], const ZP2 w2)\n" \
"{\n" \
"	const ZP2 w2_q = w2 * pq.s1;\n" \
"	FWD2_CONST(x[0], x[2], w2.s0, w2_q.s0); FWD2_CONST(x[1], x[3], w2.s0, w2_q.s0);\n" \
"	FWD2_CONST(x[4], x[6], w2.s1, w2_q.s1); FWD2_CONST(x[5], x[7], w2.s1, w2_q.s1);\n" \
"}\n" \
"\n" \
"INLINE void _fwd8(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2)\n" \
"{\n" \
"	const ZP w1_q = w1 * pq.s1;\n" \
"	FWD2_CONST(x[0], x[4], w1, w1_q); FWD2_CONST(x[1], x[5], w1, w1_q);\n" \
"	FWD2_CONST(x[2], x[6], w1, w1_q); FWD2_CONST(x[3], x[7], w1, w1_q);\n" \
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
"	_square2x4(pq, xl, w2, w2 * pq.s1);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"INLINE void square2x4l(const ZP2 pq, __local ZP * restrict const X,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2();\n" \
"	LOADL(1);\n" \
"	_square2x4(pq, xl, w2, w2 * pq.s1);\n" \
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
"	_mul2x4(pq, xl, xpl, w2, w2 * pq.s1);\n" \
"	STOREG(1);\n" \
"}\n" \
"\n" \
"INLINE void mul2x4l(const ZP2 pq, __local ZP * restrict const X, __global const ZP * restrict const xp,\n" \
"	__global const ZP * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2();\n" \
"	LOADL(1);\n" \
"	LOADPG(1);\n" \
"	_mul2x4(pq, xl, xpl, w2, w2 * pq.s1);\n" \
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
"__kernel	// TODO ZP4\n" \
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
"#define BASE2		((uint_64)(1) << (2 * LBASE))\n" \
"#define NULL_PTR	((__global const ZP *)0)\n" \
"\n" \
"// carry-in and carry-out addition\n" \
"INLINE uint_64 _addcc(const uint_64 a, const uint_64 b, bool * const carry)\n" \
"{\n" \
"	const uint_64 t = a + b;\n" \
"	const bool c1 = (t < b);\n" \
"	const uint_64 c = t + ((*carry) ? 1 : 0);\n" \
"	const bool c2 = (c < (*carry));\n" \
"	(*carry) = c1 | c2;\n" \
"	return c;\n" \
"}\n" \
"\n" \
"// carry-in and carry-out subtraction\n" \
"INLINE uint_64 _subcc(const uint_64 a, const uint_64 b, bool * const carry)\n" \
"{\n" \
"	const uint_64 t = a - b;\n" \
"	const bool c1 = (a < b);\n" \
"	const uint_64 c = t - ((*carry)  ? 1 : 0);\n" \
"	const bool c2 = (t < (*carry));\n" \
"	(*carry) = c1 | c2;\n" \
"	return c;\n" \
"}\n" \
"\n" \
"#if (PRM_COUNT == 2)\n" \
"\n" \
"typedef struct { int_64 l[1]; } big_type;		// 64-bit, size = PRM_COUNT * 32 > LN + 2 * LBASE\n" \
"\n" \
"INLINE big_type big_zero() { big_type r; r.l[0] = 0; return r; }\n" \
"INLINE big_type big_add(const big_type x, const big_type y) { big_type r; r.l[0] = x.l[0] + y.l[0]; return r; }\n" \
"INLINE big_type big_sub(const big_type x, const big_type y) { big_type r; r.l[0] = x.l[0] - y.l[0]; return r; }\n" \
"INLINE big_type big_neg(const big_type x) { big_type r; r.l[0] = -x.l[0]; return r; }\n" \
"INLINE big_type big_inc(const big_type x) { big_type r; r.l[0] = x.l[0] + 1; return r; }\n" \
"INLINE big_type big_add_pck(const big_type x, const pck_type y) { big_type r; r.l[0] = x.l[0] + (int_64)(y.r[0]); return r; }	// PCK_FACTOR = 1\n" \
"INLINE big_type big_mul_u32(const big_type x, const uint_32 y) { big_type r; r.l[0] = x.l[0] * (int_64)(y); return r; }\n" \
"INLINE big_type big_rshift(const big_type x, const int s) { big_type r; r.l[0] = x.l[0] >> s; return r; }\n" \
"\n" \
"INLINE bool big_ge_pow_2_s(const big_type x, const int s)	// x >= 2^s\n" \
"{\n" \
"	if (x.l[0] < 0) return false;\n" \
"	return ((x.l[0] >> s) != 0);\n" \
"}\n" \
"\n" \
"INLINE big_type garner2(const ZP r0, const ZP r1)\n" \
"{\n" \
"	const uint_32 u1 = r1;\n" \
"	const uint_32 u0 = submulmod(r0, u1, INVP1_P0, PQ0);	// P2 < P1\n" \
"	const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);\n" \
"\n" \
"	big_type r; r.l[0] = (int_64)(u1) + (int_64)(P1) * i0;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE big_type getK_i(const RNS_K r) { return garner2(r.r[0], r.r[1]); }\n" \
"\n" \
"#else\n" \
"\n" \
"// size = PRM_COUNT * 32 > LN + 2 * LBASE\n" \
"#define BIG_LSIZE	((PRM_COUNT - 1) / 2)\n" \
"#if (PRM_COUNT % 2 == 1)\n" \
"#define HTYPE		int_32\n" \
"#else\n" \
"#define HTYPE		int_64\n" \
"#endif\n" \
"\n" \
"#endif\n" \
"\n" \
"#if (PRM_COUNT > 2)\n" \
"\n" \
"typedef struct { uint_64 l[BIG_LSIZE]; HTYPE h; } big_type;\n" \
"\n" \
"INLINE big_type big_zero() { big_type r; for (sz_t i = 0; i < BIG_LSIZE; ++i) r.l[i] = 0; r.h = 0; return r; }\n" \
"\n" \
"INLINE big_type big_add(const big_type x, const big_type y)\n" \
"{\n" \
"	big_type r;\n" \
"// #if defined(PTX_ASM)\n" \
"// 	asm volatile (\"add.cc.u64 %0, %1, %2;\" : \"=r\" (r.s0) : \"r\" (x.s0), \"r\" (y.s0));\n" \
"// 	asm volatile (\"addc.s64 %0, %1, %2;\" : \"=r\" (r.s1) : \"r\" (x.s1), \"r\" (y.s1));\n" \
"// #else\n" \
"	bool carry = false;\n" \
"	for (sz_t i = 0; i < BIG_LSIZE; ++i) r.l[i] = _addcc(x.l[i], y.l[i], &carry);\n" \
"	r.h = x.h + y.h + (carry ? 1: 0);\n" \
"// #endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE big_type big_sub(const big_type x, const big_type y)\n" \
"{\n" \
"	big_type r;\n" \
"	bool carry = false;\n" \
"	for (sz_t i = 0; i < BIG_LSIZE; ++i) r.l[i] = _subcc(x.l[i], y.l[i], &carry);\n" \
"	r.h = x.h - y.h - (carry ? 1: 0);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE big_type big_neg(const big_type x) { return big_sub(big_zero(), x); }\n" \
"\n" \
"INLINE big_type _big_add_u64(const big_type x, const uint_64 y)\n" \
"{\n" \
"	big_type t; t.l[0] = y; for (sz_t i = 1; i < BIG_LSIZE; ++i) t.l[i] = 0; t.h = 0;\n" \
"	return big_add(x, t);\n" \
"}\n" \
"\n" \
"INLINE big_type big_inc(const big_type x) { return _big_add_u64(x, 1); }\n" \
"\n" \
"INLINE big_type big_mul_u32(const big_type x, const uint_32 y)\n" \
"{\n" \
"	big_type r;\n" \
"	uint_32 carry = 0;\n" \
"	for (sz_t i = 0; i < BIG_LSIZE; ++i)\n" \
"	{\n" \
"		const uint_32 xl = (uint_32)(x.l[i]), xh = (uint_32)(x.l[i] >> 32);\n" \
"		const uint_64 sl = xl * (uint_64)(y) + carry;\n" \
"		carry = (uint_32)(sl >> 32);\n" \
"		const uint_64 sh = xh * (uint_64)(y) + carry;\n" \
"		carry = (uint_32)(sh >> 32);\n" \
"		r.l[i] = (sh << 32) | (uint_32)(sl);\n" \
"	}\n" \
"	r.h = x.h * (HTYPE)(y) + (HTYPE)(carry);\n" \
"	// r.h = (HTYPE)(x.h * (int_64)(y) + carry);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE big_type big_rshift(const big_type x, const int s)	// s != 0\n" \
"{\n" \
"	big_type r;\n" \
"	for (sz_t i = 0; i < BIG_LSIZE - 1; ++i)\n" \
"	{\n" \
"		r.l[i] = (x.l[i] >> s) | (uint_64)(x.l[i + 1] << (64 - s));\n" \
"	}\n" \
"	const int_64 h = x.h;	// 64-bit shift\n" \
"	r.l[BIG_LSIZE - 1] = (x.l[BIG_LSIZE - 1] >> s) | (uint_64)(h << (64 - s));\n" \
"	r.h = (HTYPE)(h >> s);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"#if (PCK_FACTOR == 1)\n" \
"\n" \
"INLINE big_type big_add_pck(const big_type x, const pck_type y) { return _big_add_u64(x, y.r[0]); }\n" \
"\n" \
"#elif (PCK_FACTOR == 2)\n" \
"\n" \
"INLINE big_type big_add_pck(const big_type x, const pck_type y) { return _big_add_u64(x, y.r[0] | ((uint_64)(y.r[1]) << LBASE)); }\n" \
"\n" \
"INLINE bool big_ge_pow_2_s(const big_type x, const int s)	// x >= 2^s\n" \
"{\n" \
"	if (x.h < 0) return false;\n" \
"	// PCK_FACTOR = 2 then s < 64\n" \
"	if (x.h != 0) return true;\n" \
"	for (sz_t i = 1; i < BIG_LSIZE; ++i) if (x.l[i] != 0) return true;\n" \
"	return ((x.l[0] >> s) != 0);\n" \
"}\n" \
"\n" \
"#elif (PCK_FACTOR == 4)\n" \
"\n" \
"INLINE big_type big_add_pck(const big_type x, const pck_type y)\n" \
"{\n" \
"	const uint_64 r01 = y.r[0] | ((uint_64)(y.r[1]) << LBASE);\n" \
"	const uint_64 r23 = y.r[2] | ((uint_64)(y.r[3]) << LBASE);\n" \
"	big_type t;\n" \
"	t.l[0] = (r23 << 2*LBASE) | r01;\n" \
"	t.l[1] = r23 >> (64 - 2*LBASE);\n" \
"	for (sz_t i = 2; i < BIG_LSIZE; ++i) t.l[i] = 0;\n" \
"	t.h = 0;\n" \
"	return big_add(x, t);\n" \
"}\n" \
"\n" \
"INLINE bool big_ge_pow_2_s(const big_type x, const int s)	// x >= 2^s\n" \
"{\n" \
"	if (x.h < 0) return false;\n" \
"	// PCK_FACTOR = 4 then s < 128\n" \
"	if (x.h != 0) return true;\n" \
"	if (s < 64)\n" \
"	{\n" \
"		for (sz_t i = 1; i < BIG_LSIZE; ++i) if (x.l[i] != 0) return true;\n" \
"		return ((x.l[0] >> s) != 0);\n" \
"	}\n" \
"	else\n" \
"	{\n" \
"		for (sz_t i = 2; i < BIG_LSIZE; ++i) if (x.l[i] != 0) return true;\n" \
"		return ((x.l[1] >> (s - 64)) != 0);\n" \
"	}\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"INLINE big_type _big_mul_u64i32(const uint_64 x, const int_32 y)\n" \
"{\n" \
"	const uint_32 x0 = (uint_32)(x), x1 = (uint_32)(x >> 32);\n" \
"\n" \
"	const int_64 sl = x0 * (int_64)(y);\n" \
"	const int_64 sh = x1 * (int_64)(y) + (int_32)(sl >> 32);\n" \
"\n" \
"	big_type r;\n" \
"	r.l[0] = (uint_64)(sh << 32) | (uint_32)(sl);\n" \
"#if (BIG_LSIZE == 1)\n" \
"	r.h = (HTYPE)(sh >> 32);\n" \
"#else\n" \
"	r.l[1] = (uint_64)(sh >> 32);\n" \
"	for (sz_t i = 2; i < BIG_LSIZE; ++i) r.l[i] = (sh < 0) ? (uint_64)(-1) : 0;\n" \
"	r.h = (sh < 0) ? (HTYPE)(-1) : 0;\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE big_type _big_mul_u64i64(const uint_64 x, const int_64 y)\n" \
"{\n" \
"	const uint_32 x0 = (uint_32)(x), x1 = (uint_32)(x >> 32);\n" \
"	const uint_32 y0 = (uint_32)(y); const int_32 y1 = (int_32)(y >> 32);\n" \
"\n" \
"	const uint_64 s00 = x0 * (uint_64)(y0);\n" \
"	const uint_64 s10 = x1 * (uint_64)(y0) + (uint_32)(s00 >> 32);\n" \
"	const int_64 s01 = x0 * (int_64)(y1) + (uint_32)(s10);\n" \
"	const int_64 s11 = x1 * (int_64)(y1) + (uint_32)(s10 >> 32) + (int_32)(s01 >> 32);\n" \
"\n" \
"	big_type r;\n" \
"	r.l[0] = (uint_64)(s01 << 32) | (uint_32)(s00);\n" \
"#if (BIG_LSIZE == 1)\n" \
"	r.h = s11;\n" \
"#else\n" \
"	r.l[1] = (uint_64)(s11);\n" \
"	for (sz_t i = 2; i < BIG_LSIZE; ++i) r.l[i] = (s11 < 0) ? (uint_64)(-1) : 0;\n" \
"	r.h = (s11 < 0) ? (HTYPE)(-1) : 0;\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"#endif	// (PRM_COUNT > 2)\n" \
"\n" \
"#if (PRM_COUNT == 3)\n" \
"\n" \
"INLINE big_type garner3(const ZP r0, const ZP r1, const ZP r2)\n" \
"{\n" \
"	// convert to mixed-radix representation, we must have P3 < P2 < P1 < P0\n" \
"	const uint_32 u2 = r2;\n" \
"	const uint_32 u1 = submulmod(r1, u2, INVP2_P1, PQ1);\n" \
"	const uint_32 u0 = submulmod(submulmod(r0, u2, INVP2_P0, PQ0), u1, INVP1_P0, PQ0);\n" \
"	const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);\n" \
"\n" \
"	const uint_64 u12 = u2 + (uint_64)(P2) * u1;\n" \
"	return _big_add_u64(_big_mul_u64i32(P1 * (uint_64)(P2), i0), u12);\n" \
"}\n" \
"\n" \
"INLINE big_type getK_i(const RNS_K r) { return garner3(r.r[0], r.r[1], r.r[2]); }\n" \
"\n" \
"#elif (PRM_COUNT == 4)\n" \
"\n" \
"INLINE big_type garner4(const ZP r0, const ZP r1, const ZP r2, const ZP r3)\n" \
"{\n" \
"	// convert to mixed-radix representation, we must have P3 < P2 < P1 < P0\n" \
"	const uint_32 u3 = r3;\n" \
"	const uint_32 u2 = submulmod(r2, u3, INVP3_P2, PQ2);\n" \
"	const uint_32 u1 = submulmod(submulmod(r1, u3, INVP3_P1, PQ1), u2, INVP2_P1, PQ1);\n" \
"	const uint_32 u0 = submulmod(submulmod(submulmod(r0, u3, INVP3_P0, PQ0), u2, INVP2_P0, PQ0), u1, INVP1_P0, PQ0);\n" \
"	const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);\n" \
"\n" \
"	const uint_64 u23 = u3 + (uint_64)(P3) * u2;\n" \
"	const int_64 i01 = (int_64)(u1) + (int_64)(P1) * i0;\n" \
"	return _big_add_u64(_big_mul_u64i64(P2 * (uint_64)(P3), i01), u23);\n" \
"}\n" \
"\n" \
"INLINE big_type getK_i(const RNS_K r) { return garner4(r.r[0], r.r[1], r.r[2], r.r[3]); }\n" \
"\n" \
"#elif (PRM_COUNT == 5)\n" \
"\n" \
"INLINE big_type garner5(const ZP r0, const ZP r1, const ZP r2, const ZP r3, const ZP r4)\n" \
"{\n" \
"	// convert to mixed-radix representation, we must have P4 < P3 < P2 < P1 < P0\n" \
"	const uint_32 u4 = r4;\n" \
"	const uint_32 u3 = submulmod(r3, u4, INVP4_P3, PQ3);\n" \
"	const uint_32 u2 = submulmod(submulmod(r2, u4, INVP4_P2, PQ2), u3, INVP3_P2, PQ2);\n" \
"	const uint_32 u1 = submulmod(submulmod(submulmod(r1, u4, INVP4_P1, PQ1), u3, INVP3_P1, PQ1), u2, INVP2_P1, PQ1);\n" \
"	const uint_32 u0 = submulmod(submulmod(submulmod(submulmod(r0, u4, INVP4_P0, PQ0), u3, INVP3_P0, PQ0), u2, INVP2_P0, PQ0), u1, INVP1_P0, PQ0);\n" \
"	const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);\n" \
"\n" \
"	const uint_64 u23 = u3 + (uint_64)(P3) * u2;\n" \
"	const int_64 i01 = (int_64)(u1) + (int_64)(P1) * i0;\n" \
"	const big_type i0123 = _big_add_u64(_big_mul_u64i64(P2 * (uint_64)(P3), i01), u23);\n" \
"	return _big_add_u64(big_mul_u32(i0123, P4), u4);\n" \
"}\n" \
"\n" \
"INLINE big_type getK_i(const RNS_K r) { return garner5(r.r[0], r.r[1], r.r[2], r.r[3], r.r[4]); }\n" \
"\n" \
"#endif\n" \
"\n" \
"INLINE RNS_K encode(const RNS_K x, const ZP r) { return addmodK(mulmodK(x, RBASE_K()), setK_u32(r)); }\n" \
"\n" \
"#if (PCK_FACTOR == 1)\n" \
"\n" \
"INLINE pck_type extract_shift(big_type * const f)\n" \
"{\n" \
"	pck_type r; r.r[0] = (uint_32)((*f).l[0]) & (BASE - 1);\n" \
"	(*f) = big_rshift((*f), LBASE);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE RNS_K extract_encode(big_type * const f)\n" \
"{\n" \
"	const pck_type r = extract_shift(f);\n" \
"	int_32 ri = (int_32)(r.r[0]);\n" \
"	if (ri >= (int_32)(BASE / 2)) { ri -= (int_32)(BASE); (*f) = big_inc((*f)); }	// balanced\n" \
"	return setK_i32(ri);\n" \
"}\n" \
"\n" \
"INLINE RNS_K extract_encode_last(const big_type f)\n" \
"{\n" \
"	const int_32 r1 = (int_32)(f.l[0]);\n" \
"	return setK_i32(r1);\n" \
"}\n" \
"\n" \
"#elif (PCK_FACTOR == 2)\n" \
"\n" \
"INLINE pck_type extract_shift(big_type * const f)\n" \
"{\n" \
"	const uint_64 r01 = (*f).l[0] & (BASE2 - 1);\n" \
"	pck_type r; r.r[0] = (uint_32)(r01) & (BASE - 1); r.r[1] = (uint_32)(r01 >> LBASE);\n" \
"	(*f) = big_rshift((*f), 2 * LBASE);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE RNS_K extract_encode(big_type * const f)\n" \
"{\n" \
"	const pck_type r = extract_shift(f);\n" \
"	int_32 r1 = (int_32)(r.r[1]);\n" \
"	if (r1 >= (int_32)(BASE / 2)) { r1 -= (int_32)(BASE); (*f) = big_inc((*f)); }	// balanced\n" \
"	const RNS_K x = setK_i32(r1);\n" \
"	return encode(x, r.r[0]);\n" \
"}\n" \
"\n" \
"INLINE RNS_K extract_encode_last(const big_type f)\n" \
"{\n" \
"	const int_64 r01 = (int_64)(f.l[0]);\n" \
"	const uint_32 r0 = (uint_32)(r01) & (BASE - 1); const int_32 r1 = (int_32)(r01 >> LBASE);\n" \
"	const RNS_K x = setK_i32(r1);\n" \
"	return encode(x, r0);\n" \
"}\n" \
"\n" \
"#elif (PCK_FACTOR == 4)\n" \
"\n" \
"INLINE pck_type extract_shift(big_type * const f)\n" \
"{\n" \
"	const uint_64 r01 = (*f).l[0] & (BASE2 - 1);\n" \
"	pck_type r; r.r[0] = (uint_32)(r01) & (BASE - 1); r.r[1] = (uint_32)(r01 >> LBASE);\n" \
"	(*f) = big_rshift((*f), 2 * LBASE);\n" \
"	const uint_64 r23 = (*f).l[0] & (BASE2 - 1);\n" \
"	r.r[2] = (uint_32)(r23) & (BASE - 1); r.r[3] = (uint_32)(r23 >> LBASE);\n" \
"	(*f) = big_rshift((*f), 2 * LBASE);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE RNS_K extract_encode(big_type * const f)\n" \
"{\n" \
"	const pck_type r = extract_shift(f);\n" \
"	int_32 r3 = (int_32)(r.r[3]);\n" \
"	if (r3 >= (int_32)(BASE / 2)) { r3 -= (int_32)(BASE); (*f) = big_inc((*f)); }	// balanced\n" \
"	RNS_K x = setK_i32(r3); x = encode(x, r.r[2]); x = encode(x, r.r[1]);\n" \
"	return encode(x, r.r[0]);\n" \
"}\n" \
"\n" \
"INLINE RNS_K extract_encode_last(const big_type f)\n" \
"{\n" \
"	const uint_64 r01 = f.l[0] & (BASE2 - 1);\n" \
"	const uint_32 r0 = (uint_32)(r01) & (BASE - 1), r1 = (uint_32)(r01 >> LBASE);\n" \
"	const big_type f23 = big_rshift(f, 2 * LBASE);\n" \
"	const int_64 r23 = (int_64)(f23.l[0]);\n" \
"	const uint_32 r2 = (uint_32)(r23) & (BASE - 1); const int_32 r3 = (int_32)(r23 >> LBASE);\n" \
"	RNS_K x = setK_i32(r3); x = encode(x, r2); x = encode(x, r1);\n" \
"	return encode(x, r0);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"INLINE void carry_mul_sub_1(const __global ZP * restrict const xi, __global const ZP * restrict const yi,\n" \
"	__global big_type * restrict const carry, __local big_type * const cl, pck_type r[CARRY_LENGTH], const sz_t id, const uint_32 a)\n" \
"{\n" \
"	// assert CARRY_LENGTH = 4\n" \
"	ZP4 x4[PRM_COUNT]; for (sz_t k = 0; k < PRM_COUNT; ++k) x4[k] = load4g(xi, k * Z_OFFSET / 4);\n" \
"	RNS_K y[4]; for (sz_t k = 0; k < PRM_COUNT; ++k) { y[0].r[k] = x4[k].s0; y[1].r[k] = x4[k].s1; y[2].r[k] = x4[k].s2; y[3].r[k] = x4[k].s3; }\n" \
"\n" \
"	big_type f = big_zero();\n" \
"	for (sz_t j = 0; j < CARRY_LENGTH; ++j)\n" \
"	{\n" \
"		// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF\n" \
"		f = big_add(f, big_mul_u32(getK_i(mulmodK(y[j], NORM_K())), a));\n" \
"		if (yi != NULL_PTR) f = big_sub(f, getK_i(loadRNS_K(&yi[j])));	// TODO: K residues are not necessary here because yi[j] <= BASE^PCK_FACTOR\n" \
"		r[j] = extract_shift(&f);\n" \
"	}\n" \
"\n" \
"	const sz_t lid = id % CARRY_WG_SZ;\n" \
"	cl[lid] = f;\n" \
"	if (lid == CARRY_WG_SZ - 1) carry[id / CARRY_WG_SZ] = f;\n" \
"}\n" \
"\n" \
"INLINE void carry_2(__global ZP * restrict const xi, const __local big_type * const cl, const pck_type r[CARRY_LENGTH], const sz_t lid)\n" \
"{\n" \
"	big_type f = (lid != 0) ? cl[lid - 1] : big_zero();\n" \
"	for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)\n" \
"	{\n" \
"		f = big_add_pck(f, r[j]);\n" \
"		const RNS_K u = extract_encode(&f);\n" \
"		if (lid > 0) storeRNS_K(&xi[j], u); else storeRNS_K_pck(&xi[j], r[j]);\n" \
"	}\n" \
"\n" \
"	f = big_add_pck(f, r[CARRY_LENGTH - 1]);\n" \
"	const RNS_K u = extract_encode_last(f);\n" \
"	storeRNS_K(&xi[CARRY_LENGTH - 1], u);\n" \
"	if (lid > 0) storeRNS_K(&xi[CARRY_LENGTH - 1], u); else storeRNS_K_pck(&xi[CARRY_LENGTH - 1], r[CARRY_LENGTH - 1]);\n" \
"}\n" \
"\n" \
"INLINE void carry_3(__global ZP * restrict const xi, __global const big_type * restrict const carry, const int_32 is_nega, const sz_t id)\n" \
"{\n" \
"	// assert CARRY_LENGTH = 4\n" \
"	pck_type y[4]; loadRNS_K_pck4(y, xi);\n" \
"\n" \
"	big_type f = carry[((id == 0) ? (N_SZ / CARRY_LENGTH / CARRY_WG_SZ) : id) - 1];\n" \
"	if ((id == 0) && (is_nega != 0)) f = big_neg(f);\n" \
"	for (sz_t j = 0; j < CARRY_LENGTH - 1; ++j)\n" \
"	{\n" \
"		f = big_add_pck(f, y[j]);\n" \
"		storeRNS_K(&xi[j], extract_encode(&f));\n" \
"	}\n" \
"\n" \
"	f = big_add_pck(f, y[CARRY_LENGTH - 1]);\n" \
"	storeRNS_K(&xi[CARRY_LENGTH - 1], extract_encode_last(f));\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))\n" \
"void carry_mul(__global ZP * restrict const reg, __global big_type * restrict const carry, const uint_32 reg_x, const uint_32 a)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];\n" \
"	__local big_type cl[CARRY_WG_SZ];\n" \
"	pck_type r[CARRY_LENGTH];\n" \
"\n" \
"	carry_mul_sub_1(xi, NULL_PTR, carry, cl, r, id, a);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	carry_2(xi, cl, r, id % CARRY_WG_SZ);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))\n" \
"void carry_sub(__global ZP * restrict const reg, __global big_type * restrict const carry, const uint_32 reg_x, const uint_32 reg_y)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];\n" \
"	__global const ZP * restrict const yi = &reg[reg_y * N_SZ + CARRY_LENGTH * id];\n" \
"	__local big_type cl[CARRY_WG_SZ];\n" \
"	pck_type r[CARRY_LENGTH];\n" \
"\n" \
"	carry_mul_sub_1(xi, yi, carry, cl, r, id, 1);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	carry_2(xi, cl, r, id % CARRY_WG_SZ);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void carry_complete(__global ZP * restrict const reg, __global const big_type * restrict const carry, const uint_32 reg_x, const int_32 is_nega)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * CARRY_WG_SZ * id];\n" \
"\n" \
"	carry_3(xi, carry, is_nega, id);\n" \
"}\n" \
"\n" \
"// --- strict modulo ---\n" \
"\n" \
"INLINE void lower(__global ZP * restrict const x, const int_32 l0)\n" \
"{\n" \
"	const RNS_K hi = loadRNS_K(&x[N_SZ - 1]);	// TODO: K residues are not necessary here because yi[j] <= BASE^PCK_FACTOR\n" \
"	if (big_ge_pow_2_s(getK_i(hi), PCK_FACTOR * LBASE))	// x >= base^n => x[N_SZ - 1] >= BASE^PCK_FACTOR\n" \
"	{\n" \
"		// - R or Q\n" \
"		storeRNS_K(&x[N_SZ - 1], submodK(hi, RBASE_PCK_K()));\n" \
"\n" \
"		const RNS_K lo = loadRNS_K(&x[0]);\n" \
"		storeRNS_K(&x[0], submodK(lo, setK_i32(l0)));\n" \
"	}\n" \
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
"	const RNS_K x0 = addmodK(loadRNS_K(&x[0]), setK_u32(2));\n" \
"	storeRNS_K(&x[0], x0);\n" \
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
"	for (sz_t k = 0; k < PRM_COUNT; ++k) y[k * Z_OFFSET / 4 + id] = x[k * Z_OFFSET / 4 + id];\n" \
"}\n" \
"\n" \
"";
