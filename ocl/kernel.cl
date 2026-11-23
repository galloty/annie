/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#if __OPENCL_VERSION__ >= 120
	#define INLINE	static inline
#else
	#define INLINE
#endif

#if defined(__NV_CL_C_VERSION)
	#define PTX_ASM	1
#endif

#if !defined(N_SZ)
#define LN_SZ			10
#define N_SZ			1024u
#define LBASE			28
#define PRM_COUNT		4
#define PCK_FACTOR		2
#define P0				4194304001u
#define Q0				100663297u
#define RSQ0			232465106u
#define P1				4076863489u
#define Q1				218103809u
#define RSQ1			3444438393u
#define P2				3942645761u
#define Q2				352321537u
#define RSQ2			3810498414u
#define P3				3892314113u
#define Q3				402653185u
#define RSQ3			1703176688u
__constant uint g_norm[4] = { 3670016001u, 3567255553u, 3449815041u, 3405774849u };
__constant uint g_base[4] = { 3174786139u, 3997810352u, 1461973742u, 212897086u };
__constant uint g_baseS[4] = { 3018288268u, 499726294u, 1168408158u, 999690664u };
#define INVP1_P0		1797558821u
#define INVP2_P0		3075822917u
#define INVP2_P1		4076863457u
#define INVP3_P0		3262236431u
#define INVP3_P1		2223743698u
#define INVP3_P2		2628430422u
#define CARRY_LENGTH	4u
#define CARRY_WG_SZ		256u
#define Z_OFFSET		10240u
#define BLK16			8
#define BLK32			8
#define BLK64			8
#define BLK128			4
#define BLK256			2
#define BLK512			1
#define BLK1024 		1
#define BLK2048 		1
#define CHUNK64 		8
#define CHUNK512		4
#endif

typedef uint	sz_t;
typedef uint	uint_32;
typedef int		int_32;
typedef ulong	uint_64;
typedef long	int_64;
typedef uint2	uint2_32;
typedef uint4	uint4_32;

// --- modular arithmetic

typedef uint_32		ZP;
typedef uint2_32	ZP2;
typedef uint4_32	ZP4;

#define	PQ0		(ZP2)(P0, Q0)
#define	PQ1		(ZP2)(P1, Q1)
#define	PQ2		(ZP2)(P2, Q2)
#define	PQ3		(ZP2)(P3, Q3)

__constant ZP2 g_pq[4] = { PQ0, PQ1, PQ2, PQ3 };
__constant ZP g_rsq[4] = { RSQ0, RSQ1, RSQ2, RSQ3 };

INLINE ZP addmod(const ZP lhs, const ZP rhs, const ZP p)
{
	return lhs + rhs - ((lhs >= p - rhs) ? p : 0);
}

INLINE ZP submod(const ZP lhs, const ZP rhs, const ZP p)
{
	return lhs - rhs + ((lhs < rhs) ? p : 0);
}

// Montgomery multiplication by a constant: 1 mul + 2 mul_hi
INLINE ZP mulmod_const(const ZP lhs, const ZP rhs, const ZP rhs_q, const uint_32 p)
{
	const uint_32 mp = mul_hi(lhs * rhs_q, p);
	return submod(mul_hi(lhs, rhs), mp, p);
}

// Montgomery multiplication: 2 mul + 2 mul_hi
INLINE ZP mulmod(const ZP lhs, const ZP rhs, const ZP2 pq) { return mulmod_const(lhs, rhs, rhs * pq.s1, pq.s0); }

INLINE ZP submulmod(const ZP lhs, const ZP rhs, const ZP m, const ZP2 pq)
{
	return mulmod(submod(lhs, rhs, pq.s0), m, pq);
}

INLINE uint_32 set_i32(const int_32 i, const uint_32 p) { return (uint_32)(i) + ((i < 0) ? p : 0); }
INLINE int_32 get_i32(const uint_32 u, const uint_32 p) { return (int_32)(u - ((u >= p / 2) ? p : 0)); }

typedef struct { ZP r[PRM_COUNT]; } RNS_K;

INLINE RNS_K setK_u32(const uint_32 n) { RNS_K r; for (sz_t l = 0; l < PRM_COUNT; ++l) r.r[l] = n; return r; }
INLINE RNS_K setK_i32(const int_32 i) { RNS_K r; for (sz_t l = 0; l < PRM_COUNT; ++l) r.r[l] = set_i32(i, g_pq[l].s0); return r; }

INLINE RNS_K addmodK(const RNS_K lhs, const RNS_K rhs)
{
	RNS_K r; for (sz_t l = 0; l < PRM_COUNT; ++l) r.r[l] = addmod(lhs.r[l], rhs.r[l], g_pq[l].s0); return r;
}

INLINE RNS_K submodK(const RNS_K lhs, const RNS_K rhs)
{
	RNS_K r; for (sz_t l = 0; l < PRM_COUNT; ++l) r.r[l] = submod(lhs.r[l], rhs.r[l], g_pq[l].s0); return r;
}

INLINE RNS_K mulmodK(const RNS_K lhs, const RNS_K rhs)
{
	RNS_K r; for (sz_t l = 0; l < PRM_COUNT; ++l) r.r[l] = mulmod(lhs.r[l], rhs.r[l], g_pq[l]); return r;
}

INLINE RNS_K NORM_K() { RNS_K r; for (sz_t l = 0; l < PRM_COUNT; ++l) r.r[l] = g_norm[l]; return r; }
INLINE RNS_K RBASE_K() { RNS_K r; for (sz_t l = 0; l < PRM_COUNT; ++l) r.r[l] = g_base[l]; return r; }
INLINE RNS_K RBASE_S() { RNS_K r; for (sz_t l = 0; l < PRM_COUNT; ++l) r.r[l] = g_baseS[l]; return r; }

typedef struct { ZP r[PCK_FACTOR]; } RNS_S;

// --- I/O ---

INLINE void loadg(const sz_t n, ZP * const xl, __global const ZP * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }
INLINE void storeg(const sz_t n, __global ZP * restrict const x, const sz_t s, const ZP * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }

INLINE void loadl(const sz_t n, ZP * const xl, __local const ZP * restrict const X, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = X[l * s]; }
INLINE void storel(const sz_t n, __local ZP * restrict const X, const sz_t s, const ZP * const xl) { for (sz_t l = 0; l < n; ++l) X[l * s] = xl[l]; }

INLINE ZP load1g(__global const ZP * restrict const x, const sz_t j) { return x[j]; }
INLINE ZP2 load2g(__global const ZP * restrict const x, const sz_t j) { return ((__global const ZP2 *)x)[j]; }
INLINE ZP4 load4g(__global const ZP * restrict const x, const sz_t j) { return ((__global const ZP4 *)x)[j]; }

INLINE RNS_K loadRNS_K(__global const ZP * restrict const x) { RNS_K r; for (sz_t l = 0; l < PRM_COUNT; ++l) r.r[l] = x[l * Z_OFFSET]; return r; }
INLINE void storeRNS_K(__global ZP * restrict const x, const RNS_K r) { for (sz_t l = 0; l < PRM_COUNT; ++l) x[l * Z_OFFSET] = r.r[l]; }

INLINE RNS_S loadRNS_S(__global const ZP * restrict const x) { RNS_S r; for (sz_t l = 0; l < PCK_FACTOR; ++l) r.r[l] = x[l * Z_OFFSET]; return r; }
INLINE void storeRNS_S(__global ZP * restrict const x, const RNS_S r) { for (sz_t l = 0; l < PCK_FACTOR; ++l) x[l * Z_OFFSET] = r.r[l]; }

// --- transform/macro ---

#define FWD2(x0, x1, w) \
{ \
	const ZP t = mulmod(x1, w, pq); \
	x1 = submod(x0, t, pq.s0); x0 = addmod(x0, t, pq.s0); \
}

#define FWD2_CONST(x0, x1, w, w_q) \
{ \
	const ZP t = mulmod_const(x1, w, w_q, pq.s0); \
	x1 = submod(x0, t, pq.s0); x0 = addmod(x0, t, pq.s0); \
}

#define BCK2(x0, x1, w_i) \
{ \
	const ZP t = submod(x0, x1, pq.s0); x0 = addmod(x0, x1, pq.s0); \
	x1 = mulmod(t, w_i, pq); \
}

#define BCK2_CONST(x0, x1, w_i, w_i_q) \
{ \
	const ZP t = submod(x0, x1, pq.s0); x0 = addmod(x0, x1, pq.s0); \
	x1 = mulmod_const(t, w_i, w_i_q, pq.s0); \
}

#define SQR2(x0, x1, w, w_q) \
{ \
	const ZP t = mulmod_const(mulmod(x1, x1, pq), w, w_q, pq.s0); \
	x1 = mulmod(addmod(x0, x0, pq.s0), x1, pq); \
	x0 = addmod(mulmod(x0, x0, pq), t, pq.s0); \
}

#define SQR2N(x0, x1, w, w_q) \
{ \
	const ZP t = mulmod_const(mulmod(x1, x1, pq), w, w_q, pq.s0); \
	x1 = mulmod(addmod(x0, x0, pq.s0), x1, pq); \
	x0 = submod(mulmod(x0, x0, pq), t, pq.s0); \
}

#define MUL2(x0, x1, xp0, xp1, w, w_q) \
{ \
	const ZP t = mulmod_const(mulmod(x1, xp1, pq), w, w_q, pq.s0); \
	x1 = addmod(mulmod(x0, xp1, pq), mulmod(xp0, x1, pq), pq.s0); \
	x0 = addmod(mulmod(x0, xp0, pq), t, pq.s0); \
}

#define MUL2N(x0, x1, xp0, xp1, w, w_q) \
{ \
	const ZP t = mulmod_const(mulmod(x1, xp1, pq), w, w_q, pq.s0); \
	x1 = addmod(mulmod(x0, xp1, pq), mulmod(xp0, x1, pq), pq.s0); \
	x0 = submod(mulmod(x0, xp0, pq), t, pq.s0); \
}

// --- transform/inline ---

INLINE void _forward8(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2, const ZP4 w4)
{
	const ZP w1_q = w1 * pq.s1;
	FWD2_CONST(x[0], x[4], w1, w1_q); FWD2_CONST(x[1], x[5], w1, w1_q);
	FWD2_CONST(x[2], x[6], w1, w1_q); FWD2_CONST(x[3], x[7], w1, w1_q);
	const ZP w20_q = w2.s0 * pq.s1;
	FWD2_CONST(x[0], x[2], w2.s0, w20_q); FWD2_CONST(x[1], x[3], w2.s0, w20_q);
	const ZP w21_q = w2.s1 * pq.s1;
	FWD2_CONST(x[4], x[6], w2.s1, w21_q); FWD2_CONST(x[5], x[7], w2.s1, w21_q);
	FWD2(x[0], x[1], w4.s0); FWD2(x[2], x[3], w4.s1);
	FWD2(x[4], x[5], w4.s2); FWD2(x[6], x[7], w4.s3);
}

INLINE void _backward8(const ZP2 pq, ZP x[8], const ZP w1_i, const ZP2 w2_i, const ZP4 w4_i)
{
	BCK2(x[0], x[1], w4_i.s0); BCK2(x[2], x[3], w4_i.s1);
	BCK2(x[4], x[5], w4_i.s2); BCK2(x[6], x[7], w4_i.s3);
	const ZP w20_i_q = w2_i.s0 * pq.s1;
	BCK2_CONST(x[0], x[2], w2_i.s0, w20_i_q); BCK2_CONST(x[1], x[3], w2_i.s0, w20_i_q);
	const ZP w21_i_q = w2_i.s1 * pq.s1;
	BCK2_CONST(x[4], x[6], w2_i.s1, w21_i_q); BCK2_CONST(x[5], x[7], w2_i.s1, w21_i_q);
	const ZP w1_i_q = w1_i * pq.s1;
	BCK2_CONST(x[0], x[4], w1_i, w1_i_q); BCK2_CONST(x[1], x[5], w1_i, w1_i_q);
	BCK2_CONST(x[2], x[6], w1_i, w1_i_q); BCK2_CONST(x[3], x[7], w1_i, w1_i_q);
}

INLINE void _forward8_0(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2, const ZP4 w4, const ZP rsq)
{
	for (sz_t i = 0; i < 4; ++i) x[i] = mulmod(x[i], rsq, pq);
	_forward8(pq, x, w1, w2, w4);
}

INLINE void _square2x4(const ZP2 pq, ZP x[8], const ZP2 w2, const ZP2 w2_q)
{
	SQR2(x[0], x[1], w2.s0, w2_q.s0); SQR2N(x[2], x[3], w2.s0, w2_q.s0);
	SQR2(x[4], x[5], w2.s1, w2_q.s1); SQR2N(x[6], x[7], w2.s1, w2_q.s1);
}

INLINE void _square4x2(const ZP2 pq, ZP x[8], const ZP2 w2, const ZP2 w2_i)
{
	const ZP2 w2_q = w2 * pq.s1;
	FWD2_CONST(x[0], x[2], w2.s0, w2_q.s0); FWD2_CONST(x[1], x[3], w2.s0, w2_q.s0);
	FWD2_CONST(x[4], x[6], w2.s1, w2_q.s1); FWD2_CONST(x[5], x[7], w2.s1, w2_q.s1);
	_square2x4(pq, x, w2, w2_q);
	const ZP2 w2_i_q = w2_i * pq.s1;
	BCK2_CONST(x[0], x[2], w2_i.s0, w2_i_q.s0); BCK2_CONST(x[1], x[3], w2_i.s0, w2_i_q.s0);
	BCK2_CONST(x[4], x[6], w2_i.s1, w2_i_q.s1); BCK2_CONST(x[5], x[7], w2_i.s1, w2_i_q.s1);
}

INLINE void _square8(const ZP2 pq, ZP x[8], const ZP w1, const ZP w1_i, const ZP2 w2, const ZP2 w2_i)
{
	const ZP w1_q = w1 * pq.s1;
	FWD2_CONST(x[0], x[4], w1, w1_q); FWD2_CONST(x[1], x[5], w1, w1_q);
	FWD2_CONST(x[2], x[6], w1, w1_q); FWD2_CONST(x[3], x[7], w1, w1_q);
	_square4x2(pq, x, w2, w2_i);
	const ZP w1_i_q = w1_i * pq.s1;
	BCK2_CONST(x[0], x[4], w1_i, w1_i_q); BCK2_CONST(x[1], x[5], w1_i, w1_i_q);
	BCK2_CONST(x[2], x[6], w1_i, w1_i_q); BCK2_CONST(x[3], x[7], w1_i, w1_i_q);
}

INLINE void _mul2x4(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP2 w2, const ZP2 w2_q)
{
	MUL2(x[0], x[1], xp[0], xp[1], w2.s0, w2_q.s0); MUL2N(x[2], x[3], xp[2], xp[3], w2.s0, w2_q.s0);
	MUL2(x[4], x[5], xp[4], xp[5], w2.s1, w2_q.s1); MUL2N(x[6], x[7], xp[6], xp[7], w2.s1, w2_q.s1);
}

INLINE void _mul4x2(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP2 w2, const ZP2 w2_i)
{
	const ZP2 w2_q = w2 * pq.s1;
	FWD2_CONST(x[0], x[2], w2.s0, w2_q.s0); FWD2_CONST(x[1], x[3], w2.s0, w2_q.s0);
	FWD2_CONST(x[4], x[6], w2.s1, w2_q.s1); FWD2_CONST(x[5], x[7], w2.s1, w2_q.s1);
	_mul2x4(pq, x, xp, w2, w2_q);
	const ZP2 w2_i_q = w2_i * pq.s1;
	BCK2_CONST(x[0], x[2], w2_i.s0, w2_i_q.s0); BCK2_CONST(x[1], x[3], w2_i.s0, w2_i_q.s0);
	BCK2_CONST(x[4], x[6], w2_i.s1, w2_i_q.s1); BCK2_CONST(x[5], x[7], w2_i.s1, w2_i_q.s1);
}

INLINE void _mul8(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP w1, const ZP w1_i, const ZP2 w2, const ZP2 w2_i)
{
	const ZP w1_q = w1 * pq.s1;
	FWD2_CONST(x[0], x[4], w1, w1_q); FWD2_CONST(x[1], x[5], w1, w1_q);
	FWD2_CONST(x[2], x[6], w1, w1_q); FWD2_CONST(x[3], x[7], w1, w1_q);
	_mul4x2(pq, x, xp, w2, w2_i);
	const ZP w1_i_q = w1_i * pq.s1;
	BCK2_CONST(x[0], x[4], w1_i, w1_i_q); BCK2_CONST(x[1], x[5], w1_i, w1_i_q);
	BCK2_CONST(x[2], x[6], w1_i, w1_i_q); BCK2_CONST(x[3], x[7], w1_i, w1_i_q);
}

INLINE void _fwd4x2(const ZP2 pq, ZP x[8], const ZP2 w2)
{
	const ZP2 w2_q = w2 * pq.s1;
	FWD2_CONST(x[0], x[2], w2.s0, w2_q.s0); FWD2_CONST(x[1], x[3], w2.s0, w2_q.s0);
	FWD2_CONST(x[4], x[6], w2.s1, w2_q.s1); FWD2_CONST(x[5], x[7], w2.s1, w2_q.s1);
}

INLINE void _fwd8(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2)
{
	const ZP w1_q = w1 * pq.s1;
	FWD2_CONST(x[0], x[4], w1, w1_q); FWD2_CONST(x[1], x[5], w1, w1_q);
	FWD2_CONST(x[2], x[6], w1, w1_q); FWD2_CONST(x[3], x[7], w1, w1_q);
	_fwd4x2(pq, x, w2);
}

// ---

#define LOADG(m)	ZP xl[8]; loadg(8, xl, x, m);
#define LOADPG(m)	ZP xpl[8]; loadg(8, xpl, xp, m);
#define STOREG(m)	storeg(8, x, m, xl);

#define LOADL(m) \
	barrier(CLK_LOCAL_MEM_FENCE); \
	ZP xl[8]; loadl(8, xl, X, m);

#define STOREL(m)	storel(8, X, m, xl);

#define DECLARE_W124() \
	const ZP w1 = load1g(w, sj); const ZP2 w2 = load2g(w, sj); const ZP4 w4 = load4g(w, sj);

#define DECLARE_WI124() \
	__global const ZP * restrict const w_i = &w[N_SZ / 2]; \
	const ZP w1_i = load1g(w_i, sj); const ZP2 w2_i = load2g(w_i, sj); const ZP4 w4_i = load4g(w_i, sj);

#define DECLARE_W124_0() \
	const ZP w1 = load1g(w, 0); const ZP2 w2 = load2g(w, 1); const ZP4 w4 = load4g(w, 1);

#define DECLARE_W2()	const ZP2 w2 = load2g(w, sj);
#define DECLARE_W12()	const ZP w1 = load1g(w, sj); const ZP2 w2 = load2g(w, sj);

#define DECLARE_W2I2() \
	__global const ZP * restrict const w_i = &w[N_SZ / 2]; \
	const ZP2 w2 = load2g(w, sj), w2_i = load2g(w_i, sj);

#define DECLARE_W12I12() \
	DECLARE_W2I2(); \
	const ZP w1 = load1g(w, sj), w1_i = load1g(w_i, sj);

// ---

INLINE void forward8g(const ZP2 pq, const sz_t m, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W124();
	LOADG(m);
	_forward8(pq, xl, w1, w2, w4);
	STOREG(m);
}

INLINE void forward8i(const ZP2 pq, const sz_t ml, __local ZP * restrict const X, const sz_t mg,
	__global ZP * restrict const x, __global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W124();
	LOADG(mg);
	_forward8(pq, xl, w1, w2, w4);
	STOREL(ml);
}

INLINE void forward8l(const ZP2 pq, const sz_t m, __local ZP * restrict const X,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W124();
	LOADL(m);
	_forward8(pq, xl, w1, w2, w4);
	STOREL(m);
}

INLINE void forward8o(const ZP2 pq, const sz_t mg, __global ZP * restrict const x, const sz_t ml,
	__local ZP * restrict const X, __global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W124();
	LOADL(ml);
	_forward8(pq, xl, w1, w2, w4);
	STOREG(mg);
}

INLINE void backward8g(const ZP2 pq, const sz_t m, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_WI124();
	LOADG(m);
	_backward8(pq, xl, w1_i, w2_i, w4_i);
	STOREG(m);
}

INLINE void backward8i(const ZP2 pq, const sz_t ml, __local ZP * restrict const X, const sz_t mg,
	__global ZP * restrict const x, __global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_WI124();
	LOADG(mg);
	_backward8(pq, xl, w1_i, w2_i, w4_i);
	STOREL(ml);
}

INLINE void backward8l(const ZP2 pq, const sz_t m, __local ZP * restrict const X,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_WI124();
	LOADL(m);
	_backward8(pq, xl, w1_i, w2_i, w4_i);
	STOREL(m);
}

INLINE void backward8o(const ZP2 pq, const sz_t mg, __global ZP * restrict const x, const sz_t ml,
	__local ZP * restrict const X, __global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_WI124();
	LOADL(ml);
	_backward8(pq, xl, w1_i, w2_i, w4_i);
	STOREG(mg);
}

INLINE void forward8_0g(const ZP2 pq, const sz_t m, __global ZP * restrict const x,
	__global const ZP * restrict const w, const ZP rsq)
{
	// Montgomery form of MF of I to convert input into MF
	DECLARE_W124_0();
	LOADG(m);
	_forward8_0(pq, xl, w1, w2, w4, rsq);
	STOREG(m);
}

INLINE void forward8_0i(const ZP2 pq,const sz_t ml, __local ZP * restrict const X, const sz_t mg,
	__global ZP * restrict const x, __global const ZP * restrict const w, const ZP rsq)
{
	DECLARE_W124_0();
	LOADG(mg);
	_forward8_0(pq, xl, w1, w2, w4, rsq);
	STOREL(ml);
}

INLINE void square2x4g(const ZP2 pq, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2();
	LOADG(1);
	_square2x4(pq, xl, w2, w2 * pq.s1);
	STOREG(1);
}

INLINE void square2x4l(const ZP2 pq, __local ZP * restrict const X,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2();
	LOADL(1);
	_square2x4(pq, xl, w2, w2 * pq.s1);
	STOREL(1);
}

INLINE void square4x2g(const ZP2 pq, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2I2();
	LOADG(1);
	_square4x2(pq, xl, w2, w2_i);
	STOREG(1);
}

INLINE void square4x2l(const ZP2 pq, __local ZP * restrict const X,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2I2();
	LOADL(1);
	_square4x2(pq, xl, w2, w2_i);
	STOREL(1);
}

INLINE void square8g(const ZP2 pq, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W12I12();
	LOADG(1);
	_square8(pq, xl, w1, w1_i, w2, w2_i);
	STOREG(1);
}

INLINE void square8l(const ZP2 pq, __local ZP * restrict const X,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W12I12();
	LOADL(1);
	_square8(pq, xl, w1, w1_i, w2, w2_i);
	STOREL(1);
}

INLINE void mul2x4g(const ZP2 pq, __global ZP * restrict const x, __global const ZP * restrict const xp,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2();
	LOADG(1);
	LOADPG(1);
	_mul2x4(pq, xl, xpl, w2, w2 * pq.s1);
	STOREG(1);
}

INLINE void mul2x4l(const ZP2 pq, __local ZP * restrict const X, __global const ZP * restrict const xp,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2();
	LOADL(1);
	LOADPG(1);
	_mul2x4(pq, xl, xpl, w2, w2 * pq.s1);
	STOREL(1);
}

INLINE void mul4x2g(const ZP2 pq, __global ZP * restrict const x, __global const ZP * restrict const xp,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2I2();
	LOADG(1);
	LOADPG(1);
	_mul4x2(pq, xl, xpl, w2, w2_i);
	STOREG(1);
}

INLINE void mul4x2l(const ZP2 pq, __local ZP * restrict const X, __global const ZP * restrict const xp,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2I2();
	LOADL(1);
	LOADPG(1);
	_mul4x2(pq, xl, xpl, w2, w2_i);
	STOREL(1);
}

INLINE void mul8g(const ZP2 pq, __global ZP * restrict const x, __global const ZP * restrict const xp,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W12I12();
	LOADG(1);
	LOADPG(1);
	_mul8(pq, xl, xpl, w1, w1_i, w2, w2_i);
	STOREG(1);
}

INLINE void mul8l(const ZP2 pq, __local ZP * restrict const X, __global const ZP * restrict const xp,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W12I12();
	LOADL(1);
	LOADPG(1);
	_mul8(pq, xl, xpl, w1, w1_i, w2, w2_i);
	STOREL(1);
}

INLINE void fwd4x2g(const ZP2 pq, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2();
	LOADG(1);
	_fwd4x2(pq, xl, w2);
	STOREG(1);
}

INLINE void fwd4x2o(const ZP2 pq, __global ZP * restrict const x, __local ZP * restrict const X,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2();
	LOADL(1);
	_fwd4x2(pq, xl, w2);
	STOREG(1);
}

INLINE void fwd8g(const ZP2 pq, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W12();
	LOADG(1);
	_fwd8(pq, xl, w1, w2);
	STOREG(1);
}

INLINE void fwd8o(const ZP2 pq, __global ZP * restrict const x, __local ZP * restrict const X,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W12();
	LOADL(1);
	_fwd8(pq, xl, w1, w2);
	STOREG(1);
}

// --- transform/macro ---

#define DECLARE_VAR_REG() \
	const sz_t gid = (sz_t)get_global_id(0), lid = gid / (N_SZ / 8), id = gid % (N_SZ / 8); \
	const ZP2 pq = g_pq[lid]; \
	__global ZP * restrict const reg = &regm[lid * Z_OFFSET]; \
	__global ZP * restrict const x = &reg[reg_x * N_SZ]; \
	__global const ZP * restrict const w = &reg[reg_w * N_SZ];

#define DECLARE_VARP_REG()	__global const ZP * restrict const xp = &reg[reg_xp * N_SZ];

// --- transform without local mem ---

#if (LN_SZ >= 21)
__kernel	// TODO ZP4
void forward8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const int_32 lm, const uint_32 s)
{
	DECLARE_VAR_REG();
	const sz_t m = 1u << lm, sj = s + (id >> lm), k = 7 * (id & ~(m - 1)) + id;
	forward8g(pq, m, &x[k], w, sj);
}

__kernel
void backward8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const int_32 lm, const uint_32 s)
{
	DECLARE_VAR_REG();
	const sz_t m = 1u << lm, sj = s + (id >> lm), k = 7 * (id & ~(m - 1)) + id;
	backward8g(pq, m, &x[k], w, sj);
}
#endif

#if (LN_SZ <= 12)
__kernel
void forward8_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t m = N_SZ / 8, k = id;
	forward8_0g(pq, m, &x[k], w, g_rsq[lid]);
}

__kernel
void backward8_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t m = N_SZ / 8, sj = 1, k = id;
	backward8g(pq, m, &x[k], w, sj);
}
#endif

// ---

#if (LN_SZ == 4)
__kernel
void square2x4(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	square2x4g(pq, &x[k], w, sj);
}
#endif

#if (LN_SZ == 5)
__kernel
void square4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	square4x2g(pq, &x[k], w, sj);
}
#endif

#if (LN_SZ == 6)
__kernel
void square8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	square8g(pq, &x[k], w, sj);
}
#endif

// ---

#if (LN_SZ == 4)
__kernel
void mul2x4(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	mul2x4g(pq, &x[k], &xp[k], w, sj);
}
#endif

#if (LN_SZ == 5)
__kernel
void mul4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	mul4x2g(pq, &x[k], &xp[k], w, sj);
}
#endif

#if (LN_SZ == 6)
__kernel
void mul8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	mul8g(pq, &x[k], &xp[k], w, sj);
}
#endif

// ---

#if (LN_SZ == 5)
__kernel
void fwd4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	fwd4x2g(pq, &x[k], w, sj);
}
#endif

#if (LN_SZ == 6)
__kernel
void fwd8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	fwd8g(pq, &x[k], w, sj);
}
#endif

#if (LN_SZ == 7)
__kernel
void fwd16(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k2 = 7 * (id & ~(2 - 1)) + id;
	forward8g(pq, 2, &x[k2], w, sj / 2);
}
#endif

// --- transform using local mem ---

#define DECLARE_VAR_FB(N, CHUNK_N) \
	__local ZP X[N * CHUNK_N]; \
	DECLARE_VAR_REG(); \
	const size_t chunk_id = id % CHUNK_N, local_id = (id / CHUNK_N) % (N / 8), block_id = id & ~(N / 8 * CHUNK_N - 1); \
	__global ZP * restrict const xt = &x[block_id / (N / 8) + chunk_id]; \
	__local ZP * const Xt = &X[N * chunk_id];

#if (LN_SZ > 12) && (LN_SZ <= 16)

#define DECLARE_VAR_FB64() \
	const sz_t kl0 = local_id, k0 = kl0 * (N_SZ / 64), ml0 = 8, m0 = ml0 * (N_SZ / 64); \
	const sz_t kl8 = local_id * 8, k8 = kl8 * (N_SZ / 64), ml8 = 1, m8 = ml8 * (N_SZ / 64); \
	const sz_t sj8 = 8 + local_id;

__kernel __attribute__((reqd_work_group_size(64 / 8 * CHUNK64, 1, 1)))
void forward64_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_FB(64, CHUNK64);
	DECLARE_VAR_FB64();

	forward8_0i(pq, ml0, &Xt[kl0], m0, &xt[k0], w, g_rsq[lid]);
	forward8o(pq, m8, &xt[k8], ml8, &Xt[kl8], w, sj8);
}

__kernel __attribute__((reqd_work_group_size(64 / 8 * CHUNK64, 1, 1)))
void backward64_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_FB(64, CHUNK64);
	DECLARE_VAR_FB64();

	backward8i(pq, ml8, &Xt[kl8], m8, &xt[k8], w, sj8);
	backward8o(pq, m0, &xt[k0], ml0, &Xt[kl0], w, 1);
}
#endif

#if (LN_SZ > 16)

#define DECLARE_VAR_FB512() \
	const sz_t kl0 = local_id, k0 = kl0 * (N_SZ / 512), ml0 = 64, m0 = ml0 * (N_SZ / 512); \
	const sz_t kl8 = (local_id % 8) + (local_id / 8) * 64, ml8 = 8; \
	const sz_t kl64 = local_id * 8, k64 = kl64 * (N_SZ / 512), ml64 = 1, m64 = ml64 * (N_SZ / 512); \
	const sz_t sj64 = 64 + local_id, sj8 = 8 + (local_id / 8) % 8;

__kernel __attribute__((reqd_work_group_size(512 / 8 * CHUNK512, 1, 1)))
void forward512_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_FB(512, CHUNK512);
	DECLARE_VAR_FB512();

	forward8_0i(pq, ml0, &Xt[kl0], m0, &xt[k0], w, g_rsq[lid]);
	forward8l(pq, ml8, &Xt[kl8], w, sj8);
	forward8o(pq, m64, &xt[k64], ml64, &Xt[kl64], w, sj64);
}

__kernel __attribute__((reqd_work_group_size(512 / 8 * CHUNK512, 1, 1)))
void backward512_0(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_FB(512, CHUNK512);
	DECLARE_VAR_FB512();

	backward8i(pq, ml64, &Xt[kl64], m64, &xt[k64], w, sj64);
	backward8l(pq, ml8, &Xt[kl8], w, sj8);
	backward8o(pq, m0, &xt[k0], ml0, &Xt[kl0], w, 1);
}
#endif

// ---

#define DECLARE_VAR_SQRMUL(N, BLK_N) \
	__local ZP X[N * BLK_N]; \
	DECLARE_VAR_REG(); \
	const sz_t block_id = (id / (N / 8)) % BLK_N; \
	__local ZP * const Xb = &X[N * block_id]; \
	const sz_t s = N_SZ / 8, sj = s + id, k = 8 * id;

#if (LN_SZ == 7)

#define DECLARE_VAR_SQRMUL16() \
	DECLARE_VAR_SQRMUL(16, BLK16); \
	const sz_t k2 = 7 * (id & ~(2 - 1)) + id;

__kernel __attribute__((reqd_work_group_size(16 / 8 * BLK16, 1, 1)))
void square16(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL16();

	forward8i(pq, 2, &Xb[k2 % 16], 2, &x[k2], w, sj / 2);
	square2x4l(pq, &Xb[k % 16], w, sj);
	backward8o(pq, 2, &x[k2], 2, &Xb[k2 % 16], w, sj / 2);
}

__kernel __attribute__((reqd_work_group_size(16 / 8 * BLK16, 1, 1)))
void mul16(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_SQRMUL16();
	DECLARE_VARP_REG();

	forward8i(pq, 2, &Xb[k2 % 16], 2, &x[k2], w, sj / 2);
	mul2x4l(pq, &Xb[k % 16], &xp[k], w, sj);
	backward8o(pq, 2, &x[k2], 2, &Xb[k2 % 16], w, sj / 2);
}
#endif

#if (LN_SZ == 8)

#define DECLARE_VAR_SQRMUL32() \
	DECLARE_VAR_SQRMUL(32, BLK32); \
	const sz_t k4 = 7 * (id & ~(4 - 1)) + id;

__kernel __attribute__((reqd_work_group_size(32 / 8 * BLK32, 1, 1)))
void square32(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL32();

	forward8i(pq, 4, &Xb[k4 % 32], 4, &x[k4], w, sj / 4);
	square4x2l(pq, &Xb[k % 32], w, sj);
	backward8o(pq, 4, &x[k4], 4, &Xb[k4 % 32], w, sj / 4);
}

__kernel __attribute__((reqd_work_group_size(32 / 8 * BLK32, 1, 1)))
void mul32(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_SQRMUL32();
	DECLARE_VARP_REG();

	forward8i(pq, 4, &Xb[k4 % 32], 4, &x[k4], w, sj / 4);
	mul4x2l(pq, &Xb[k % 32], &xp[k], w, sj);
	backward8o(pq, 4, &x[k4], 4, &Xb[k4 % 32], w, sj / 4);
}

__kernel __attribute__((reqd_work_group_size(32 / 8 * BLK32, 1, 1)))
void fwd32(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL32();

	forward8i(pq, 4, &Xb[k4 % 32], 4, &x[k4], w, sj / 4);
	fwd4x2o(pq, &x[k], &Xb[k % 32], w, sj);
}
#endif

#if (LN_SZ == 9)

#define DECLARE_VAR_SQRMUL64() \
	DECLARE_VAR_SQRMUL(64, BLK64); \
	const sz_t k8 = 7 * (id & ~(8 - 1)) + id;

__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))
void square64(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL64();

	forward8i(pq, 8, &Xb[k8 % 64], 8, &x[k8], w, sj / 8);
	square8l(pq, &Xb[k % 64], w, sj);
	backward8o(pq, 8, &x[k8], 8, &Xb[k8 % 64], w, sj / 8);
}

__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))
void mul64(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_SQRMUL64();
	DECLARE_VARP_REG();

	forward8i(pq, 8, &Xb[k8 % 64], 8, &x[k8], w, sj / 8);
	mul8l(pq, &Xb[k % 64], &xp[k], w, sj);
	backward8o(pq, 8, &x[k8], 8, &Xb[k8 % 64], w, sj / 8);
}

__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))
void fwd64(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL64();

	forward8i(pq, 8, &Xb[k8 % 64], 8, &x[k8], w, sj / 8);
	fwd8o(pq, &x[k], &Xb[k % 64], w, sj);
}
#endif

#if (LN_SZ == 10) || (LN_SZ == 13)

#define DECLARE_VAR_SQRMUL128() \
	DECLARE_VAR_SQRMUL(128, BLK128); \
	const sz_t k2 = 7 * (id & ~(2 - 1)) + id, k16 = 7 * (id & ~(16 - 1)) + id;

__kernel __attribute__((reqd_work_group_size(128 / 8 * BLK128, 1, 1)))
void square128(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL128();

	forward8i(pq, 16, &Xb[k16 % 128], 16, &x[k16], w, sj / 16);
	forward8l(pq, 2, &Xb[k2 % 128], w, sj / 2);
	square2x4l(pq, &Xb[k % 128], w, sj);
	backward8l(pq, 2, &Xb[k2 % 128], w, sj / 2);
	backward8o(pq, 16, &x[k16], 16, &Xb[k16 % 128], w, sj / 16);
}

__kernel __attribute__((reqd_work_group_size(128 / 8 * BLK128, 1, 1)))
void mul128(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_SQRMUL128();
	DECLARE_VARP_REG();

	forward8i(pq, 16, &Xb[k16 % 128], 16, &x[k16], w, sj / 16);
	forward8l(pq, 2, &Xb[k2 % 128], w, sj / 2);
	mul2x4l(pq, &Xb[k % 128], &xp[k], w, sj);
	backward8l(pq, 2, &Xb[k2 % 128], w, sj / 2);
	backward8o(pq, 16, &x[k16], 16, &Xb[k16 % 128], w, sj / 16);
}

__kernel __attribute__((reqd_work_group_size(128 / 8 * BLK128, 1, 1)))
void fwd128(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL128();

	forward8i(pq, 16, &Xb[k16 % 128], 16, &x[k16], w, sj / 16);
	forward8o(pq, 2, &x[k2], 2, &Xb[k2 % 128], w, sj / 2);
}
#endif

#if (LN_SZ == 11) || (LN_SZ == 14) || (LN_SZ == 17)

#define DECLARE_VAR_SQRMUL256() \
	DECLARE_VAR_SQRMUL(256, BLK256); \
	const sz_t k4 = 7 * (id & ~(4 - 1)) + id, k32 = 7 * (id & ~(32 - 1)) + id;

__kernel __attribute__((reqd_work_group_size(256 / 8 * BLK256, 1, 1)))
void square256(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL256();

	forward8i(pq, 32, &Xb[k32 % 256], 32, &x[k32], w, sj / 32);
	forward8l(pq, 4, &Xb[k4 % 256], w, sj / 4);
	square4x2l(pq, &Xb[k % 256], w, sj);
	backward8l(pq, 4, &Xb[k4 % 256], w, sj / 4);
	backward8o(pq, 32, &x[k32], 32, &Xb[k32 % 256], w, sj / 32);
}

__kernel __attribute__((reqd_work_group_size(256 / 8 * BLK256, 1, 1)))
void mul256(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_SQRMUL256();
	DECLARE_VARP_REG();

	forward8i(pq, 32, &Xb[k32 % 256], 32, &x[k32], w, sj / 32);
	forward8l(pq, 4, &Xb[k4 % 256], w, sj / 4);
	mul4x2l(pq, &Xb[k % 256], &xp[k], w, sj);
	backward8l(pq, 4, &Xb[k4 % 256], w, sj / 4);
	backward8o(pq, 32, &x[k32], 32, &Xb[k32 % 256], w, sj / 32);
}

__kernel __attribute__((reqd_work_group_size(256 / 8 * BLK256, 1, 1)))
void fwd256(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL256();

	forward8i(pq, 32, &Xb[k32 % 256], 32, &x[k32], w, sj / 32);
	forward8l(pq, 4, &Xb[k4 % 256], w, sj / 4);
	fwd4x2o(pq, &x[k], &Xb[k % 256], w, sj);
}
#endif

#if (LN_SZ == 12) || (LN_SZ == 15) || (LN_SZ == 18) || (LN_SZ == 21)

#define DECLARE_VAR_SQRMUL512() \
	DECLARE_VAR_SQRMUL(512, BLK512); \
	const sz_t k8 = 7 * (id & ~(8 - 1)) + id, k64 = 7 * (id & ~(64 - 1)) + id;

__kernel __attribute__((reqd_work_group_size(512 / 8 * BLK512, 1, 1)))
void square512(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL512();

	forward8i(pq, 64, &Xb[k64 % 512], 64, &x[k64], w, sj / 64);
	forward8l(pq, 8, &Xb[k8 % 512], w, sj / 8);
	square8l(pq, &Xb[k % 512], w, sj);
	backward8l(pq, 8, &Xb[k8 % 512], w, sj / 8);
	backward8o(pq, 64, &x[k64], 64, &Xb[k64 % 512], w, sj / 64);
}

__kernel __attribute__((reqd_work_group_size(512 / 8 * BLK512, 1, 1)))
void mul512(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_SQRMUL512();
	DECLARE_VARP_REG();

	forward8i(pq, 64, &Xb[k64 % 512], 64, &x[k64], w, sj / 64);
	forward8l(pq, 8, &Xb[k8 % 512], w, sj / 8);
	mul8l(pq, &Xb[k % 512], &xp[k], w, sj);
	backward8l(pq, 8, &Xb[k8 % 512], w, sj / 8);
	backward8o(pq, 64, &x[k64], 64, &Xb[k64 % 512], w, sj / 64);
}

__kernel __attribute__((reqd_work_group_size(512 / 8 * BLK512, 1, 1)))
void fwd512(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL512();

	forward8i(pq, 64, &Xb[k64 % 512], 64, &x[k64], w, sj / 64);
	forward8l(pq, 8, &Xb[k8 % 512], w, sj / 8);
	fwd8o(pq, &x[k], &Xb[k % 512], w, sj);
}
#endif

#if (LN_SZ == 16) || (LN_SZ == 19) || (LN_SZ == 22)

#define DECLARE_VAR_SQRMUL1024() \
	DECLARE_VAR_SQRMUL(1024, BLK1024); \
	const sz_t k2 = 7 * (id & ~(2 - 1)) + id, k16 = 7 * (id & ~(16 - 1)) + id, k128 = 7 * (id & ~(128 - 1)) + id;

__kernel __attribute__((reqd_work_group_size(1024 / 8 * BLK1024, 1, 1)))
void square1024(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL1024();

	forward8i(pq, 128, &Xb[k128 % 1024], 128, &x[k128], w, sj / 128);
	forward8l(pq, 16, &Xb[k16 % 1024], w, sj / 16);
	forward8l(pq, 2, &Xb[k2 % 1024], w, sj / 2);
	square2x4l(pq, &Xb[k % 1024], w, sj);
	backward8l(pq, 2, &Xb[k2 % 1024], w, sj / 2);
	backward8l(pq, 16, &Xb[k16 % 1024], w, sj / 16);
	backward8o(pq, 128, &x[k128], 128, &Xb[k128 % 1024], w, sj / 128);
}

__kernel __attribute__((reqd_work_group_size(1024 / 8 * BLK1024, 1, 1)))
void mul1024(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_SQRMUL1024();
	DECLARE_VARP_REG();

	forward8i(pq, 128, &Xb[k128 % 1024], 128, &x[k128], w, sj / 128);
	forward8l(pq, 16, &Xb[k16 % 1024], w, sj / 16);
	forward8l(pq, 2, &Xb[k2 % 1024], w, sj / 2);
	mul2x4l(pq, &Xb[k % 1024], &xp[k], w, sj);
	backward8l(pq, 2, &Xb[k2 % 1024], w, sj / 2);
	backward8l(pq, 16, &Xb[k16 % 1024], w, sj / 16);
	backward8o(pq, 128, &x[k128], 128, &Xb[k128 % 1024], w, sj / 128);
}

__kernel __attribute__((reqd_work_group_size(1024 / 8 * BLK1024, 1, 1)))
void fwd1024(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL1024();

	forward8i(pq, 128, &Xb[k128 % 1024], 128, &x[k128], w, sj / 128);
	forward8l(pq, 16, &Xb[k16 % 1024], w, sj / 16);
	forward8o(pq, 2, &x[k2], 2, &Xb[k2 % 1024], w, sj / 2);
}
#endif

#if (LN_SZ == 20) || (LN_SZ == 23)

#define DECLARE_VAR_SQRMUL2048() \
	DECLARE_VAR_SQRMUL(2048, BLK2048); \
	const sz_t k4 = 7 * (id & ~(4 - 1)) + id, k32 = 7 * (id & ~(32 - 1)) + id, k256 = 7 * (id & ~(256 - 1)) + id;

__kernel __attribute__((reqd_work_group_size(2048 / 8 * BLK2048, 1, 1)))
void square2048(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL2048();

	forward8i(pq, 256, &Xb[k256 % 2048], 256, &x[k256], w, sj / 256);
	forward8l(pq, 32, &Xb[k32 % 2048], w, sj / 32);
	forward8l(pq, 4, &Xb[k4 % 2048], w, sj / 4);
	square4x2l(pq, &Xb[k % 2048], w, sj);
	backward8l(pq, 4, &Xb[k4 % 2048], w, sj / 4);
	backward8l(pq, 32, &Xb[k32 % 2048], w, sj / 32);
	backward8o(pq, 256, &x[k256], 256, &Xb[k256 % 2048], w, sj / 256);
}

__kernel __attribute__((reqd_work_group_size(2048 / 8 * BLK2048, 1, 1)))
void mul2048(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_SQRMUL2048();
	DECLARE_VARP_REG();

	forward8i(pq, 256, &Xb[k256 % 2048], 256, &x[k256], w, sj / 256);
	forward8l(pq, 32, &Xb[k32 % 2048], w, sj / 32);
	forward8l(pq, 4, &Xb[k4 % 2048], w, sj / 4);
	mul4x2l(pq, &Xb[k % 2048], &xp[k], w, sj);
	backward8l(pq, 4, &Xb[k4 % 2048], w, sj / 4);
	backward8l(pq, 32, &Xb[k32 % 2048], w, sj / 32);
	backward8o(pq, 256, &x[k256], 256, &Xb[k256 % 2048], w, sj / 256);
}

__kernel __attribute__((reqd_work_group_size(2048 / 8 * BLK2048, 1, 1)))
void fwd2048(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL2048();

	forward8i(pq, 256, &Xb[k256 % 2048], 256, &x[k256], w, sj / 256);
	forward8l(pq, 32, &Xb[k32 % 2048], w, sj / 32);
	forward8l(pq, 4, &Xb[k4 % 2048], w, sj / 4);
	fwd4x2o(pq, &x[k], &Xb[k % 2048], w, sj);
}
#endif

// --- carry ---

#define BASE		(1u << LBASE)
#define NULL_PTR	((__global const ZP *)0)

// Chinese remainder theorem

INLINE int_32 garner1(const ZP r0)
{
	return get_i32(r0, P0);
}

INLINE int_64 garner2(const ZP r0, const ZP r1)
{
	const uint_32 u1 = r1;
	const uint_32 u0 = submulmod(r0, u1, INVP1_P0, PQ0);	// P2 < P1
	const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);

	return (int_64)(u1) + (int_64)(P1) * i0;
}

#if (PRM_COUNT == 2)	// PCK_FACTOR = 1

typedef int_64 big_type;	// 64-bit, size = PRM_COUNT * 32 > LN + 2 * LBASE
typedef uint_32 pck_type;	// vector type of size PCK_FACTOR
typedef int_32 small_type;	// 32-bit, size = PCK_FACTOR * 32 > PCK_FACTOR * LBASE

INLINE big_type big_zero() { return 0; }
INLINE big_type big_neg(const big_type x) { return -x; }
INLINE big_type big_add(const big_type x, const big_type y) { return x + y; }
INLINE big_type big_inc(const big_type x) { return x + 1; }
INLINE big_type big_add_pck(const big_type x, const pck_type r) { return x + (big_type)(r); }
INLINE big_type big_add_small(const big_type x, const small_type y) { return x + y; }
INLINE big_type big_sub_small(const big_type x, const small_type y) { return x - y; }
INLINE big_type big_mul_u32(const big_type x, const uint_32 y) { return x * (big_type)(y); }

INLINE small_type getS_i(const RNS_S r) { return garner1(r.r[0]); }
INLINE big_type getK_i(const RNS_K r) { return garner2(r.r[0], r.r[1]); }

INLINE pck_type extract_r(big_type * const f)
{
	const pck_type r = (uint_32)(*f) & (BASE - 1);
	(*f) >>= LBASE;
	return r;
}

INLINE RNS_K extract_encode(big_type * const f)
{
	const pck_type re = extract_r(f);
	int_32 ri = (int_32)(re);
	if (ri >= (int_32)(BASE / 2)) { ri -= (int_32)(BASE); (*f) = big_inc((*f)); }	// balanced
	return setK_i32(ri);
}

INLINE RNS_K extract_encode_last(const big_type f)
{
	const int_32 r1 = (int_32)(f);
	return setK_i32(r1);
}

#elif (PRM_COUNT == 3)	// PCK_FACTOR = 2

#define BASE2		((uint_64)(1) << (2 * LBASE))

typedef struct { uint_64 s0; int_32 s1; } big_type;	// 96-bit, size = PRM_COUNT * 32 > LN + 2 * LBASE
typedef uint_64 pck_type;	// vector type of size PCK_FACTOR
typedef int_64 small_type;	// 64-bit, size = PCK_FACTOR * 32 > PCK_FACTOR * LBASE

INLINE big_type big_zero() { big_type r; r.s0 = 0; r.s1 = 0; return r; }

INLINE big_type big_neg(const big_type x)
{
	big_type r;
	r.s0 = -x.s0;
	r.s1 = -x.s1 - ((x.s0 != 0) ? 1 : 0);
	return r;
}

INLINE big_type big_add(const big_type x, const big_type y)
{
	big_type r;
// #if defined(PTX_ASM)
// 	asm volatile ("add.cc.u64 %0, %1, %2;" : "=r" (r.s0) : "r" (x.s0), "r" (y.s0));
// 	asm volatile ("addc.s64 %0, %1, %2;" : "=r" (r.s1) : "r" (x.s1), "r" (y.s1));
// #else
	r.s0 = x.s0 + y.s0;
	r.s1 = x.s1 + (int_32)y.s1 + ((r.s0 < x.s0) ? 1 : 0);
// #endif
	return r;
}

INLINE big_type _big_add_u64(const big_type x, const uint_64 y)
{
	big_type t; t.s0 = y; t.s1 = 0;
	return big_add(x, t);
}

INLINE big_type _big_add_s64(const big_type x, const int_64 y)
{
	big_type t; t.s0 = (uint_64)(y); t.s1 = (y < 0) ? -1 : 0;
	return big_add(x, t);
}

INLINE big_type big_inc(const big_type x) { return _big_add_u64(x, 1); }
INLINE big_type big_add_pck(const big_type x, const pck_type r) { return _big_add_u64(x, r); }
INLINE big_type big_add_small(const big_type x, const small_type y) { return _big_add_s64(x, y); }
INLINE big_type big_sub_small(const big_type x, const small_type y) { return _big_add_s64(x, -y); }

INLINE big_type big_mul_u32(const big_type x, const uint_32 y)
{
	const big_type xp = (x.s1 < 0) ? big_neg(x) : x;
	big_type r;
	r.s0 = xp.s0 * y;
	r.s1 = (int_32)((uint_32)(xp.s1) * y + mul_hi(xp.s0, (uint_64)(y)));
	return (x.s1 < 0) ? big_neg(r) : r;
}

INLINE big_type _big_rshift(const big_type x, const int s)	// s != 0
{
	big_type r;
	const int_64 s1 = x.s1;	// 64-bit shift
	r.s0 = (x.s0 >> s) | (uint_64)(s1 << (64 - s));
	r.s1 = (int_32)(s1 >> s);
	return r;
}

INLINE big_type _big_mul_u64i32(const uint_64 x, const int_32 y)
{
	const uint_64 yp = abs(y);
	const uint_64 l = (uint_32)(x) * (uint_64)(yp);
	const uint_64 h = (uint_32)(x >> 32) * (uint_64)(yp) + (l >> 32);
	big_type r;
	r.s0 = (uint_32)(l) | (h << 32);
	r.s1 = (int_32)(h >> 32);
	return (y < 0) ? big_neg(r) : r;
}

INLINE big_type garner3(const ZP r0, const ZP r1, const ZP r2)
{
	// convert to mixed-radix representation, we must have P3 < P2 < P1 < P0
	const uint_32 u2 = r2;
	const uint_32 u1 = submulmod(r1, u2, INVP2_P1, PQ1);
	const uint_32 u0 = submulmod(submulmod(r0, u2, INVP2_P0, PQ0), u1, INVP1_P0, PQ0);
	const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);

	const uint_64 u12 = u2 + (uint_64)(P2) * u1;
	return _big_add_u64(_big_mul_u64i32(P1 * (uint_64)(P2), i0), u12);
}

INLINE small_type getS_i(const RNS_S r) { return garner2(r.r[0], r.r[1]); }
INLINE big_type getK_i(const RNS_K r) { return garner3(r.r[0], r.r[1], r.r[2]); }

INLINE pck_type extract_r(big_type * const f)
{
	const pck_type r = (*f).s0 & (BASE2 - 1);
	(*f) = _big_rshift((*f), 2 * LBASE);
	return r;
}

INLINE RNS_K extract_encode(big_type * const f)
{
	const pck_type re = extract_r(f);
	const uint_32 r1 = (uint_32)(re) & (BASE - 1);
	int_32 r2 = (int_32)(re >> LBASE);
	if (r2 >= (int_32)(BASE / 2)) { r2 -= (int_32)(BASE); (*f) = big_inc((*f)); }	// balanced
	return addmodK(setK_u32(r1), mulmodK(setK_i32(r2), RBASE_K()));
}

INLINE RNS_K extract_encode_last(const big_type f)
{
	const uint_32 r1 = (uint_32)(f.s0) & (BASE - 1);
	const int_32 r2 = (int_32)(f.s0 >> LBASE);
	return addmodK(setK_u32(r1), mulmodK(setK_i32(r2), RBASE_K()));
}

#elif (PRM_COUNT == 4)	// PCK_FACTOR = 2

#define BASE2		((uint_64)(1) << (2 * LBASE))

typedef struct { uint_64 s0; int_64 s1; } big_type;	// 128-bit, size = PRM_COUNT * 32 > LN + 2 * LBASE
typedef uint_64 pck_type;	// vector type of size PCK_FACTOR
typedef int_64 small_type;	// 64-bit, size = PCK_FACTOR * 32 > PCK_FACTOR * LBASE

INLINE big_type big_zero() { big_type r; r.s0 = 0; r.s1 = 0; return r; }

INLINE big_type big_neg(const big_type x)
{
	big_type r;
	r.s0 = -x.s0;
	r.s1 = -x.s1 - ((x.s0 != 0) ? 1 : 0);
	return r;
}

INLINE big_type big_add(const big_type x, const big_type y)
{
	big_type r;
// #if defined(PTX_ASM)
// 	asm volatile ("add.cc.u64 %0, %1, %2;" : "=r" (r.s0) : "r" (x.s0), "r" (y.s0));
// 	asm volatile ("addc.s64 %0, %1, %2;" : "=r" (r.s1) : "r" (x.s1), "r" (y.s1));
// #else
	r.s0 = x.s0 + y.s0;
	r.s1 = x.s1 + y.s1 + ((r.s0 < x.s0) ? 1 : 0);
// #endif
	return r;
}

INLINE big_type _big_add_u64(const big_type x, const uint_64 y)
{
	big_type t; t.s0 = y; t.s1 = 0;
	return big_add(x, t);
}

INLINE big_type _big_add_s64(const big_type x, const int_64 y)
{
	big_type t; t.s0 = (uint_64)(y); t.s1 = (y < 0) ? -1 : 0;
	return big_add(x, t);
}

INLINE big_type big_inc(const big_type x) { return _big_add_u64(x, 1); }
INLINE big_type big_add_pck(const big_type x, const pck_type r) { return _big_add_u64(x, r); }
INLINE big_type big_add_small(const big_type x, const small_type y) { return _big_add_s64(x, y); }
INLINE big_type big_sub_small(const big_type x, const small_type y) { return _big_add_s64(x, -y); }

INLINE big_type big_mul_u32(const big_type x, const uint_32 y)
{
	const big_type xp = (x.s1 < 0) ? big_neg(x) : x;
	big_type r;
	r.s0 = xp.s0 * y;
	r.s1 = (int_64)((uint_64)(xp.s1) * y + mul_hi(xp.s0, (uint_64)(y)));
	return (x.s1 < 0) ? big_neg(r) : r;
}

INLINE big_type _big_rshift(const big_type x, const int s)	// s != 0
{
	big_type r;
	r.s0 = (x.s0 >> s) | (uint_64)(x.s1 << (64 - s));
	r.s1 = x.s1 >> s;
	return r;
}

INLINE big_type _big_mul_u64i64(const uint_64 x, const int_64 y)
{
	const uint_64 yp = abs(y);
	big_type r;
	r.s0 = x * yp;
	r.s1 = (int_64)(mul_hi(x, yp));
	return (y < 0) ? big_neg(r) : r;
}

INLINE big_type garner4(const ZP r0, const ZP r1, const ZP r2, const ZP r3)
{
	// convert to mixed-radix representation, we must have P3 < P2 < P1 < P0
	const uint_32 u3 = r3;
	const uint_32 u2 = submulmod(r2, u3, INVP3_P2, PQ2);
	const uint_32 u1 = submulmod(submulmod(r1, u3, INVP3_P1, PQ1), u2, INVP2_P1, PQ1);
	const uint_32 u0 = submulmod(submulmod(submulmod(r0, u3, INVP3_P0, PQ0), u2, INVP2_P0, PQ0), u1, INVP1_P0, PQ0);
	const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);

	const uint_64 u23 = u3 + (uint_64)(P3) * u2;
	const int_64 i01 = (int_64)(u1) + (int_64)(P1) * i0;
	return _big_add_u64(_big_mul_u64i64(P2 * (uint_64)(P3), i01), u23);
}

INLINE small_type getS_i(const RNS_S r) { return garner2(r.r[0], r.r[1]); }
INLINE big_type getK_i(const RNS_K r) { return garner4(r.r[0], r.r[1], r.r[2], r.r[3]); }

INLINE pck_type extract_r(big_type * const f)
{
	const pck_type r = (*f).s0 & (BASE2 - 1);
	(*f) = _big_rshift((*f), 2 * LBASE);
	return r;
}

INLINE RNS_K extract_encode(big_type * const f)
{
	const pck_type re = extract_r(f);
	const uint_32 r1 = (uint_32)(re) & (BASE - 1);
	int_32 r2 = (int_32)(re >> LBASE);
	if (r2 >= (int_32)(BASE / 2)) { r2 -= (int_32)(BASE); (*f) = big_inc((*f)); }	// balanced
	return addmodK(setK_u32(r1), mulmodK(setK_i32(r2), RBASE_K()));
}

INLINE RNS_K extract_encode_last(const big_type f)
{
	const uint_32 r1 = (uint_32)(f.s0) & (BASE - 1);
	const int_32 r2 = (int_32)(f.s0 >> LBASE);
	return addmodK(setK_u32(r1), mulmodK(setK_i32(r2), RBASE_K()));
}

#endif

INLINE void carry_mul_sub_1(const __global ZP * restrict const xi, __global const ZP * restrict const yi,
	__global big_type * restrict const carry, __local big_type * const cl, pck_type r[CARRY_LENGTH], const sz_t id, const uint_32 a)
{
	// assert CARRY_LENGTH = 4
	ZP4 x4[PRM_COUNT]; for (sz_t l = 0; l < PRM_COUNT; ++l) x4[l] = load4g(xi, l * Z_OFFSET / 4);
	RNS_K y[4]; for (sz_t l = 0; l < PRM_COUNT; ++l) { y[0].r[l] = x4[l].s0; y[1].r[l] = x4[l].s1; y[2].r[l] = x4[l].s2; y[3].r[l] = x4[l].s3; }

	big_type f = big_zero();
	for (sz_t j = 0; j < CARRY_LENGTH; ++j)
	{
		// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF
		f = big_add(f, big_mul_u32(getK_i(mulmodK(y[j], NORM_K())), a));
		if (yi != NULL_PTR) f = big_sub_small(f, getS_i(loadRNS_S(&yi[j])));
		r[j] = extract_r(&f);
	}

	const sz_t lid = id % CARRY_WG_SZ;
	cl[lid] = f;
	if (lid == CARRY_WG_SZ - 1) carry[id / CARRY_WG_SZ] = f;
}

INLINE void carry_2(__global ZP * restrict const xi, const __local big_type * const cl, const pck_type r[CARRY_LENGTH], const sz_t lid)
{
	big_type f = (lid != 0) ? cl[lid - 1] : big_zero();
	for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)
	{
		f = big_add_pck(f, r[j]);
		const RNS_K u = extract_encode(&f);
		storeRNS_K(&xi[j], u);
	}

	f = big_add_pck(f, r[CARRY_LENGTH - 1]);
	const RNS_K u = extract_encode_last(f);
	storeRNS_K(&xi[CARRY_LENGTH - 1], u);
}

INLINE void carry_3(__global ZP * restrict const xi, __global const big_type * restrict const carry, const int_32 is_nega, const sz_t id)
{
	// assert CARRY_LENGTH = 4
	ZP4 x4[PCK_FACTOR]; for (sz_t l = 0; l < PCK_FACTOR; ++l) x4[l] = load4g(xi, l * Z_OFFSET / 4);
	RNS_S y[4]; for (sz_t l = 0; l < PCK_FACTOR; ++l) { y[0].r[l] = x4[l].s0; y[1].r[l] = x4[l].s1; y[2].r[l] = x4[l].s2; y[3].r[l] = x4[l].s3; }

	big_type f = carry[((id == 0) ? (N_SZ / CARRY_LENGTH / CARRY_WG_SZ) : id) - 1];
	if ((id == 0) && (is_nega != 0)) f = big_neg(f);
	for (sz_t j = 0; j < CARRY_LENGTH - 1; ++j)
	{
		f = big_add_small(f, getS_i(y[j]));
		const RNS_K u = extract_encode(&f);
		storeRNS_K(&xi[j], u);
	}

	f = big_add_small(f, getS_i(y[CARRY_LENGTH - 1]));
	const RNS_K u = extract_encode_last(f);
	storeRNS_K(&xi[CARRY_LENGTH - 1], u);
}

__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))
void carry_mul(__global ZP * restrict const reg, __global big_type * restrict const carry, const uint_32 reg_x, const uint_32 a)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];
	__local big_type cl[CARRY_WG_SZ];
	pck_type r[CARRY_LENGTH];

	carry_mul_sub_1(xi, NULL_PTR, carry, cl, r, id, a);
	barrier(CLK_LOCAL_MEM_FENCE);
	carry_2(xi, cl, r, id % CARRY_WG_SZ);
}

__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))
void carry_sub(__global ZP * restrict const reg, __global big_type * restrict const carry, const uint_32 reg_x, const uint_32 reg_y)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];
	__global const ZP * restrict const yi = &reg[reg_y * N_SZ + CARRY_LENGTH * id];
	__local big_type cl[CARRY_WG_SZ];
	pck_type r[CARRY_LENGTH];

	carry_mul_sub_1(xi, yi, carry, cl, r, id, 1);
	barrier(CLK_LOCAL_MEM_FENCE);
	carry_2(xi, cl, r, id % CARRY_WG_SZ);
}

__kernel
void carry_complete(__global ZP * restrict const reg, __global const big_type * restrict const carry, const uint_32 reg_x, const int_32 is_nega)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * CARRY_WG_SZ * id];

	carry_3(xi, carry, is_nega, id);
}

// --- strict modulo ---

INLINE void lower(__global ZP * restrict const x, const int_32 l0)
{
	const small_type base = (small_type)(1) << (PCK_FACTOR * LBASE);
	if (getS_i(loadRNS_S(&x[N_SZ - 1])) >= base)	// >= base^n
	{
		// - R or Q
		const RNS_K hi = submodK(loadRNS_K(&x[N_SZ - 1]), RBASE_S());
		storeRNS_K(&x[N_SZ - 1], hi);

		const RNS_K lo = submodK(loadRNS_K(&x[0]), setK_i32(l0));
		storeRNS_K(&x[0], lo);
	}
}

__kernel
void mod_Q(__global ZP * restrict const reg, const uint_32 reg_x)
{
	__global ZP * restrict const x = &reg[reg_x * N_SZ];

	lower(x, 1);	// < Q - 1
}

__kernel
void mod_R_add2(__global ZP * restrict const reg, const uint_32 reg_x)
{
	__global ZP * restrict const x = &reg[reg_x * N_SZ];

	lower(x, -1);		// <= R

	const RNS_K x0 = addmodK(loadRNS_K(&x[0]), setK_u32(2));
	storeRNS_K(&x[0], x0);
}

// --- misc ---

__kernel
void set(__global ZP4 * restrict const reg, const uint_32 reg_y, const uint_32 reg_x)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global ZP4 * restrict const y = &reg[reg_y * N_SZ / 4];
	__global const ZP4 * restrict const x = &reg[reg_x * N_SZ / 4];

	for (sz_t l = 0; l < 4; ++l) y[l * Z_OFFSET / 4 + id] = x[l * Z_OFFSET / 4 + id];
}

