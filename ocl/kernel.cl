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
#define N_SZ			65536u
#define LBASE			22
#define P1				4194304001u
#define Q1				100663297u
#define RSQ1			232465106u
#define P2				4076863489u
#define Q2				218103809u
#define RSQ2			3444438393u
#define INVP2_P1		1797558821u
#define P1P2			17099604843443519489ul
#define NORM1			4194176001u
#define NORM2			4076739073u
#define CARRY_LENGTH	4u
#define CARRY_WG_SZ     256u
#define Z_OFFSET		655360u
#define BLK16			32
#define BLK32			16
#define BLK64			8
#define BLK128			4
#define BLK256			2
#define BLK512			1
#define BLK1024			1
#define BLK2048			1
#define CHUNK64			32
#define CHUNK512		4
#endif
#define BASE			(1u << LBASE)

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

#define	PQ1		(ZP2)(P1, Q1)
#define	PQ2		(ZP2)(P2, Q2)

__constant ZP2 g_pq[2] = { PQ1, PQ2 };
__constant ZP g_rsq[2] = { RSQ1, RSQ2 };

INLINE ZP addmod(const ZP lhs, const ZP rhs, const ZP p)
{
	return lhs + rhs - ((lhs >= p - rhs) ? p : 0);
}

INLINE ZP submod(const ZP lhs, const ZP rhs, const ZP p)
{
	return lhs - rhs + ((lhs < rhs) ? p : 0);
}

// 2 mul + 2 mul_hi
INLINE ZP mulmod(const ZP lhs, const ZP rhs, const ZP2 pq)
{
	const uint_64 t = lhs * (uint_64)(rhs);
	const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);
	const uint_32 mp = mul_hi(lo * pq.s1, pq.s0);
	return submod(hi, mp, pq.s0);
}

INLINE int_32 get_int(const uint_32 n, const uint_32 p) { return (int_32)(n - ((n >= p / 2) ? p : 0)); }
INLINE uint_32 set_int(const int_32 i, const uint_32 p) { return (uint_32)(i) + ((i < 0) ? p : 0); }

// --- I/O ---

INLINE void loadg(const sz_t n, ZP * const xl, __global const ZP * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }
INLINE void storeg(const sz_t n, __global ZP * restrict const x, const sz_t s, const ZP * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }

INLINE void loadl(const sz_t n, ZP * const xl, __local const ZP * restrict const X, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = X[l * s]; }
INLINE void storel(const sz_t n, __local ZP * restrict const X, const sz_t s, const ZP * const xl) { for (sz_t l = 0; l < n; ++l) X[l * s] = xl[l]; }

INLINE ZP load1g(__global const ZP * restrict const x, const sz_t j) { return x[j]; }
INLINE ZP2 load2g(__global const ZP * restrict const x, const sz_t j) { return ((__global const ZP2 *)x)[j]; }
INLINE ZP4 load4g(__global const ZP * restrict const x, const sz_t j) { return ((__global const ZP4 *)x)[j]; }

INLINE void store4g(__global ZP * restrict const w, const sz_t j, const ZP4 x4) { ((__global ZP4 *)w)[j] = x4; }

// --- transform/macro ---

#define FWD2(x0, x1, w) \
{ \
	const ZP t = mulmod(x1, w, pq); \
	x1 = submod(x0, t, pq.s0); x0 = addmod(x0, t, pq.s0); \
}

#define BCK2(x0, x1, w_i) \
{ \
	const ZP t = submod(x0, x1, pq.s0); x0 = addmod(x0, x1, pq.s0); \
	x1 = mulmod(t, w_i, pq); \
}

#define SQR2(x0, x1, w) \
{ \
	const ZP t = mulmod(mulmod(x1, x1, pq), w, pq); \
	x1 = mulmod(addmod(x0, x0, pq.s0), x1, pq); \
	x0 = addmod(mulmod(x0, x0, pq), t, pq.s0); \
}

#define SQR2N(x0, x1, w) \
{ \
	const ZP t = mulmod(mulmod(x1, x1, pq), w, pq); \
	x1 = mulmod(addmod(x0, x0, pq.s0), x1, pq); \
	x0 = submod(mulmod(x0, x0, pq), t, pq.s0); \
}

#define MUL2(x0, x1, xp0, xp1, w) \
{ \
	const ZP t = mulmod(mulmod(x1, xp1, pq), w, pq); \
	x1 = addmod(mulmod(x0, xp1, pq), mulmod(xp0, x1, pq), pq.s0); \
	x0 = addmod(mulmod(x0, xp0, pq), t, pq.s0); \
}

#define MUL2N(x0, x1, xp0, xp1, w) \
{ \
	const ZP t = mulmod(mulmod(x1, xp1, pq), w, pq); \
	x1 = addmod(mulmod(x0, xp1, pq), mulmod(xp0, x1, pq), pq.s0); \
	x0 = submod(mulmod(x0, xp0, pq), t, pq.s0); \
}

// --- transform/inline ---

INLINE void _forward8(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2, const ZP4 w4)
{
	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);
	FWD2(x[0], x[2], w2.s0); FWD2(x[1], x[3], w2.s0); FWD2(x[4], x[6], w2.s1); FWD2(x[5], x[7], w2.s1);
	FWD2(x[0], x[1], w4.s0); FWD2(x[2], x[3], w4.s1); FWD2(x[4], x[5], w4.s2); FWD2(x[6], x[7], w4.s3);
}

INLINE void _backward8(const ZP2 pq, ZP x[8], const ZP w1_i, const ZP2 w2_i, const ZP4 w4_i)
{
	BCK2(x[0], x[1], w4_i.s0); BCK2(x[2], x[3], w4_i.s1); BCK2(x[4], x[5], w4_i.s2); BCK2(x[6], x[7], w4_i.s3);
	BCK2(x[0], x[2], w2_i.s0); BCK2(x[1], x[3], w2_i.s0); BCK2(x[4], x[6], w2_i.s1); BCK2(x[5], x[7], w2_i.s1);
	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);
}

INLINE void _forward8_0(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2, const ZP4 w4, const ZP rsq)
{
	for (sz_t i = 0; i < 4; ++i) x[i] = mulmod(x[i], rsq, pq);
	_forward8(pq, x, w1, w2, w4);
}

INLINE void _square2x4(const ZP2 pq, ZP x[8], const ZP2 w2)
{
	SQR2(x[0], x[1], w2.s0); SQR2N(x[2], x[3], w2.s0); SQR2(x[4], x[5], w2.s1); SQR2N(x[6], x[7], w2.s1);
}

INLINE void _square4x2(const ZP2 pq, ZP x[8], const ZP2 w2, const ZP2 w2_i)
{
	FWD2(x[0], x[2], w2.s0); FWD2(x[1], x[3], w2.s0); FWD2(x[4], x[6], w2.s1); FWD2(x[5], x[7], w2.s1);
	_square2x4(pq, x, w2);
	BCK2(x[0], x[2], w2_i.s0); BCK2(x[1], x[3], w2_i.s0); BCK2(x[4], x[6], w2_i.s1); BCK2(x[5], x[7], w2_i.s1);
}

INLINE void _square8(const ZP2 pq, ZP x[8], const ZP w1, const ZP w1_i, const ZP2 w2, const ZP2 w2_i)
{
	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);
	_square4x2(pq, x, w2, w2_i);
	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);
}

INLINE void _mul2x4(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP2 w2)
{
	MUL2(x[0], x[1], xp[0], xp[1], w2.s0); MUL2N(x[2], x[3], xp[2], xp[3], w2.s0); MUL2(x[4], x[5], xp[4], xp[5], w2.s1); MUL2N(x[6], x[7], xp[6], xp[7], w2.s1);
}

INLINE void _mul4x2(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP2 w2, const ZP2 w2_i)
{
	FWD2(x[0], x[2], w2.s0); FWD2(x[1], x[3], w2.s0); FWD2(x[4], x[6], w2.s1); FWD2(x[5], x[7], w2.s1);
	_mul2x4(pq, x, xp, w2);
	BCK2(x[0], x[2], w2_i.s0); BCK2(x[1], x[3], w2_i.s0); BCK2(x[4], x[6], w2_i.s1); BCK2(x[5], x[7], w2_i.s1);
}

INLINE void _mul8(const ZP2 pq, ZP x[8], const ZP xp[8], const ZP w1, const ZP w1_i, const ZP2 w2, const ZP2 w2_i)
{
	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);
	_mul4x2(pq, x, xp, w2, w2_i);
	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);
}

INLINE void _fwd4x2(const ZP2 pq, ZP x[8], const ZP2 w2)
{
	FWD2(x[0], x[2], w2.s0); FWD2(x[1], x[3], w2.s0); FWD2(x[4], x[6], w2.s1); FWD2(x[5], x[7], w2.s1);
}

INLINE void _fwd8(const ZP2 pq, ZP x[8], const ZP w1, const ZP2 w2)
{
	FWD2(x[0], x[4], w1); FWD2(x[2], x[6], w1); FWD2(x[1], x[5], w1); FWD2(x[3], x[7], w1);
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
	_square2x4(pq, xl, w2);
	STOREG(1);
}

INLINE void square2x4l(const ZP2 pq, __local ZP * restrict const X,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2();
	LOADL(1);
	_square2x4(pq, xl, w2);
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
	_mul2x4(pq, xl, xpl, w2);
	STOREG(1);
}

INLINE void mul2x4l(const ZP2 pq, __local ZP * restrict const X, __global const ZP * restrict const xp,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2();
	LOADL(1);
	LOADPG(1);
	_mul2x4(pq, xl, xpl, w2);
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

__kernel
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

// ---

__kernel
void square2x4(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	square2x4g(pq, &x[k], w, sj);
}

__kernel
void square4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	square4x2g(pq, &x[k], w, sj);
}

__kernel
void square8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	square8g(pq, &x[k], w, sj);
}

// ---

__kernel
void mul2x4(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	mul2x4g(pq, &x[k], &xp[k], w, sj);
}

__kernel
void mul4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	mul4x2g(pq, &x[k], &xp[k], w, sj);
}

__kernel
void mul8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	mul8g(pq, &x[k], &xp[k], w, sj);
}

// ---

__kernel
void fwd4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	fwd4x2g(pq, &x[k], w, sj);
}

__kernel
void fwd8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	fwd8g(pq, &x[k], w, sj);
}

__kernel
void fwd16(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	const sz_t sj = N_SZ / 8 + id, k2 = 7 * (id & ~(2 - 1)) + id;
	forward8g(pq, 2, &x[k2], w, sj / 2);
}

// --- transform using local mem ---

#define DECLARE_VAR_FB(N, CHUNK_N) \
	__local ZP X[N * CHUNK_N]; \
	DECLARE_VAR_REG(); \
	const size_t chunk_id = id % CHUNK_N, local_id = (id / CHUNK_N) % (N / 8), block_id = id & ~(N / 8 * CHUNK_N - 1); \
	__global ZP * restrict const xt = &x[block_id / (N / 8) + chunk_id]; \
	__local ZP * const Xt = &X[N * chunk_id];

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

// ---

#define DECLARE_VAR_SQRMUL(N, BLK_N) \
	__local ZP X[N * BLK_N]; \
	DECLARE_VAR_REG(); \
	const sz_t block_id = (id / (N / 8)) % BLK_N; \
	__local ZP * const Xb = &X[N * block_id]; \
	const sz_t s = N_SZ / 8, sj = s + id, k = 8 * id;

#define DECLARE_VAR_SQRMUL16() \
	DECLARE_VAR_SQRMUL(16, BLK16); \
	const sz_t k2 = 7 * (id & ~(2 - 1)) + id;

#define DECLARE_VAR_SQRMUL32() \
	DECLARE_VAR_SQRMUL(32, BLK32); \
	const sz_t k4 = 7 * (id & ~(4 - 1)) + id;

#define DECLARE_VAR_SQRMUL64() \
	DECLARE_VAR_SQRMUL(64, BLK64); \
	const sz_t k8 = 7 * (id & ~(8 - 1)) + id;

#define DECLARE_VAR_SQRMUL128() \
	DECLARE_VAR_SQRMUL(128, BLK128); \
	const sz_t k2 = 7 * (id & ~(2 - 1)) + id, k16 = 7 * (id & ~(16 - 1)) + id;

#define DECLARE_VAR_SQRMUL256() \
	DECLARE_VAR_SQRMUL(256, BLK256); \
	const sz_t k4 = 7 * (id & ~(4 - 1)) + id, k32 = 7 * (id & ~(32 - 1)) + id;

#define DECLARE_VAR_SQRMUL512() \
	DECLARE_VAR_SQRMUL(512, BLK512); \
	const sz_t k8 = 7 * (id & ~(8 - 1)) + id, k64 = 7 * (id & ~(64 - 1)) + id;

#define DECLARE_VAR_SQRMUL1024() \
	DECLARE_VAR_SQRMUL(1024, BLK1024); \
	const sz_t k2 = 7 * (id & ~(2 - 1)) + id, k16 = 7 * (id & ~(16 - 1)) + id, k128 = 7 * (id & ~(128 - 1)) + id;

#define DECLARE_VAR_SQRMUL2048() \
	DECLARE_VAR_SQRMUL(2048, BLK2048); \
	const sz_t k4 = 7 * (id & ~(4 - 1)) + id, k32 = 7 * (id & ~(32 - 1)) + id, k256 = 7 * (id & ~(256 - 1)) + id;


__kernel __attribute__((reqd_work_group_size(16 / 8 * BLK16, 1, 1)))
void square16(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL16();

	forward8i(pq, 2, &Xb[k2 % 16], 2, &x[k2], w, sj / 2);
	square2x4l(pq, &Xb[k % 16], w, sj);
	backward8o(pq, 2, &x[k2], 2, &Xb[k2 % 16], w, sj / 2);
}

__kernel __attribute__((reqd_work_group_size(32 / 8 * BLK32, 1, 1)))
void square32(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL32();

	forward8i(pq, 4, &Xb[k4 % 32], 4, &x[k4], w, sj / 4);
	square4x2l(pq, &Xb[k % 32], w, sj);
	backward8o(pq, 4, &x[k4], 4, &Xb[k4 % 32], w, sj / 4);
}

__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))
void square64(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL64();

	forward8i(pq, 8, &Xb[k8 % 64], 8, &x[k8], w, sj / 8);
	square8l(pq, &Xb[k % 64], w, sj);
	backward8o(pq, 8, &x[k8], 8, &Xb[k8 % 64], w, sj / 8);
}

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

// ---

__kernel __attribute__((reqd_work_group_size(16 / 8 * BLK16, 1, 1)))
void mul16(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_SQRMUL16();
	DECLARE_VARP_REG();

	forward8i(pq, 2, &Xb[k2 % 16], 2, &x[k2], w, sj / 2);
	mul2x4l(pq, &Xb[k % 16], &xp[k], w, sj);
	backward8o(pq, 2, &x[k2], 2, &Xb[k2 % 16], w, sj / 2);
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

__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))
void mul64(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w, const uint_32 reg_xp)
{
	DECLARE_VAR_SQRMUL64();
	DECLARE_VARP_REG();

	forward8i(pq, 8, &Xb[k8 % 64], 8, &x[k8], w, sj / 8);
	mul8l(pq, &Xb[k % 64], &xp[k], w, sj);
	backward8o(pq, 8, &x[k8], 8, &Xb[k8 % 64], w, sj / 8);
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

// ---

__kernel __attribute__((reqd_work_group_size(32 / 8 * BLK32, 1, 1)))
void fwd32(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL32();

	forward8i(pq, 4, &Xb[k4 % 32], 4, &x[k4], w, sj / 4);
	fwd4x2o(pq, &x[k], &Xb[k % 32], w, sj);
}

__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))
void fwd64(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL64();

	forward8i(pq, 8, &Xb[k8 % 64], 8, &x[k8], w, sj / 8);
	fwd8o(pq, &x[k], &Xb[k % 64], w, sj);
}

__kernel __attribute__((reqd_work_group_size(128 / 8 * BLK128, 1, 1)))
void fwd128(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL128();

	forward8i(pq, 16, &Xb[k16 % 128], 16, &x[k16], w, sj / 16);
	forward8o(pq, 2, &x[k2], 2, &Xb[k2 % 128], w, sj / 2);
}

__kernel __attribute__((reqd_work_group_size(256 / 8 * BLK256, 1, 1)))
void fwd256(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL256();

	forward8i(pq, 32, &Xb[k32 % 256], 32, &x[k32], w, sj / 32);
	forward8l(pq, 4, &Xb[k4 % 256], w, sj / 4);
	fwd4x2o(pq, &x[k], &Xb[k % 256], w, sj);
}

__kernel __attribute__((reqd_work_group_size(512 / 8 * BLK512, 1, 1)))
void fwd512(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL512();

	forward8i(pq, 64, &Xb[k64 % 512], 64, &x[k64], w, sj / 64);
	forward8l(pq, 8, &Xb[k8 % 512], w, sj / 8);
	fwd8o(pq, &x[k], &Xb[k % 512], w, sj);
}

__kernel __attribute__((reqd_work_group_size(1024 / 8 * BLK1024, 1, 1)))
void fwd1024(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_w)
{
	DECLARE_VAR_SQRMUL1024();

	forward8i(pq, 128, &Xb[k128 % 1024], 128, &x[k128], w, sj / 128);
	forward8l(pq, 16, &Xb[k16 % 1024], w, sj / 16);
	forward8o(pq, 2, &x[k2], 2, &Xb[k2 % 1024], w, sj / 2);
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

// --- carry ---

// Chinese remainder theorem
INLINE int_64 garner2(const ZP r1, const ZP r2)
{
	const ZP u = mulmod(submod(r1, r2, P1), INVP2_P1, PQ1);	// P2 < P1
	const uint_64 n = r2 + u * (uint_64)(P2);
	const bool b = (n > P1P2 / 2);
	return (int_64)(n - (b ? P1P2 : 0));
}

#define NULL_PTR	((__global const ZP *)0)

INLINE void carry_mul_sub_1(const __global ZP * restrict const xi, __global const ZP * restrict const yi,
	__global int_64 * restrict const carry, __local int_64 * const cl, uint_32 r[CARRY_LENGTH], const sz_t id, const int_32 a)
{
	// assert CARRY_LENGTH = 4
	const ZP4 x1_4 = load4g(xi, 0), x2_4 = load4g(xi, Z_OFFSET / 4);
	ZP x1[4]; x1[0] = x1_4.s0; x1[1] = x1_4.s1; x1[2] = x1_4.s2; x1[3] = x1_4.s3;
	ZP x2[4]; x2[0] = x2_4.s0; x2[1] = x2_4.s1; x2[2] = x2_4.s2; x2[3] = x2_4.s3;

	int_64 f = 0;
	for (sz_t j = 0; j < CARRY_LENGTH; ++j)
	{
		// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF
		const ZP u1 = mulmod(x1[j], NORM1, PQ1), u2 = mulmod(x2[j], NORM2, PQ2);
		f += a * garner2(u1, u2);
		if (yi != NULL_PTR) f -= get_int(yi[j], P1);
		r[j] = (uint_32)(f) & (BASE - 1);
		f >>= LBASE;
	}

	const sz_t lid = id % CARRY_WG_SZ;
	cl[lid] = f;

	if (lid == CARRY_WG_SZ - 1) carry[id / CARRY_WG_SZ] = f;
}

INLINE void carry_2(__global ZP * restrict const xi, const __local int_64 * const cl,
	uint_32 r[CARRY_LENGTH], const sz_t id)
{
	const sz_t lid = id % CARRY_WG_SZ;
	int_64 f = (lid != 0) ? cl[lid - 1] : (int_64)(0);

	for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)
	{
		f += (int_64)(r[j]);
		r[j] = (uint_32)(f) & (BASE - 1);
		f >>= LBASE;
		if (f == 0) break;
	}
	f += (int_64)(r[CARRY_LENGTH - 1]);

	// assert CARRY_LENGTH = 4
	ZP4 x_4; x_4.s0 = r[0]; x_4.s1 = r[1]; x_4.s2 = r[2];
	x_4.s3 = set_int((int_32)f, P1);
	store4g(xi, 0, x_4);
	x_4.s3 = set_int((int_32)f, P2);
	store4g(xi, Z_OFFSET / 4, x_4);
}

__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))
void carry_mul(__global ZP * restrict const reg, __global int_64 * restrict const carry, const uint_32 reg_x, const int_32 a)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];
	__local int_64 cl[CARRY_WG_SZ];
	uint_32 r[CARRY_LENGTH];

	carry_mul_sub_1(xi, NULL_PTR, carry, cl, r, id, a);
	barrier(CLK_LOCAL_MEM_FENCE);
	carry_2(xi, cl, r, id);
}

__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))
void carry_sub(__global ZP * restrict const reg, __global int_64 * restrict const carry, const uint_32 reg_x, const uint_32 reg_y)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];
	__global const ZP * restrict const yi = &reg[reg_y * N_SZ + CARRY_LENGTH * id];
	__local int_64 cl[CARRY_WG_SZ];
	uint_32 r[CARRY_LENGTH];

	carry_mul_sub_1(xi, yi, carry, cl, r, id, 1);
	barrier(CLK_LOCAL_MEM_FENCE);
	carry_2(xi, cl, r, id);
}

__kernel
void carry_complete(__global ZP * restrict const reg, __global const int_64 * restrict const carry, const uint_32 reg_x, const int_32 is_nega)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global ZP * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * CARRY_WG_SZ * id];
	uint_32 r[CARRY_LENGTH];

	int_64 f = carry[((id == 0) ? (N_SZ / CARRY_LENGTH / CARRY_WG_SZ) : id) - 1];
	if ((id == 0) && (is_nega != 0)) f = -f;

	// assert CARRY_LENGTH = 4
	const ZP4 x1_4 = load4g(xi, 0);
	r[0] = x1_4.s0; r[1] = x1_4.s1; r[2] = x1_4.s2; r[3] = x1_4.s3;

	for (sz_t j = 0; j < CARRY_LENGTH - 1; ++j)
	{
		f += (int_64)(r[j]);
		r[j] = (uint_32)(f) & (BASE - 1);
		f >>= LBASE;
	}
	f += (int_64)(r[CARRY_LENGTH - 1]);

	// assert CARRY_LENGTH = 4
	ZP4 x_4; x_4.s0 = r[0]; x_4.s1 = r[1]; x_4.s2 = r[2];
	x_4.s3 = set_int((int_32)f, P1);
	store4g(xi, 0, x_4);
	x_4.s3 = set_int((int_32)f, P2);
	store4g(xi, Z_OFFSET / 4, x_4);
}

// --- strict modulo ---

INLINE void set_rns(__global ZP * restrict const x, const int i)
{
	x[0] = set_int(i, P1); x[0 + Z_OFFSET] = set_int(i, P2);
}

// b-1, b-1, ..., b-1, b + a, ... => b, 0, ..., 0, a, ...
// fix[0] = index of the digit below the run of b-1 (fix_scan), fix[1] = fix[0] if fix_zero must clear the run, -1 otherwise
__kernel
void fix_reset(__global ZP * restrict const reg, __global int_32 * restrict const fix)
{
	fix[0] = 0; fix[1] = -1;
}

__kernel
void fix_scan(__global ZP * restrict const reg, __global int_32 * restrict const fix, const uint_32 reg_x)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global const ZP * restrict const x = &reg[reg_x * N_SZ];

	if ((id == 0) || (id > N_SZ - 2) || (x[N_SZ - 1] != BASE - 1)) return;
	if ((x[id] != BASE - 1) && ((id == N_SZ - 2) || (x[id + 1] == BASE - 1))) atomic_max(&fix[0], (int_32)(id));
}

__kernel
void fix_zero(__global ZP * restrict const reg, __global const int_32 * restrict const fix, const uint_32 reg_x)
{
	const sz_t id = (sz_t)get_global_id(0);
	const int_32 i = fix[1];

	if ((i < 0) || (id <= (sz_t)(i)) || (id > N_SZ - 2)) return;
	__global ZP * restrict const x = &reg[reg_x * N_SZ];
	x[id] = 0; x[id + Z_OFFSET] = 0;
}

INLINE void fix_base(__global ZP * restrict const x, __global int_32 * restrict const fix)
{
	if (x[N_SZ - 1] == BASE - 1)
	{
		const sz_t i = (sz_t)(fix[0]);
		if (get_int(x[i], P1) >= (int_32)(BASE))
		{
			x[i] -= BASE; x[i + Z_OFFSET] -= BASE;
			fix[1] = (int_32)(i);
			x[N_SZ - 1] = BASE; x[N_SZ - 1 + Z_OFFSET] = BASE;
		}
	}
}

__kernel
void mod_Q(__global ZP * restrict const reg, __global int_32 * restrict const fix, const uint_32 reg_x)
{
	__global ZP * restrict const x = &reg[reg_x * N_SZ];

	fix_base(x, fix);

	int_32 hi = get_int(x[N_SZ - 1], P1);
	if (hi < 0)
	{
		int_32 lo = get_int(x[0], P1);
		hi += (int_32)(BASE); lo += 1;	// + Q
		set_rns(&x[N_SZ - 1], hi);
		set_rns(&x[0], lo);
	}
	else if (hi >= (int_32)(BASE))
	{
		int_32 lo = get_int(x[0], P1);
		hi -= (int_32)(BASE); lo -= 1;	// - Q
		set_rns(&x[N_SZ - 1], hi);
		set_rns(&x[0], lo);
	}
}

__kernel
void mod_R_add2(__global ZP * restrict const reg, __global int_32 * restrict const fix, const uint_32 reg_x)
{
	__global ZP * restrict const x = &reg[reg_x * N_SZ];

	fix_base(x, fix);

	int_32 hi = get_int(x[N_SZ - 1], P1), lo = get_int(x[0], P1);
	if (hi < 0)
	{
		int_32 lo = get_int(x[0], P1);
		hi += (int_32)(BASE); lo -= 1;	// + R
		set_rns(&x[N_SZ - 1], hi);
	}
	else if (hi >= (int_32)(BASE))
	{
		hi -= (int_32)(BASE); lo += 1;	// - R
		set_rns(&x[N_SZ - 1], hi);
	}
	set_rns(&x[0], lo + 2);
}

// --- misc ---

__kernel
void set(__global ZP4 * restrict const reg, const uint_32 reg_y, const uint_32 reg_x)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global ZP4 * restrict const y = &reg[reg_y * N_SZ / 4];
	__global const ZP4 * restrict const x = &reg[reg_x * N_SZ / 4];

	y[id] = x[id]; y[id + Z_OFFSET / 4] = x[id + Z_OFFSET / 4];
}

__kernel
void extend(__global ZP4 * restrict const reg)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global ZP4 * restrict const x = &reg[0];

	const ZP4 u = x[id]; ZP4 r;
	r.s0 = set_int(get_int(u.s0, P1), P2);
	r.s1 = set_int(get_int(u.s1, P1), P2);
	r.s2 = set_int(get_int(u.s2, P1), P2);
	r.s3 = set_int(get_int(u.s3, P1), P2);
	x[id + Z_OFFSET / 4] = r;
}
