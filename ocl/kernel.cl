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
#define LOADGP(m)	ZP xpl[8]; loadg(8, xpl, xp, m);
#define STOREG(m)	storeg(8, x, m, xl);

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


INLINE void forward8g(const ZP2 pq, const sz_t m, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W124();
	LOADG(m);
	_forward8(pq, xl, w1, w2, w4);
	STOREG(m);
}

INLINE void backward8g(const ZP2 pq, const sz_t m, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_WI124();
	LOADG(m);
	_backward8(pq, xl, w1_i, w2_i, w4_i);
	STOREG(m);
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

INLINE void square2x4g(const ZP2 pq, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2();
	LOADG(1);
	_square2x4(pq, xl, w2);
	STOREG(1);
}

INLINE void square4x2g(const ZP2 pq, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2I2();
	LOADG(1);
	_square4x2(pq, xl, w2, w2_i);
	STOREG(1);
}

INLINE void square8g(const ZP2 pq, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W12I12();
	LOADG(1);
	_square8(pq, xl, w1, w1_i, w2, w2_i);
	STOREG(1);
}

INLINE void mul2x4g(const ZP2 pq, __global ZP * restrict const x, __global const ZP * restrict const xp,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2();
	LOADG(1);
	LOADGP(1);
	_mul2x4(pq, xl, xpl, w2);
	STOREG(1);
}

INLINE void mul4x2g(const ZP2 pq, __global ZP * restrict const x, __global const ZP * restrict const xp,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2I2();
	LOADG(1);
	LOADGP(1);
	_mul4x2(pq, xl, xpl, w2, w2_i);
	STOREG(1);
}

INLINE void mul8g(const ZP2 pq, __global ZP * restrict const x, __global const ZP * restrict const xp,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W12I12();
	LOADG(1);
	LOADGP(1);
	_mul8(pq, xl, xpl, w1, w1_i, w2, w2_i);
	STOREG(1);
}

INLINE void fwd4x2g(const ZP2 pq, __global ZP * restrict const x,
	__global const ZP * restrict const w, const sz_t sj)
{
	DECLARE_W2();
	LOADG(1);
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

__kernel
void mul2x4(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	mul2x4g(pq, &x[k], &xp[k], w, sj);
}

__kernel
void mul4x2(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	mul4x2g(pq, &x[k], &xp[k], w, sj);
}

__kernel
void mul8(__global ZP * restrict const regm, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t sj = N_SZ / 8 + id, k = 8 * id;
	mul8g(pq, &x[k], &xp[k], w, sj);
}

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

// --- transform using local mem ---

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
INLINE void fix_base(__global ZP * restrict const x)
{
	if (x[N_SZ - 1] == BASE - 1)
	{
		sz_t i = 2; for (; i < N_SZ; ++i) { if (x[N_SZ - i] != BASE - 1) break; }
		if (get_int(x[N_SZ - i], P1) >= (int_32)(BASE))
		{
			x[N_SZ - i] -= BASE; x[N_SZ - i + Z_OFFSET] -= BASE;
			for (sz_t j = i - 1; j >= 2; --j)
			{
				x[N_SZ - j] = 0; x[N_SZ - j + Z_OFFSET] = 0;
			}
			x[N_SZ - 1] = BASE; x[N_SZ - 1 + Z_OFFSET] = BASE;
		}
	}
}

__kernel
void mod_Q(__global ZP * restrict const reg, const uint_32 reg_x)
{
	__global ZP * restrict const x = &reg[reg_x * N_SZ];

	fix_base(x);

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
void mod_R_add2(__global ZP * restrict const reg, const uint_32 reg_x)
{
	__global ZP * restrict const x = &reg[reg_x * N_SZ];

	fix_base(x);

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