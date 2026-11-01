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
#endif
#define BASE			(1u << LBASE)

typedef uint	sz_t;
typedef uint	uint_32;
typedef int		int_32;
typedef ulong	uint_64;
typedef long	int_64;
typedef uint2	uint2_32;
typedef uint4	uint4_32;
typedef uint8	uint8_32;

// --- modular arithmetic

#define RNS		uint2_32
#define RNS2	uint4_32
#define RNS4	uint8_32

INLINE uint_32 _addmod(const uint_32 lhs, const uint_32 rhs, const uint_32 p)
{
	return lhs + rhs - ((lhs >= p - rhs) ? p : 0);
}

INLINE uint_32 _submod(const uint_32 lhs, const uint_32 rhs, const uint_32 p)
{
	return lhs - rhs + ((lhs < rhs) ? p : 0);
}

// 2 mul + 2 mul_hi
INLINE uint_32 _mulmod(const uint_32 lhs, const uint_32 rhs, const uint_32 p, const uint_32 q)
{
	const uint_64 t = lhs * (uint_64)(rhs);
	const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);
	const uint_32 mp = mul_hi(lo * q, p);
	return _submod(hi, mp, p);
}

INLINE int_32 _get_int(const uint_32 n, const uint_32 p) { return (int_32)(n - ((n >= p / 2) ? p : 0)); }
INLINE uint_32 _set_int(const int_32 i, const uint_32 p) { return (uint_32)(i) + ((i < 0) ? p : 0); }

INLINE RNS addmod(const RNS lhs, const RNS rhs)
{
	return (RNS)(_addmod(lhs.s0, rhs.s0, P1), _addmod(lhs.s1, rhs.s1, P2));
}

INLINE RNS submod(const RNS lhs, const RNS rhs)
{
	return (RNS)(_submod(lhs.s0, rhs.s0, P1), _submod(lhs.s1, rhs.s1, P2));
}

INLINE RNS mulmod(const RNS lhs, const RNS rhs)
{
	return (RNS)(_mulmod(lhs.s0, rhs.s0, P1, Q1), _mulmod(lhs.s1, rhs.s1, P2, Q2));
}

INLINE RNS set_int(const int_32 i)
{
	return (RNS)(_set_int(i, P1), _set_int(i, P2));
}

INLINE int_32 get_int(const RNS lhs)
{
	return _get_int(lhs.s0, P1);
}

// --- I/O ---

INLINE void _loadg(const sz_t n, RNS * const xl, __global const RNS * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }
INLINE void _storeg(const sz_t n, __global RNS * restrict const x, const sz_t s, const RNS * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }

INLINE RNS2 _load2g(__global const RNS * restrict const w, const sz_t j) { return ((__global const RNS2 *)w)[j]; }
INLINE RNS4 _load4g(__global const RNS * restrict const w, const sz_t j) { return ((__global const RNS4 *)w)[j]; }

// --- transform/macro ---

#define FWD2(x0, x1, w) \
{ \
	const RNS t = mulmod(x1, w); \
	x1 = submod(x0, t); x0 = addmod(x0, t); \
}

#define BCK2(x0, x1, w_i) \
{ \
	const RNS t = submod(x0, x1); x0 = addmod(x0, x1); \
	x1 = mulmod(t, w_i); \
}

#define SQR2(x0, x1, w) \
{ \
	const RNS t = mulmod(mulmod(x1, x1), w); \
	x1 = mulmod(addmod(x0, x0), x1); \
	x0 = addmod(mulmod(x0, x0), t); \
}

#define SQR2N(x0, x1, w) \
{ \
	const RNS t = mulmod(mulmod(x1, x1), w); \
	x1 = mulmod(addmod(x0, x0), x1); \
	x0 = submod(mulmod(x0, x0), t); \
}

#define MUL2(x0, x1, xp0, xp1, w) \
{ \
	const RNS t = mulmod(mulmod(x1, xp1), w); \
	x1 = addmod(mulmod(x0, xp1), mulmod(xp0, x1)); \
	x0 = addmod(mulmod(x0, xp0), t); \
}

#define MUL2N(x0, x1, xp0, xp1, w) \
{ \
	const RNS t = mulmod(mulmod(x1, xp1), w); \
	x1 = addmod(mulmod(x0, xp1), mulmod(xp0, x1)); \
	x0 = submod(mulmod(x0, xp0), t); \
}

// --- transform/inline ---

INLINE void _forward8(RNS x[8], const RNS w1, const RNS2 w2, const RNS4 w4)
{
	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);
	FWD2(x[0], x[2], w2.s01); FWD2(x[1], x[3], w2.s01); FWD2(x[4], x[6], w2.s23); FWD2(x[5], x[7], w2.s23);
	FWD2(x[0], x[1], w4.s01); FWD2(x[2], x[3], w4.s23); FWD2(x[4], x[5], w4.s45); FWD2(x[6], x[7], w4.s67);
}

INLINE void _backward8(RNS x[8], const RNS w1_i, const RNS2 w2_i, const RNS4 w4_i)
{
	BCK2(x[0], x[1], w4_i.s01); BCK2(x[2], x[3], w4_i.s23); BCK2(x[4], x[5], w4_i.s45); BCK2(x[6], x[7], w4_i.s67);
	BCK2(x[0], x[2], w2_i.s01); BCK2(x[1], x[3], w2_i.s01); BCK2(x[4], x[6], w2_i.s23); BCK2(x[5], x[7], w2_i.s23);
	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);
}

INLINE void _forward8_0(RNS x[8], const RNS w1, const RNS2 w2, const RNS4 w4)
{
	for (sz_t i = 0; i < 4; ++i) x[i] = mulmod(x[i], (RNS)(RSQ1, RSQ2));
	_forward8(x, w1, w2, w4);
}

INLINE void _square2x4(RNS x[8], const RNS2 w2)
{
	SQR2(x[0], x[1], w2.s01); SQR2N(x[2], x[3], w2.s01); SQR2(x[4], x[5], w2.s23); SQR2N(x[6], x[7], w2.s23);
}

INLINE void _square4x2(RNS x[8], const RNS2 w2, const RNS2 w2_i)
{
	FWD2(x[0], x[2], w2.s01); FWD2(x[1], x[3], w2.s01); FWD2(x[4], x[6], w2.s23); FWD2(x[5], x[7], w2.s23);
	_square2x4(x, w2);
	BCK2(x[0], x[2], w2_i.s01); BCK2(x[1], x[3], w2_i.s01); BCK2(x[4], x[6], w2_i.s23); BCK2(x[5], x[7], w2_i.s23);
}

INLINE void _square8(RNS x[8], const RNS w1, const RNS w1_i, const RNS2 w2, const RNS2 w2_i)
{
	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);
	_square4x2(x, w2, w2_i);
	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);
}

INLINE void _mul2x4(RNS x[8], const RNS xp[8], const RNS2 w2)
{
	MUL2(x[0], x[1], xp[0], xp[1], w2.s01); MUL2N(x[2], x[3], xp[2], xp[3], w2.s01); MUL2(x[4], x[5], xp[4], xp[5], w2.s23); MUL2N(x[6], x[7], xp[6], xp[7], w2.s23);
}

INLINE void _mul4x2(RNS x[8], const RNS xp[8], const RNS2 w2, const RNS2 w2_i)
{
	FWD2(x[0], x[2], w2.s01); FWD2(x[1], x[3], w2.s01); FWD2(x[4], x[6], w2.s23); FWD2(x[5], x[7], w2.s23);
	_mul2x4(x, xp, w2);
	BCK2(x[0], x[2], w2_i.s01); BCK2(x[1], x[3], w2_i.s01); BCK2(x[4], x[6], w2_i.s23); BCK2(x[5], x[7], w2_i.s23);
}

INLINE void _mul8(RNS x[8], const RNS xp[8], const RNS w1, const RNS w1_i, const RNS2 w2, const RNS2 w2_i)
{
	FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);
	_mul4x2(x, xp, w2, w2_i);
	BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);
}

INLINE void _fwd4x2(RNS x[8], const RNS2 w2)
{
	FWD2(x[0], x[2], w2.s01); FWD2(x[1], x[3], w2.s01); FWD2(x[4], x[6], w2.s23); FWD2(x[5], x[7], w2.s23);
}

INLINE void _fwd8(RNS x[8], const RNS w1, const RNS2 w2)
{
	FWD2(x[0], x[4], w1); FWD2(x[2], x[6], w1); FWD2(x[1], x[5], w1); FWD2(x[3], x[7], w1);
	_fwd4x2(x, w2);
}

// ---

// --- transform/macro ---

// --- transform without local mem ---

__kernel
void forward8(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w, const int_32 lm, const uint_32 s)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];
	__global const RNS * restrict const w = &reg[reg_w * N_SZ];
	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, sj = s + (id >> lm), k = 7 * (id & ~(m - 1)) + id;

	const RNS w1 = w[sj];
	const RNS2 w2 = _load2g(w, sj);
	const RNS4 w4 = _load4g(w, sj);

	RNS xl[8]; _loadg(8, xl, &x[k], m);
	_forward8(xl, w1, w2, w4);
	_storeg(8, &x[k], m, xl);
}

__kernel
void backward8(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w, const int_32 lm, const uint_32 s)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];
	__global const RNS * restrict const w_i = &reg[reg_w * N_SZ + N_SZ / 2];
	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, sj = s + (id >> lm), k = 7 * (id & ~(m - 1)) + id;

	const RNS w1_i = w_i[sj];
	const RNS2 w2_i = _load2g(w_i, sj);
	const RNS4 w4_i = _load4g(w_i, sj);

	RNS xl[8]; _loadg(8, xl, &x[k], m);
	_backward8(xl, w1_i, w2_i, w4_i);
	_storeg(8, &x[k], m, xl);
}

__kernel
void forward8_0(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];
	__global const RNS * restrict const w = &reg[reg_w * N_SZ];
	const sz_t id = (sz_t)get_global_id(0), m = N_SZ / 8, k = id;

	// Montgomery form of MF of I to convert input into MF
	const RNS w1 = w[0];
	const RNS2 w2 = _load2g(w, 1);
	const RNS4 w4 = _load4g(w, 1);

	RNS xl[8]; _loadg(8, xl, &x[k], m);
	_forward8_0(xl, w1, w2, w4);
	_storeg(8, &x[k], m, xl);
}

__kernel
void square2x4(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];
	__global const RNS * restrict const w = &reg[reg_w * N_SZ];
	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;

	const RNS2 w2 = _load2g(w, sj);

	RNS xl[8]; _loadg(8, xl, &x[k], m);
	_square2x4(xl, w2);
	_storeg(8, &x[k], m, xl);
}

__kernel
void square4x2(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];
	__global const RNS * restrict const w = &reg[reg_w * N_SZ];
	__global const RNS * restrict const w_i = &w[N_SZ / 2];
	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;

	const RNS2 w2 = _load2g(w, sj), w2_i = _load2g(w_i, sj);

	RNS xl[8]; _loadg(8, xl, &x[k], m);
	_square4x2(xl, w2, w2_i);
	_storeg(8, &x[k], m, xl);
}

__kernel
void square8(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];
	__global const RNS * restrict const w = &reg[reg_w * N_SZ];
	__global const RNS * restrict const w_i = &w[N_SZ / 2];
	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;

	const RNS w1 = w[sj], w1_i = w_i[sj];
	const RNS2 w2 = _load2g(w, sj), w2_i = _load2g(w_i, sj);

	RNS xl[8]; _loadg(8, xl, &x[k], m);
	_square8(xl, w1, w1_i, w2, w2_i);
	_storeg(8, &x[k], m, xl);
}

__kernel
void mul2x4(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];
	__global const RNS * restrict const xp = &reg[reg_xp * N_SZ];
	__global const RNS * restrict const w = &reg[reg_w * N_SZ];
	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;

	const RNS2 w2 = _load2g(w, sj);

	RNS xl[8]; _loadg(8, xl, &x[k], m);
	RNS xpl[8]; _loadg(8, xpl, &xp[k], m);
	_mul2x4(xl, xpl, w2);
	_storeg(8, &x[k], m, xl);
}

__kernel
void mul4x2(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];
	__global const RNS * restrict const xp = &reg[reg_xp * N_SZ];
	__global const RNS * restrict const w = &reg[reg_w * N_SZ];
	__global const RNS * restrict const w_i = &w[N_SZ / 2];
	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;

	const RNS2 w2 = _load2g(w, sj), w2_i = _load2g(w_i, sj);

	RNS xl[8]; _loadg(8, xl, &x[k], m);
	RNS xpl[8]; _loadg(8, xpl, &xp[k], m);
	_mul4x2(xl, xpl, w2, w2_i);
	_storeg(8, &x[k], m, xl);
}

__kernel
void mul8(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_xp, const uint_32 reg_w)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];
	__global const RNS * restrict const xp = &reg[reg_xp * N_SZ];
	__global const RNS * restrict const w = &reg[reg_w * N_SZ];
	__global const RNS * restrict const w_i = &w[N_SZ / 2];
	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;

	const RNS w1 = w[sj], w1_i = w_i[sj];
	const RNS2 w2 = _load2g(w, sj), w2_i = _load2g(w_i, sj);

	RNS xl[8]; _loadg(8, xl, &x[k], m);
	RNS xpl[8]; _loadg(8, xpl, &xp[k], m);
	_mul8(xl, xpl, w1, w1_i, w2, w2_i);
	_storeg(8, &x[k], m, xl);
}

__kernel
void fwd4x2(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];
	__global const RNS * restrict const w = &reg[reg_w * N_SZ];
	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;

	const RNS2 w2 = _load2g(w, sj);

	RNS xl[8]; _loadg(8, xl, &x[k], m);
	_fwd4x2(xl, w2);
	_storeg(8, &x[k], m, xl);
}

__kernel
void fwd8(__global RNS * restrict const reg, const uint_32 reg_x, const uint_32 reg_w)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];
	__global const RNS * restrict const w = &reg[reg_w * N_SZ];
	const sz_t id = (sz_t)get_global_id(0), m = 1, sj = N_SZ / 8 + id, k = 8 * id;

	const RNS w1 = w[sj];
	const RNS2 w2 = _load2g(w, sj);

	RNS xl[8]; _loadg(8, xl, &x[k], m);
	_fwd8(xl, w1, w2);
	_storeg(8, &x[k], m, xl);
}

// --- transform using local mem ---

// --- carry ---

// Chinese remainder theorem
INLINE int_64 garner2(const RNS r)
{
	const uint_32 r1 = r.s0, r2 = r.s1;
	const uint_32 u = _mulmod(_submod(r1, r2, P1), INVP2_P1, P1, Q1);	// P2 < P1
	const uint_64 n = r2 + u * (uint_64)(P2);
	const bool b = (n > P1P2 / 2);
	return (int_64)(n - (b ? P1P2 : 0));
}

INLINE void carry_mul_1(const __global RNS * restrict const xi, __global int_64 * restrict const carry,
	__local int_64 * const cl, uint_32 r[CARRY_LENGTH], const sz_t id, const int_32 a)
{
	int_64 f = 0;
	for (sz_t j = 0; j < CARRY_LENGTH; ++j)
	{
		// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF
		const RNS u = mulmod(xi[j], (RNS)(NORM1, NORM2));
		f += a * garner2(u);
		r[j] = (uint_32)(f) & (BASE - 1);
		f >>= LBASE;
	}

	const sz_t lid = id % CARRY_WG_SZ;
	cl[lid] = f;

	if (lid == CARRY_WG_SZ - 1) carry[id / CARRY_WG_SZ] = f;
}

INLINE void carry_sub_1(const __global RNS * restrict const xi, __global const RNS * restrict const yi, __global int_64 * restrict const carry,
	__local int_64 * const cl, uint_32 r[CARRY_LENGTH], const sz_t id)
{
	int_64 f = 0;
	for (sz_t j = 0; j < CARRY_LENGTH; ++j)
	{
		// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF
		const RNS u = mulmod(xi[j], (RNS)(NORM1, NORM2));
		f += garner2(u);
		f -= get_int(yi[j]);
		r[j] = (uint_32)(f) & (BASE - 1);
		f >>= LBASE;
	}

	const sz_t lid = id % CARRY_WG_SZ;
	cl[lid] = f;

	if (lid == CARRY_WG_SZ - 1) carry[id / CARRY_WG_SZ] = f;
}

INLINE void carry_2(__global RNS * restrict const xi, const __local int_64 * const cl,
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

	for (size_t j = 0; j < CARRY_LENGTH - 1; ++j) xi[j] = (RNS)(r[j], r[j]);
	f += (int_64)(r[CARRY_LENGTH - 1]);
	xi[CARRY_LENGTH - 1] = set_int((int_32)f);
}

__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))
void carry_mul(__global RNS * restrict const reg, __global int_64 * restrict const carry, const uint_32 reg_x, const int_32 a)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global RNS * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];
	__local int_64 cl[CARRY_WG_SZ];
	uint_32 r[CARRY_LENGTH];

	carry_mul_1(xi, carry, cl, r, id, a);
	barrier(CLK_LOCAL_MEM_FENCE);
	carry_2(xi, cl, r, id);
}

__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))
void carry_sub(__global RNS * restrict const reg, __global int_64 * restrict const carry, const uint_32 reg_x, const uint_32 reg_y)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global RNS * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * id];
	__global const RNS * restrict const yi = &reg[reg_y * N_SZ + CARRY_LENGTH * id];
	__local int_64 cl[CARRY_WG_SZ];
	uint_32 r[CARRY_LENGTH];

	carry_sub_1(xi, yi, carry, cl, r, id);
	barrier(CLK_LOCAL_MEM_FENCE);
	carry_2(xi, cl, r, id);
}

__kernel
void carry_complete(__global RNS * restrict const reg, __global const int_64 * restrict const carry, const uint_32 reg_x, const int_32 is_nega)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global RNS * restrict const xi = &reg[reg_x * N_SZ + CARRY_LENGTH * CARRY_WG_SZ * id];

	int_64 f = carry[((id == 0) ? (N_SZ / CARRY_LENGTH / CARRY_WG_SZ) : id) - 1];
	if ((id == 0) && (is_nega != 0)) f = -f;

	for (sz_t j = 0; j < CARRY_LENGTH - 1; ++j)
	{
		f += (int_64)(xi[j].s0);
		const uint_32 r = (uint_32)(f) & (BASE - 1);
		xi[j] = (RNS)(r, r);
		f >>= LBASE;
	}
	f += (int_64)(xi[CARRY_LENGTH - 1].s0);
	xi[CARRY_LENGTH - 1] = set_int((int_32)f);
}

// --- strict modulo ---

// b-1, b-1, ..., b-1, b + a, ... => b, 0, ..., 0, a, ...
INLINE void fix_base(__global RNS * restrict const x)
{
	if (x[N_SZ - 1].s0 == BASE - 1)
	{
		sz_t i = 2; for (; i < N_SZ; ++i) { if (x[N_SZ - i].s0 != BASE - 1) break; }
		if (get_int(x[N_SZ - i]) >= (int_32)(BASE))
		{
			x[N_SZ - i] -= (RNS)(BASE, BASE);
			for (sz_t j = i - 1; j >= 2; --j) x[N_SZ - j] = (RNS)(0, 0);
			x[N_SZ - 1] = (RNS)(BASE, BASE);
		}
	}
}

__kernel
void mod_Q(__global RNS * restrict const reg, const uint_32 reg_x)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];

	fix_base(x);

	int_32 hi = get_int(x[N_SZ - 1]);
	if (hi >= (int_32)(BASE))
	{
		int_32 lo = get_int(x[0]);
		while (hi >= (int_32)(BASE)) { hi -= (int_32)(BASE); lo -= 1; }	// - Q
		x[N_SZ - 1] = set_int(hi); x[0] = set_int(lo);
	}
}

__kernel
void mod_R_add2(__global RNS * restrict const reg, const uint_32 reg_x)
{
	__global RNS * restrict const x = &reg[reg_x * N_SZ];

	fix_base(x);

	int_32 hi = get_int(x[N_SZ - 1]), lo = get_int(x[0]);
	if (hi >= (int_32)(BASE))
	{
		while (hi >= (int_32)(BASE)) { hi -= (int_32)(BASE); lo += 1; }	// - R
		x[N_SZ - 1] = set_int(hi);
	}
	x[0] = set_int(lo + 2);
}

// --- misc ---

__kernel
void set(__global RNS * restrict const reg, const uint_32 reg_y, const uint_32 reg_x)
{
	const sz_t id = (sz_t)get_global_id(0);
	__global RNS * restrict const y = &reg[reg_y * N_SZ];
	__global const RNS * restrict const x = &reg[reg_x * N_SZ];
	y[id] = x[id];
}
