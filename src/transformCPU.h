/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>

#include <gmp.h>

#include "transform.h"

#define BALANCED		true
#define CARRY_LENGTH	4u

#define PRM_COUNT		5
#define PCK_FACTOR		4

typedef uint32_t	uint_32;
typedef int32_t		int_32;
typedef uint64_t	uint_64;
typedef int64_t		int_64;

inline uint_32 mul_hi(const uint_32 x, const uint_32 y) { return uint_32((x * uint_64(y)) >> 32); }
inline uint_64 mul_hi(const uint_64 x, const uint_64 y) { return uint_64((x * __int128_t(y)) >> 64); }

// carry-in and carry-out addition
inline uint_64 _addcc(const uint_64 a, const uint_64 b, bool * const carry)
{
	const uint_64 t = a + b;
	const bool c1 = (t < b);
	const uint_64 c = t + ((*carry) ? 1 : 0);
	const bool c2 = (c < (*carry));
	(*carry) = c1 | c2;
	return c;
}

// carry-in and carry-out subtraction
inline uint_64 _subcc(const uint_64 a, const uint_64 b, bool * const carry)
{
	const uint_64 t = a - b;
	const bool c1 = (a < b);
	const uint_64 c = t - ((*carry)  ? 1 : 0);
	const bool c2 = (t < (*carry));
	(*carry) = c1 | c2;
	return c;
}

typedef struct { uint_32 r[PCK_FACTOR]; } pck_type;

#if (PRM_COUNT == 4)

// 128-bit
#define BIG_LSIZE	1
#define HTYPE		int_64

#elif (PRM_COUNT == 5)

// 160-bit
#define BIG_LSIZE	2
#define HTYPE		int_32

#endif

// typedef signed _BitInt(160) big_check;

// void print_big_check(const big_check & x)
// {
// 	big_check t = x;
// 	uint32_t d[5];
// 	for (size_t i = 0; i < 5; ++i) { d[i] = uint32_t(t); t >>= 32; }
// 	for (size_t i = 0; i < 5; ++i) { printf("%08x ", d[5 - 1 - i]); }
// 	printf("- ");
// }

// inline big_check get_check(const big_type & x) { big_check r = x.h; for (size_t i = 0; i < BIG_LSIZE; ++i) r = (r << 64) | x.l[BIG_LSIZE - 1 - i]; return r; }

typedef struct { uint_64 l[BIG_LSIZE]; HTYPE h; } big_type;

inline big_type big_zero() { big_type r; for (size_t i = 0; i < BIG_LSIZE; ++i) r.l[i] = 0; r.h = 0; return r; }

inline big_type big_add(const big_type & x, const big_type & y)
{
	big_type r;
	bool carry = false;
	for (size_t i = 0; i < BIG_LSIZE; ++i) r.l[i] = _addcc(x.l[i], y.l[i], &carry);
	r.h = x.h + y.h + (carry ? 1: 0);
	return r;
}

inline big_type big_sub(const big_type & x, const big_type & y)
{
	big_type r;
	bool carry = false;
	for (size_t i = 0; i < BIG_LSIZE; ++i) r.l[i] = _subcc(x.l[i], y.l[i], &carry);
	r.h = x.h - y.h - (carry ? 1: 0);
	return r;
}

inline big_type big_neg(const big_type & x) { return big_sub(big_zero(), x); }

inline big_type _big_add_u64(const big_type & x, const uint_64 & y)
{
	big_type t; t.l[0] = y; for (size_t i = 1; i < BIG_LSIZE; ++i) t.l[i] = 0; t.h = 0;
	return big_add(x, t);
}

inline big_type big_inc(const big_type x) { return _big_add_u64(x, 1); }

inline big_type big_mul_u32(const big_type x, const uint_32 y)
{
	big_type r;
	uint_32 carry = 0;
	for (size_t i = 0; i < BIG_LSIZE; ++i)
	{
		const uint_32 xl = (uint_32)(x.l[i]), xh = (uint_32)(x.l[i] >> 32);
		const uint_64 sl = xl * (uint_64)(y) + carry;
		carry = (uint_32)(sl >> 32);
		const uint_64 sh = xh * (uint_64)(y) + carry;
		carry = (uint_32)(sh >> 32);
		r.l[i] = (sh << 32) | (uint_32)(sl);
	}
	r.h = x.h * (HTYPE)(y) + (HTYPE)(carry);
	// r.h = (HTYPE)(x.h * (int_64)(y) + carry);
	return r;
}

inline big_type big_rshift(const big_type x, const int s)	// s != 0
{
	big_type r;
	for (size_t i = 0; i < BIG_LSIZE - 1; ++i)
	{
		r.l[i] = (x.l[i] >> s) | (uint_64)(x.l[i + 1] << (64 - s));
	}
	const int_64 h = x.h;	// 64-bit shift
	r.l[BIG_LSIZE - 1] = (x.l[BIG_LSIZE - 1] >> s) | (uint_64)(h << (64 - s));
	r.h = (HTYPE)(h >> s);
	return r;
}

#if (PCK_FACTOR == 2)

inline big_type big_add_pck(const big_type & x, const pck_type & y, const int lbase) { return _big_add_u64(x, y.r[0] | ((uint_64)(y.r[1]) << lbase)); }

inline bool big_ge_pow_2_s(const big_type x, const int s)	// x >= 2^s
{
	if (x.h < 0) return false;
	// PCK_FACTOR = 2 then s < 64
	if (x.h != 0) return true;
	return ((x.l[0] >> s) != 0);
}

#elif (PCK_FACTOR == 4)

inline big_type big_add_pck(const big_type & x, const pck_type & y, const int lbase)
{
	big_type t; t.l[0] = y.r[2] | ((uint_64)(y.r[3]) << lbase);
	t.l[1] = t.l[0] >> (64 - 2*lbase); t.l[0] <<= 2*lbase;
	t.l[0] |= y.r[0] | ((uint_64)(y.r[1]) << lbase);
	t.h = 0;
	return big_add(x, t);
}

inline bool big_ge_pow_2_s(const big_type x, const int s)	// x >= 2^s
{
	if (x.h < 0) return false;
	// PCK_FACTOR = 4 then s < 128
	if (x.h != 0) return true;
	if (s < 64)
	{
		for (size_t i = 1; i < BIG_LSIZE; ++i) if (x.l[i] != 0) return true;
		return ((x.l[0] >> s) != 0);
	}
	else
	{
		for (size_t i = 2; i < BIG_LSIZE; ++i) if (x.l[i] != 0) return true;
		return ((x.l[1] >> (s - 64)) != 0);
	}
}

#endif

inline big_type _big_mul_u64i64(const uint_64 x, const int_64 y)
{
	const uint_32 x0 = (uint_32)(x), x1 = (uint_32)(x >> 32);
	const uint_32 y0 = (uint_32)(y); const int_32 y1 = (int_32)(y >> 32);

	const uint_64 s00 = x0 * (uint_64)(y0);
	const uint_64 s10 = x1 * (uint_64)(y0) + (uint_32)(s00 >> 32);
	const int_64 s01 = x0 * (int_64)(y1) + (uint_32)(s10);
	const int_64 s11 = x1 * (int_64)(y1) + (uint_32)(s10 >> 32) + (int_32)(s01 >> 32);

	big_type r;
	r.l[0] = (uint_64)(s01 << 32) | (uint_32)(s00);
#if (PRM_COUNT == 4)
	r.h = s11;
#elif (PRM_COUNT == 5)
	r.l[1] = (uint_64)(s11);
	r.h = (s11 < 0) ? (HTYPE)(-1) : 0;
#endif
	return r;
}

// The prime finite field Z/PZ
template<uint_32 _P, uint_32 _PRIM_ROOT>
class ZPT
{
private:
	uint_32 _n;

private:
	// p * q = 1 (mod 2^32) (Newton's method)
	static constexpr uint_32 _invert(const uint_32 p)
	{
		uint_32 q = 1, prev = 0;
		while (q != prev) { prev = q; q *= 2 - p * q; }
		return q;
	}

	static const uint_32 _Q = _invert(_P);

	static uint_32 _add(const uint_32 lhs, const uint_32 rhs) { return lhs + rhs - ((lhs >= _P - rhs) ? _P : 0); }
	static uint_32 _sub(const uint_32 lhs, const uint_32 rhs) { return lhs - rhs + ((lhs < rhs) ? _P : 0); }

	// Montgomery multiplication by a constant: 1 mul + 2 mul_hi
	static uint_32 _mul_const(const uint_32 lhs, const uint_32 rhs, const uint_32 rhs_Q)
	{
		const uint_32 mp = mul_hi(lhs * rhs_Q, _P);
		return _sub(mul_hi(lhs, rhs), mp);
	}

	// Montgomery multiplication: 2 mul + 2 mul_hi
	static uint_32 _mul(const uint_32 lhs, const uint_32 rhs) { return _mul_const(lhs, rhs,  rhs * _Q); }

	// 2^32 mod p = Montgomery form of 1
	static const uint_32 _one = (uint_32)(((uint_64)(1) << 32) % _P);
	// (2^32)^2 mod p
	static const uint_32 _rsq = (uint_32)((_one * (uint_64)(_one)) % _P);

	ZPT pow(const uint_32 e) const
	{
		if (e == 0) return one();
		ZPT r = one(), y = *this;
		for (uint_32 i = e; i > 1; i /= 2)
		{
			if (i % 2 != 0) r *= y;
			y *= y;
		}
		r *= y;
		return r;
	}

public:
	ZPT() {}
	explicit ZPT(const uint_32 n) : _n(n) {}

	static uint_32 P() { return _P; }
	static uint_32 Q() { return _Q; }

	static ZPT one() { return ZPT(_one); }
	static ZPT rsq() { return ZPT(_rsq); }

	uint_32 get() const { return _n; }

	ZPT & set_i32(const int_32 i) { _n = (uint_32)(i) + ((i < 0) ? _P : 0); return *this; }

	// Conversion into / out of Montgomery form
	ZPT toMonty() const { return ZPT(_mul(_n, _rsq)); }
	// ZPT fromMonty() const { return ZPT(_mul(_n, 1)); }

	bool operator==(const ZPT & rhs) const { return (_n == rhs._n); }

	ZPT & operator+=(const ZPT & rhs) { _n = _add(_n, rhs._n); return *this; }
	ZPT & operator-=(const ZPT & rhs) { _n = _sub(_n, rhs._n); return *this; }
	ZPT & operator*=(const ZPT & rhs) { _n = _mul(_n, rhs._n); return *this; }

	ZPT operator+(const ZPT & rhs) const { return ZPT(_add(_n, rhs._n)); }
	ZPT operator-(const ZPT & rhs) const { return ZPT(_sub(_n, rhs._n)); }
	ZPT operator*(const ZPT & rhs) const { return ZPT(_mul(_n, rhs._n)); }

	ZPT mul_Q() const { return ZPT(_n * _Q); }
	ZPT mul_const(const ZPT & rhs, const ZPT & rhs_Q) const { return ZPT(_mul_const(_n, rhs._n, rhs_Q._n)); }

	ZPT invert() const { return pow(_P - 2); }
	static ZPT primroot_n(const int ln) { return ZPT(_PRIM_ROOT).toMonty().pow((_P - 1) >> ln); }
	static ZPT norm(const int ln) { return ZPT(_P - ((_P - 1) >> ln)); }
};

using ZP0 = ZPT<(125u << 25) + 1, 3>;
using ZP1 = ZPT<(243u << 24) + 1, 7>;
using ZP2 = ZPT<(235u << 24) + 1, 3>;
using ZP3 = ZPT<( 29u << 27) + 1, 3>;
using ZP4 = ZPT<( 13u << 28) + 1, 3>;

// The residue number system {P0, P1, ...}

#if (PRM_COUNT == 4)

class RNS_K
{
private:
	ZP0 _r0; ZP1 _r1; ZP2 _r2; ZP3 _r3;

private:
	explicit RNS_K(const ZP0 & r0, const ZP1 & r1, const ZP2 & r2, const ZP3 & r3) : _r0(r0), _r1(r1), _r2(r2), _r3(r3) {}

	// Chinese remainder theorem
	static big_type garner4(const ZP0 & r0, const ZP1 & r1, const ZP2 & r2, const ZP3 & r3)
	{
		static const uint_32 P0 = ZP0::P(), P1 = ZP1::P(), P2 = ZP2::P(), P3 = ZP3::P();
		// we must have P4 < P3 < P2 < P1
		static const ZP2 INVP3_P2 = ZP2(P3).toMonty().invert();
		static const ZP1 INVP3_P1 = ZP1(P3).toMonty().invert();
		static const ZP1 INVP2_P1 = ZP1(P2).toMonty().invert();
		static const ZP0 INVP3_P0 = ZP0(P3).toMonty().invert();
		static const ZP0 INVP2_P0 = ZP0(P2).toMonty().invert();
		static const ZP0 INVP1_P0 = ZP0(P1).toMonty().invert();

		// convert to mixed-radix representation
		const uint_32 u3 = r3.get();
		const uint_32 u2 = ((r2 - ZP2(u3)) * INVP3_P2).get();
		const uint_32 u1 = (((r1 - ZP1(u3)) * INVP3_P1 - ZP1(u2)) * INVP2_P1).get();
		const uint_32 u0 = ((((r0 - ZP0(u3)) * INVP3_P0 - ZP0(u2)) * INVP2_P0 - ZP0(u1)) * INVP1_P0).get();
		const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);

		const uint_64 u23 = u3 + (uint_64)(P3) * u2;
		const int_64 i01 = (int_64)(u1) + (int_64)(P1) * i0;
		return _big_add_u64(_big_mul_u64i64(P2 * (uint_64)(P3), i01), u23);
	}

public:
	RNS_K() {}
	explicit RNS_K(const uint_32 n) : _r0(n), _r1(n), _r2(n), _r3(n) {}
	explicit RNS_K(const uint_32 n[4]) : _r0(n[0]), _r1(n[1]), _r2(n[2]), _r3(n[3]) {}
	explicit RNS_K(const pck_type & r) : _r0(r.r[0]), _r1(r.r[1]) {} // PCK_FACTOR = 2

	static RNS_K one() { return RNS_K(ZP0::one(), ZP1::one(), ZP2::one(), ZP3::one()); }

	uint_32 get_r(const size_t i) const
	{
		if (i == 0) return _r0.get();
		if (i == 1) return _r1.get();
		if (i == 2) return _r2.get();
		if (i == 3) return _r3.get();
		return 0;
	}

	pck_type get_pck() const { pck_type r; r.r[0] = _r0.get(); r.r[1] = _r1.get(); return r; } // PCK_FACTOR = 2

	big_type getK_i() const { return garner4(_r0, _r1, _r2, _r3); }

	RNS_K & set_i32(const int_32 i) { _r0.set_i32(i); _r1.set_i32(i); _r2.set_i32(i); _r3.set_i32(i); return *this; }

	// Conversion into Montgomery form
	RNS_K toMonty() const { return RNS_K(_r0.toMonty(), _r1.toMonty(), _r2.toMonty(), _r3.toMonty()); }

	bool operator==(const RNS_K & rhs) const { return (_r0 == rhs._r0) && (_r1 == rhs._r1) && (_r2 == rhs._r2) && (_r3 == rhs._r3); }
	bool operator!=(const RNS_K & rhs) const { return !(*this == rhs); }

	RNS_K & operator+=(const RNS_K & rhs) { _r0 += rhs._r0; _r1 += rhs._r1; _r2 += rhs._r2; _r3 += rhs._r3; return *this; }
	RNS_K & operator-=(const RNS_K & rhs) { _r0 -= rhs._r0; _r1 -= rhs._r1; _r2 -= rhs._r2; _r3 -= rhs._r3; return *this; }
	RNS_K & operator*=(const RNS_K & rhs) { _r0 *= rhs._r0; _r1 *= rhs._r1; _r2 *= rhs._r2; _r3 *= rhs._r3; return *this; }

	RNS_K operator+(const RNS_K & rhs) const { RNS_K r = *this; r += rhs; return r; }
	RNS_K operator-(const RNS_K & rhs) const { RNS_K r = *this; r -= rhs; return r; }
	RNS_K operator*(const RNS_K & rhs) const { RNS_K r = *this; r *= rhs; return r; }

	RNS_K mul_Q() const { return RNS_K(_r0.mul_Q(), _r1.mul_Q(), _r2.mul_Q(), _r3.mul_Q()); }
	RNS_K mul_const(const RNS_K & rhs, const RNS_K & rhs_Q) const
	{
		return RNS_K(_r0.mul_const(rhs._r0, rhs_Q._r0), _r1.mul_const(rhs._r1, rhs_Q._r1),
		           _r2.mul_const(rhs._r2, rhs_Q._r2), _r3.mul_const(rhs._r3, rhs_Q._r3));
	}

	RNS_K invert() const { return RNS_K(_r0.invert(), _r1.invert(), _r2.invert(), _r3.invert()); }
	static RNS_K primroot_n(const int ln) { return RNS_K(ZP0::primroot_n(ln), ZP1::primroot_n(ln), ZP2::primroot_n(ln), ZP3::primroot_n(ln)); }
	static RNS_K norm(const int ln) { return RNS_K(ZP0::norm(ln), ZP1::norm(ln), ZP2::norm(ln), ZP3::norm(ln)); }

	RNS_K encode(const uint32_t & rhs, const RNS_K & base) const { RNS_K r = *this; r *= base; r += RNS_K(rhs); return r; }
};

#elif (PRM_COUNT == 5)

class RNS_K
{
private:
	ZP0 _r0; ZP1 _r1; ZP2 _r2; ZP3 _r3; ZP4 _r4;
	
private:
	explicit RNS_K(const ZP0 & r0, const ZP1 & r1, const ZP2 & r2, const ZP3 & r3, const ZP4 & r4) : _r0(r0), _r1(r1), _r2(r2), _r3(r3), _r4(r4) {}

	// Chinese remainder theorem
	static big_type garner5(const ZP0 & r0, const ZP1 & r1, const ZP2 & r2, const ZP3 & r3, const ZP4 & r4)
	{
		static const uint_32 P0 = ZP0::P(), P1 = ZP1::P(), P2 = ZP2::P(), P3 = ZP3::P(), P4 = ZP4::P();
		// we must have P4 < P3 < P2 < P1
		static const ZP3 INVP4_P3 = ZP3(P4).toMonty().invert();
		static const ZP2 INVP4_P2 = ZP2(P4).toMonty().invert();
		static const ZP2 INVP3_P2 = ZP2(P3).toMonty().invert();
		static const ZP1 INVP4_P1 = ZP1(P4).toMonty().invert();
		static const ZP1 INVP3_P1 = ZP1(P3).toMonty().invert();
		static const ZP1 INVP2_P1 = ZP1(P2).toMonty().invert();
		static const ZP0 INVP4_P0 = ZP0(P4).toMonty().invert();
		static const ZP0 INVP3_P0 = ZP0(P3).toMonty().invert();
		static const ZP0 INVP2_P0 = ZP0(P2).toMonty().invert();
		static const ZP0 INVP1_P0 = ZP0(P1).toMonty().invert();

		// convert to mixed-radix representation, we must have P4 < P3 < P2 < P1 < P0
		const uint_32 u4 = r4.get();
		const uint_32 u3 = ((r3 - ZP3(u4)) * INVP4_P3).get();
		const uint_32 u2 = (((r2 - ZP2(u4)) * INVP4_P2 - ZP2(u3)) * INVP3_P2).get();
		const uint_32 u1 = ((((r1 - ZP1(u4)) * INVP4_P1 - ZP1(u3)) * INVP3_P1 - ZP1(u2)) * INVP2_P1).get();
		const uint_32 u0 = (((((r0 - ZP0(u4)) * INVP4_P0 - ZP0(u3)) * INVP3_P0 - ZP0(u2)) * INVP2_P0  - ZP0(u1)) * INVP1_P0).get();
		const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);

		const uint_64 u23 = u3 + (uint_64)(P3) * u2;
		const int_64 i01 = (int_64)(u1) + (int_64)(P1) * i0;
		const big_type i0123 = _big_add_u64(_big_mul_u64i64(P2 * (uint_64)(P3), i01), u23);
		return _big_add_u64(big_mul_u32(i0123, P4), u4);
	}

public:
	RNS_K() {}
	explicit RNS_K(const uint_32 n) : _r0(n), _r1(n), _r2(n), _r3(n), _r4(n) {}
	explicit RNS_K(const uint_32 n[5]) : _r0(n[0]), _r1(n[1]), _r2(n[2]), _r3(n[3]), _r4(n[4]) {}
	explicit RNS_K(const pck_type & r) : _r0(r.r[0]), _r1(r.r[1]), _r2(r.r[2]), _r3(r.r[3]), _r4(0) {} // PCK_FACTOR = 4

	static RNS_K one() { return RNS_K(ZP0::one(), ZP1::one(), ZP2::one(), ZP3::one(), ZP4::one()); }

	uint_32 get_r(const size_t i) const
	{
		if (i == 0) return _r0.get();
		if (i == 1) return _r1.get();
		if (i == 2) return _r2.get();
		if (i == 3) return _r3.get();
		if (i == 4) return _r4.get();
		return 0;
	}

	pck_type get_pck() const { pck_type r; r.r[0] = _r0.get(); r.r[1] = _r1.get(); r.r[2] = _r2.get(); r.r[3] = _r3.get(); return r; } // PCK_FACTOR = 4

	big_type getK_i() const { return garner5(_r0, _r1, _r2, _r3, _r4); }

	RNS_K & set_i32(const int_32 i) { _r0.set_i32(i); _r1.set_i32(i); _r2.set_i32(i); _r3.set_i32(i);  _r4.set_i32(i); return *this; }

	// Conversion into Montgomery form
	RNS_K toMonty() const { return RNS_K(_r0.toMonty(), _r1.toMonty(), _r2.toMonty(), _r3.toMonty(), _r4.toMonty()); }

	bool operator==(const RNS_K & rhs) const { return (_r0 == rhs._r0) && (_r1 == rhs._r1) && (_r2 == rhs._r2) && (_r3 == rhs._r3) && (_r4 == rhs._r4); }
	bool operator!=(const RNS_K & rhs) const { return !(*this == rhs); }

	RNS_K & operator+=(const RNS_K & rhs) { _r0 += rhs._r0; _r1 += rhs._r1; _r2 += rhs._r2; _r3 += rhs._r3; _r4 += rhs._r4; return *this; }
	RNS_K & operator-=(const RNS_K & rhs) { _r0 -= rhs._r0; _r1 -= rhs._r1; _r2 -= rhs._r2; _r3 -= rhs._r3; _r4 -= rhs._r4; return *this; }
	RNS_K & operator*=(const RNS_K & rhs) { _r0 *= rhs._r0; _r1 *= rhs._r1; _r2 *= rhs._r2; _r3 *= rhs._r3; _r4 *= rhs._r4; return *this; }

	RNS_K operator+(const RNS_K & rhs) const { RNS_K r = *this; r += rhs; return r; }
	RNS_K operator-(const RNS_K & rhs) const { RNS_K r = *this; r -= rhs; return r; }
	RNS_K operator*(const RNS_K & rhs) const { RNS_K r = *this; r *= rhs; return r; }

	RNS_K mul_Q() const { return RNS_K(_r0.mul_Q(), _r1.mul_Q(), _r2.mul_Q(), _r3.mul_Q(), _r4.mul_Q()); }
	RNS_K mul_const(const RNS_K & rhs, const RNS_K & rhs_Q) const
	{
		return RNS_K(_r0.mul_const(rhs._r0, rhs_Q._r0), _r1.mul_const(rhs._r1, rhs_Q._r1),
		             _r2.mul_const(rhs._r2, rhs_Q._r2), _r3.mul_const(rhs._r3, rhs_Q._r3),
		             _r4.mul_const(rhs._r4, rhs_Q._r4));
	}

	RNS_K invert() const { return RNS_K(_r0.invert(), _r1.invert(), _r2.invert(), _r3.invert(), _r4.invert()); }
	static RNS_K primroot_n(const int ln) { return RNS_K(ZP0::primroot_n(ln), ZP1::primroot_n(ln), ZP2::primroot_n(ln), ZP3::primroot_n(ln), ZP4::primroot_n(ln)); }
	static RNS_K norm(const int ln) { return RNS_K(ZP0::norm(ln), ZP1::norm(ln), ZP2::norm(ln), ZP3::norm(ln), ZP4::norm(ln)); }

	RNS_K encode(const uint32_t & rhs, const RNS_K & base) const { RNS_K r = *this; r *= base; r += RNS_K(rhs); return r; }
};

#endif

#if (PCK_FACTOR == 2)

class RNS_S
{
public:
	typedef int_64 small_type;

private:
	ZP0 _r0; ZP1 _r1;

private:
	// Chinese remainder theorem
	static int_64 garner2(const ZP0 & r0, const ZP1 & r1)
	{
		static const uint_32 P0 = ZP0::P(), P1 = ZP1::P();
		static const ZP0 INVP1_P0 = ZP0(P1).toMonty().invert();

		// convert to mixed-radix representation
		const uint_32 u1 = r1.get();
		const uint_32 u0 = ((r0 - ZP0(u1)) * INVP1_P0).get();
		const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);

		return (int_64)(u1) + (int_64)(P1) * i0;
	}

public:
	RNS_S() {}
	explicit RNS_S(const uint_32 n) : _r0(n), _r1(n) {}
	explicit RNS_S(const uint_32 * const n) : _r0(n[0]), _r1(n[1]) {}

	small_type getS_i() const { return garner2(_r0, _r1); }
};

#elif (PCK_FACTOR == 4)

class RNS_S
{
public:
	typedef __int128_t small_type;

private:
	ZP0 _r0; ZP1 _r1; ZP2 _r2; ZP3 _r3;

private:
	// Chinese remainder theorem
	static __int128_t garner4(const ZP0 & r0, const ZP1 & r1, const ZP2 & r2, const ZP3 & r3)
	{
		static const uint_32 P0 = ZP0::P(), P1 = ZP1::P(), P2 = ZP2::P(), P3 = ZP3::P();
		static const ZP2 INVP3_P2 = ZP2(P3).toMonty().invert();
		static const ZP1 INVP3_P1 = ZP1(P3).toMonty().invert();
		static const ZP1 INVP2_P1 = ZP1(P2).toMonty().invert();
		static const ZP0 INVP3_P0 = ZP0(P3).toMonty().invert();
		static const ZP0 INVP2_P0 = ZP0(P2).toMonty().invert();
		static const ZP0 INVP1_P0 = ZP0(P1).toMonty().invert();

		// convert to mixed-radix representation
		const uint_32 u3 = r3.get();
		const uint_32 u2 = ((r2 - ZP2(u3)) * INVP3_P2).get();
		const uint_32 u1 = (((r1 - ZP1(u3)) * INVP3_P1 - ZP1(u2)) * INVP2_P1).get();
		const uint_32 u0 = ((((r0 - ZP0(u3)) * INVP3_P0 - ZP0(u2)) * INVP2_P0 - ZP0(u1)) * INVP1_P0).get();
		const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);

		const uint_64 u23 = u3 + (uint_64)(P3) * u2;
		const int_64 i01 = (int_64)(u1) + (int_64)(P1) * i0;
		return (__int128_t)(P2 * (uint_64)(P3)) * i01 + u23;
	}

public:
	RNS_S() {}
	explicit RNS_S(const uint_32 n) : _r0(n), _r1(n), _r2(n), _r3(n) {}
	explicit RNS_S(const uint_32 * const n) : _r0(n[0]), _r1(n[1]), _r2(n[2]), _r3(n[3]) {}

	small_type getS_i() const { return garner4(_r0, _r1, _r2, _r3); }
};

#endif

class transformCPU : public transform
{
private:
	static const size_t reg_count = 3;
	int _ln = 0;
	int _lbase = 0;
	size_t _prm_count, _pck_factor;
	RNS_K _norm;
	bool _init_failed = false;
	mpz_t _Nz;
	RNS_K * _N_Q = nullptr; RNS_K * _Ninv_R = nullptr; RNS_K * _one = nullptr;
	RNS_K * _t_Q = nullptr; RNS_K * _t = nullptr;
	RNS_K * _reg[reg_count];
	RNS_K * _w_R; RNS_K * _wi_R; RNS_K * _w_Q; RNS_K * _wi_Q;
	big_type * _carry;
	uint_32 * _buf = nullptr;
#if defined(CHECK_THEOREM)
	double _logq_min = 1e100, _logq_max = -1e100, _logN = 0;
#endif

private:
	static void _load(const size_t n, RNS_K * const xl, const RNS_K * const x, const size_t s) { for (size_t l = 0; l < n; ++l) xl[l] = x[l * s]; }
	static void _store(const size_t n, RNS_K * const x, const size_t s, const RNS_K * const xl) { for (size_t l = 0; l < n; ++l) x[l * s] = xl[l]; }

	static void FWD2(RNS_K & x0, RNS_K & x1, const RNS_K & w) { const RNS_K t = x1 * w; x1 = x0 - t; x0 = x0 + t; }
	static void BCK2(RNS_K & x0, RNS_K & x1, const RNS_K & w_i) { const RNS_K t = x0 - x1; x0 = x0 + x1; x1 = t * w_i; }

	static void FWD2_const(RNS_K & x0, RNS_K & x1, const RNS_K & w, const RNS_K w_Q)
	{
		const RNS_K t = x1.mul_const(w, w_Q); x1 = x0 - t; x0 = x0 + t;
	}
	static void BCK2_const(RNS_K & x0, RNS_K & x1, const RNS_K & w_i, const RNS_K & w_i_Q)
	{
		const RNS_K t = x0 - x1; x0 = x0 + x1; x1 = t.mul_const(w_i, w_i_Q);
	}

	static void SQR2(RNS_K & x0, RNS_K & x1, const RNS_K & w, const RNS_K w_Q)
	{
		const RNS_K t = (x1 * x1).mul_const(w, w_Q); x1 = (x0 + x0) * x1; x0 = x0 * x0 + t;
	}
	static void SQR2N(RNS_K & x0, RNS_K & x1, const RNS_K & w, const RNS_K w_Q)
	{
		const RNS_K t = (x1 * x1).mul_const(w, w_Q); x1 = (x0 + x0) * x1; x0 = x0 * x0 - t;
	}

	static void MUL2(RNS_K & x0, RNS_K & x1, const RNS_K & xp0, const RNS_K & xp1, const RNS_K & w, const RNS_K w_Q)
	{
		const RNS_K t = (x1 * xp1).mul_const(w, w_Q); x1 = x0 * xp1 + xp0 * x1; x0 = x0 * xp0 + t;
	}
	static void MUL2N(RNS_K & x0, RNS_K & x1, const RNS_K & xp0, const RNS_K & xp1, const RNS_K & w, const RNS_K w_Q)
	{
		const RNS_K t = (x1 * xp1).mul_const(w, w_Q); x1 = x0 * xp1 + xp0 * x1; x0 = x0 * xp0 - t;
	}

	static void _forward8(RNS_K x[8], const RNS_K & w1, const RNS_K w2[2], const RNS_K w4[4])
	{
		const RNS_K w1_Q = w1.mul_Q();
		FWD2_const(x[0], x[4], w1, w1_Q); FWD2_const(x[1], x[5], w1, w1_Q); FWD2_const(x[2], x[6], w1, w1_Q); FWD2_const(x[3], x[7], w1, w1_Q);
		const RNS_K w20_Q = w2[0].mul_Q();
		FWD2_const(x[0], x[2], w2[0], w20_Q); FWD2_const(x[1], x[3], w2[0], w20_Q);
		const RNS_K w21_Q = w2[1].mul_Q();
		FWD2_const(x[4], x[6], w2[1], w21_Q); FWD2_const(x[5], x[7], w2[1], w21_Q);
		FWD2(x[0], x[1], w4[0]); FWD2(x[2], x[3], w4[1]); FWD2(x[4], x[5], w4[2]); FWD2(x[6], x[7], w4[3]);
	}

	static void _backward8(RNS_K x[8], const RNS_K w1_i, const RNS_K w2_i[2], const RNS_K w4_i[4])
	{
		BCK2(x[0], x[1], w4_i[0]); BCK2(x[2], x[3], w4_i[1]); BCK2(x[4], x[5], w4_i[2]); BCK2(x[6], x[7], w4_i[3]);
		const RNS_K w20_i_Q = w2_i[0].mul_Q();
		BCK2_const(x[0], x[2], w2_i[0], w20_i_Q); BCK2_const(x[1], x[3], w2_i[0], w20_i_Q);
		const RNS_K w21_i_Q = w2_i[1].mul_Q();
		BCK2_const(x[4], x[6], w2_i[1], w21_i_Q); BCK2_const(x[5], x[7], w2_i[1], w21_i_Q);
		const RNS_K w1_i_Q = w1_i.mul_Q();
		BCK2_const(x[0], x[4], w1_i, w1_i_Q); BCK2_const(x[1], x[5], w1_i, w1_i_Q); BCK2_const(x[2], x[6], w1_i, w1_i_Q); BCK2_const(x[3], x[7], w1_i, w1_i_Q);
	}

	static void _forward8_0(RNS_K x[8], const RNS_K & w1, const RNS_K w2[2], const RNS_K w4[4])
	{
		for (size_t i = 0; i < 4; ++i) x[i] = x[i].toMonty();
		_forward8(x, w1, w2, w4);
	}

	static void _square2x4(RNS_K x[8], const RNS_K w2[2], const RNS_K w2_Q[2])
	{
		SQR2(x[0], x[1], w2[0], w2_Q[0]); SQR2N(x[2], x[3], w2[0], w2_Q[0]);
		SQR2(x[4], x[5], w2[1], w2_Q[1]); SQR2N(x[6], x[7], w2[1], w2_Q[1]);
	}

	static void _square4x2(RNS_K x[8], const RNS_K w2[2], const RNS_K w2_i[2])
	{
		RNS_K w2_Q[2]; w2_Q[0] = w2[0].mul_Q(); w2_Q[1] = w2[1].mul_Q();
		FWD2_const(x[0], x[2], w2[0], w2_Q[0]); FWD2_const(x[1], x[3], w2[0], w2_Q[0]);
		FWD2_const(x[4], x[6], w2[1], w2_Q[1]); FWD2_const(x[5], x[7], w2[1], w2_Q[1]);
		_square2x4(x, w2, w2_Q);
		RNS_K w2_i_Q[2]; w2_i_Q[0] = w2_i[0].mul_Q(); w2_i_Q[1] = w2_i[1].mul_Q();
		BCK2_const(x[0], x[2], w2_i[0], w2_i_Q[0]); BCK2_const(x[1], x[3], w2_i[0], w2_i_Q[0]);
		BCK2_const(x[4], x[6], w2_i[1], w2_i_Q[1]); BCK2_const(x[5], x[7], w2_i[1], w2_i_Q[1]);
	}

	static void _square8(RNS_K x[8], const RNS_K & w1, const RNS_K & w1_i, const RNS_K w2[2], const RNS_K w2_i[2])
	{
		const RNS_K w1_Q = w1.mul_Q();
		FWD2_const(x[0], x[4], w1, w1_Q); FWD2_const(x[1], x[5], w1, w1_Q); FWD2_const(x[2], x[6], w1, w1_Q); FWD2_const(x[3], x[7], w1, w1_Q);
		_square4x2(x, w2, w2_i);
		const RNS_K w1_i_Q = w1_i.mul_Q();
		BCK2_const(x[0], x[4], w1_i, w1_i_Q); BCK2_const(x[1], x[5], w1_i, w1_i_Q); BCK2_const(x[2], x[6], w1_i, w1_i_Q); BCK2_const(x[3], x[7], w1_i, w1_i_Q);
	}

	static void _mul2x4(RNS_K x[8], const RNS_K xp[8], const RNS_K w2[2], const RNS_K w2_Q[2])
	{
		MUL2(x[0], x[1], xp[0], xp[1], w2[0], w2_Q[0]); MUL2N(x[2], x[3], xp[2], xp[3], w2[0], w2_Q[0]);
		MUL2(x[4], x[5], xp[4], xp[5], w2[1], w2_Q[1]); MUL2N(x[6], x[7], xp[6], xp[7], w2[1], w2_Q[1]);
	}

	static void _mul4x2(RNS_K x[8], const RNS_K xp[8], const RNS_K w2[2], const RNS_K w2_i[2])
	{
		RNS_K w2_Q[2]; w2_Q[0] = w2[0].mul_Q(); w2_Q[1] = w2[1].mul_Q();
		FWD2_const(x[0], x[2], w2[0], w2_Q[0]); FWD2_const(x[1], x[3], w2[0], w2_Q[0]);
		FWD2_const(x[4], x[6], w2[1], w2_Q[1]); FWD2_const(x[5], x[7], w2[1], w2_Q[1]);
		_mul2x4(x, xp, w2, w2_Q);
		RNS_K w2_i_Q[2]; w2_i_Q[0] = w2_i[0].mul_Q(); w2_i_Q[1] = w2_i[1].mul_Q();
		BCK2_const(x[0], x[2], w2_i[0], w2_i_Q[0]); BCK2_const(x[1], x[3], w2_i[0], w2_i_Q[0]);
		BCK2_const(x[4], x[6], w2_i[1], w2_i_Q[1]); BCK2_const(x[5], x[7], w2_i[1], w2_i_Q[1]);
	}

	static void _mul8(RNS_K x[8], const RNS_K xp[8], const RNS_K & w1, const RNS_K & w1_i, const RNS_K w2[2], const RNS_K w2_i[2])
	{
		const RNS_K w1_Q = w1.mul_Q();
		FWD2_const(x[0], x[4], w1, w1_Q); FWD2_const(x[1], x[5], w1, w1_Q); FWD2_const(x[2], x[6], w1, w1_Q); FWD2_const(x[3], x[7], w1, w1_Q);
		_mul4x2(x, xp, w2, w2_i);
		const RNS_K w1_i_Q = w1_i.mul_Q();
		BCK2_const(x[0], x[4], w1_i, w1_i_Q); BCK2_const(x[1], x[5], w1_i, w1_i_Q); BCK2_const(x[2], x[6], w1_i, w1_i_Q); BCK2_const(x[3], x[7], w1_i, w1_i_Q);
	}

	static void _fwd4x2(RNS_K x[8], const RNS_K w2[2])
	{
		RNS_K w2_Q[2]; w2_Q[0] = w2[0].mul_Q(); w2_Q[1] = w2[1].mul_Q();
		FWD2_const(x[0], x[2], w2[0], w2_Q[0]); FWD2_const(x[1], x[3], w2[0], w2_Q[0]);
		FWD2_const(x[4], x[6], w2[1], w2_Q[1]); FWD2_const(x[5], x[7], w2[1], w2_Q[1]);
	}

	static void _fwd8(RNS_K x[8], const RNS_K & w1, const RNS_K w2[2])
	{
		const RNS_K w1_Q = w1.mul_Q();
		FWD2_const(x[0], x[4], w1, w1_Q); FWD2_const(x[1], x[5], w1, w1_Q); FWD2_const(x[2], x[6], w1, w1_Q); FWD2_const(x[3], x[7], w1, w1_Q);
		_fwd4x2(x, w2);
	}

private:
	static void forward8(RNS_K * const x, const RNS_K * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const RNS_K w1 = w[1 * (s + j)];
			RNS_K w2[2]; _load(2, w2, &w[2 * (s + j)], 1);
			RNS_K w4[4]; _load(4, w4, &w[4 * (s + j)], 1);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 8 * m * j + i;
				RNS_K zl[8]; _load(8, zl, &x[k], m);
				_forward8(zl, w1, w2, w4);
				_store(8, &x[k], m, zl);
			}
		}
	}

	static void backward8(RNS_K * const z, const RNS_K * const w_i, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const RNS_K w1_i = w_i[1 * (s + j)];
			RNS_K w2_i[2]; _load(2, w2_i, &w_i[2 * (s + j)], 1);
			RNS_K w4_i[4]; _load(4, w4_i, &w_i[4 * (s + j)], 1);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 8 * m * j + i;
				RNS_K zl[8]; _load(8, zl, &z[k], m);
				_backward8(zl, w1_i, w2_i, w4_i);
				_store(8, &z[k], m, zl);
			}
		}
	}

	static void forward8_0(RNS_K * const x, const RNS_K * const w, const size_t n_8)
	{
		// Montgomery form of MF of I to convert input into MF
		const RNS_K w1 = w[0];
		RNS_K w2[2]; _load(2, w2, &w[2], 1);
		RNS_K w4[4]; _load(4, w4, &w[4], 1);

		for (size_t i = 0; i < n_8; ++i)
		{
			const size_t k = i;
			RNS_K xl[8]; _load(8, xl, &x[k], n_8);
			_forward8_0(xl, w1, w2, w4);
			_store(8, &x[k], n_8, xl);
		}
	}

	static void square2x4(RNS_K * const x, const RNS_K * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			RNS_K w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS_K w2_Q[2]; w2_Q[0] = w2[0].mul_Q(); w2_Q[1] = w2[1].mul_Q();

			RNS_K xl[8]; _load(8, xl, &x[8 * j], 1);
			_square2x4(xl, w2, w2_Q);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void square4x2(RNS_K * const x, const RNS_K * const w, const RNS_K * const w_i, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			RNS_K w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS_K w2_i[2]; _load(2, w2_i, &w_i[2 * (n_8 + j)], 1);

			RNS_K xl[8]; _load(8, xl, &x[8 * j], 1);
			_square4x2(xl, w2, w2_i);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void square8(RNS_K * const x, const RNS_K * const w, const RNS_K * const w_i, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const RNS_K w1 = w[1 * (n_8 + j)];
			const RNS_K w1_i = w_i[1 * (n_8 + j)];
			RNS_K w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS_K w2_i[2]; _load(2, w2_i, &w_i[2 * (n_8 + j)], 1);

			RNS_K xl[8]; _load(8, xl, &x[8 * j], 1);
			_square8(xl, w1, w1_i, w2, w2_i);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void mul2x4(RNS_K * const x, const RNS_K * const xp, const RNS_K * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			RNS_K w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS_K w2_Q[2]; w2_Q[0] = w2[0].mul_Q(); w2_Q[1] = w2[1].mul_Q();

			RNS_K zl[8]; _load(8, zl, &x[8 * j], 1);
			RNS_K zpl[8]; _load(8, zpl, &xp[8 * j], 1);
			_mul2x4(zl, zpl, w2, w2_Q);
			_store(8, &x[8 * j], 1, zl);
		}
	}

	static void mul4x2(RNS_K * const x, const RNS_K * const xp, const RNS_K * const w, const RNS_K * const w_i, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			RNS_K w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS_K w2_i[2]; _load(2, w2_i, &w_i[2 * (n_8 + j)], 1);

			RNS_K xl[8]; _load(8, xl, &x[8 * j], 1);
			RNS_K xpl[8]; _load(8, xpl, &xp[8 * j], 1);
			_mul4x2(xl, xpl, w2, w2_i);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void mul8(RNS_K * const x, const RNS_K * const xp, const RNS_K * const w, const RNS_K * const w_i, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const RNS_K w1 = w[1 * (n_8 + j)];
			const RNS_K w1_i = w_i[1 * (n_8 + j)];
			RNS_K w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS_K w2_i[2]; _load(2, w2_i, &w_i[2 * (n_8 + j)], 1);

			RNS_K xl[8]; _load(8, xl, &x[8 * j], 1);
			RNS_K xpl[8]; _load(8, xpl, &xp[8 * j], 1);
			_mul8(xl, xpl, w1, w1_i, w2, w2_i);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void fwd4x2(RNS_K * const x, const RNS_K * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			RNS_K w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);

			RNS_K xl[8]; _load(8, xl, &x[8 * j], 1);
			_fwd4x2(xl, w2);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void fwd8(RNS_K * const x, const RNS_K * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const RNS_K w1 = w[1 * (n_8 + j)];
			RNS_K w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);

			RNS_K xl[8]; _load(8, xl, &x[8 * j], 1);
			_fwd8(xl, w1, w2);
			_store(8, &x[8 * j], 1, xl);
		}
	}

private:
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	// modulo x^n - 1
	void roots_R(RNS_K * const w, RNS_K * const wi)
	{
		const size_t n = size_t(1) << _ln;

		RNS_K sr_s = RNS_K::primroot_n(_ln), sr_si = sr_s.invert();	// (-1)^{1/s}
		for (size_t s = n / 2; s >= 1; s /= 2)
		{
			RNS_K t_j = RNS_K::one(), ti_j = RNS_K::one();
			for (size_t j = 0; j < s; ++j)
			{
				const size_t jr = bitrev(j, s);
				w[s + jr] = t_j; wi[s + jr] = ti_j;
				t_j *= sr_s; ti_j *= sr_si;
			}

			sr_s *= sr_s; sr_si *= sr_si;
		}

		w[0] = w[1].toMonty();
	}

	// modulo x^n + 1
	void roots_Q(RNS_K * const w, RNS_K * const wi)
	{
		const size_t n = size_t(1) << _ln;

		RNS_K sr_2s = RNS_K::primroot_n(_ln + 1), sr_2si = sr_2s.invert();	// (-1)^{1/2s}
		for (size_t s = n / 2; s >= 1; s /= 2)
		{
			const RNS_K sr_s = sr_2s * sr_2s, sr_si = sr_2si * sr_2si;	// (-1)^{1/s}

			RNS_K t_j = sr_2s, ti_j = sr_2si;
			for (size_t j = 0; j < s; ++j)
			{
				const size_t jr = bitrev(j, s);
				w[s + jr] = t_j; wi[s + jr] = ti_j;
				t_j *= sr_s; ti_j *= sr_si;
			}

			sr_2s = sr_s; sr_2si = sr_si;
		}

		w[0] = w[1].toMonty();
	}

	void set(RNS_K * const y, const RNS_K * const x)
	{
		if (x == y) return;

		const size_t n = size_t(1) << _ln;
		for (size_t k = 0; k < n; ++k) y[k] = x[k];
	}

	static constexpr mp_limb_t read_limb(const mp_limb_t * const x_ptr, const size_t x_size, const size_t i)
	{
		return (i < x_size) ? x_ptr[i] : 0;
	}

	static constexpr uint_32 read_word(const mp_limb_t * const x_ptr, const size_t x_size, const size_t bit_index)
	{
		const size_t limb_index = bit_index / 64;
		const int limb_shift = int(bit_index % 64);
		const mp_limb_t rl = read_limb(x_ptr, x_size, limb_index), rh = read_limb(x_ptr, x_size, limb_index + 1);
		mp_limb_t r = rl >> limb_shift;
		if (limb_shift != 0) r |= rh << (64 - limb_shift);
		return uint_32(r);
	}

	static constexpr void write_word(mp_limb_t * const x_ptr, const size_t bit_index, const uint_64 r)
	{
		const size_t limb_index = bit_index / 64;
		const int limb_shift = int(bit_index % 64);
		x_ptr[limb_index] |= mp_limb_t(r) << limb_shift;
		if (limb_shift != 0) x_ptr[limb_index + 1] |= mp_limb_t(r) >> (64 - limb_shift);
	}

	void set_mpz(RNS_K * const y, const mpz_t & x)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const uint_32 base = 1u << lbase;
		const size_t pck_factor = _pck_factor;
		uint_32 * const buf = _buf;

		const size_t x_size = mpz_size(x);
		const mp_limb_t * const x_ptr = mpz_limbs_read(x);

		size_t bit_index = 0;
		for (size_t i = 0; i < n; ++i)
		{
			for (size_t l = 0; l < pck_factor; ++l)
			{
				buf[l * n + i] = read_word(x_ptr, x_size, bit_index) & (base - 1);
				bit_index += size_t(lbase);
			}
		}

		const RNS_K rbase = RNS_K(1u << lbase).toMonty();
		for (size_t i = 0; i < n; ++i)
		{
			size_t j = pck_factor - 1;
			RNS_K r = RNS_K(buf[j * n + i]); while (j > 0) {--j; r = r.encode(buf[j * n + i], rbase); }
			y[i] = r;
		}
	}

	void get_mpz(mpz_t & y, const RNS_K * const x)
	{
		const size_t n = size_t(1) << _ln;
		const size_t pck_factor = _pck_factor;
		const int lbase = _lbase, lbase_pck = int(pck_factor) * lbase;
		const uint_32 base = 1u << lbase;
		uint_32 * const buf = _buf;

		for (size_t l = 0; l < pck_factor; ++l)
		{
			for (size_t i = 0; i < n; ++i) buf[l * n + i] = x[i].get_r(l);
		}

		const size_t y_size = (size_t(lbase_pck) << _ln) / 64 + 2;
		mp_limb_t * const y_ptr = mpz_limbs_write(y, mp_size_t(y_size));
		for (size_t i = 0; i < y_size; ++i) y_ptr[i] = 0;

		long fr = 0;
		if (pck_factor == 2)
		{
			const uint_64 base2 = base * uint_64(base);
			RNS_S::small_type f = 0;
			size_t bit_index = 0;
			for (size_t i = 0; i < n; ++i)
			{
				uint_32 u[PCK_FACTOR]; for (size_t l = 0; l < PCK_FACTOR; ++l) u[l] = buf[l * n + i];
				f += RNS_S(u).getS_i();

				const uint_64 r = (uint_64)(f) & (base2 - 1);
				f >>= 2 * lbase;

				write_word(y_ptr, bit_index, r);
				bit_index += size_t(2 * lbase);
			}
			fr = long(f);
		}
		else if (pck_factor == 4)
		{
			const uint_64 base2 = base * uint_64(base);
			RNS_S::small_type f = 0;
			size_t bit_index = 0;
			for (size_t i = 0; i < n; ++i)
			{
				uint_32 u[PCK_FACTOR]; for (size_t l = 0; l < PCK_FACTOR; ++l) u[l] = buf[l * n + i];
				f += RNS_S(u).getS_i();

				const uint_64 r0 = (uint_64)(f) & (base2 - 1);
				f >>= 2 * lbase;
				const uint_64 r1 = (uint_64)(f) & (base2 - 1);
				f >>= 2 * lbase;

				write_word(y_ptr, bit_index, r0);
				bit_index += size_t(2 * lbase);
				write_word(y_ptr, bit_index, r1);
				bit_index += size_t(2 * lbase);
			}
			fr = long(f);
		}

		mp_size_t s = mp_size_t(y_size);
		while ((s > 0) && (y_ptr[s - 1] == 0)) --s;
 		mpz_limbs_finish(y, s);

		if (fr != 0)
		{
			std::cout << "warning" << std::endl;
			mpz_t t; mpz_init_set_si(t, fr);
			mpz_mul_2exp(t, t, mp_bitcnt_t(lbase_pck) << _ln);
			mpz_add(y, y, t);
			mpz_clear(t);
		}
	}

	size_t NTT_forward8(RNS_K * const x, const RNS_K * const w)
	{
		const size_t n = size_t(1) << _ln;

		forward8_0(x, w, n / 8);
		size_t m = n / 8 / 8, s = 8;
		for (; m > 1; m /= 8, s *= 8) forward8(x, w, m, s);
		return s;
	}

	void NTT_inverse8(RNS_K * const x, const RNS_K * const w_i, const size_t s0)
	{
		const size_t n = size_t(1) << _ln;

		for (size_t s = s0 / 8, m = n / 8 / s; m <= n / 8; m *= 8, s /= 8) backward8(x, w_i, m, s);
	}

	size_t NTT_forward(RNS_K * const x, const RNS_K * const w)
	{
		const size_t s0 = NTT_forward8(x, w);

		const size_t n = size_t(1) << _ln;
		if (s0 == n / 8) fwd8(x, w, n / 8);
		else if (s0 == n / 4) fwd4x2(x, w, n / 8);

		return s0;
	}

	void square(RNS_K * const x, const RNS_K * const w, const RNS_K * const w_i, const size_t s0)
	{
		const size_t n = size_t(1) << _ln;

		if (s0 == n / 8) square8(x, w, w_i, n / 8);
		else if (s0 == n / 4) square4x2(x, w, w_i, n / 8);
		else if (s0 == n / 2) square2x4(x, w, n / 8);
	}

	void mul(RNS_K * const y, const RNS_K * const x, const RNS_K * const w, const RNS_K * const w_i, const size_t s0)
	{
		const size_t n = size_t(1) << _ln;

		if (s0 == n / 8) mul8(y, x, w, w_i, n / 8);
		else if (s0 == n / 4) mul4x2(y, x, w, w_i, n / 8);
		else if (s0 == n / 2) mul2x4(y, x, w, n / 8);
	}

#if (PCK_FACTOR == 2)

	pck_type extract_shift(big_type * const f, const int lbase, const uint_32 base, const uint_64 base2)
	{
		const uint_64 r01 = (*f).l[0] & (base2 - 1);
		pck_type r; r.r[0] = (uint_32)(r01) & (base - 1); r.r[1] = (uint_32)(r01 >> lbase);
		(*f) = big_rshift((*f), 2 * lbase);
		return r;
	}

	RNS_K extract_encode(big_type * const f, const int lbase, const uint_32 base, const uint_64 base2, const RNS_K & rbase)
	{
		const pck_type r = extract_shift(f, lbase, base, base2);
		int_32 r1 = (int_32)(r.r[1]);
		if (r1 >= (int_32)(base / 2)) { r1 -= (int_32)(base); (*f) = big_inc((*f)); }	// balanced
		const RNS_K x = RNS_K().set_i32(r1);
		return x.encode(r.r[0], rbase);
	}

	RNS_K extract_encode_last(const big_type & f, const int lbase, const uint_32 base, const uint_64, const RNS_K & rbase)
	{
		const int_64 r01 = (int_64)(f.l[0]);
		const uint_32 r0 = (uint_32)(r01) & (base - 1); const int_32 r1 = (int_32)(r01 >> lbase);
		const RNS_K x = RNS_K().set_i32(r1);
		return x.encode(r0, rbase);
	}

#elif (PCK_FACTOR == 4)

	pck_type extract_shift(big_type * const f, const int lbase, const uint_32 base, const uint_64 base2)
	{
		const uint_64 r01 = (*f).l[0] & (base2 - 1);
		pck_type r; r.r[0] = (uint_32)(r01) & (base - 1); r.r[1] = (uint_32)(r01 >> lbase);
		(*f) = big_rshift((*f), 2 * lbase);
		const uint_64 r23 = (*f).l[0] & (base2 - 1);
		r.r[2] = (uint_32)(r23) & (base - 1); r.r[3] = (uint_32)(r23 >> lbase);
		(*f) = big_rshift((*f), 2 * lbase);
		return r;
	}

	RNS_K extract_encode(big_type * const f, const int lbase, const uint_32 base, const uint_64 base2, const RNS_K & rbase)
	{
		const pck_type r = extract_shift(f, lbase, base, base2);
		int_32 r3 = (int_32)(r.r[3]);
		if (r3 >= (int_32)(base / 2)) { r3 -= (int_32)(base); (*f) = big_inc((*f)); }	// balanced
		RNS_K x = RNS_K().set_i32(r3); x = x.encode(r.r[2], rbase); x = x.encode(r.r[1], rbase);
		return x.encode(r.r[0], rbase);
	}

	RNS_K extract_encode_last(const big_type & f, const int lbase, const uint_32 base, const uint_64 base2, const RNS_K & rbase)
	{
		const uint_64 r01 = f.l[0] & (base2 - 1);
		const uint_32 r0 = (uint_32)(r01) & (base - 1), r1 = (uint_32)(r01 >> lbase);
		const big_type f23 = big_rshift(f, 2 * lbase);
		const int_64 r23 = (int_64)(f23.l[0]);
		const uint_32 r2 = (uint_32)(r23) & (base - 1); const int_32 r3 = (int_32)(r23 >> lbase);
		RNS_K x = RNS_K().set_i32(r3); x = x.encode(r2, rbase); x = x.encode(r1, rbase);
		return x.encode(r0, rbase);
	}

#endif

	void carry_mul_sub(RNS_K * const x, const RNS_K * const y, big_type * const carry, const uint_32 a)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const RNS_K & norm = _norm;
		const uint_32 base = 1u << lbase;
		const uint_64 base2 = (uint_64)(1) << (2 * lbase);
		// const RNS_K rbase = RNS_K(base).toMonty();
	
		for (size_t i = 0; i < n / CARRY_LENGTH; ++i)
		{
			big_type f = big_zero();
			for (size_t j = 0; j < CARRY_LENGTH; ++j)
			{
				const size_t k = i * CARRY_LENGTH + j;
				// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF
				f = big_add(f, big_mul_u32((x[k] * norm).getK_i(), a));
				if (y != nullptr) f = big_sub(f, y[k].getK_i());	// TODO getS_i
				const pck_type r = extract_shift(&f, lbase, base, base2);
				x[k] = RNS_K(r);
			}

			carry[i] = f;
		}
	}

	void carry_complete(RNS_K * const x, big_type * const carry, const bool nega)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const uint_32 base = 1u << lbase;
		const uint_64 base2 = (uint_64)(1) << (2 * lbase);
		const RNS_K rbase = RNS_K(base).toMonty();

		for (size_t i = 0; i < n / CARRY_LENGTH; ++i)
		{
			big_type f = carry[((i == 0) ? (n / CARRY_LENGTH) : i) - 1];
			if ((i == 0) && nega) f = big_neg(f);

			for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)
			{
				const size_t k = i * CARRY_LENGTH + j;
				f = big_add_pck(f, x[k].get_pck(), lbase);	// x[k] >= 0
				x[k] = extract_encode(&f, lbase, base, base2, rbase);
			}

			const size_t k = i * CARRY_LENGTH + CARRY_LENGTH - 1;
			const pck_type r = x[k].get_pck();
			f = big_add_pck(f, r, lbase);	// x[k] >= 0
			x[k] = extract_encode_last(f, lbase, base, base2, rbase);
		}
	}

	// b-1, b-1, ..., b-1, b + a, ... => b, 0, ..., 0, a, ... 
	void fix_base(RNS_K * const x, const char * const who)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const size_t pck_factor = _pck_factor;
		const RNS_S::small_type base_pck = RNS_S::small_type(1) << (int(pck_factor) * lbase);
		const RNS_K base = RNS_K(1u << lbase), rbase = base.toMonty();
		RNS_K rbase_pck = base; for (size_t l = 1; l < pck_factor; ++l) rbase_pck *= rbase;
		const RNS_K rbase_pck_m1 = rbase_pck - RNS_K(1);

#if defined(CHECK_THEOREM)
		static size_t i_max = 2;
#else
		(void)who;
#endif
		if (x[n - 1] == rbase_pck_m1)
		{
			size_t i = 2; for (; i < n; ++i) { if (x[n - i] != rbase_pck_m1) break; }
#if defined(CHECK_THEOREM)
			if (i > i_max) { i_max = i; std::cout << "fix_base " << who << ": i = " << i << " / " << n << std::endl; }
#endif
			uint_32 hi_i[PCK_FACTOR]; for (size_t l = 0; l < PCK_FACTOR; ++l) hi_i[l] = x[n - i].get_r(l);
			if (RNS_S(hi_i).getS_i() >= base_pck)	// >= base^n
			{
				x[n - i] -= rbase_pck;
				for (size_t j = i - 1; j >= 2; --j) x[n - j] = RNS_K(0u);
				x[n - 1] = rbase_pck;
			}
		}
	}

	void lower(RNS_K * const x, const int_32 l0)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const size_t pck_factor = _pck_factor;
		const RNS_K base = RNS_K(1u << lbase), rbase = base.toMonty();
		RNS_K rbase_pck = base; for (size_t l = 1; l < pck_factor; ++l) rbase_pck *= rbase;

		if (big_ge_pow_2_s(x[n - 1].getK_i(), int(pck_factor) * lbase))	// x >= base^n => x[N_SZ - 1] >= BASE^PCK_FACTOR
		{
			// - R or Q
			x[n - 1] -= rbase_pck;
			x[0] -= RNS_K().set_i32(l0);
		}
	}

	void positive(RNS_K * const x, const int_32 l0)
	{
		const size_t n = size_t(1) << _ln;
		const RNS_K base = RNS_K(1u << _lbase), rbase = base.toMonty();
		const size_t pck_factor = _pck_factor;
		RNS_K rbase_pck = base; for (size_t l = 1; l < pck_factor; ++l) rbase_pck *= rbase;

		for (size_t i = 0; i < n; ++i)
		{
			uint_32 hi_i[PCK_FACTOR]; for (size_t l = 0; l < PCK_FACTOR; ++l) hi_i[l] = x[n - 1 - i].get_r(l);
			const int_64 u = RNS_S(hi_i).getS_i();
			if (u > 0) return;
			if (u < 0)
			{
				// + R or Q
				x[n - 1] += rbase_pck;
				x[0] += RNS_K().set_i32(l0);
				return;
			}
		}
	}

	void mod_Q(RNS_K * const x)
	{
#if !defined(BALANCED)
		fix_base(x, "Q");
#endif
		lower(x, 1);	// < Q - 1
#if !defined(BALANCED)
		positive(x, 1);	// >= 0
#endif

#if defined(CHECK_THEOREM)
		const size_t n = size_t(1) << _ln;
		const size_t pck_factor = _pck_factor;

		double log_q = 0;
		for (size_t i = 0; i < n; ++i)
		{
			uint_32 hi_i[pck_factor]; for (size_t l = 0; l < pck_factor; ++l) hi_i[l] = x[n - 1 - i].get_r(l);
			const int_64 u = RNS_S(hi_i).getS_i();
			if (u != 0)
			{
#if !defined(BALANCED)
				if (u < 0) throw std::runtime_error("Q_min < 0");
#endif
				mpz_t t; mpz_init(t);
				get_mpz(t, x);
				if (mpz_sgn(t) < 0) throw std::runtime_error("Q_min < 0");
				long ex; const double mx = mpz_get_d_2exp(&ex, t);
				log_q = std::log2(mx) + ex;
				mpz_clear(t);
				break;
			}
		}
		if (log_q < _logq_min) _logq_min = log_q;
		if (log_q > _logq_max)
		{
			_logq_max = log_q;
			const double q_max = pow(2.0, _logq_max - _logN);
			if (q_max > 4)
			{
				std::ostringstream ss; ss << "Q_max / N = " << q_max << " = 2^" << _logq_max - _logN;
				throw std::runtime_error(ss.str());
			}
		}
#endif
	}

	void mod_R_add2(RNS_K * const x)
	{
#if !defined(BALANCED)
		fix_base(x, "R");
#endif
		lower(x, -1);		// <= R
#if !defined(BALANCED)
		positive(x, -1);	// >= 0
#endif

		x[0] += RNS_K(2);
	}

	// Modular arithmetic

	void square_mul_mod_R(RNS_K * const y, const RNS_K * const x, const uint_32 a)
	{
		set(y, x); const size_t s0 = NTT_forward8(y, _w_R);
		square(y, _w_R, _wi_R, s0);
		NTT_inverse8(y, _wi_R, s0);
		carry_mul_sub(y, nullptr, _carry, a);
		carry_complete(y, _carry, false);
	}

	void square_mul_mod_Q(RNS_K * const y, const RNS_K * const x, const uint_32 a)
	{
		set(y, x); const size_t s0 = NTT_forward8(y, _w_Q);
		square(y, _w_Q, _wi_Q, s0);
		NTT_inverse8(y, _wi_Q, s0);
		carry_mul_sub(y, nullptr, _carry, a);
		carry_complete(y, _carry, true);
	}

	void mul_mod_R(RNS_K * const y, const RNS_K * const xt)
	{
		const size_t s0 = NTT_forward8(y, _w_R);
		mul(y, xt, _w_R, _wi_R, s0);
		NTT_inverse8(y, _wi_R, s0);
 		carry_mul_sub(y, nullptr, _carry, 1);
		carry_complete(y, _carry, false);
	}

	void mul_mod_Q(RNS_K * const y, const RNS_K * const xt)
	{
		const size_t s0 = NTT_forward8(y, _w_Q);
		mul(y, xt, _w_Q, _wi_Q, s0);
		NTT_inverse8(y, _wi_Q, s0);
 		carry_mul_sub(y, nullptr, _carry, 1);
		carry_complete(y, _carry, true);
	}

	void mul_sub_mod_Q(RNS_K * const y, const RNS_K * const xt, const RNS_K * const z)
	{
		const size_t s0 = NTT_forward8(y, _w_Q);
		mul(y, xt, _w_Q, _wi_Q, s0);
		NTT_inverse8(y, _wi_Q, s0);
 		carry_mul_sub(y, z, _carry, 1);
		carry_complete(y, _carry, true);
	}

	// Montgomery-McLaughlin-Gallot-Woltman multiplication

	void reduceMMGW(RNS_K * const x)
	{
		mul_mod_R(x, _Ninv_R);	// m'
		mod_R_add2(x);

		mul_sub_mod_Q(x, _N_Q, _t_Q);
		mod_Q(x);
	}

	void mulMMGW(RNS_K * const x, const RNS_K * const y)
	{
		set(_t, y); NTT_forward(_t, _w_Q);
		set(_t_Q, x); mul_mod_Q(_t_Q, _t);
		set(_t, y); NTT_forward(_t, _w_R);
		mul_mod_R(x, _t);
		reduceMMGW(x);
	}

	void squareMMGW(RNS_K * const x, const uint_32 a)
	{
		square_mul_mod_Q(_t_Q, x, a);
		square_mul_mod_R(x, x, a);
		reduceMMGW(x);
	}

public:
	void set1() override
	{
		set(_reg[Reg0], _one);
	}

	void get(mpz_t & z, const size_t src) override
	{
		// Conversion out of Montgomery-McLaughlin-Gallot-Woltman form
		set(_t_Q, _reg[src]);
		set(_t, _reg[src]);
		reduceMMGW(_t);
		get_mpz(z, _t);

		// 0 <= z < mod
		mpz_mod(z, z, _Nz);
	}

	void copy(const size_t dst, const size_t src) override
	{
		set(_reg[dst], _reg[src]);
	}

	void square_mul(const int32_t a) override
	{
		squareMMGW(_reg[Reg0], uint_32(a));
	}

	void mul(const size_t dst, const size_t src) override
	{
		mulMMGW(_reg[dst], _reg[src]);
	}

	bool init_failed() const override { return _init_failed; }
	void get_factor(mpz_t & z) const override { mpz_set(z, _Nz); }

	bool has_checkpoint() const override { return true; }

	bool read_checkpoint(file & cfile) override
	{
		const size_t n = size_t(1) << _ln;
		const size_t prm_count = _prm_count;
		uint_32 * const buf = _buf;

		for (size_t j = 0; j < 2; ++j)
		{
			if (!cfile.read(reinterpret_cast<char *>(buf), prm_count * n * sizeof(uint_32))) return false;
			RNS_K * const reg = _reg[j];
			for (size_t i = 0; i < n; ++i)
			{
				uint_32 u[PRM_COUNT]; for (size_t k = 0; k < PRM_COUNT; ++k) u[k] = buf[k * n + i];
				reg[i] = RNS_K(u);
			}
		}

		return true;
	}

	void save_checkpoint(file & cfile) override
	{
		const size_t n = size_t(1) << _ln;
		const size_t prm_count = _prm_count;
		uint_32 * const buf = _buf;

		for (size_t j = 0; j < 2; ++j)
		{
			const RNS_K * const reg = _reg[j];
			for (size_t i = 0; i < n; ++i)
			{
				const RNS_K & r = reg[i];
				for (size_t k = 0; k < prm_count; ++k) buf[k * n + i] = r.get_r(k);
			}
			if (!cfile.write(reinterpret_cast<const char *>(buf), prm_count * n * sizeof(uint_32))) break;
		}
	}

	static void get_n_base(int & rln, int & rlbase, const int pck_factor, const double log2_16aN, const double log2_p_2a)
	{
		for (int lbase = 31; lbase >= 2; --lbase)
		{
			// log2(base^n) = lbase * n
			int ln = 4; while (double(lbase * pck_factor) * (1 << ln) < log2_16aN) ++ln;

			// A necessary condition is a * n * (base - 1)^2 < Prod_i(Pi) / 2
			const int log2_base = lbase * pck_factor;
			if (ln + 2 * log2_base < log2_p_2a)
			{
				rln = ln;
				rlbase = lbase;
				break;
			}
		}
	}

	static constexpr size_t get_s(const size_t K)
	{
		int s = -1; for (size_t k = K - 1; k != 0; k /= 2) ++s;
		return size_t(1) << s;
	}

public:
	transformCPU(const mpz_t & N, const int32_t a) : transform()
	{
		_prm_count = PRM_COUNT; _pck_factor = PCK_FACTOR;

		// R = base^n - 1, Q = base^n + 1 such that base^n > 16 * a * N and n is a power of two
		long e_n; const double m_n = mpz_get_d_2exp(&e_n, N);
		const double log2_16aN = std::log2(16.0 * a) + std::log2(m_n) + e_n;

		double log2_p_2a = std::log2(double(ZP0::P())) + std::log2(double(ZP1::P())) + std::log2(double(ZP2::P())) + std::log2(double(ZP3::P())) - std::log2(2.0 * a);
		if (PRM_COUNT == 5) log2_p_2a += std::log2(double(ZP4::P()));
		int ln = 0; get_n_base(ln, _lbase, _pck_factor, log2_16aN, log2_p_2a);
		_ln = ln;

		const size_t n = size_t(1) << ln;
		const mp_bitcnt_t lbase = mp_bitcnt_t(_lbase), lbase_pck = mp_bitcnt_t(_pck_factor) * mp_bitcnt_t(lbase);
		_norm = RNS_K::norm(ln - 1);

		_buf = new uint32_t[_prm_count * n];

		_N_Q = new RNS_K[n]; _Ninv_R = new RNS_K[n]; _one = new RNS_K[n];
		_t_Q = new RNS_K[n];  _t = new RNS_K[n];
		for (size_t i = 0; i < reg_count; ++i) _reg[i] = new RNS_K[n];

		_w_R = new RNS_K[n]; _wi_R = new RNS_K[n];
		roots_R(_w_R, _wi_R);
		_w_Q = new RNS_K[n]; _wi_Q = new RNS_K[n];
		roots_Q(_w_Q, _wi_Q);

		_carry = new big_type[n / CARRY_LENGTH];

		mpz_t R, t; mpz_inits(_Nz, R, t, nullptr);

#if defined(CHECK_THEOREM)
		long eN; const double mN = mpz_get_d_2exp(&eN, N);
		_logN = std::log2(mN) + eN;
#endif
		mpz_set_ui(R, 1); mpz_mul_2exp(R, R, lbase_pck << ln); mpz_sub_ui(R, R, 1);
		if (mpz_invert(t, N, R) == 0)
		{
			mpz_gcd(_Nz, N, R);
			_init_failed = true;
			mpz_clears(R, t, nullptr);
			return;
		}
		else
		{
			mpz_set(_Nz, N);
			set_mpz(_N_Q, N);
			NTT_forward(_N_Q, _w_Q);
			set_mpz(_Ninv_R, t);
			NTT_forward(_Ninv_R, _w_R);
		}
#if defined(CHECK_THEOREM)
		long e_r; const double m_r = mpz_get_d_2exp(&e_r, R);
		std::cout << "N = " << m_n << " * 2^" << e_n << ", R = 2^" << e_r << " - 1, N/R = " << m_n / m_r << " * 2^" << e_n - e_r << std::endl;
#endif
		// The Montgomery-McLaughlin-Gallot-Woltman form of x is R/2 * x mod N
		mpz_mod(t, R, N);
		if (mpz_odd_p(t) != 0) mpz_add(t, t, N);
		mpz_div_2exp(t, t, 1);
		set_mpz(_one, t);

		mpz_clears(R, t, nullptr);
	}

	virtual ~transformCPU()
	{
#if defined(CHECK_THEOREM)
		const double q_min = pow(2.0, _logq_min - _logN), q_max = pow(2.0, _logq_max - _logN);
		std::cout << "Q_min / N = " << q_min << ", Q_max / N = " << q_max << std::endl;
#endif
		mpz_clear(_Nz);

		delete[] _N_Q; delete[] _Ninv_R; delete[] _one;
		delete[] _t_Q; delete[] _t;
		for (size_t i = 0; i < reg_count; ++i) delete[] _reg[i];

		delete[] _w_R; delete[] _wi_R;
		delete[] _w_Q; delete[] _wi_Q;

		delete[] _carry;
		delete[] _buf;
	}

	std::string get_type() const override { return ""; }
	std::string get_data_size() const override { return ""; }
	std::string get_cache_size() const override { return ""; }

	std::string get_info_short() const override
	{
		const size_t n = size_t(1) << _ln;
		std::ostringstream ss; ss << _prm_count << "-prime NTT of length " << n << ", packed-base = 2^{" << _pck_factor << "x" << _lbase << "}";
		return (ss.str());
	}

	std::string get_info() const override
	{
		std::ostringstream ss; ss << "Montgomery-McLaughlin-Gallot-Woltman reduction of " << get_info_short() << std::endl;
		return (ss.str());
	}
};
