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

typedef uint32_t	uint_32;
typedef int32_t		int_32;
typedef uint64_t	uint_64;
typedef int64_t		int_64;

static uint_32 mul_hi(const uint_32 x, const uint_32 y) { return uint_32((x * uint_64(y)) >> 32); }
static uint_64 mul_hi(const uint_64 x, const uint_64 y) { return uint_64((x * __int128_t(y)) >> 64); }

// 128-bit arithmetic

typedef struct { uint_64 s0; int_64 s1; } int_128;

static int_128 int128_set(const uint_64 l, const int_64 h) { int_128 r; r.s0 = l; r.s1 = h; return r; }

static int_128 int128_neg(const int_128 x) { int_128 r; r.s0 = -x.s0; r.s1 = -x.s1 - ((x.s0 != 0) ? 1 : 0); return r; }

static int_128 int128_add(const int_128 x, const int_128 y)
{
	int_128 r;
	r.s0 = x.s0 + y.s0;
	r.s1 = x.s1 + y.s1 + ((r.s0 < x.s0) ? 1 : 0);
	return r;
}

static int_128 int128_add_u64(const int_128 x, const uint_64 y)
{
	int_128 r;
	r.s0 = x.s0 + y;
	r.s1 = x.s1 + ((r.s0 < x.s0) ? 1 : 0);
	return r;
}

static int_128 int128_sub_i64(const int_128 x, const int_64 y)
{
	int_128 r;
	r.s0 = x.s0 - (uint_64)(y);
	r.s1 = x.s1 - ((y < 0) ? -1 : 0) - ((x.s0 < (uint_64)(y)) ? 1 : 0);
	return r;
}

static int_128 int128_rshift(const int_128 x, const int s)	// s != 0
{
	int_128 r;
	r.s0 = (x.s0 >> s) | (uint_64)(x.s1 << (64 - s));
	r.s1 = x.s1 >> s;
	return r;
}

static int_128 int128_mul_u32(const int_128 x, const uint_32 y)
{
	int_128 r;
	r.s0 = x.s0 * y;
	r.s1 = x.s1 * (int_64)(y) + (int_64)(mul_hi(x.s0, (uint_64)(y)));
	return r;
}

static int_128 int128_mul_u64i64(const uint_64 x, const int_64 y)
{
	int_128 r;
	r.s0 = x * (uint_64)(y);
	r.s1 = (int_64)(mul_hi(x, (uint_64)(y)) - ((y < 0) ? x : 0));
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

	ZPT & set_i(const int_32 i) { _n = (uint_32)(i) + ((i < 0) ? _P : 0); return *this; }

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

using ZP1 = ZPT<(125u << 25) + 1, 3>;
using ZP2 = ZPT<(243u << 24) + 1, 7>;
using ZP3 = ZPT<(235u << 24) + 1, 3>;
using ZP4 = ZPT<( 29u << 27) + 1, 3>;

// The residue number system {P1, P2}
class RNS
{
private:
	ZP1 _r1;
	ZP2 _r2;
	ZP3 _r3;
	ZP4 _r4;

private:
	explicit RNS(const ZP1 & r1, const ZP2 & r2, const ZP3 & r3, const ZP4 & r4) : _r1(r1), _r2(r2), _r3(r3), _r4(r4) {}

	// Chinese remainder theorem

	static int_64 garner2(const ZP1 & r1, const ZP2 & r2)
	{
		static const uint_32 P1 = ZP1::P(), P2 = ZP2::P();
		static const ZP1 INVP2_P1 = ZP1(P2).toMonty().invert();	// P2 < P1

		// convert to mixed-radix representation
		const uint_32 u2 = r2.get();
		const uint_32 u1 = ((r1 - ZP1(u2)) * INVP2_P1).get();
		const int_32 i1 = (u1 > P1 / 2) ? (int_32)(u1 - P1) : (int_32)(u1);

		return (int_64)(u2) + (int_64)(P2) * i1;
	}

	static int_128 garner4(const ZP1 & r1, const ZP2 & r2, const ZP3 & r3, const ZP4 & r4)
	{
		static const uint_32 P1 = ZP1::P(), P2 = ZP2::P(), P3 = ZP3::P(), P4 = ZP4::P();
		// we must have P4 < P3 < P2 < P1
		static const ZP3 INVP4_P3 = ZP3(P4).toMonty().invert();
		static const ZP2 INVP4_P2 = ZP2(P4).toMonty().invert();
		static const ZP2 INVP3_P2 = ZP2(P3).toMonty().invert();
		static const ZP1 INVP4_P1 = ZP1(P4).toMonty().invert();
		static const ZP1 INVP3_P1 = ZP1(P3).toMonty().invert();
		static const ZP1 INVP2_P1 = ZP1(P2).toMonty().invert();
		static const uint_64 P34 = P3 * (uint_64)(P4);

		// convert to mixed-radix representation
		const uint_32 u4 = r4.get();
		const uint_32 u3 = ((r3 - ZP3(u4)) * INVP4_P3).get();
		const uint_32 u2 = (((r2 - ZP2(u4)) * INVP4_P2 - ZP2(u3)) * INVP3_P2).get();
		const uint_32 u1 = ((((r1 - ZP1(u4)) * INVP4_P1 - ZP1(u3)) * INVP3_P1 - ZP1(u2)) * INVP2_P1).get();
		const int_32 i1 = (u1 > P1 / 2) ? (int_32)(u1 - P1) : (int_32)(u1);

		const uint_64 u34 = u4 + (uint_64)(P4) * u3;
		const int_64 i12 = (int_64)(u2) + (int_64)(P2) * i1;
		return int128_add_u64(int128_mul_u64i64(P34, i12), u34);
	}

public:
	RNS() {}
	explicit RNS(const uint_32 n) : _r1(n), _r2(n), _r3(n), _r4(n) {}
	explicit RNS(const uint_32 n1, const uint_32 n2, const uint_32 n3, const uint_32 n4) : _r1(n1), _r2(n2), _r3(n3), _r4(n4) {}

	static RNS one() { return RNS(ZP1::one(), ZP2::one(), ZP3::one(), ZP4::one()); }

	uint_32 get_r1() const { return _r1.get(); }
	uint_32 get_r2() const { return _r2.get(); }
	uint_32 get_r3() const { return _r3.get(); }
	uint_32 get_r4() const { return _r4.get(); }

	int_64 get_i64() const { return garner2(_r1, _r2); }
	int_128 get_i128() const { return garner4(_r1, _r2, _r3, _r4); }

	RNS & set_i(const int_32 i) { _r1.set_i(i); _r2.set_i(i); _r3.set_i(i); _r4.set_i(i); return *this; }

	// Conversion into Montgomery form
	RNS toMonty() const { return RNS(_r1.toMonty(), _r2.toMonty(), _r3.toMonty(), _r4.toMonty()); }

	bool operator==(const RNS & rhs) const { return (_r1 == rhs._r1) && (_r2 == rhs._r2) && (_r3 == rhs._r3) && (_r4 == rhs._r4); }
	bool operator!=(const RNS & rhs) const { return !(*this == rhs); }

	RNS & operator+=(const RNS & rhs) { _r1 += rhs._r1; _r2 += rhs._r2; _r3 += rhs._r3; _r4 += rhs._r4; return *this; }
	RNS & operator-=(const RNS & rhs) { _r1 -= rhs._r1; _r2 -= rhs._r2; _r3 -= rhs._r3; _r4 -= rhs._r4; return *this; }
	RNS & operator*=(const RNS & rhs) { _r1 *= rhs._r1; _r2 *= rhs._r2; _r3 *= rhs._r3; _r4 *= rhs._r4; return *this; }

	RNS operator+(const RNS & rhs) const { RNS r = *this; r += rhs; return r; }
	RNS operator-(const RNS & rhs) const { RNS r = *this; r -= rhs; return r; }
	RNS operator*(const RNS & rhs) const { RNS r = *this; r *= rhs; return r; }

	RNS mul_Q() const { return RNS(_r1.mul_Q(), _r2.mul_Q(), _r3.mul_Q(), _r4.mul_Q()); }
	RNS mul_const(const RNS & rhs, const RNS & rhs_Q) const
	{
		return RNS(_r1.mul_const(rhs._r1, rhs_Q._r1), _r2.mul_const(rhs._r2, rhs_Q._r2),
		           _r3.mul_const(rhs._r3, rhs_Q._r3), _r4.mul_const(rhs._r4, rhs_Q._r4));
	}

	RNS invert() const { return RNS(_r1.invert(), _r2.invert(), _r3.invert(), _r4.invert()); }
	static RNS primroot_n(const int ln) { return RNS(ZP1::primroot_n(ln), ZP2::primroot_n(ln), ZP3::primroot_n(ln), ZP4::primroot_n(ln)); }
	static RNS norm(const int ln) { return RNS(ZP1::norm(ln), ZP2::norm(ln), ZP3::norm(ln), ZP4::norm(ln)); }

	// void display() const { std::cout << "(" << _r1.get() << ", " << _r2.get() << ", " << _r3.get() << ", " << _r4.get() << ")"; }
};

class transformCPU : public transform
{
private:
	static const size_t reg_count = 3;
	int _ln = 0;
	int _lbase = 0;
	RNS _rbase, _rbase2;
	bool _init_failed = false;
	RNS _norm;
	mpz_t _Nz;
	RNS * _N_Q = nullptr; RNS * _Ninv_R = nullptr; RNS * _one = nullptr;
	RNS * _t_Q = nullptr; RNS * _t = nullptr;
	RNS * _reg[reg_count];
	RNS * _w_R; RNS * _wi_R; RNS * _w_Q; RNS * _wi_Q;
	int_128 * _carry;
#if defined(CHECK_THEOREM)
	double _logq_min = 1e100, _logq_max = -1e100, _logN = 0;
#endif

private:
	static void _load(const size_t n, RNS * const xl, const RNS * const x, const size_t s) { for (size_t l = 0; l < n; ++l) xl[l] = x[l * s]; }
	static void _store(const size_t n, RNS * const x, const size_t s, const RNS * const xl) { for (size_t l = 0; l < n; ++l) x[l * s] = xl[l]; }

	static void FWD2(RNS & x0, RNS & x1, const RNS & w) { const RNS t = x1 * w; x1 = x0 - t; x0 = x0 + t; }
	static void BCK2(RNS & x0, RNS & x1, const RNS & w_i) { const RNS t = x0 - x1; x0 = x0 + x1; x1 = t * w_i; }

	static void FWD2_const(RNS & x0, RNS & x1, const RNS & w, const RNS w_Q)
	{
		const RNS t = x1.mul_const(w, w_Q); x1 = x0 - t; x0 = x0 + t;
	}
	static void BCK2_const(RNS & x0, RNS & x1, const RNS & w_i, const RNS & w_i_Q)
	{
		const RNS t = x0 - x1; x0 = x0 + x1; x1 = t.mul_const(w_i, w_i_Q);
	}

	static void SQR2(RNS & x0, RNS & x1, const RNS & w, const RNS w_Q)
	{
		const RNS t = (x1 * x1).mul_const(w, w_Q); x1 = (x0 + x0) * x1; x0 = x0 * x0 + t;
	}
	static void SQR2N(RNS & x0, RNS & x1, const RNS & w, const RNS w_Q)
	{
		const RNS t = (x1 * x1).mul_const(w, w_Q); x1 = (x0 + x0) * x1; x0 = x0 * x0 - t;
	}

	static void MUL2(RNS & x0, RNS & x1, const RNS & xp0, const RNS & xp1, const RNS & w, const RNS w_Q)
	{
		const RNS t = (x1 * xp1).mul_const(w, w_Q); x1 = x0 * xp1 + xp0 * x1; x0 = x0 * xp0 + t;
	}
	static void MUL2N(RNS & x0, RNS & x1, const RNS & xp0, const RNS & xp1, const RNS & w, const RNS w_Q)
	{
		const RNS t = (x1 * xp1).mul_const(w, w_Q); x1 = x0 * xp1 + xp0 * x1; x0 = x0 * xp0 - t;
	}

	static void _forward8(RNS x[8], const RNS & w1, const RNS w2[2], const RNS w4[4])
	{
		const RNS w1_Q = w1.mul_Q();
		FWD2_const(x[0], x[4], w1, w1_Q); FWD2_const(x[1], x[5], w1, w1_Q); FWD2_const(x[2], x[6], w1, w1_Q); FWD2_const(x[3], x[7], w1, w1_Q);
		const RNS w20_Q = w2[0].mul_Q();
		FWD2_const(x[0], x[2], w2[0], w20_Q); FWD2_const(x[1], x[3], w2[0], w20_Q);
		const RNS w21_Q = w2[1].mul_Q();
		FWD2_const(x[4], x[6], w2[1], w21_Q); FWD2_const(x[5], x[7], w2[1], w21_Q);
		FWD2(x[0], x[1], w4[0]); FWD2(x[2], x[3], w4[1]); FWD2(x[4], x[5], w4[2]); FWD2(x[6], x[7], w4[3]);
	}

	static void _backward8(RNS x[8], const RNS w1_i, const RNS w2_i[2], const RNS w4_i[4])
	{
		BCK2(x[0], x[1], w4_i[0]); BCK2(x[2], x[3], w4_i[1]); BCK2(x[4], x[5], w4_i[2]); BCK2(x[6], x[7], w4_i[3]);
		const RNS w20_i_Q = w2_i[0].mul_Q();
		BCK2_const(x[0], x[2], w2_i[0], w20_i_Q); BCK2_const(x[1], x[3], w2_i[0], w20_i_Q);
		const RNS w21_i_Q = w2_i[1].mul_Q();
		BCK2_const(x[4], x[6], w2_i[1], w21_i_Q); BCK2_const(x[5], x[7], w2_i[1], w21_i_Q);
		const RNS w1_i_Q = w1_i.mul_Q();
		BCK2_const(x[0], x[4], w1_i, w1_i_Q); BCK2_const(x[1], x[5], w1_i, w1_i_Q); BCK2_const(x[2], x[6], w1_i, w1_i_Q); BCK2_const(x[3], x[7], w1_i, w1_i_Q);
	}

	static void _forward8_0(RNS x[8], const RNS & w1, const RNS w2[2], const RNS w4[4])
	{
		for (size_t i = 0; i < 4; ++i) x[i] = x[i].toMonty();
		_forward8(x, w1, w2, w4);
	}

	static void _square2x4(RNS x[8], const RNS w2[2], const RNS w2_Q[2])
	{
		SQR2(x[0], x[1], w2[0], w2_Q[0]); SQR2N(x[2], x[3], w2[0], w2_Q[0]);
		SQR2(x[4], x[5], w2[1], w2_Q[1]); SQR2N(x[6], x[7], w2[1], w2_Q[1]);
	}

	static void _square4x2(RNS x[8], const RNS w2[2], const RNS w2_i[2])
	{
		RNS w2_Q[2]; w2_Q[0] = w2[0].mul_Q(); w2_Q[1] = w2[1].mul_Q();
		FWD2_const(x[0], x[2], w2[0], w2_Q[0]); FWD2_const(x[1], x[3], w2[0], w2_Q[0]);
		FWD2_const(x[4], x[6], w2[1], w2_Q[1]); FWD2_const(x[5], x[7], w2[1], w2_Q[1]);
		_square2x4(x, w2, w2_Q);
		RNS w2_i_Q[2]; w2_i_Q[0] = w2_i[0].mul_Q(); w2_i_Q[1] = w2_i[1].mul_Q();
		BCK2_const(x[0], x[2], w2_i[0], w2_i_Q[0]); BCK2_const(x[1], x[3], w2_i[0], w2_i_Q[0]);
		BCK2_const(x[4], x[6], w2_i[1], w2_i_Q[1]); BCK2_const(x[5], x[7], w2_i[1], w2_i_Q[1]);
	}

	static void _square8(RNS x[8], const RNS & w1, const RNS & w1_i, const RNS w2[2], const RNS w2_i[2])
	{
		const RNS w1_Q = w1.mul_Q();
		FWD2_const(x[0], x[4], w1, w1_Q); FWD2_const(x[1], x[5], w1, w1_Q); FWD2_const(x[2], x[6], w1, w1_Q); FWD2_const(x[3], x[7], w1, w1_Q);
		_square4x2(x, w2, w2_i);
		const RNS w1_i_Q = w1_i.mul_Q();
		BCK2_const(x[0], x[4], w1_i, w1_i_Q); BCK2_const(x[1], x[5], w1_i, w1_i_Q); BCK2_const(x[2], x[6], w1_i, w1_i_Q); BCK2_const(x[3], x[7], w1_i, w1_i_Q);
	}

	static void _mul2x4(RNS x[8], const RNS xp[8], const RNS w2[2], const RNS w2_Q[2])
	{
		MUL2(x[0], x[1], xp[0], xp[1], w2[0], w2_Q[0]); MUL2N(x[2], x[3], xp[2], xp[3], w2[0], w2_Q[0]);
		MUL2(x[4], x[5], xp[4], xp[5], w2[1], w2_Q[1]); MUL2N(x[6], x[7], xp[6], xp[7], w2[1], w2_Q[1]);
	}

	static void _mul4x2(RNS x[8], const RNS xp[8], const RNS w2[2], const RNS w2_i[2])
	{
		RNS w2_Q[2]; w2_Q[0] = w2[0].mul_Q(); w2_Q[1] = w2[1].mul_Q();
		FWD2_const(x[0], x[2], w2[0], w2_Q[0]); FWD2_const(x[1], x[3], w2[0], w2_Q[0]);
		FWD2_const(x[4], x[6], w2[1], w2_Q[1]); FWD2_const(x[5], x[7], w2[1], w2_Q[1]);
		_mul2x4(x, xp, w2, w2_Q);
		RNS w2_i_Q[2]; w2_i_Q[0] = w2_i[0].mul_Q(); w2_i_Q[1] = w2_i[1].mul_Q();
		BCK2_const(x[0], x[2], w2_i[0], w2_i_Q[0]); BCK2_const(x[1], x[3], w2_i[0], w2_i_Q[0]);
		BCK2_const(x[4], x[6], w2_i[1], w2_i_Q[1]); BCK2_const(x[5], x[7], w2_i[1], w2_i_Q[1]);
	}

	static void _mul8(RNS x[8], const RNS xp[8], const RNS & w1, const RNS & w1_i, const RNS w2[2], const RNS w2_i[2])
	{
		const RNS w1_Q = w1.mul_Q();
		FWD2_const(x[0], x[4], w1, w1_Q); FWD2_const(x[1], x[5], w1, w1_Q); FWD2_const(x[2], x[6], w1, w1_Q); FWD2_const(x[3], x[7], w1, w1_Q);
		_mul4x2(x, xp, w2, w2_i);
		const RNS w1_i_Q = w1_i.mul_Q();
		BCK2_const(x[0], x[4], w1_i, w1_i_Q); BCK2_const(x[1], x[5], w1_i, w1_i_Q); BCK2_const(x[2], x[6], w1_i, w1_i_Q); BCK2_const(x[3], x[7], w1_i, w1_i_Q);
	}

	static void _fwd4x2(RNS x[8], const RNS w2[2])
	{
		RNS w2_Q[2]; w2_Q[0] = w2[0].mul_Q(); w2_Q[1] = w2[1].mul_Q();
		FWD2_const(x[0], x[2], w2[0], w2_Q[0]); FWD2_const(x[1], x[3], w2[0], w2_Q[0]);
		FWD2_const(x[4], x[6], w2[1], w2_Q[1]); FWD2_const(x[5], x[7], w2[1], w2_Q[1]);
	}

	static void _fwd8(RNS x[8], const RNS & w1, const RNS w2[2])
	{
		const RNS w1_Q = w1.mul_Q();
		FWD2_const(x[0], x[4], w1, w1_Q); FWD2_const(x[1], x[5], w1, w1_Q); FWD2_const(x[2], x[6], w1, w1_Q); FWD2_const(x[3], x[7], w1, w1_Q);
		_fwd4x2(x, w2);
	}

private:
	static void forward8(RNS * const x, const RNS * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const RNS w1 = w[1 * (s + j)];
			RNS w2[2]; _load(2, w2, &w[2 * (s + j)], 1);
			RNS w4[4]; _load(4, w4, &w[4 * (s + j)], 1);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 8 * m * j + i;
				RNS zl[8]; _load(8, zl, &x[k], m);
				_forward8(zl, w1, w2, w4);
				_store(8, &x[k], m, zl);
			}
		}
	}

	static void backward8(RNS * const z, const RNS * const w_i, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const RNS w1_i = w_i[1 * (s + j)];
			RNS w2_i[2]; _load(2, w2_i, &w_i[2 * (s + j)], 1);
			RNS w4_i[4]; _load(4, w4_i, &w_i[4 * (s + j)], 1);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 8 * m * j + i;
				RNS zl[8]; _load(8, zl, &z[k], m);
				_backward8(zl, w1_i, w2_i, w4_i);
				_store(8, &z[k], m, zl);
			}
		}
	}

	static void forward8_0(RNS * const x, const RNS * const w, const size_t n_8)
	{
		// Montgomery form of MF of I to convert input into MF
		const RNS w1 = w[0];
		RNS w2[2]; _load(2, w2, &w[2], 1);
		RNS w4[4]; _load(4, w4, &w[4], 1);

		for (size_t i = 0; i < n_8; ++i)
		{
			const size_t k = i;
			RNS xl[8]; _load(8, xl, &x[k], n_8);
			_forward8_0(xl, w1, w2, w4);
			_store(8, &x[k], n_8, xl);
		}
	}

	static void square2x4(RNS * const x, const RNS * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			RNS w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS w2_Q[2]; w2_Q[0] = w2[0].mul_Q(); w2_Q[1] = w2[1].mul_Q();

			RNS xl[8]; _load(8, xl, &x[8 * j], 1);
			_square2x4(xl, w2, w2_Q);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void square4x2(RNS * const x, const RNS * const w, const RNS * const w_i, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			RNS w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS w2_i[2]; _load(2, w2_i, &w_i[2 * (n_8 + j)], 1);

			RNS xl[8]; _load(8, xl, &x[8 * j], 1);
			_square4x2(xl, w2, w2_i);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void square8(RNS * const x, const RNS * const w, const RNS * const w_i, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const RNS w1 = w[1 * (n_8 + j)];
			const RNS w1_i = w_i[1 * (n_8 + j)];
			RNS w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS w2_i[2]; _load(2, w2_i, &w_i[2 * (n_8 + j)], 1);

			RNS xl[8]; _load(8, xl, &x[8 * j], 1);
			_square8(xl, w1, w1_i, w2, w2_i);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void mul2x4(RNS * const x, const RNS * const xp, const RNS * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			RNS w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS w2_Q[2]; w2_Q[0] = w2[0].mul_Q(); w2_Q[1] = w2[1].mul_Q();

			RNS zl[8]; _load(8, zl, &x[8 * j], 1);
			RNS zpl[8]; _load(8, zpl, &xp[8 * j], 1);
			_mul2x4(zl, zpl, w2, w2_Q);
			_store(8, &x[8 * j], 1, zl);
		}
	}

	static void mul4x2(RNS * const x, const RNS * const xp, const RNS * const w, const RNS * const w_i, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			RNS w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS w2_i[2]; _load(2, w2_i, &w_i[2 * (n_8 + j)], 1);

			RNS xl[8]; _load(8, xl, &x[8 * j], 1);
			RNS xpl[8]; _load(8, xpl, &xp[8 * j], 1);
			_mul4x2(xl, xpl, w2, w2_i);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void mul8(RNS * const x, const RNS * const xp, const RNS * const w, const RNS * const w_i, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const RNS w1 = w[1 * (n_8 + j)];
			const RNS w1_i = w_i[1 * (n_8 + j)];
			RNS w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			RNS w2_i[2]; _load(2, w2_i, &w_i[2 * (n_8 + j)], 1);

			RNS xl[8]; _load(8, xl, &x[8 * j], 1);
			RNS xpl[8]; _load(8, xpl, &xp[8 * j], 1);
			_mul8(xl, xpl, w1, w1_i, w2, w2_i);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void fwd4x2(RNS * const x, const RNS * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			RNS w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);

			RNS xl[8]; _load(8, xl, &x[8 * j], 1);
			_fwd4x2(xl, w2);
			_store(8, &x[8 * j], 1, xl);
		}
	}

	static void fwd8(RNS * const x, const RNS * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const RNS w1 = w[1 * (n_8 + j)];
			RNS w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);

			RNS xl[8]; _load(8, xl, &x[8 * j], 1);
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
	void roots_R(RNS * const w, RNS * const wi)
	{
		const size_t n = size_t(1) << _ln;

		RNS sr_s = RNS::primroot_n(_ln), sr_si = sr_s.invert();	// (-1)^{1/s}
		for (size_t s = n / 2; s >= 1; s /= 2)
		{
			RNS t_j = RNS::one(), ti_j = RNS::one();
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
	void roots_Q(RNS * const w, RNS * const wi)
	{
		const size_t n = size_t(1) << _ln;

		RNS sr_2s = RNS::primroot_n(_ln + 1), sr_2si = sr_2s.invert();	// (-1)^{1/2s}
		for (size_t s = n / 2; s >= 1; s /= 2)
		{
			const RNS sr_s = sr_2s * sr_2s, sr_si = sr_2si * sr_2si;	// (-1)^{1/s}

			RNS t_j = sr_2s, ti_j = sr_2si;
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

	void set(RNS * const y, const RNS * const x)
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

	static constexpr void write_word(mp_limb_t * const x_ptr, const size_t bit_index, const uint_32 r)
	{
		const size_t limb_index = bit_index / 64;
		const int limb_shift = int(bit_index % 64);
		x_ptr[limb_index] |= mp_limb_t(r) << limb_shift;
		if (limb_shift != 0) x_ptr[limb_index + 1] |= mp_limb_t(r) >> (64 - limb_shift);
	}

	void set_mpz(RNS * const y, const mpz_t & x)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const uint_32 base = 1u << lbase;
		const RNS rbase = _rbase;

		const size_t x_size = mpz_size(x);
		const mp_limb_t * const x_ptr = mpz_limbs_read(x);

		size_t bit_index = 0;
		for (size_t i = 0; i < n; ++i)
		{
			const uint_32 r1 = read_word(x_ptr, x_size, bit_index) & (base - 1);
			bit_index += size_t(lbase);
			const uint_32 r2 = read_word(x_ptr, x_size, bit_index) & (base - 1);
			bit_index += size_t(lbase);
			y[i] = RNS(r1) + RNS(r2) * rbase;
		}
	}

	void get_mpz(mpz_t & y, const RNS * const x)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase, lbase2 = 2 * lbase;
		const uint_32 base = 1u << lbase;

		const size_t y_size = (size_t(lbase2) << _ln) / 64 + 2;
		mp_limb_t * const y_ptr = mpz_limbs_write(y, mp_size_t(y_size));
		for (size_t i = 0; i < y_size; ++i) y_ptr[i] = 0;

		int_64 f = 0;
		size_t bit_index = 0;
		for (size_t i = 0; i < n; ++i)
		{
			f += x[i].get_i64();
			const uint_32 r1 = (uint_32)(f) & (base - 1);
			f >>= lbase;
			const uint_32 r2 = (uint_32)(f) & (base - 1);
			f >>= lbase;

			write_word(y_ptr, bit_index, r1);
			bit_index += size_t(lbase);
			write_word(y_ptr, bit_index, r2);
			bit_index += size_t(lbase);
		}

		mp_size_t s = mp_size_t(y_size);
		while ((s > 0) && (y_ptr[s - 1] == 0)) --s;
 		mpz_limbs_finish(y, s);

		if (f != 0)
		{
			mpz_t t; mpz_init_set_si(t, long(f));
			mpz_mul_2exp(t, t, mp_bitcnt_t(lbase2) << _ln);
			mpz_add(y, y, t);
			mpz_clear(t);
		}
	}

	size_t NTT_forward8(RNS * const x, const RNS * const w)
	{
		const size_t n = size_t(1) << _ln;

		forward8_0(x, w, n / 8);
		size_t m = n / 8 / 8, s = 8;
		for (; m > 1; m /= 8, s *= 8) forward8(x, w, m, s);
		return s;
	}

	void NTT_inverse8(RNS * const x, const RNS * const w_i, const size_t s0)
	{
		const size_t n = size_t(1) << _ln;

		for (size_t s = s0 / 8, m = n / 8 / s; m <= n / 8; m *= 8, s /= 8) backward8(x, w_i, m, s);
	}

	size_t NTT_forward(RNS * const x, const RNS * const w)
	{
		const size_t s0 = NTT_forward8(x, w);

		const size_t n = size_t(1) << _ln;
		if (s0 == n / 8) fwd8(x, w, n / 8);
		else if (s0 == n / 4) fwd4x2(x, w, n / 8);

		return s0;
	}

	void square(RNS * const x, const RNS * const w, const RNS * const w_i, const size_t s0)
	{
		const size_t n = size_t(1) << _ln;

		if (s0 == n / 8) square8(x, w, w_i, n / 8);
		else if (s0 == n / 4) square4x2(x, w, w_i, n / 8);
		else if (s0 == n / 2) square2x4(x, w, n / 8);
	}

	void mul(RNS * const y, const RNS * const x, const RNS * const w, const RNS * const w_i, const size_t s0)
	{
		const size_t n = size_t(1) << _ln;

		if (s0 == n / 8) mul8(y, x, w, w_i, n / 8);
		else if (s0 == n / 4) mul4x2(y, x, w, w_i, n / 8);
		else if (s0 == n / 2) mul2x4(y, x, w, n / 8);
	}

	void carry_mul_sub(RNS * const x, const RNS * const y, int_128 * const carry, const uint_32 a)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const RNS & norm = _norm;
		const uint_32 base = 1u << lbase;
		const RNS rbase = RNS(base).toMonty();
	
		for (size_t i = 0; i < n / CARRY_LENGTH; ++i)
		{
			int_128 f = int128_set(0u, 0);
			for (size_t j = 0; j < CARRY_LENGTH; ++j)
			{
				const size_t k = i * CARRY_LENGTH + j;
				// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF
				f = int128_add(f, int128_mul_u32((x[k] * norm).get_i128(), a));
				if (y != nullptr) f = int128_sub_i64(f, y[k].get_i64());
				const uint_32 r1 = (uint_32)(f.s0) & (base - 1);
				f = int128_rshift(f, lbase);
				const uint_32 r2 = (uint_32)(f.s0) & (base - 1);
				f = int128_rshift(f, lbase);
				x[k] = RNS(r1) + RNS(r2) * rbase;
			}
			carry[i] = f;
		}
	}

	void carry_complete(RNS * const x, int_128 * const carry, const bool nega)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const uint_32 base = 1u << lbase;
		const RNS rbase = RNS(base).toMonty();

		for (size_t i = 0; i < n / CARRY_LENGTH; ++i)
		{
			int_128 f = carry[((i == 0) ? (n / CARRY_LENGTH) : i) - 1];
			if ((i == 0) && nega) f = int128_neg(f);

			for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)
			{
				const size_t k = i * CARRY_LENGTH + j;
				f = int128_add_u64(f, (uint_64)(x[k].get_i64()));	// x[k] >= 0
				const uint_32 r1 = (uint_32)(f.s0) & (base - 1);
				f = int128_rshift(f, lbase);
				int_32 r2 = (int_32)((uint_32)(f.s0) & (base - 1));
				f = int128_rshift(f, lbase);
#if defined(BALANCED)
				if (r2 >= (int_32)(base / 2)) { r2 -= (int_32)(base); f = int128_add_u64(f, 1); }
#endif
				x[k] = RNS(r1) + RNS().set_i(r2) * rbase;
			}
			const size_t k = i * CARRY_LENGTH + CARRY_LENGTH - 1;
			f = int128_add_u64(f, (uint_64)(x[k].get_i64()));	// x[k] >= 0
			const uint_32 r1 = (uint_32)(f.s0) & (base - 1);
			f = int128_rshift(f, lbase);
			const int_32 r2 = (int_32)(f.s0);
			x[k] = RNS(r1) + RNS().set_i(r2) * rbase;
		}
	}

	// b-1, b-1, ..., b-1, b + a, ... => b, 0, ..., 0, a, ... 
	void fix_base(RNS * const x, const char * const who)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase, lbase2 = 2 * lbase;
		const int_64 base2 = int_64(1) << lbase2;
		const RNS rbase2 = _rbase2, rbase2_m1 = rbase2 - RNS(1);

#if defined(CHECK_THEOREM)
		static size_t i_max = 2;
#else
		(void)who;
#endif
		if (x[n - 1] == rbase2_m1)
		{
			size_t i = 2; for (; i < n; ++i) { if (x[n - i] != rbase2_m1) break; }
#if defined(CHECK_THEOREM)
			if (i > i_max) { i_max = i; std::cout << "fix_base " << who << ": i = " << i << " / " << n << std::endl; }
#endif
			if (x[n - i].get_i64() >= base2)
			{
				x[n - i] -= rbase2;
				for (size_t j = i - 1; j >= 2; --j) x[n - j] = RNS(0u);
				x[n - 1] = rbase2;
			}
		}
	}

	void lower(RNS * const x, const int_32 l0)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase, lbase2 = 2 * lbase;
		const int_64 base2 = int_64(1) << lbase2;

		if (x[n - 1].get_i64() >= base2)	// >= base^n
		{
			// - R or Q
			x[n - 1] -= _rbase2;
			x[0] -= RNS().set_i(l0);
		}
	}

	void positive(RNS * const x, const int_32 l0)
	{
		const size_t n = size_t(1) << _ln;

		for (size_t i = 0; i < n; ++i)
		{
			const int_64 u = x[n - 1 - i].get_i64();
			if (u > 0) return;
			if (u < 0)
			{
				// + R or Q
				x[n - 1] += _rbase2;
				x[0] += RNS().set_i(l0);
				return;
			}
		}
	}

	void mod_Q(RNS * const x)
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

		double log_q = 0;
		for (size_t i = 0; i < n; ++i)
		{
			const size_t j = n - 1 - i;
			int_64 u = x[j].get_i64();
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

	void mod_R_add2(RNS * const x)
	{
#if !defined(BALANCED)
		fix_base(x, "R");
#endif
		lower(x, -1);		// <= R
#if !defined(BALANCED)
		positive(x, -1);	// >= 0
#endif

		x[0] += RNS(2);
	}

	// Modular arithmetic

	// void display(const RNS * const x, const char * const str) const
	// {
	// 	const size_t n = size_t(1) << _ln;
	// 	std::cout << str << ": ";
	// 	for (size_t i = 0; i < n; ++i) x[i].display();
	// 	std::cout << std::endl;
	// }

	void square_mul_mod_R(RNS * const y, const RNS * const x, const uint_32 a)
	{
		set(y, x); const size_t s0 = NTT_forward8(y, _w_R);
		square(y, _w_R, _wi_R, s0);
		NTT_inverse8(y, _wi_R, s0);
		carry_mul_sub(y, nullptr, _carry, a);
		carry_complete(y, _carry, false);
	}

	void square_mul_mod_Q(RNS * const y, const RNS * const x, const uint_32 a)
	{
		set(y, x); const size_t s0 = NTT_forward8(y, _w_Q);
		square(y, _w_Q, _wi_Q, s0);
		NTT_inverse8(y, _wi_Q, s0);
		carry_mul_sub(y, nullptr, _carry, a);
		carry_complete(y, _carry, true);
	}

	void mul_mod_R(RNS * const y, const RNS * const xt)
	{
		const size_t s0 = NTT_forward8(y, _w_R);
		mul(y, xt, _w_R, _wi_R, s0);
		NTT_inverse8(y, _wi_R, s0);
 		carry_mul_sub(y, nullptr, _carry, 1);
		carry_complete(y, _carry, false);
	}

	void mul_mod_Q(RNS * const y, const RNS * const xt)
	{
		const size_t s0 = NTT_forward8(y, _w_Q);
		mul(y, xt, _w_Q, _wi_Q, s0);
		NTT_inverse8(y, _wi_Q, s0);
 		carry_mul_sub(y, nullptr, _carry, 1);
		carry_complete(y, _carry, true);
	}

	void mul_sub_mod_Q(RNS * const y, const RNS * const xt, const RNS * const z)
	{
		const size_t s0 = NTT_forward8(y, _w_Q);
		mul(y, xt, _w_Q, _wi_Q, s0);
		NTT_inverse8(y, _wi_Q, s0);
 		carry_mul_sub(y, z, _carry, 1);
		carry_complete(y, _carry, true);
	}

	// Montgomery-McLaughlin-Gallot-Woltman multiplication

	void reduceMMGW(RNS * const x)
	{
		mul_mod_R(x, _Ninv_R);	// m'
		mod_R_add2(x);

		mul_sub_mod_Q(x, _N_Q, _t_Q);
		mod_Q(x);
	}

	void mulMMGW(RNS * const x, const RNS * const y)
	{
		set(_t, y); NTT_forward(_t, _w_Q);
		set(_t_Q, x); mul_mod_Q(_t_Q, _t);
		set(_t, y); NTT_forward(_t, _w_R);
		mul_mod_R(x, _t);
		reduceMMGW(x);
	}

	void squareMMGW(RNS * const x, const uint_32 a)
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

		uint_32 * const buf = new uint_32[4 * n];
		for (size_t i = 0; i < 2; ++i)
		{
			if (!cfile.read(reinterpret_cast<char *>(buf), 4 * n * sizeof(uint_32))) { delete[] buf; return false; }
			RNS * const reg = _reg[i];
			for (size_t k = 0; k < n; ++k) reg[k] = RNS(buf[k + 0 * n], buf[k + 1 * n], buf[k + 2 * n], buf[k + 3 * n]);
		}
		delete[] buf;

		return true;
	}

	void save_checkpoint(file & cfile) override
	{
		const size_t n = size_t(1) << _ln;

		uint_32 * const buf = new uint_32[4 * n];
		for (size_t i = 0; i < 2; ++i)
		{
			const RNS * const reg = _reg[i];
			for (size_t k = 0; k < n; ++k)
			{
				const RNS & r = reg[k];
				buf[k + 0 * n] = r.get_r1(); buf[k + 1 * n] = r.get_r2();
				buf[k + 2 * n] = r.get_r3(); buf[k + 3 * n] = r.get_r4();
			}
			if (!cfile.write(reinterpret_cast<const char *>(buf), 4 * n * sizeof(uint_32))) break;
		}
		delete[] buf;
	}

public:
	transformCPU(const mpz_t & N, const int32_t a) : transform()
	{
		mpz_t R, t; mpz_inits(_Nz, R, t, nullptr);

		// R = base^n - 1, Q = base^n + 1 such that base^n > 16 * a * N and n is a power of two
		long e_n; const double m_n = mpz_get_d_2exp(&e_n, N);
		const double log2_16aN = std::log2(16.0 * a) + std::log2(m_n) + e_n;

		const double p_2a = ZP1::P() * double(ZP2::P()) * double(ZP3::P()) * double(ZP4::P()) / 2 / a;
		int ln;
		for (int lbase2 = 62; lbase2 >= 2; lbase2 -= 2)
		{
			// log2(base^n) = lbase * n
			ln = 4; while (double(lbase2) * (1 << ln) < log2_16aN) ++ln;

			// A necessary condition is a * n * (base - 1)^2 < Prod_i(Pi) / 2
			const double dbase2 = double(uint64_t(1) << lbase2);
			if ((1 << ln) * (dbase2 - 1) * (dbase2 - 1) < p_2a)
			{
				_ln = ln;
				_lbase = lbase2 / 2;
				break;
			}
		}
		const size_t n = size_t(1) << ln;
		const mp_bitcnt_t lbase = mp_bitcnt_t(_lbase), lbase2 = 2 * lbase;
		const RNS base = RNS(1u << lbase);
		_rbase = base.toMonty();
		_rbase2 = base * _rbase;
		_norm = RNS::norm(ln - 1);

		_N_Q = new RNS[n]; _Ninv_R = new RNS[n]; _one = new RNS[n];
		_t_Q = new RNS[n];  _t = new RNS[n];
		for (size_t i = 0; i < reg_count; ++i) _reg[i] = new RNS[n];

		_w_R = new RNS[n]; _wi_R = new RNS[n];
		roots_R(_w_R, _wi_R);
		_w_Q = new RNS[n]; _wi_Q = new RNS[n];
		roots_Q(_w_Q, _wi_Q);

		_carry = new int_128[n / CARRY_LENGTH];

#if defined(CHECK_THEOREM)
		long eN; const double mN = mpz_get_d_2exp(&eN, N);
		_logN = std::log2(mN) + eN;
#endif
		mpz_set_ui(R, 1); mpz_mul_2exp(R, R, lbase2 << ln); mpz_sub_ui(R, R, 1);
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
	}

	std::string get_type() const override { return ""; }
	std::string get_data_size() const override { return ""; }
	std::string get_cache_size() const override { return ""; }

	std::string get_info_short() const override
	{
		const size_t n = size_t(1) << _ln;
		std::ostringstream ss; ss << "length " << n << "x2x" << _lbase << " bits";
		return (ss.str());
	}

	std::string get_info() const override
	{
		std::ostringstream ss; ss << "Montgomery-McLaughlin-Gallot-Woltman reduction of " << get_info_short() << std::endl;
		return (ss.str());
	}
};
