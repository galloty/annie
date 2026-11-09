/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <gmp.h>

#include "transform.h"

#define BALANCED	true

// The prime finite field Z/PZ
template<uint32_t P, uint32_t Q, uint32_t R, uint32_t RSQ, uint32_t H>
class ZPT
{
private:
	uint32_t _n;

private:
	static uint32_t _add(const uint32_t a, const uint32_t b) { return a + b - ((a >= P - b) ? P : 0); }
	static uint32_t _sub(const uint32_t a, const uint32_t b) { return a - b + ((a < b) ? P : 0); }

	// Montgomery multiplication: 2 mul + 2 mul_hi
	static uint32_t _mul(const uint32_t a, const uint32_t b)
	{
		const uint64_t t = a * uint64_t(b);
		const uint32_t lo = uint32_t(t), hi = uint32_t(t >> 32);
		const uint32_t mp = uint32_t(((lo * Q) * uint64_t(P)) >> 32);
		return _sub(hi, mp);
	}

	ZPT pow(const uint64_t e) const
	{
		if (e == 0) return ZPT(R);
		ZPT r = ZPT(R), y = *this;
		for (uint64_t i = e; i > 1; i /= 2)
		{
			if (i % 2 != 0) r *= y;
			y *= y;
		}
		r *= y;
		return r;
	}

public:
	ZPT() {}
	explicit ZPT(const uint32_t n) : _n(n) {}

	static ZPT one() { return ZPT(R); }

	uint32_t get() const { return _n; }
	int32_t get_int() const { return (int32_t)(_n - ((_n >= P / 2) ? P : 0)); }
	void set_int(const int32_t i) { _n = (uint32_t)(i) + ((i < 0) ? P : 0); }

	// Conversion into / out of Montgomery form
	ZPT toMonty() const { return ZPT(_mul(_n, RSQ)); }
	// ZPT fromMonty() const { return ZPT(_mul(_n, 1)); }

	ZPT & operator+=(const ZPT & rhs) { _n = _add(_n, rhs._n); return *this; }
	ZPT & operator-=(const ZPT & rhs) { _n = _sub(_n, rhs._n); return *this; }
	ZPT & operator*=(const ZPT & rhs) { _n = _mul(_n, rhs._n); return *this; }

	ZPT operator+(const ZPT & rhs) const { return ZPT(_add(_n, rhs._n)); }
	ZPT operator-(const ZPT & rhs) const { return ZPT(_sub(_n, rhs._n)); }
	ZPT operator*(const ZPT & rhs) const { return ZPT(_mul(_n, rhs._n)); }

	ZPT invert() const { return pow(P - 2); }
	static ZPT primroot_n(const int ln) { return ZPT(H).pow((P - 1) >> ln); }
	static ZPT norm(const int ln) { return ZPT(P - ((P - 1) >> ln)); }
};

#define P1			(125 * (1u << 25) + 1)
#define	Q1			100663297u		// p * q = 1 (mod 2^32)
#define	R1			100663295u		// 2^32 mod p = Montgomery form of 1
#define	RSQ1		232465106u		// (2^32)^2 mod p
#define	H1			301989885u		// Montgomery form of the primitive root 3

#define P2			(243 * (1u << 24) + 1)
#define	Q2			218103809u
#define	R2			218103807u
#define	RSQ2		3444438393u
#define	H2			1526726649u		// MF of the primitive root 7

#define CARRY_LENGTH	4u

using ZP1 = ZPT<P1, Q1, R1, RSQ1, H1>;
using ZP2 = ZPT<P2, Q2, R2, RSQ2, H2>;

// The residue number system {P1, P2}
class RNS
{
private:
	ZP1 _r1;
	ZP2 _r2;

private:
	explicit RNS(const ZP1 & r1, const ZP2 & r2) : _r1(r1), _r2(r2) {}

	// Chinese remainder theorem
	static int64_t garner(const ZP1 & r1, const ZP2 & r2)
	{
		const ZP1 invP2_P1 = ZP1(1797558821u);	// Montgomery form of 1 / P2 (mod P1)
		const uint64_t P1P2 = P1 * uint64_t(P2);

		const ZP1 u = (r1 - ZP1(r2.get())) * invP2_P1;	// P2 < P1
		const uint64_t n = r2.get() + u.get() * uint64_t(P2);
		return (n > P1P2 / 2) ? int64_t(n - P1P2) : int64_t(n);
	}

public:
	RNS() {}
	explicit RNS(const uint32_t n) : _r1(n), _r2(n) {}

	static RNS one() { return RNS(ZP1::one(), ZP2::one()); }

	uint32_t get() const { return _r1.get(); }
	int64_t geti() const { return garner(_r1, _r2); }
	void set_int(const int32_t i) { _r1.set_int(i); _r2.set_int(i); }
	int32_t get_int() const { return _r1.get_int(); }

	// Conversion into / out of Montgomery form
	RNS toMonty() const { return RNS(_r1.toMonty(), _r2.toMonty()); }

	RNS & operator+=(const RNS & rhs) { _r1 += rhs._r1; _r2 += rhs._r2; return *this; }
	RNS & operator-=(const RNS & rhs) { _r1 -= rhs._r1; _r2 -= rhs._r2; return *this; }
	RNS & operator*=(const RNS & rhs) { _r1 *= rhs._r1; _r2 *= rhs._r2; return *this; }

	RNS operator+(const RNS & rhs) const { RNS r = *this; r += rhs; return r; }
	RNS operator-(const RNS & rhs) const { RNS r = *this; r -= rhs; return r; }
	RNS operator*(const RNS & rhs) const { RNS r = *this; r *= rhs; return r; }

	RNS invert() const { return RNS(_r1.invert(), _r2.invert()); }
	static RNS primroot_n(const int ln) { return RNS(ZP1::primroot_n(ln), ZP2::primroot_n(ln)); }
	static RNS norm(const int ln) { return RNS(ZP1::norm(ln), ZP2::norm(ln)); }
};

class transformCPU : public transform
{
private:
	static const size_t reg_count = 3;
	int _ln = 0;
	int _lbase = 0;
	bool _init_failed = false;
	RNS _norm;
	mpz_t _Nz;
	RNS * _N_Q = nullptr; RNS * _Ninv_R = nullptr; RNS * _one = nullptr;
	RNS * _t_Q = nullptr; RNS * _t = nullptr;
	RNS * _reg[reg_count];
	RNS * _w_R; RNS * _wi_R; RNS * _w_Q; RNS * _wi_Q;
	int64_t * _carry;
#if defined(CHECK_THEOREM)
	double _logq_min = 1e100, _logq_max = -1e100, _logN = 0;
#endif

private:
	static void _load(const size_t n, RNS * const xl, const RNS * const x, const size_t s) { for (size_t l = 0; l < n; ++l) xl[l] = x[l * s]; }
	static void _store(const size_t n, RNS * const x, const size_t s, const RNS * const xl) { for (size_t l = 0; l < n; ++l) x[l * s] = xl[l]; }

	static void FWD2(RNS & x0, RNS & x1, const RNS & w) { const RNS t = x1 * w; x1 = x0 - t; x0 = x0 + t; }
	static void BCK2(RNS & x0, RNS & x1, const RNS & w_i) { const RNS t = x0 - x1; x0 = x0 + x1; x1 = t * w_i; }

	static void SQR2(RNS & x0, RNS & x1, const RNS & w) { const RNS t = x1 * x1 * w; x1 = (x0 + x0) * x1; x0 = x0 * x0 + t; }
	static void SQR2N(RNS & x0, RNS & x1, const RNS & w) { const RNS t = x1 * x1 * w; x1 = (x0 + x0) * x1; x0 = x0 * x0 - t; }

	static void MUL2(RNS & x0, RNS & x1, const RNS & xp0, const RNS & xp1, const RNS & w) { const RNS t = (x1 * xp1) * w; x1 = x0 * xp1 + xp0 * x1; x0 = x0 * xp0 + t; }
	static void MUL2N(RNS & x0, RNS & x1, const RNS & xp0, const RNS & xp1, const RNS & w) { const RNS t = (x1 * xp1) * w; x1 = x0 * xp1 + xp0 * x1; x0 = x0 * xp0 - t; }

	static void _forward8(RNS x[8], const RNS & w1, const RNS w2[2], const RNS w4[4])
	{
		FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);
		FWD2(x[0], x[2], w2[0]); FWD2(x[1], x[3], w2[0]); FWD2(x[4], x[6], w2[1]); FWD2(x[5], x[7], w2[1]);
		FWD2(x[0], x[1], w4[0]); FWD2(x[2], x[3], w4[1]); FWD2(x[4], x[5], w4[2]); FWD2(x[6], x[7], w4[3]);
	}

	static void _backward8(RNS x[8], const RNS w1_i, const RNS w2_i[2], const RNS w4_i[4])
	{
		BCK2(x[0], x[1], w4_i[0]); BCK2(x[2], x[3], w4_i[1]); BCK2(x[4], x[5], w4_i[2]); BCK2(x[6], x[7], w4_i[3]);
		BCK2(x[0], x[2], w2_i[0]); BCK2(x[1], x[3], w2_i[0]); BCK2(x[4], x[6], w2_i[1]); BCK2(x[5], x[7], w2_i[1]);
		BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);
	}

	static void _forward8_0(RNS x[8], const RNS & w1, const RNS w2[2], const RNS w4[4])
	{
		for (size_t i = 0; i < 4; ++i) x[i] = x[i].toMonty();
		_forward8(x, w1, w2, w4);
	}

	static void _square2x4(RNS x[8], const RNS w2[2])
	{
		SQR2(x[0], x[1], w2[0]); SQR2N(x[2], x[3], w2[0]); SQR2(x[4], x[5], w2[1]); SQR2N(x[6], x[7], w2[1]);
	}

	static void _square4x2(RNS x[8], const RNS w2[2], const RNS w2_i[2])
	{
		FWD2(x[0], x[2], w2[0]); FWD2(x[1], x[3], w2[0]); FWD2(x[4], x[6], w2[1]); FWD2(x[5], x[7], w2[1]);
		_square2x4(x, w2);
		BCK2(x[0], x[2], w2_i[0]); BCK2(x[1], x[3], w2_i[0]); BCK2(x[4], x[6], w2_i[1]); BCK2(x[5], x[7], w2_i[1]);
	}

	static void _square8(RNS x[8], const RNS & w1, const RNS & w1_i, const RNS w2[2], const RNS w2_i[2])
	{
		FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);
		_square4x2(x, w2, w2_i);
		BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);
	}

	static void _mul2x4(RNS x[8], const RNS xp[8], const RNS w2[2])
	{
		MUL2(x[0], x[1], xp[0], xp[1], w2[0]); MUL2N(x[2], x[3], xp[2], xp[3], w2[0]); MUL2(x[4], x[5], xp[4], xp[5], w2[1]); MUL2N(x[6], x[7], xp[6], xp[7], w2[1]);
	}

	static void _mul4x2(RNS x[8], const RNS xp[8], const RNS w2[2], const RNS w2_i[2])
	{
		FWD2(x[0], x[2], w2[0]); FWD2(x[1], x[3], w2[0]); FWD2(x[4], x[6], w2[1]); FWD2(x[5], x[7], w2[1]);
		_mul2x4(x, xp, w2);
		BCK2(x[0], x[2], w2_i[0]); BCK2(x[1], x[3], w2_i[0]); BCK2(x[4], x[6], w2_i[1]); BCK2(x[5], x[7], w2_i[1]);
	}

	static void _mul8(RNS x[8], const RNS xp[8], const RNS & w1, const RNS & w1_i, const RNS w2[2], const RNS w2_i[2])
	{
		FWD2(x[0], x[4], w1); FWD2(x[1], x[5], w1); FWD2(x[2], x[6], w1); FWD2(x[3], x[7], w1);
		_mul4x2(x, xp, w2, w2_i);
		BCK2(x[0], x[4], w1_i); BCK2(x[1], x[5], w1_i); BCK2(x[2], x[6], w1_i); BCK2(x[3], x[7], w1_i);
	}

	static void _fwd4x2(RNS x[8], const RNS w2[2])
	{
		FWD2(x[0], x[2], w2[0]); FWD2(x[1], x[3], w2[0]); FWD2(x[4], x[6], w2[1]); FWD2(x[5], x[7], w2[1]);
	}

	static void _fwd8(RNS x[8], const RNS & w1, const RNS w2[2])
	{
		FWD2(x[0], x[4], w1); FWD2(x[2], x[6], w1); FWD2(x[1], x[5], w1); FWD2(x[3], x[7], w1);
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

			RNS xl[8]; _load(8, xl, &x[8 * j], 1);
			_square2x4(xl, w2);
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

			RNS zl[8]; _load(8, zl, &x[8 * j], 1);
			RNS zpl[8]; _load(8, zpl, &xp[8 * j], 1);
			_mul2x4(zl, zpl, w2);
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

	void set_mpz(RNS * const y, const mpz_t & x)
	{
		const size_t n = size_t(1) << _ln;
		const mp_bitcnt_t lbase = mp_bitcnt_t(_lbase);

		mpz_t s, r; mpz_init_set(s, x); mpz_init(r);
		for (size_t i = 0; i < n; ++i)
		{
			mpz_mod_2exp(r, s, lbase);
			y[i] = RNS(uint32_t(mpz_get_ui(r)));
			mpz_div_2exp(s, s, lbase);
		}
		mpz_clears(s, r, nullptr);
	}

	void get_mpz(mpz_t & y, const RNS * const x)
	{
		const size_t n = size_t(1) << _ln;
		const mp_bitcnt_t lbase = mp_bitcnt_t(_lbase);

		mpz_set_ui(y, 0);
		for (size_t i = n; i > 0; --i)
		{
			mpz_mul_2exp(y, y, lbase);
			const int32_t u = x[i - 1].get_int();
			if (u > 0) mpz_add_ui(y, y, (unsigned long int)(u));
			else if (u < 0) mpz_sub_ui(y, y, (unsigned long int)(-u));
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

	void carry_mul_sub(RNS * const x, const RNS * const y, int64_t * const carry, const int32_t a)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const RNS & norm = _norm;
		const uint32_t base = (1u << lbase);
	
		for (size_t i = 0; i < n / CARRY_LENGTH; ++i)
		{
			int64_t f = 0;
			for (size_t j = 0; j < CARRY_LENGTH; ++j)
			{
				const size_t k = i * CARRY_LENGTH + j;
				// norm is not converted into Montgomery form such that x[i] * norm is converted out of MF
				f += a * (x[k] * norm).geti();
				if (y != nullptr) f -= int64_t(y[k].get_int());
				x[k] = RNS(uint32_t(f) & (base - 1));
				f >>= lbase;
			}
			carry[i] = f;
		}
	}

	void carry_complete(RNS * const x, int64_t * const carry, const bool nega)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const uint32_t base = (1u << lbase);

		for (size_t i = 0; i < n / CARRY_LENGTH; ++i)
		{
			int64_t f = carry[((i == 0) ? (n / CARRY_LENGTH) : i) - 1];
			if ((i == 0) && nega) f = -f;

			for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)
			{
				const size_t k = i * CARRY_LENGTH + j;
				f += int64_t(x[k].get());
				int32_t r = int32_t(uint32_t(f) & (base - 1));
				f >>= lbase;
#if defined(BALANCED)
				if (r >= int32_t(base / 2)) { r -= int32_t(base); f += 1; }
#endif
				x[k].set_int(r);
			}
			const size_t k = i * CARRY_LENGTH + CARRY_LENGTH - 1;
			f += int64_t(x[k].get());
			x[k].set_int(int32_t(f));	// -1 <= x_k <= base
		}
	}

	// b-1, b-1, ..., b-1, b + a, ... => b, 0, ..., 0, a, ...
	void fix_base(RNS * const x, const char * const who)
	{
		const size_t n = size_t(1) << _ln;
		const uint32_t base = (1u << _lbase);
#if defined(CHECK_THEOREM)
		static size_t i_max = 2;
#else
		(void)who;
#endif
		if (x[n - 1].get() == base - 1)
		{
			size_t i = 2; for (; i < n; ++i) { if (x[n - i].get() != base - 1) break; }
#if defined(CHECK_THEOREM)
			if (i > i_max) { i_max = i; std::cout << "fix_base " << who << ": i = " << i << " / " << n << std::endl; }
#endif
			if (x[n - i].get_int() >= int32_t(base))
			{
				x[n - i] -= RNS(base);
				for (size_t j = i - 1; j >= 2; --j) x[n - j] = RNS(0);
				x[n - 1] = RNS(base);
			}
		}
	}

	void lower(RNS * const x, const int32_t l0)
	{
		const size_t n = size_t(1) << _ln;
		const uint32_t base = (1u << _lbase);

		int32_t hi = x[n - 1].get_int();
		if (hi >= int32_t(base))	// >= base^n
		{
			// - R or Q
			x[n - 1].set_int(hi - int32_t(base));
			x[0].set_int(x[0].get_int() - l0);
		}
	}

	void positive(RNS * const x, const int32_t l0)
	{
		const size_t n = size_t(1) << _ln;
		const uint32_t base = (1u << _lbase);

		for (size_t i = 0; i < n; ++i)
		{
			const int32_t u = x[n - 1 - i].get_int();
			if (u > 0) return;
			if (u < 0)
			{
				// + R or Q
				x[n - 1].set_int(x[n - 1].get_int() + int32_t(base));
				x[0].set_int(x[0].get_int() + l0);
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
		const int lbase = _lbase;
		const double base = double(1u << _lbase);

		double log_q = 0;
		for (size_t i = 0; i < n; ++i)
		{
			const size_t j = n - 1 - i;
			int32_t u = x[j].get_int();
			if (u != 0)
			{
				if (u < 0)
				{
#if !defined(BALANCED)
					throw std::runtime_error("Q_min < 0");
#else
					mpz_t t; mpz_init(t);
					get_mpz(t, x);
					if (mpz_sgn(t) < 0) throw std::runtime_error("Q_min < 0");
					long ex; const double mx = mpz_get_d_2exp(&ex, t);
					log_q = std::log2(mx) + ex;
					mpz_clear(t);
#endif
				}
				else
				{
					const double du = (j > 0) ? double(u) + x[j - 1].get_int() / base : double(u);
					if (du != 0) log_q = std::log2(std::fabs(du)) + j * double(lbase);
				}
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

		x[0].set_int(x[0].get_int() + 2);
	}

	// Modular arithmetic

	void square_mul_mod_R(RNS * const y, const RNS * const x, const int32_t a)
	{
		set(y, x); const size_t s0 = NTT_forward8(y, _w_R);
		square(y, _w_R, _wi_R, s0);
		NTT_inverse8(y, _wi_R, s0);
		carry_mul_sub(y, nullptr, _carry, a);
		carry_complete(y, _carry, false);
	}

	void square_mul_mod_Q(RNS * const y, const RNS * const x, const int32_t a)
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

	void squareMMGW(RNS * const x, const int32_t a)
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
		squareMMGW(_reg[Reg0], a);
	}

	void mul(const size_t dst, const size_t src) override
	{
		mulMMGW(_reg[dst], _reg[src]);
	}

	bool init_failed() const override { return _init_failed; }
	void get_factor(mpz_t & z) const override { mpz_set(z, _Nz); }

	bool has_checkpoint() const { return true; }

	bool read_checkpoint(file & cfile) override
	{
		const size_t n = size_t(1) << _ln;

		int32_t * const buf = new int32_t[n];
		for (size_t i = 0; i < 2; ++i)
		{
			if (!cfile.read(reinterpret_cast<char *>(buf), n * sizeof(int32_t))) { delete[] buf; return false; }
			RNS * const reg = _reg[i];
			for (size_t k = 0; k < n; ++k) reg[k].set_int(buf[k]);
		}
		delete[] buf;

		return true;
	}

	void save_checkpoint(file & cfile) const override
	{
		const size_t n = size_t(1) << _ln;

		int32_t * const buf = new int32_t[n];
		for (size_t i = 0; i < 2; ++i)
		{
			const RNS * const reg = _reg[i];
			for (size_t k = 0; k < n; ++k) buf[k] = reg[k].get_int();
			if (!cfile.write(reinterpret_cast<const char *>(buf), n * sizeof(int32_t))) break;
		}
		delete[] buf;
	}

public:
	transformCPU(const mpz_t & N, const int32_t a) : transform()
	{
		mpz_t R, t; mpz_inits(_Nz, R, t, nullptr);

		// R = base^n - 1, Q = base^n + 1 such that base^n > 16 * a * N and n is a power of two
		const size_t m_2 = mpz_sizeinbase(N, 2) + size_t(32 - __builtin_clz(uint32_t(a))) + 4;
		const uint64_t p_2a = (P1 * uint64_t(P2)) / 2 / uint32_t(a);
		int ln;
		for (unsigned int lbase = 31; lbase >= 1; --lbase)
		{
			// base is a power of two
			const uint32_t base = (1u << lbase);
			const size_t m = (m_2 + lbase - 1) / lbase;
			ln = 4; while ((size_t(1) << ln) < m) ++ln;
			// A necessary condition is a * n * (base - 1)^2 < (P1 * P2) / 2
			if ((base - 1) * uint64_t(base - 1) < (p_2a >> ln))
			{
				_ln = ln;
				_lbase = int(lbase);
				break;
			}
		}
		_norm = RNS::norm(ln - 1);
		const size_t n = size_t(1) << ln;
		const mp_bitcnt_t lbase = mp_bitcnt_t(_lbase);

		_N_Q = new RNS[n]; _Ninv_R = new RNS[n]; _one = new RNS[n];
		_t_Q = new RNS[n];  _t = new RNS[n];
		for (size_t i = 0; i < reg_count; ++i) _reg[i] = new RNS[n];

		_w_R = new RNS[n]; _wi_R = new RNS[n];
		roots_R(_w_R, _wi_R);
		_w_Q = new RNS[n]; _wi_Q = new RNS[n];
		roots_Q(_w_Q, _wi_Q);

		_carry = new int64_t[n / CARRY_LENGTH];

#if defined(CHECK_THEOREM)
		long eN; const double mN = mpz_get_d_2exp(&eN, N);
		_logN = std::log2(mN) + eN;
#endif
		mpz_set_ui(R, 1); mpz_mul_2exp(R, R, lbase << ln); mpz_sub_ui(R, R, 1);
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
		long e_n; const double m_n = mpz_get_d_2exp(&e_n, N);
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

	std::string get_info() const override
	{
		const size_t n = size_t(1) << _ln;
		std::ostringstream ss; ss << "Montgomery-McLaughlin-Gallot-Woltman reduction of length " << n << "x" << _lbase << " bits" << std::endl;
		return (ss.str());
	}
};
