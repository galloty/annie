/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <gmp.h>

#include "transform.h"

class ZP
{
protected:
	uint32_t _n;

public:
	ZP() {}
	explicit ZP(const uint32_t n) : _n(n) {}

	uint32_t get() const { return _n; }
};

// The prime finite field Z/PZ
template<uint32_t P, uint32_t Q, uint32_t R, uint32_t RSQ, uint32_t H>
class ZPT : public ZP
{
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
	explicit ZPT(const uint32_t n) : ZP(n) {}

	static ZPT one() { return ZPT(R); }

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
	RNS _norm;
	mpz_t _Nz;
	RNS * _N_Q = nullptr; RNS * _Ninv_R = nullptr; RNS * _one = nullptr;
	RNS * _t_R = nullptr; RNS * _t_Q = nullptr; RNS * _t = nullptr;
	RNS * _reg[reg_count];
	RNS * _w_R; RNS * _wi_R; RNS * _w_Q; RNS * _wi_Q;
	bool _init_failed = false;

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
		const int ln = _ln;
		const size_t n = size_t(1) << ln;

		RNS sr_s = RNS::primroot_n(ln), sr_si = sr_s.invert();	// (-1)^{1/s}
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
		const int ln = _ln;
		const size_t n = size_t(1) << ln;

		RNS sr_2s = RNS::primroot_n(ln + 1), sr_2si = sr_2s.invert();	// (-1)^{1/2s}
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
		const size_t n = size_t(1) << _ln;

		for (size_t k = 0; k < n; ++k) y[k] = x[k];
	}

	void set_mpz(RNS * const y, const mpz_t & x) const
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

	void get_mpz(mpz_t & y, const RNS * const x) const
	{
		const size_t n = size_t(1) << _ln;
		const mp_bitcnt_t lbase = mp_bitcnt_t(_lbase);

		mpz_set_ui(y, 0);
		for (size_t i = n; i > 0; --i)
		{
			mpz_mul_2exp(y, y, lbase);
			mpz_add_ui(y, y, x[i - 1].get());
		}
	}

	void NTT_forward(RNS * const x, const RNS * const w) const
	{
		const size_t n = size_t(1) << _ln;

		// Montgomery form of MF of I to convert input into MF
		const RNS w1 = w[0];
		for (size_t k = 0; k < n / 2; ++k)
		{
			const RNS u0 = x[k + 0 * n / 2].toMonty(), u1 = w1 * x[k + 1 * n / 2];
			x[k + 0 * n / 2] = u0 + u1;
			x[k + 1 * n / 2] = u0 - u1;
		}

		for (size_t m = n / 4, s = 2; m >= 1; m /= 2, s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const RNS wsj = w[s + j];
				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m * j + i;
					const RNS u0 = x[k + 0 * m], u1 = wsj * x[k + 1 * m];
					x[k + 0 * m] = u0 + u1;
					x[k + 1 * m] = u0 - u1;
				}
			}
		}
	}

	void NTT_inverse(RNS * const x, const RNS * const w_i) const
	{
		const size_t n = size_t(1) << _ln;

		for (size_t m = 1, s = n / 2; s >= 1; m *= 2, s /= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const RNS wsj_i = w_i[s + j];
				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m * j + i;
					const RNS u0 = x[k + 0 * m], u1 = x[k + 1 * m];
					x[k + 0 * m] = u0 + u1;
					x[k + 1 * m] = (u0 - u1) * wsj_i;
				}
			}
		}
	}

	void mul_pointwise(RNS * const y, const RNS * const x)
	{
		const size_t n = size_t(1) << _ln;

		for (size_t k = 0; k < n; ++k) y[k] *= x[k];
	}

	void carry_complete(RNS * const x, const int64_t carry, const bool nega) const
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const uint32_t base = (1u << lbase);

		int64_t f = carry;
		while (f != 0)
		{
			if (nega) f = -f;
			for (size_t i = 0; i < n; ++i)
			{
				f += int64_t(x[i].get());
				x[i] = RNS(uint32_t(f) & (base - 1));
				f >>= lbase;
				if (f == 0) return;
			}
		}
	}

	int64_t carry_mul(RNS * const x, const int32_t a) const
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const uint32_t base = (1u << lbase);
		// Not converted into Montgomery form such that x[i] * norm is converted out of MF
		const RNS norm = _norm;

		int64_t f = 0;
		for (size_t i = 0; i < n; ++i)
		{
			f += a * (x[i] * norm).geti();
			x[i] = RNS(uint32_t(f) & (base - 1));
			f >>= lbase;
		}

		return f;
	}

	int64_t carry_sub(RNS * const x, const RNS * const y) const
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const uint32_t base = (1u << lbase);
		// Not converted into Montgomery form such that x[i] * norm is converted out of MF
		const RNS norm = _norm;

		int64_t f = 0;
		for (size_t i = 0; i < n; ++i)
		{
			f += (x[i] * norm).geti();
			f -= int64_t(y[i].get());
			x[i] = RNS(uint32_t(f) & (base - 1));
			f >>= lbase;
		}

		return f;
	}

	// Modular arithmetic

	void square_mul_mod_R(RNS * const z, const RNS * const x, const int32_t a)
	{
		set(z, x);
		NTT_forward(z, _w_R);
		mul_pointwise(z, z);
		NTT_inverse(z, _wi_R);
 		const int64_t carry = carry_mul(z, a);
		carry_complete(z, carry, false);
	}

	void square_mul_mod_Q(RNS * const z, const RNS * const x, const int32_t a)
	{
		set(z, x);
		NTT_forward(z, _w_Q);
		mul_pointwise(z, z);
		NTT_inverse(z, _wi_Q);
 		const int64_t carry = carry_mul(z, a);
		carry_complete(z, carry, true);
	}

	void mul_mod_R(RNS * const z, const RNS * const x, const RNS * const yt)
	{
		set(z, x);
		NTT_forward(z, _w_R);
		mul_pointwise(z, yt);
		NTT_inverse(z, _wi_R);
 		const int64_t carry = carry_mul(z, 1);
		carry_complete(z, carry, false);
	}

	void mul_sub_mod_Q(RNS * const z, const RNS * const x, const RNS * const yt, const RNS * const s)
	{
		set(z, x);
		NTT_forward(z, _w_Q);
		mul_pointwise(z, yt);
		NTT_inverse(z, _wi_Q);
 		const int64_t carry = (s != nullptr) ? carry_sub(z, s) : carry_mul(z, 1);
		carry_complete(z, carry, true);
	}

	// Montgomery-McLaughlin-Gallot-Woltman multiplication

	void reduceMMGW(RNS * const x)
	{
		mul_mod_R(x, _t_R, _Ninv_R);	// m'

		x[0] = RNS(x[0].get() + 2);
		mul_sub_mod_Q(x, x, _N_Q, _t_Q);
	}

	void mulMMGW(RNS * const x, const RNS * const y)
	{
		set(_t, y); NTT_forward(_t, _w_R);
		mul_mod_R(_t_R, x, _t);
		set(_t, y); NTT_forward(_t, _w_Q);
		mul_sub_mod_Q(_t_Q, x, _t, nullptr);
		reduceMMGW(x);
	}

	void squareMMGW(RNS * const x, const int32_t a)
	{
		square_mul_mod_R(_t_R, x, a);
		square_mul_mod_Q(_t_Q, x, a);
		reduceMMGW(x);
	}

public:
	void set1() override
	{
		set(_reg[0], _one);
	}

	void get(mpz_t & z, const size_t src) override
	{
		// Conversion out of Montgomery-McLaughlin-Gallot-Woltman form
		set(_t_R, _reg[src]);
		set(_t_Q, _reg[src]);
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
		squareMMGW(_reg[0], a);
	}

	void mul(const size_t dst, const size_t src) override
	{
		mulMMGW(_reg[dst], _reg[src]);
	}

	bool init_failed() const override { return _init_failed; }
	void get_factor(mpz_t & z) const override { mpz_set(z, _Nz); }

public:
	transformCPU(const mpz_t & N, const int32_t a, const bool verbose) : transform()
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
			ln = 1; while ((size_t(1) << ln) < m) ++ln;
			// A necessary condition is a * n * (base - 1)^2 < (P1 * P2) / 2
			if ((base - 1) * uint64_t(base - 1) < (p_2a >> ln))
			{
				_ln = ln;
				_lbase = int(lbase);
				break;
			}
		}
		_norm = RNS::norm(ln);
		const size_t n = (size_t(1) << ln);

		_N_Q = new RNS[n]; _Ninv_R = new RNS[n]; _one = new RNS[n];
		_t_R = new RNS[n]; _t_Q = new RNS[n];  _t = new RNS[n];
		for (size_t i = 0; i < reg_count; ++i) _reg[i] = new RNS[n];

		_w_R = new RNS[n]; _wi_R = new RNS[n];
		roots_R(_w_R, _wi_R);
		_w_Q = new RNS[n]; _wi_Q = new RNS[n];
		roots_Q(_w_Q, _wi_Q);

		mpz_set_ui(R, 1); mpz_mul_2exp(R, R, mp_bitcnt_t(_lbase) << ln); mpz_sub_ui(R, R, 1);
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

		// The Montgomery-McLaughlin-Gallot-Woltman form of x is R/2 * x mod N
		mpz_mod(t, R, N);
		if (mpz_odd_p(t) != 0) mpz_add(t, t, N);
		mpz_div_2exp(t, t, 1);
		set_mpz(_one, t);

		mpz_clears(R, t, nullptr);

		if (verbose) std::cout << "Montgomery-McLaughlin-Gallot-Woltman reduction of length " << n << "x" << _lbase << " bits" << std::endl;
	}

	virtual ~transformCPU()
	{
		mpz_clear(_Nz);

		delete[] _N_Q; delete[] _Ninv_R; delete[] _one;
		delete[] _t_R; delete[] _t_Q; delete[] _t;
		for (size_t i = 0; i < reg_count; ++i) delete[] _reg[i];

		delete[] _w_R; delete[] _wi_R;
		delete[] _w_Q; delete[] _wi_Q;
	}
};
