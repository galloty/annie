/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <gmp.h>

#include "transform.h"

#define LBASE	16
#define BASE	(1 << LBASE)

// The prime finite field with p = 2^64 - 2^32 + 1
class Zp
{
private:
	uint64_t _n;

public:
	static const uint64_t p = (((1ull << 32) - 1) << 32) + 1;

private:
	explicit Zp(const uint64_t n) : _n(n) {}

	Zp mulmod(const Zp & rhs) const
	{
		const __uint128_t t = _n * __uint128_t(rhs._n);
		const uint64_t hi = uint64_t(t >> 64), lo = uint64_t(t);

		// hih * 2^96 + hil * 2^64 + lo = lo + hil * 2^32 - (hih + hil)
		Zp r = Zp((lo >= p) ? lo - p : lo);
		r += Zp(hi << 32);	// lhs * rhs < p^2 => hi * 2^32 < p^2 / 2^32 < p.
		r -= Zp((hi >> 32) + uint32_t(hi));
		return r;
	}

public:
	Zp() {}

	uint32_t get() const { return uint32_t(_n); }
	void set(const uint32_t n) { _n = n; }

	int64_t geti() const { return (_n >= p / 2) ? int64_t(_n - p) : int64_t(_n); }

	Zp & operator+=(const Zp & rhs) { const uint64_t c = (_n >= p - rhs._n) ? p : 0; _n += rhs._n; _n -= c; return *this; }
	Zp & operator-=(const Zp & rhs) { const uint64_t c = (_n < rhs._n) ? p : 0; _n -= rhs._n; _n += c; return *this; }
	Zp & operator*=(const Zp & rhs) { *this = mulmod(rhs); return *this; }

	Zp operator+(const Zp & rhs) const { Zp r = *this; r += rhs; return r; }
	Zp operator-(const Zp & rhs) const { Zp r = *this; r -= rhs; return r; }
	Zp operator*(const Zp & rhs) const { return mulmod(rhs); }

	Zp half() const { return (_n % 2 == 0) ? Zp(_n / 2) : Zp((_n + 1) / 2 + (p - 1) / 2); }

	Zp pow(const uint64_t e) const
	{
		if (e == 0) return Zp(1);

		Zp r = Zp(1), y = *this;
		for (uint64_t i = e; i > 1; i /= 2)
		{
			if (i % 2 != 0) r *= y;
			y *= y;
		}
		r *= y;

		return r;
	}

	Zp invert() const { return Zp(pow(p - 2)); }
	static const Zp primroot_n(const size_t n) { return Zp(7).pow((p - 1) / n); }
};

class transformCPU : public transform
{
private:
	static const size_t reg_count = 3;
	size_t _n;
	mpz_t _Nz;
	std::vector<Zp> _N, _Ninv, _one;
	std::vector<Zp> _t_R, _t_Q;
	std::vector<Zp> _reg[reg_count];
	std::vector<Zp> _w_R, _wi_R, _w_Q, _wi_Q;
	bool _init_failed = false;

private:
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	// modulo x^n - 1
	void roots_R(Zp * const w, Zp * const wi)
	{
		const size_t n = _n;

		Zp sr_s = Zp::primroot_n(n), sr_si = sr_s.invert();	// (-1)^{1/s}
		for (size_t s = n / 2; s >= 1; s /= 2)
		{
			Zp t_j, ti_j; t_j.set(1u); ti_j.set(1u);
			for (size_t j = 0; j < s; ++j)
			{
				const size_t jr = bitrev(j, s);
				w[s + jr] = t_j; wi[s + jr] = ti_j;
				t_j *= sr_s; ti_j *= sr_si;
			}

			sr_s *= sr_s; sr_si *= sr_si;
		}
	}

	// modulo x^n + 1
	void roots_Q(Zp * const w, Zp * const wi)
	{
		const size_t n = _n;

		Zp sr_2s = Zp::primroot_n(2 * n), sr_2si = sr_2s.invert();	// (-1)^{1/2s}
		for (size_t s = n / 2; s >= 1; s /= 2)
		{
			const Zp sr_s = sr_2s * sr_2s, sr_si = sr_2si * sr_2si;	// (-1)^{1/s}

			Zp t_j = sr_2s, ti_j = sr_2si;
			for (size_t j = 0; j < s; ++j)
			{
				const size_t jr = bitrev(j, s);
				w[s + jr] = t_j; wi[s + jr] = ti_j;
				t_j *= sr_s; ti_j *= sr_si;
			}

			sr_2s = sr_s; sr_2si = sr_si;
		}
	}

	void set(Zp * const y, const Zp * const x)
	{
		const size_t n = _n;
		for (size_t k = 0; k < n; ++k) y[k] = x[k];
	}

	void set_mpz(Zp * const y, const mpz_t & x) const
	{
		const size_t n = _n;

		mpz_t s, r; mpz_init_set(s, x); mpz_init(r);
		for (size_t i = 0; i < n; ++i)
		{
			mpz_mod_2exp(r, s, LBASE);
			y[i].set(mpz_get_ui(r));
			mpz_div_2exp(s, s, LBASE);
		}
		mpz_clears(s, r, nullptr);
	}

	void get_mpz(mpz_t & y, const Zp * const x) const
	{
		const size_t n = _n;

		mpz_set_ui(y, 0);
		for (size_t i = n; i > 0; --i)
		{
			mpz_mul_2exp(y, y, LBASE);
			mpz_add_ui(y, y, x[i - 1].get());
		}
	}

	void NTT_forward(Zp * const x, const Zp * const w) const
	{
		const size_t n = _n;

		for (size_t m = n / 2, s = 1; m > 0; m /= 2, s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Zp sr = w[s + j];
				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m * j + i;
					const Zp u0 = x[k + 0 * m], u1 = sr * x[k + 1 * m];
					x[k + 0 * m] = u0 + u1;
					x[k + 1 * m] = u0 - u1;
				}
			}
		}
	}

	void NTT_inverse(Zp * const x, const Zp * const w_i) const
	{
		const size_t n = _n;

		for (size_t m = 1, s = n / 2; s > 0; m *= 2, s /= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Zp sr_inv = w_i[s + j];
				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m * j + i;
					const Zp u0 = x[k + 0 * m], u1 = x[k + 1 * m];
					x[k + 0 * m] = Zp(u0 + u1).half();	// TODO
					x[k + 1 * m] = Zp(u0 - u1).half() * sr_inv;
				}
			}
		}
	}

	void mul_pointwise(Zp * const y, const Zp * const x)
	{
		const size_t n = _n;
		for (size_t k = 0; k < n; ++k) y[k] *= x[k];
	}

	void carry_mul(Zp * const x, const int32_t a, const bool nega) const
	{
		const size_t n = _n;

		int64_t f = 0;
		for (size_t i = 0; i < n; ++i)
		{
			f += a * x[i].geti();
			int64_t r = f % BASE;	// TODO
			if (r < 0) r += BASE;
			f = (f - r) / BASE;
			x[i].set(uint32_t(r));
		}

		while (f != 0)
		{
			if (nega) f = -f;
			for (size_t i = 0; i < n; ++i)
			{
				f += x[i].geti();
				int64_t r = f % BASE;
				if (r < 0) r += BASE;
				f = (f - r) / BASE;
				x[i].set(uint32_t(r));
				if (f == 0) return;
			}
		}
	}

	void carry_sub(Zp * const x, const Zp * const Q, const bool nega) const
	{
		const size_t n = _n;

		int64_t f = 0;
		for (size_t i = 0; i < n; ++i)
		{
			f += x[i].geti();
			if (Q != nullptr) f -= Q[i].geti();
			int64_t r = f % BASE;
			if (r < 0) r += BASE;
			f = (f - r) / BASE;
			x[i].set(uint32_t(r));
		}

		while (f != 0)
		{
			if (nega) f = -f;
			for (size_t i = 0; i < n; ++i)
			{
				f += x[i].geti();
				int64_t r = f % BASE;
				if (r < 0) r += BASE;
				f = (f - r) / BASE;
				x[i].set(uint32_t(r));
				if (f == 0) return;
			}
		}
	}

	// Modular arithmetic

	void square_mul_mod_R(Zp * const z, const Zp * const x, const int32_t a)
	{
		set(z, x);
		NTT_forward(z, _w_R.data());
		mul_pointwise(z, z);
		NTT_inverse(z, _wi_R.data());
 		carry_mul(z, a, false);
	}

	void square_mul_mod_Q(Zp * const z, const Zp * const x, const int32_t a)
	{
		set(z, x);
		NTT_forward(z, _w_Q.data());
		mul_pointwise(z, z);
		NTT_inverse(z, _wi_Q.data());
 		carry_mul(z, a, true);
	}

	void mul_mod_R(Zp * const z, const Zp * const x, const Zp * const y)
	{
		set(z, x);
		NTT_forward(z, _w_R.data());
		std::vector<Zp> t(_n); set(t.data(), y);
		NTT_forward(t.data(), _w_R.data());
		mul_pointwise(z, t.data());
		NTT_inverse(z, _wi_R.data());
 		carry_mul(z, 1, false);
	}

	void mul_sub_mod_Q(Zp * const z, const Zp * const x, const Zp * const y, const Zp * const s)
	{
		set(z, x);
		NTT_forward(z, _w_Q.data());
		std::vector<Zp> t(_n); set(t.data(), y);
		NTT_forward(t.data(), _w_Q.data());
		mul_pointwise(z, t.data());
		NTT_inverse(z, _wi_Q.data());
 		carry_sub(z, s, true);
	}

	// Montgomery-McLaughlin-Gallot-Woltman multiplication

	void reduceMMGW(Zp * const x)
	{
		mul_mod_R(x, _t_R.data(), _Ninv.data());	// m'

		x[0].set(x[0].get() + 2);
		mul_sub_mod_Q(x, x, _N.data(), _t_Q.data());
	}

	void mulMMGW(Zp * const x, const Zp * const y)
	{
		mul_mod_R(_t_R.data(), x, y);
		mul_sub_mod_Q(_t_Q.data(), x, y, nullptr);
		reduceMMGW(x);
	}

	void squareMMGW(Zp * const x, const int32_t a)
	{
		square_mul_mod_R(_t_R.data(), x, a);
		square_mul_mod_Q(_t_Q.data(), x, a);
		reduceMMGW(x);
	}

public:
	void set1() override
	{
		set(_reg[0].data(), _one.data());
	}

	void get(mpz_t & z, const size_t src) override
	{
		// Conversion out of Montgomery-McLaughlin-Gallot-Woltman form
		set(_t_R.data(), _reg[src].data());
		set(_t_Q.data(), _reg[src].data());
		std::vector<Zp> t(_n);
		reduceMMGW(t.data());
		get_mpz(z, t.data());

		// 0 <= z < mod
		mpz_mod(z, z, _Nz);
	}

	void copy(const size_t dst, const size_t src) override
	{
		set(_reg[dst].data(), _reg[src].data());
	}

	void square_mul(const int32_t a) override
	{
		squareMMGW(_reg[0].data(), a);
	}

	void mul(const size_t dst, const size_t src) override
	{
		mulMMGW(_reg[dst].data(), _reg[src].data());
	}

	bool init_failed() const override { return _init_failed; }
	void get_factor(mpz_t & z) const override { mpz_set(z, _Nz); }

public:
	transformCPU(const mpz_t & N, const int32_t a, const bool verbose) : transform()
	{
		mpz_t R, t; mpz_inits(_Nz, R, t, nullptr);

		// R = BASE^n - 1, Q = BASE^n + 1 such that BASE^n > 16 * a * N and n is a power of two
		const size_t m = (mpz_sizeinbase(N, 2) + size_t(32 - __builtin_clz(uint32_t(a))) + 4 + LBASE - 1) / LBASE;
		size_t n = 1; while (n < m) n *= 2; _n = n;

		_N.resize(n); _Ninv.resize(n); _one.resize(n);
		_t_R.resize(n); _t_Q.resize(n);
		for (size_t i = 0; i < reg_count; ++i) _reg[i].resize(n);

		mpz_set_ui(R, 1); mpz_mul_2exp(R, R, mp_bitcnt_t(n * LBASE)); mpz_sub_ui(R, R, 1);
		if (mpz_invert(t, N, R) == 0)
		{
			mpz_gcd(_Nz, N, R);
			_init_failed = true;
			return;
		}
		else
		{
			mpz_set(_Nz, N);
			set_mpz(_N.data(), N);
			set_mpz(_Ninv.data(), t);
		}

		// The Montgomery-McLaughlin-Gallot-Woltman form of x is R/2 * x mod N
		mpz_set(t, R); mpz_mod(t, t, N);
		if (mpz_odd_p(t) != 0) mpz_add(t, t, N);
		mpz_div_2exp(t, t, 1);
		set_mpz(_one.data(), t);

		mpz_clears(R, t, nullptr);

		_w_R.resize(n); _wi_R.resize(n);
		roots_R(_w_R.data(), _wi_R.data());
		_w_Q.resize(n); _wi_Q.resize(n);
		roots_Q(_w_Q.data(), _wi_Q.data());

		if (verbose) std::cout << "Montgomery-McLaughlin-Gallot-Woltman reduction of length " << n << "x" << LBASE << " bits" << std::endl;
	}

	virtual ~transformCPU()
	{
		mpz_clear(_Nz);
	}
};
