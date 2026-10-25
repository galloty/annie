/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <iostream>

#include "transform.h"

class transformGMP : public transform
{
private:
	static const size_t reg_count = 3;
	mp_bitcnt_t _n;
	mpz_t _N, _Ninv, _R, _Q, _one;
	mpz_t _t_R, _t_Q, _t;
	mpz_t _reg[reg_count];
	bool _init_failed = false;

private:
	// Modular arithmetic

	// mod R = 2^n - 1: 0 <= z <= R
	void mod_R(mpz_t & z)
	{
		const mp_bitcnt_t n = _n;
		while (mpz_sizeinbase(z, 2) > n)
		{
			mpz_div_2exp(_t, z, n);
			mpz_mod_2exp(z, z, n);
			mpz_add(z, z, _t);
		}
	}

	// mod Q = 2^n + 1: 0 <= z <= Q - 2
	void mod_Q(mpz_t & z)
	{
		const mp_bitcnt_t n = _n;
		while (mpz_sizeinbase(z, 2) > 2 * n)
		{
			mpz_div_2exp(_t, z, 2 * n);
			mpz_mod_2exp(z, z, 2 * n);
			mpz_add(z, z, _t);
		}

		mpz_div_2exp(_t, z, n);
		mpz_mod_2exp(z, z, n);
		mpz_sub(z, z, _t);

		if ((mpz_sgn(z) < 0)) mpz_add(z, z, _Q);
	}

	void square_mul_mod_R(mpz_t & z, const mpz_t & x, const int32_t a)
	{
		mpz_mul_si(z, x, a);
		mpz_mul(z, z, x);
		mod_R(z);
	}

	void square_mul_mod_Q(mpz_t & z, const mpz_t & x, const int32_t a)
	{
		mpz_mul_si(z, x, a);
		mpz_mul(z, z, x);
		mod_Q(z);
	}

	void mul_mod_R(mpz_t & z, const mpz_t & x, const mpz_t & y)
	{
		mpz_mul(z, x, y);
		mod_R(z);
	}

	void mul_mod_Q(mpz_t & z, const mpz_t & x, const mpz_t & y)
	{
		mpz_mul(z, x, y);
		mod_Q(z);
	}

	void sub_mod_Q(mpz_t & z, const mpz_t & x, const mpz_t & y)
	{
		mpz_sub(z, x, y);
		if ((mpz_sgn(z) < 0)) mpz_add(z, z, _Q);
	}

	// Montgomery-McLaughlin-Gallot-Woltman multiplication

	void reduceMMGW(mpz_t & x)
	{
		mul_mod_R(x, _t_R, _Ninv);	// m'

		mpz_add_ui(x, x, 2);
		mul_mod_Q(x, x, _N);
		sub_mod_Q(x, x, _t_Q);
	}

	void mulMMGW(mpz_t & x, const mpz_t & y)
	{
		mul_mod_R(_t_R, x, y);
		mul_mod_Q(_t_Q, x, y);
		reduceMMGW(x);
	}

	void squareMMGW(mpz_t & x, const int32_t a)
	{
		square_mul_mod_R(_t_R, x, a);
		square_mul_mod_Q(_t_Q, x, a);
		reduceMMGW(x);
	}

public:
	void set1() override
	{
		mpz_set(_reg[0], _one);
	}

	void get(mpz_t & z, const size_t src) override
	{
		// Conversion out of Montgomery-McLaughlin-Gallot-Woltman form
		mpz_set(_t_R, _reg[src]);
		mpz_set(_t_Q, _reg[src]);
		reduceMMGW(z);
		// 0 <= z < mod
		mpz_mod(z, z, _N);
	}

	void copy(const size_t dst, const size_t src) override
	{
		mpz_set(_reg[dst], _reg[src]);
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
	void get_factor(mpz_t & z) const override { mpz_set(z, _Ninv); }

	bool read_checkpoint(file &) override { return false; }
	void save_checkpoint(file &) const override {}

public:
	transformGMP(const mpz_t & N, const int32_t a, const bool verbose) : transform()
	{
		mpz_init_set(_N, N); mpz_inits(_Ninv, _R, _Q, _one, _t_R, _t_Q, _t, nullptr);
		for (size_t i = 0; i < reg_count; ++i) mpz_init(_reg[i]);

		// R = 2^n - 1, Q = 2^n + 1 such that 2^n > 16 * a * N
		_n = mp_bitcnt_t(mpz_sizeinbase(N, 2) + size_t(32 - __builtin_clz(uint32_t(a))) + 4);
		mpz_set_ui(_R, 1); mpz_mul_2exp(_R, _R, _n); mpz_set(_Q, _R);
		mpz_sub_ui(_R, _R, 1); mpz_add_ui(_Q, _Q, 1);
		if (mpz_invert(_Ninv, N, _R) == 0)
		{
			mpz_gcd(_Ninv, N, _R);
			_init_failed = true;
			return;
		}

		// The Montgomery-McLaughlin-Gallot-Woltman form of x is R/2 * x mod N
		mpz_set(_one, _R); mpz_mod(_one, _one, _N);
		if (mpz_odd_p(_one) != 0) mpz_add(_one, _one, _N);
		mpz_div_2exp(_one, _one, 1);

		if (verbose) std::cout << "Montgomery-McLaughlin-Gallot-Woltman reduction of length " << _n << " bits" << std::endl;
	}

	virtual ~transformGMP()
	{
		mpz_clears(_N, _Ninv, _R, _Q, _one, _t_R, _t_Q, _t, nullptr);
		for (size_t i = 0; i < reg_count; ++i) mpz_clear(_reg[i]);
	}
};
