/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <gmp.h>

#include "transform.h"

class transformGMP : public transform
{
private:
	static const size_t reg_count = 3;
	mpz_t _mod, _regm;
	mpz_t _reg[reg_count];

public:
	void set(const uint32_t a) override
	{
		mpz_set_ui(_reg[0], a);
	}

	void get(mpz_t & z, const size_t src) const override
	{
		mpz_set(z, _reg[src]);
	}

	void copy(const size_t dst, const size_t src) override
	{
		mpz_set(_reg[dst], _reg[src]);
	}

	void square_mul(const int32_t a) override
	{
		mpz_t & r = _reg[0];
		mpz_mul(r, r, r);
		if (a != 1) mpz_mul_si(r, r, a);
		mpz_mod(r, r, _mod);
	}

	void init_multiplicand(const size_t src) override
	{
		mpz_set(_regm, _reg[src]);
	}

	void mul(const size_t dst) override
	{
		mpz_t & r = _reg[dst];
		mpz_mul(r, r, _regm);
		mpz_mod(r, r, _mod);
	}

public:
	transformGMP(const mpz_t & mod) : transform(mod)
	{
		mpz_init_set(_mod, mod); mpz_init(_regm);
		for (size_t i = 0; i < reg_count; ++i) mpz_init(_reg[i]);
	}

	virtual ~transformGMP()
	{
		mpz_clear(_mod); mpz_clear(_regm);
		for (size_t i = 0; i < reg_count; ++i) mpz_clear(_reg[i]);
	}
};
