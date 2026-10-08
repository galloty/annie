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
	static const size_t reg_count = 1;
	mpz_t _mod;
	mpz_t _reg[reg_count];

public:
	void set(const uint32_t a) override
	{
		mpz_set_ui(_reg[0], a);
	}

	void square_mul(const int32_t a) override
	{
		mpz_t & r = _reg[0];
		mpz_mul(r, r, r);
		if (a != 1) mpz_mul_si(r, r, a);
		mpz_mod(r, r, _mod);
	}

	bool is_minus_one() const override
	{
		mpz_t t; mpz_init(t);
		mpz_add_ui(t, _reg[0], 1); if (mpz_cmp(t, _mod) >= 0) mpz_sub(t, t, _mod);
		const bool b = (mpz_sgn(t) == 0);
		mpz_clear(t);
		return b;
	}

	uint64_t get_res64() const override
	{
		return (mpz_size(_reg[0]) != 0) ? uint64_t(mpz_getlimbn(_reg[0], 0)) : uint64_t(0);
	}

public:
	transformGMP(const mpz_t & mod)
	{
		mpz_init_set(_mod, mod);
		for (size_t i = 0; i < reg_count; ++i) mpz_init(_reg[i]);
	}

	virtual ~transformGMP()
	{
		for (size_t i = 0; i < reg_count; ++i) mpz_clear(_reg[i]);
		mpz_clear(_mod);
	}
};
