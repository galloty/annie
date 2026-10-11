/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

class transform
{
private:
	mpz_t _mod;

private:
	static constexpr uint64_t rotl64(const uint64_t x, const uint8_t n) { return (x << n) | (x >> (-n & 63)); }

public:
	virtual void set(const uint32_t a) = 0;							// r_0 = a
	virtual void get(mpz_t & z, const size_t src) const = 0;		// z = r_0
	virtual void copy(const size_t dst, const size_t src) = 0;		// r_dst = r_src
	virtual void square_mul(const int32_t a) = 0;					// r_0 = a * r_0^2
	virtual void init_multiplicand(const size_t src) = 0;			// r_m = transform(r_src)
	virtual void mul(const size_t dst) = 0;							// r_dst *= r_m

	bool is_minus_one(const size_t src) const	// r_src ?= -1
	{
		mpz_t t; mpz_init(t); get(t, src);
		mpz_add_ui(t, t, 1); if (mpz_cmp(t, _mod) >= 0) mpz_sub(t, t, _mod);
		const bool b = (mpz_sgn(t) == 0);
		mpz_clear(t);
		return b;
	}

	uint64_t get_res64(const size_t src) const	// r_src mod 2^64
	{
		mpz_t t; mpz_init(t); get(t, src);
		const uint64_t res = (mpz_size(t) != 0) ? uint64_t(mpz_getlimbn(t, 0)) : uint64_t(0);
		mpz_clear(t);
		return res;
	}

	uint64_t gethash64(const size_t src) const
	{
		mpz_t t; mpz_init(t); get(t, src);
		uint64_t hash = 0;
		for (size_t i = 0, size = mpz_size(t); i < size; ++i)
		{
			const uint64_t a_i = uint64_t(mpz_getlimbn(t, i));
			hash += a_i;
			hash ^= rotl64(a_i + 0xc39d8a0552b073e8ull, (17 * a_i + 5) % 64);
		}
		mpz_clear(t);
		return hash;
	}

public:
	transform(const mpz_t & mod) { mpz_init_set(_mod, mod); }
	virtual ~transform() { mpz_clear(_mod);}
};
