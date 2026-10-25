/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <gmp.h>

#include "file.h"

class transform
{
public:
	virtual void set1() = 0;										// r_0 = 1
	virtual void get(mpz_t & z, const size_t src) = 0;				// z = r_0
	virtual void copy(const size_t dst, const size_t src) = 0;		// r_dst = r_src
	virtual void square_mul(const int32_t a) = 0;					// r_0 = a * r_0^2
	virtual void mul(const size_t dst, const size_t src) = 0;		// r_dst *= r_src
	virtual bool init_failed() const = 0;
	virtual void get_factor(mpz_t & z) const = 0;

	virtual bool read_checkpoint(file & cfile) = 0;
	virtual void save_checkpoint(file & cfile) const = 0;

public:
	transform() {}
	virtual ~transform() {}
};
