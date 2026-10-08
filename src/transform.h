/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

class transform
{
public:
	virtual void set(const uint32_t a) = 0;					// r_0 = a
	virtual void square_mul(const int32_t a) = 0;			// r_0 = a*r_0^2

	virtual bool is_minus_one() const = 0;					// r_0 ?= -1
	virtual uint64_t get_res64() const = 0;					// r_0 mod 2^64

public:
	transform() {}
	virtual ~transform() {}
};
