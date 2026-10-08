/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <iomanip>
#include <cmath>
#include <atomic>

#include <gmp.h>

#include "transformGMP.h"

class annie
{
public:
	enum class EReturn { Success, Failed, Aborted };

private:
	struct deleter { void operator()(const annie * const p) { delete p; } };

public:
	annie() {}
	virtual ~annie() {}

private:
	transform * _transform = nullptr;
	std::atomic_bool _quit = false;

protected:
	bool quitting() const
	{
		const bool b = _quit.load(std::memory_order_relaxed);
		if (b) std::atomic_thread_fence(std::memory_order_acquire);
		return b;
	}

public:
	// static size_t display_devices() { /*return itransform::display_devices();*/ }

	void quit()
	{
		std::atomic_thread_fence(std::memory_order_release);
		_quit.store(true, std::memory_order_relaxed);
	}

private:
	// void display_info(const bool full) const
	// {
		// std::ostringstream ss; ss << "Using " << _transform->get_type() << " implementation";
		// if (!_transform->get_gpu_type().empty()) ss << "." << std::endl << "Running on " << _transform->get_gpu_type();
		// else if (VSIZE / 8 > 1) ss << ", " << (VSIZE / 8) << " threads";
		// if (full)
		// {
		// 	ss << ", data size: " << std::lrint(_transform->get_data_size() / (1024 * 1024.0)) << " MB";
		// 	ss << ", cache size: " << std::setprecision(3) << _transform->get_cache_size() / (1024 * 1024.0) << " MB";
		// }
		// ss << "." << std::endl;
		// pio::print(ss.str());
	// }

	static std::string uint64_to_string(const uint64_t u)
	{
		std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << u;
		return ss.str();
	}

	static constexpr int ilog2_32(const uint32_t n) { return (n == 0) ? -1 : (31 - __builtin_clz(n)); }

	static int B_GerbiczLi(const size_t size)
	{
		const size_t L = (size_t(2) << (ilog2_32(uint32_t(size)) / 2));
		return int((size - 1) / L) + 1;
	}

public:
	EReturn check(const mpz_t & n, const std::string & expression)
	{
		int32_t a = 2;
		while (mpz_si_kronecker(a, n) != -1) { ++a; const int32_t s = std::lround(std::sqrt(double(a))); if (s * s == a) ++a; }

		transform * const p_transform = new transformGMP(n);
		_transform = p_transform;

		mpz_t exponent; mpz_init(exponent);
		mpz_set(exponent, n); mpz_sub_ui(exponent, exponent, 1); mpz_fdiv_q_2exp(exponent, exponent, 1);

		p_transform->set(1);

		const int i0 = int(mpz_sizeinbase(exponent, 2)) - 1;
		for (int i = i0; i >= 0; --i)
		{
			if (quitting()) return EReturn::Aborted;

			p_transform->square_mul((mpz_tstbit(exponent, mp_bitcnt_t(i)) != 0) ? a : 1);
		}

		const bool is_prp = p_transform->is_minus_one();

		p_transform->square_mul(1);
		p_transform->get_res64();
		const uint64_t res64 = p_transform->get_res64();

		mpz_clear(exponent);

		const char * const ns = (expression.size() <= 40) ? expression.c_str() : "input";
		std::cout << ns << " is ";
		if (is_prp) std::cout << "a probable prime";
		else std::cout << "composite, res64 = " << uint64_to_string(res64);
		std::cout << ", a = " << a << "." << std::endl;

		delete _transform;
		_transform = nullptr;

		return EReturn::Success;
	}
};
