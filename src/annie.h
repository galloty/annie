/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>
#include <atomic>

#include <gmp.h>

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

public:
	EReturn check(const mpz_t & n, const std::string & expression)
	{
		long a = 2;
		while (mpz_si_kronecker(a, n) != -1) { ++a; const long s = std::lround(std::sqrt(double(a))); if (s * s == a) ++a; }

		mpz_t base, exponent, res;
		mpz_inits(base, exponent, res, nullptr);
		mpz_set_si(base, a);
		mpz_set(exponent, n); mpz_sub_ui(exponent, exponent, 1); mpz_fdiv_q_2exp(exponent, exponent, 1);
		mpz_powm(res, base, exponent, n);
		mpz_set(base, n); mpz_sub_ui(base, base, 1);
		const bool is_prp = (mpz_cmp(res, base) == 0);
		mpz_clears(base, exponent, res, nullptr);

		const char * const ns = (expression.size() <= 40) ? expression.c_str() : "input";
		const char * const status = is_prp ? "a probable prime" : "composite";
		std::cout << ns << " is " << status << ", a = " << a << "." << std::endl;

		return EReturn::Success;
	}
};
