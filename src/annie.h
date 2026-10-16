/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <string>
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

	static std::string mpz_to_short_string(const mpz_t & n)
	{
		const size_t size = mpz_sizeinbase(n, 10);
		std::string str; str.resize(size + 80);
		mpz_get_str(str.data(), 10, n);
		if (size <= 40) return str;
		return str.substr(0, 8) + "[" + std::to_string(size - 16) + " other digits]" + str.substr(size - 8, 8);
	}

	static constexpr int ilog2_32(const uint32_t n) { return (n == 0) ? -1 : (31 - __builtin_clz(n)); }

	static int B_GerbiczLi(const size_t size)
	{
		const size_t L = (size_t(2) << (ilog2_32(uint32_t(size)) / 2));
		return int((size - 1) / L) + 1;
	}

public:
	EReturn check(const mpz_t & n, const std::string & expression, const bool verbose = true)
	{
		const std::string ns = (expression.size() <= 40) ? expression : mpz_to_short_string(n);

		int32_t a = 2;
		while (mpz_si_kronecker(a, n) != -1) { ++a; const int32_t s = std::lround(std::sqrt(double(a))); if (s * s == a) ++a; }

		if (verbose) std::cout << "Running Euler probable primality test of " << ns << " to base " << a << std::endl;

		transform * const p_transform = new transformGMP(n, a, verbose);
		if (p_transform->init_failed())
		{
			if (verbose)
			{
				mpz_t f; mpz_init(f);
				p_transform->get_factor(f);
				const size_t size = mpz_sizeinbase(f, 10);
				std::string str; str.resize(size + 80);
				mpz_get_str(str.data(), 10, f);
				mpz_clear(f);
				std::cout << str << " divides " << ns << "." << std::endl;
			}
			delete _transform;
			return EReturn::Success;
		}

		_transform = p_transform;

		mpz_t exponent; mpz_init(exponent);
		mpz_set(exponent, n); mpz_sub_ui(exponent, exponent, 1); mpz_fdiv_q_2exp(exponent, exponent, 1);

		const int B_GL = B_GerbiczLi(mpz_sizeinbase(exponent, 2));

		p_transform->set1();
		p_transform->copy(1, 0);	// d(t)

		const int i0 = int(mpz_sizeinbase(exponent, 2)) - 1;
		for (int i = i0; i >= 0; --i)
		{
			if (quitting()) return EReturn::Aborted;

			p_transform->square_mul((mpz_tstbit(exponent, mp_bitcnt_t(i)) != 0) ? a : 1);

			if ((i % B_GL == 0) && (i / B_GL != 0))
			{
				p_transform->init_multiplicand(0);
				p_transform->mul(1);	// d(t)
			}
		}

		// save result to multiplicand
		p_transform->init_multiplicand(0);

		const bool is_prp = p_transform->is_minus_one(0);
		p_transform->square_mul(1);
		const uint64_t res64 = p_transform->get_res64(0);

		// d(t + 1) = d(t) * result
		p_transform->copy(2, 1);
		p_transform->mul(2);

		mpz_t res; mpz_init_set_ui(res, 0);
		mpz_t t; mpz_init(t);
		while (mpz_sgn(exponent) != 0)
		{
			mpz_mod_2exp(t, exponent, mp_bitcnt_t(B_GL));
			mpz_add(res, res, t);
			mpz_div_2exp(exponent, exponent, mp_bitcnt_t(B_GL));
		}
		mpz_clear(t); mpz_clear(exponent);

		const int res_size = int(mpz_sizeinbase(res, 2));

		// a^{res - 2^B}
		p_transform->set1();
		for (int i = res_size - 1; i >= B_GL; --i)
		{
			if (quitting()) { mpz_clear(res); return EReturn::Aborted; }
			p_transform->square_mul((mpz_tstbit(res, mp_bitcnt_t(i)) != 0) ? a : 1);
		}

		// a^{res - 2^B} * d(t)
		p_transform->init_multiplicand(1);
		p_transform->mul(0);

		// (a^{res - 2^B} * d(t))^2^B = a^res * d(t)^{2^B}
		for (int i = B_GL - 1; i >= 0; --i)
		{
			if (quitting()) return EReturn::Aborted;
			p_transform->square_mul(((i < res_size) && (mpz_tstbit(res, mp_bitcnt_t(i)) != 0)) ? a : 1);
		}

		mpz_clear(res);

		// d(t)^{2^B} * a^res ?= d(t + 1)
		const uint64_t h1 = p_transform->gethash64(0);
		const uint64_t h2 = p_transform->gethash64(2);
		const bool success = (h1 == h2);

		if (success)
		{
			if (verbose)
			{
				std::cout << ns << " is ";
				if (is_prp) std::cout << "a probable prime";
				else std::cout << "composite, res64 = " << uint64_to_string(res64);
				std::cout << ", a = " << a << "." << std::endl;
			}
			else if (is_prp) std::cout << ns << " is a probable prime, a = " << a << "." << std::endl;
		}

		delete _transform;
		_transform = nullptr;

		return success ? EReturn::Success : EReturn::Failed;
	}
};
