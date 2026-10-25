/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <string>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <atomic>

#include <gmp.h>

#include "transformGMP.h"
#include "transformCPU.h"
#include "file.h"
#include "timer.h"

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
	transform * _transform = nullptr;
	std::string _checkpoint_filename;
	int _print_range = 0, _print_i = 0;

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

	static size_t digits(const mpz_t & n)
	{
		const size_t sizeinbase10 = mpz_sizeinbase(n, 10);	// The result is exact or one too big
		std::string str; str.resize(sizeinbase10 + 64);
		mpz_get_str(str.data(), 10, n);
		return std::strlen(str.c_str());
	}

	static std::string mpz_to_short_string(const mpz_t & n)
	{
		const size_t sizeinbase10 = mpz_sizeinbase(n, 10);
		std::string str; str.resize(sizeinbase10 + 64);
		mpz_get_str(str.data(), 10, n);
		const size_t size = std::strlen(str.c_str());
		if (size <= 40) return str;
		return str.substr(0, 8) + "[" + std::to_string(size - 16) + " other digits]" + str.substr(size - 8, 8);
	}

	static constexpr int ilog2_32(const uint32_t n) { return (n == 0) ? -1 : (31 - __builtin_clz(n)); }

	static int B_GerbiczLi(const size_t size)
	{
		const size_t L = (size_t(2) << (ilog2_32(uint32_t(size)) / 2));
		return int((size - 1) / L) + 1;
	}

	void init_print_progress(const int i0, const int i_start)
	{
		_print_range = i0; _print_i = i_start;
	}

	int print_progress(const double elapsed_time, const double display_time, const int i)
	{
		if (_print_i == i) return 1;
		const double percent = double(_print_range - i) / _print_range;
		const double mul_time = display_time / (_print_i - i); _print_i = i;
		const int dcount = std::max(int(1.0 / mul_time), 2);
		const double remaining_time = mul_time * i, expected_time = elapsed_time / percent;
		std::cout << std::setprecision(3) << percent * 100.0 << "% done, " << timer::format_time(remaining_time)
				<< "/" << timer::format_time(expected_time) << " remaining, " << mul_time * 1e3 << " ms/bit.        \r";
		return dcount;
	}

	static void endline() { std::cout << std::endl; }
	static void clearline() { std::cout << "                                                            \r" << std::flush; }

	int _read_checkpoint(const std::string & filename, int & i, double & elapsed_time)
	{
		file checkpoint_file(filename);
		if (!checkpoint_file.exists()) return -1;

		int version = 0;
		if (!checkpoint_file.read(reinterpret_cast<char *>(&version), sizeof(version))) return -2;
		if (version != 1) return -2;
		if (!checkpoint_file.read(reinterpret_cast<char *>(&i), sizeof(i))) return -2;
		if (!checkpoint_file.read(reinterpret_cast<char *>(&elapsed_time), sizeof(elapsed_time))) return -2;
		if (!_transform->read_checkpoint(checkpoint_file)) return -2;
		if (!checkpoint_file.check_crc32()) return -2;
		return 0;
	}

	bool read_checkpoint(int & i, double & elapsed_time)
	{
		std::string checkpoint_file = _checkpoint_filename;
		int error = _read_checkpoint(checkpoint_file, i, elapsed_time);
		if (error < -1)
		{
			std::ostringstream ss; ss << "Error: " << checkpoint_file << ": invalid checkpoint." << std::endl;
			std::cerr << ss.str();
		}
		checkpoint_file += ".old";
		if (error < 0)
		{
			error = _read_checkpoint(checkpoint_file, i, elapsed_time);
			if (error < -1)
			{
				std::ostringstream ss; ss << "Error: " << checkpoint_file << ": invalid checkpoint." << std::endl;
				std::cerr << ss.str();
			}
		}
		return (error == 0);
	}

	void save_checkpoint(const int i, const double elapsed_time) const
	{
		const std::string checkpoint_file = _checkpoint_filename, old_checkpoint_file = checkpoint_file + ".old", new_checkpoint_file = checkpoint_file + ".new";

		{
			file checkpoint_file(new_checkpoint_file, "wb", false);
			int version = 1;
			if (!checkpoint_file.write(reinterpret_cast<const char *>(&version), sizeof(version))) return;
			if (!checkpoint_file.write(reinterpret_cast<const char *>(&i), sizeof(i))) return;
			if (!checkpoint_file.write(reinterpret_cast<const char *>(&elapsed_time), sizeof(elapsed_time))) return;
			_transform->save_checkpoint(checkpoint_file);
			checkpoint_file.write_crc32();
		}

		std::remove(old_checkpoint_file.c_str());

		struct stat s;
		if ((stat(checkpoint_file.c_str(), &s) == 0) && (std::rename(checkpoint_file.c_str(), old_checkpoint_file.c_str()) != 0))	// file exists and cannot rename it
		{
			std::cerr << "Error: cannot save checkpoint." << std::endl;
		}

		if (std::rename(new_checkpoint_file.c_str(), checkpoint_file.c_str()) != 0)
		{
			std::cerr << "Error: cannot save checkpoint." << std::endl;
		}
	}

	void clear_checkpoints() const
	{
		const std::string checkpoint_file = _checkpoint_filename;
		std::remove(checkpoint_file.c_str());
		std::remove(std::string(checkpoint_file + ".old").c_str());
	}

public:
	EReturn check(const mpz_t & n, const std::string & expression, const bool verbose = true)
	{
		const size_t nsize = digits(n);
		const std::string ns = (expression.size() <= 40) ? expression : mpz_to_short_string(n);

		if (nsize <= 15)
		{
			const int r = mpz_probab_prime_p(n, 20);
			if (r != 0)
			{
				const char * const prp = (r == 1) ? "a probable " : "";
				std::cout << ns << " is " << prp << "prime." << std::endl;
			}
			else if (verbose) std::cout << ns << " is composite." << std::endl;
			return EReturn::Success;
		}

		if (mpz_even_p(n) != 0)
		{
			if (verbose) std::cout << ns << " is composite." << std::endl;
			return EReturn::Success;
		}

		int32_t a = 2;
		while (mpz_si_kronecker(a, n) != -1) { ++a; const int32_t s = std::lround(std::sqrt(double(a))); if (s * s == a) ++a; }

		if (verbose) std::cout << "Running Euler probable primality test of " << ns << " to base " << a << " [" << nsize << " digits]" << std::endl;

		transform * const p_transform = (nsize < 80) ? (transform *)(new transformGMP(n, a, verbose)) : (transform *)(new transformCPU(n, a, verbose));
		if (p_transform->init_failed())
		{
			if (verbose)
			{
				mpz_t f; mpz_init(f);
				p_transform->get_factor(f);
				std::string str; str.resize(mpz_sizeinbase(f, 10) + 64);
				mpz_get_str(str.data(), 10, f);
				mpz_clear(f);
				std::cout << str << " divides " << ns << "." << std::endl;
			}
			delete _transform;
			return EReturn::Success;
		}

		_transform = p_transform;

		std::ostringstream ss; ss << "a_" << nsize << ".chk";
		_checkpoint_filename = ss.str();

		mpz_t exponent; mpz_init(exponent);
		mpz_set(exponent, n); mpz_sub_ui(exponent, exponent, 1); mpz_fdiv_q_2exp(exponent, exponent, 1);

		const int B_GL = B_GerbiczLi(mpz_sizeinbase(exponent, 2));

		int ri = 0; double restored_time = 0;
		const bool found = read_checkpoint(ri, restored_time);
		if (!found)
		{
			ri = 0; restored_time = 0;
			p_transform->set1();
			p_transform->copy(1, 0);	// d(t)
		}
		else std::cout << "Resuming from a checkpoint." << std::endl;

		watch chrono(found ? restored_time : 0);
		const int i0 = int(mpz_sizeinbase(exponent, 2)) - 1;
		const int i_start = found ? ri : i0;
		init_print_progress(i0, i_start);
		int dcount = 100;
		
		for (int i = i_start; i >= 0; --i)
		{
			if (quitting())
			{
				save_checkpoint(i, chrono.get_elapsed_time());
				endline();
				return EReturn::Aborted;
			}

			if (i % dcount == 0)
			{
				const double elapsed_time = chrono.get_elapsed_time(), displayTime = chrono.get_display_time();
				if (displayTime >= 10) { dcount = print_progress(elapsed_time, displayTime, i); chrono.reset_display_time(); }
				if (chrono.get_record_time() > 600) { save_checkpoint(i, chrono.get_elapsed_time()); chrono.reset_record_time(); }
			}

			p_transform->square_mul((mpz_tstbit(exponent, mp_bitcnt_t(i)) != 0) ? a : 1);

			if ((i % B_GL == 0) && (i / B_GL != 0))
			{
				p_transform->mul(1, 0);	// d(t)
			}
		}

		// d(t + 1) = d(t) * result
		p_transform->copy(2, 0);
		p_transform->mul(2, 1);

		mpz_t t; mpz_init(t);

		// result ?= -1
		p_transform->get(t, 0);
		mpz_add_ui(t, t, 1); if (mpz_cmp(t, n) >= 0) mpz_sub(t, t, n);
		const bool is_prp = (mpz_sgn(t) == 0);

		// a^(n - 1) mod n
		p_transform->square_mul(1);

		// 64-bit residue 
		p_transform->get(t, 0);
		const uint64_t res64 = (mpz_size(t) != 0) ? uint64_t(mpz_getlimbn(t, 0)) : uint64_t(0);

		mpz_t res; mpz_init_set_ui(res, 0);
		while (mpz_sgn(exponent) != 0)
		{
			mpz_mod_2exp(t, exponent, mp_bitcnt_t(B_GL));
			mpz_add(res, res, t);
			mpz_div_2exp(exponent, exponent, mp_bitcnt_t(B_GL));
		}
		mpz_clears(t, exponent, nullptr);

		const int res_size = int(mpz_sizeinbase(res, 2));

		// a^{res - 2^B}
		p_transform->set1();
		for (int i = res_size - 1; i >= B_GL; --i)
		{
			if (quitting()) { mpz_clear(res); endline(); return EReturn::Aborted; }
			p_transform->square_mul((mpz_tstbit(res, mp_bitcnt_t(i)) != 0) ? a : 1);
		}

		// a^{res - 2^B} * d(t)
		p_transform->mul(0, 1);

		// (a^{res - 2^B} * d(t))^2^B = a^res * d(t)^{2^B}
		for (int i = B_GL - 1; i >= 0; --i)
		{
			if (quitting()) { mpz_clear(res); endline(); return EReturn::Aborted; }
			p_transform->square_mul(((i < res_size) && (mpz_tstbit(res, mp_bitcnt_t(i)) != 0)) ? a : 1);
		}

		mpz_clear(res);

		// d(t)^{2^B} * a^res ?= d(t + 1)
		mpz_t r1, r2; mpz_inits(r1, r2, nullptr);
		p_transform->get(r1, 0);
		p_transform->get(r2, 2);
		const bool success = (mpz_cmp(r1, r2) == 0);
		mpz_clears(r1, r2, nullptr);

		delete _transform;
		_transform = nullptr;

		const double elapsed_time = chrono.get_elapsed_time();
		clearline();

		if (success)
		{
			if (is_prp)
			{
				std::cout << ns << " is a probable prime, a = " << a << ", time = " << timer::format_time(elapsed_time) << "." << std::endl;
			}
			else if (verbose)
			{
				std::cout << ns << " is composite, res64 = " << uint64_to_string(res64) << ", a = " << a
					<< ", time = " << timer::format_time(elapsed_time) << "." << std::endl;
			}
		}

		clear_checkpoints();
		return success ? EReturn::Success : EReturn::Failed;
	}
};
