/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <gmp.h>

#include "transform.h"
#include "engine.h"
#include "ocl/kernel.h"

#define finline	__attribute__((always_inline))

// The prime finite field Z/PZ
template<uint32 P, uint32 Q, uint32 MR, uint32 RSQ, uint32 MPR>
class ZPT : public ZP
{
private:
	finline static uint32 _add(const uint32 a, const uint32 b) { return a + b - ((a >= P - b) ? P : 0); }
	finline static uint32 _sub(const uint32 a, const uint32 b) { return a - b + ((a < b) ? P : 0); }

	// Montgomery multiplication: 2 mul + 2 mul_hi
	finline static uint32 _mul(const uint32 a, const uint32 b)
	{
		const uint64 t = a * uint64(b);
		const uint32 lo = uint32(t), hi = uint32(t >> 32);
		const uint32 mp = uint32(((lo * Q) * uint64(P)) >> 32);
		return _sub(hi, mp);
	}

	static ZPT one() { return ZPT(MR); }

	// Conversion into / out of Montgomery form
	ZPT toMonty() const { return ZPT(_mul(_n, RSQ)); }
	// ZPT fromMonty() const { return ZPT(_mul(_n, 1)); }

	ZPT & operator*=(const ZPT & rhs) { _n = _mul(_n, rhs._n); return *this; }
	ZPT operator*(const ZPT & rhs) const { ZPT r = *this; r *= rhs; return r; }

	ZPT pow(const uint64 e) const
	{
		if (e == 0) return ZPT(MR);
		ZPT r = ZPT(MR), y = *this;
		for (uint64 i = e; i > 1; i /= 2)
		{
			if (i % 2 != 0) r *= y;
			y *= y;
		}
		r *= y;
		return r;
	}

	ZPT invert() const { return pow(P - 2); }
	static ZPT primroot_n(const int ln) { return ZPT(MPR).pow((P - 1) >> ln); }

	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

public:
	ZPT() {}
	explicit ZPT(const ZP & n) : ZP(n) {}
	explicit ZPT(const uint32 n) : ZP(n) {}

	int32 get_int() const { return int32(_n - ((_n >= P / 2) ? P : 0)); }

	static ZPT norm(const int ln) { return ZPT(P - ((P - 1) >> ln)); }

	// modulo x^n - 1
	static void set_roots_R(ZP * const w_R, ZP * const wi_R, const int ln)
	{
		const size_t n = size_t(1) << ln;

		ZPT sr_s = primroot_n(ln - 1), sr_si = sr_s.invert();	// (-1)^{1/s}
		for (size_t s = n / 4; s >= 1; s /= 2)
		{
			ZPT t_j = one(), ti_j = one();
			for (size_t j = 0; j < s; ++j)
			{
				const size_t jr = bitrev(j, s);
				w_R[s + jr] = t_j; wi_R[s + jr] = ti_j;
				t_j *= sr_s; ti_j *= sr_si;
			}

			sr_s *= sr_s; sr_si *= sr_si;
		}

		w_R[0] = ZPT(w_R[1]).toMonty();
	}

	// modulo x^n + 1
	static void set_roots_Q(ZP * const w_Q, ZP * const wi_Q, const int ln)
	{
		const size_t n = size_t(1) << ln;

		ZPT sr_2s = primroot_n(ln), sr_2si = sr_2s.invert();	// (-1)^{1/2s}
		for (size_t s = n / 4; s >= 1; s /= 2)
		{
			const ZPT sr_s = sr_2s * sr_2s, sr_si = sr_2si * sr_2si;	// (-1)^{1/s}

			ZPT t_j = sr_2s, ti_j = sr_2si;
			for (size_t j = 0; j < s; ++j)
			{
				const size_t jr = bitrev(j, s);
				w_Q[s + jr] = t_j; wi_Q[s + jr] = ti_j;
				t_j *= sr_s; ti_j *= sr_si;
			}

			sr_2s = sr_s; sr_2si = sr_si;
		}

		w_Q[0] = ZPT(w_Q[1]).toMonty();
	}
};

#define P1			(125 * (1u << 25) + 1)
#define	Q1			100663297u		// p * q = 1 (mod 2^32)
#define	MR1			100663295u		// 2^32 mod p = Montgomery form of 1
#define	RSQ1		232465106u		// (2^32)^2 mod p
#define	MPR1		301989885u		// Montgomery form of the primitive root 3

#define P2			(243 * (1u << 24) + 1)
#define	Q2			218103809u
#define	MR2			218103807u
#define	RSQ2		3444438393u
#define	MPR2		1526726649u		// MF of the primitive root 7

#define	INVP2_P1	1797558821u		// Montgomery form of 1 / P2 (mod P1)

using ZP1 = ZPT<P1, Q1, MR1, RSQ1, MPR1>;
using ZP2 = ZPT<P2, Q2, MR2, RSQ2, MPR2>;

class transformGPU : public transform
{
private:
	static const size_t Reg_w_R = 3, Reg_w_Q = 4, Reg_N_Q = 5, Reg_Ninv_R = 6, Reg_t_Q = 7, Reg_t = 8, Reg_one = 9, Reg_count = 10;
	int _ln = 0;
	int _lbase = 0;
	bool _init_failed = false;
	mpz_t _Nz;
	engine * _engine = nullptr;
	ZP * _buf = nullptr;
	std::string _hinfo;

private:
	finline static constexpr mp_limb_t read_limb(const mp_limb_t * const x_ptr, const size_t x_size, const size_t i)
	{
		return (i < x_size) ? x_ptr[i] : 0;
	}

	finline static constexpr uint32 read_word(const mp_limb_t * const x_ptr, const size_t x_size, const size_t bit_index)
	{
		const size_t limb_index = bit_index / 64;
		const int limb_shift = int(bit_index % 64);
		const mp_limb_t rl = read_limb(x_ptr, x_size, limb_index), rh = read_limb(x_ptr, x_size, limb_index + 1);
		mp_limb_t r = rl >> limb_shift;
		if (limb_shift != 0) r |= rh << (64 - limb_shift);
		return uint32(r);
	}

	finline static constexpr void write_word(mp_limb_t * const x_ptr, const size_t bit_index, const uint32 r)
	{
		const size_t limb_index = bit_index / 64;
		const int limb_shift = int(bit_index % 64);
		x_ptr[limb_index] |= mp_limb_t(r) << limb_shift;
		if (limb_shift != 0) x_ptr[limb_index + 1] |= mp_limb_t(r) >> (64 - limb_shift);
	}

	void set_mpz(const size_t reg_y, const mpz_t & x)
	{
		const size_t n = size_t(1) << _ln;
		const int lbase = _lbase;
		const uint32 base = 1u << lbase;
		ZP * const buf = _buf;

		const size_t x_size = mpz_size(x);
		const mp_limb_t * const x_ptr = mpz_limbs_read(x);

		size_t bit_index = 0;
		for (size_t i = 0; i < n; ++i)
		{
			const uint32 r = read_word(x_ptr, x_size, bit_index);
			buf[i] = ZP(r & (base - 1));
			bit_index += size_t(lbase);
		}

		engine * const eng = _engine;
		eng->write_memory(reg_y, buf);
		eng->write_memory(Reg_count + reg_y, buf);
	}

	void get_mpz(mpz_t & y, const size_t reg_x)
	{
		const int ln = _ln;
		const size_t n = size_t(1) << ln;
		const int lbase = int(_lbase);
		const uint32 base = 1u << lbase;
		ZP * const buf = _buf;

		_engine->read_memory(reg_x, buf);

		const size_t y_size = (size_t(lbase) << ln) / 64 + 2;
		mp_limb_t * const y_ptr = mpz_limbs_write(y, y_size);
		for (size_t i = 0; i < y_size; ++i) y_ptr[i] = 0;


		int64 f = 0;
		size_t bit_index = 0;
		for (size_t i = 0; i < n; ++i)
		{
			f += ZP1(buf[i]).get_int();
			const uint32 r = uint32_t(f) & (base - 1);
			f >>= lbase;

			write_word(y_ptr, bit_index, r);
			bit_index += size_t(lbase);
		}

		mp_size_t s = mp_size_t(y_size);
		while ((s > 0) && (y_ptr[s - 1] == 0)) --s;
 		mpz_limbs_finish(y, s);

		if (f != 0)
		{
			mpz_t t; mpz_init_set_si(t, long(f));
			mpz_mul_2exp(t, t, mp_bitcnt_t(lbase) << ln);
			mpz_add(y, y, t);
			mpz_clear(t);
		}
	}

	finline void NTT_forward8(const size_t reg_x, const size_t reg_w)
	{
		const int ln = _ln;
		engine * const eng = _engine;

		if (ln <= 12) eng->forward8_0(reg_x, reg_w);
		else if (ln <= 16) eng->forward64_0(reg_x, reg_w);
		else eng->forward512_0(reg_x, reg_w);

		if (ln >= 21) eng->forward8(reg_x, reg_w, ln - 3 - 9, 512);
	}

	finline void NTT_inverse8(const size_t reg_x, const size_t reg_w)
	{
		const int ln = _ln;
		engine * const eng = _engine;

		if (ln >= 21) eng->backward8(reg_x, reg_w, ln - 3 - 9, 512);

		if (ln <= 12) eng->backward8_0(reg_x, reg_w);
		else if (ln <= 16) eng->backward64_0(reg_x, reg_w);
		else eng->backward512_0(reg_x, reg_w);
	}

	finline void NTT_forward(const size_t reg_x, const size_t reg_w)
	{
		NTT_forward8(reg_x, reg_w);
		_engine->fwd(reg_x, reg_w);
	}

	finline void square(const size_t reg_x, const size_t reg_w)
	{
		NTT_forward8(reg_x, reg_w);
		_engine->square(reg_x, reg_w);
		NTT_inverse8(reg_x, reg_w);
	}

	finline void mul_const(const size_t reg_y, const size_t reg_x, const size_t reg_w)
	{
		NTT_forward8(reg_y, reg_w);
		_engine->mul(reg_y, reg_x, reg_w);
		NTT_inverse8(reg_y, reg_w);
	}

	finline void mul(const size_t reg_y, const size_t reg_x, const size_t reg_w)
	{
		NTT_forward(reg_x, reg_w);
		mul_const(reg_y, reg_x, reg_w);
	}

	finline void carry_mul(const size_t reg_x, const int32 a, const bool nega)
	{
		engine * const eng = _engine;
		eng->carry_mul(reg_x, a);
		eng->carry_complete(reg_x, nega);
	}

	finline void carry_sub(const size_t reg_x, const size_t reg_y, const bool nega)
	{
		engine * const eng = _engine;
		eng->carry_sub(reg_x, reg_y);
		eng->carry_complete(reg_x, nega);
	}

	finline void mod_Q(const size_t reg_x) { _engine->mod_Q(reg_x); }
	finline void mod_R_add2(const size_t reg_x) { _engine->mod_R_add2(reg_x); }

	finline void set(const size_t reg_y, const size_t reg_x) { _engine->set(reg_y, reg_x); }

	// Montgomery-McLaughlin-Gallot-Woltman multiplication

	void reduceMMGW(const size_t dst)
	{
		// m' = t_R * Ninv mod R (dst = t_R)
		mul_const(dst, Reg_Ninv_R, Reg_w_R);
 		carry_mul(dst, 1, false);

		// 0 <= m' < R is a necessary condition, m' += 2
		mod_R_add2(dst);

		// x = x * N_Q - t_Q mod Q
		mul_const(dst, Reg_N_Q, Reg_w_Q);
 		carry_sub(dst, Reg_t_Q, true);

		// 0 <= dst < Q is a necessary condition
		mod_Q(dst);
	}

	void mulMMGW(const size_t dst, const size_t src)
	{
		// t_Q = x * y mod Q
		set(Reg_t_Q, dst);
		set(Reg_t, src);
		mul(Reg_t_Q, Reg_t, Reg_w_Q);
 		carry_mul(Reg_t_Q, 1, true);

		// dst = x * y mod R
		set(Reg_t, src);
		mul(dst, Reg_t, Reg_w_R);
 		carry_mul(dst, 1, false);

		reduceMMGW(dst);
	}

	void squareMMGW(const int32 a)
	{
		// t_Q = a * x^2 mod Q
		set(Reg_t_Q, Reg0);
		square(Reg_t_Q, Reg_w_Q);
 		carry_mul(Reg_t_Q, a, true);

		// Reg0 = a * x^2 mod R
		square(Reg0, Reg_w_R);
 		carry_mul(Reg0, a, false);

		reduceMMGW(Reg0);
	}

public:
	void set1() override
	{
		_engine->set(Reg0, Reg_one);
	}

	void get(mpz_t & z, const size_t src) override
	{
		// Conversion out of Montgomery-McLaughlin-Gallot-Woltman form
		engine * const eng = _engine;
		eng->set(Reg_t_Q, src);
		eng->set(Reg_t, src);
		reduceMMGW(Reg_t);
		get_mpz(z, Reg_t);

		// 0 <= z < N
		mpz_mod(z, z, _Nz);
	}

	void copy(const size_t dst, const size_t src) override
	{
		_engine->set(dst, src);
	}

	void square_mul(const int32_t a) override
	{
		squareMMGW(int32(a));
	}

	void mul(const size_t dst, const size_t src) override
	{
		mulMMGW(dst, src);
	}

	bool init_failed() const override { return _init_failed; }
	void get_factor(mpz_t & z) const override { mpz_set(z, _Nz); }

	bool has_checkpoint() const { return true; }

	bool read_checkpoint(file & cfile) override
	{
		const size_t n = size_t(1) << _ln;
		ZP * const buf = _buf;
		const bool success = cfile.read(reinterpret_cast<char *>(buf), 2 * n * sizeof(ZP));
		if (success)
		{
			engine * const eng = _engine;
			eng->write_memory(Reg0, buf, 2);
			eng->extend01();
		}
		return success;
	}

	void save_checkpoint(file & cfile) const override
	{
		const size_t n = size_t(1) << _ln;
		ZP * const buf = _buf;
		_engine->read_memory(Reg0, buf, 2);
		cfile.write(reinterpret_cast<const char *>(buf), 2 * n * sizeof(ZP));
	}

public:
	std::string get_type() const { return _engine->getType(); }
	std::string get_data_size() const
	{
		const size_t data_size = _engine->get_data_size();
		std::ostringstream ss; ss << "data size: ";
		if (data_size < (10u << 20)) ss << data_size / (1u << 10) << " KB"; else ss << data_size / (1u << 20) << " MB";
		return ss.str();
	}
	std::string get_cache_size() const
	{
		const size_t cache_size = _engine->get_cache_size();
		std::ostringstream ss; ss << "cache size: ";
		if (cache_size < (10u << 20)) ss << cache_size / (1u << 10) << " KB"; else ss << cache_size / (1u << 20) << " MB";
		return ss.str();
	}
	std::string get_info_short() const
	{
		const size_t n = size_t(1) << _ln;
		std::ostringstream ss; ss << "length " << n << "x" << _lbase << " bits";
		return (ss.str());
	}

	transformGPU(const mpz_t & N, const int32_t a, const size_t device_id) : transform()
	{
		mpz_t R, t; mpz_inits(_Nz, R, t, nullptr);

		// R = base^n - 1, Q = base^n + 1 such that base^n > 16 * a * N and n is a power of two
		const size_t m_2 = mpz_sizeinbase(N, 2) + size_t(32 - __builtin_clz((unsigned int)(a))) + 4;
		int ln;
		for (unsigned int lbase = 31; lbase >= 1; --lbase)
		{
			// base is a power of two
			const uint32 base = (1u << lbase);
			const size_t m = (m_2 + lbase - 1) / lbase;
			ln = 4; while ((size_t(1) << ln) < m) ++ln;
			// A necessary condition is a * n * base^2 < (P1 * P2) / 2
			const double dbase = double(base);
			if (double(size_t(a) << ln) * dbase * dbase < (P1 * uint64(P2)) / 2)
			{
				_ln = ln;
				_lbase = int(lbase);
				break;
			}
		}

		if (ln > 23) throw std::runtime_error("NTT size > 2^23 is not supported");
		const size_t n = size_t(1) << ln;
		const mp_bitcnt_t lbase = mp_bitcnt_t(_lbase);

		_buf = new ZP[2 * n];

		mpz_set_ui(R, 1); mpz_mul_2exp(R, R, lbase << ln); mpz_sub_ui(R, R, 1);
		if (mpz_invert(t, N, R) == 0)
		{
			mpz_gcd(_Nz, N, R);
			_init_failed = true;
			mpz_clears(R, t, nullptr);
			return;
		}

		_engine = new engine(ln, device_id);
		engine * const eng = _engine;

		std::ostringstream src;

		src << "#define LN_SZ\t" << ln << std::endl;
		src << "#define N_SZ\t" << n << "u" << std::endl;
		src << "#define LBASE\t" << lbase << std::endl;

		src << "#define P1\t" << P1 << "u" << std::endl;
		src << "#define Q1\t" << Q1 << "u" << std::endl;
		src << "#define RSQ1\t" << RSQ1 << "u" << std::endl;

		src << "#define P2\t" << P2 << "u" << std::endl;
		src << "#define Q2\t" << Q2 << "u" << std::endl;
		src << "#define RSQ2\t" << RSQ2 << "u" << std::endl;

		src << "#define INVP2_P1\t" << INVP2_P1 << "u" << std::endl;

		src << "#define P1P2\t" << (P1 * uint64(P2)) << "ul" << std::endl;

		// Not converted into Montgomery form such that output is converted out of MF
		src << "#define NORM1\t" << ZP1::norm(ln - 1).get() << "u" << std::endl;
		src << "#define NORM2\t" << ZP2::norm(ln - 1).get() << "u" << std::endl;

		src << "#define CARRY_LENGTH\t" << CARRY_LENGTH << "u"  << std::endl;
		src << "#define CARRY_WG_SZ\t" << eng->get_carry_workgroup_size() << "u" << std::endl;

		src << "#define Z_OFFSET\t" << (Reg_count * n) << "u" << std::endl;

		src << "#define BLK16\t" << BLK16 << std::endl;
		src << "#define BLK32\t" << BLK32 << std::endl;
		src << "#define BLK64\t" << BLK64 << std::endl;
		src << "#define BLK128\t" << BLK128 << std::endl;
		src << "#define BLK256\t" << BLK256 << std::endl;
		src << "#define BLK512\t" << BLK512 << std::endl;
		src << "#define BLK1024\t" << BLK1024 << std::endl;
		src << "#define BLK2048\t" << BLK2048 << std::endl;

		src << "#define CHUNK64\t" << CHUNK64 << std::endl;
		src << "#define CHUNK512\t" << CHUNK512 << std::endl;

		if (!eng->readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		eng->loadProgram(src.str());
		eng->alloc_memory(Reg_count);
		eng->create_kernels();

		ZP1::set_roots_R(&_buf[0 * (n / 2)], &_buf[1 * (n / 2)], ln);
		ZP1::set_roots_Q(&_buf[2 * (n / 2)], &_buf[3 * (n / 2)], ln);
		eng->write_memory(Reg_w_R, _buf, 2);

		ZP2::set_roots_R(&_buf[0 * (n / 2)], &_buf[1 * (n / 2)], ln);
		ZP2::set_roots_Q(&_buf[2 * (n / 2)], &_buf[3 * (n / 2)], ln);
		eng->write_memory(Reg_count + Reg_w_R, _buf, 2);

		mpz_set(_Nz, N);
		set_mpz(Reg_N_Q, N);
		NTT_forward(Reg_N_Q, Reg_w_Q);
		set_mpz(Reg_Ninv_R, t);
		NTT_forward(Reg_Ninv_R, Reg_w_R);

		// The Montgomery-McLaughlin-Gallot-Woltman form of x is R/2 * x mod N
		mpz_mod(t, R, N);
		if (mpz_odd_p(t) != 0) mpz_add(t, t, N);
		mpz_div_2exp(t, t, 1);
		set_mpz(Reg_one, t);

		mpz_clears(R, t, nullptr);

		std::ostringstream ssi; ssi << get_type() << std::endl << get_data_size() << ", " << get_cache_size();
		_hinfo = ssi.str();
	}

	virtual ~transformGPU()
	{
		mpz_clear(_Nz);

		engine * const eng = _engine;
		eng->release_kernels();
		eng->release_memory();
		eng->clearProgram();
		delete eng;

		delete[] _buf;
	}

	std::string get_info() const override
	{
		std::ostringstream ss; ss << "Montgomery-McLaughlin-Gallot-Woltman reduction of " << get_info_short() << std::endl << _hinfo << std::endl;
		return (ss.str());
	}
};
