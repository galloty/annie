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

// The prime finite field Z/PZ
template<uint32 P, uint32 Q, uint32 MR, uint32 RSQ, uint32 MPR>
class ZPT : public ZP
{
private:
	static uint32 _add(const uint32 a, const uint32 b) { return a + b - ((a >= P - b) ? P : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { return a - b + ((a < b) ? P : 0); }

	// Montgomery multiplication: 2 mul + 2 mul_hi
	static uint32 _mul(const uint32 a, const uint32 b)
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

	uint32 get() const { return _n; }
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
	int _lm0 = 0;
	size_t _s0 = 0;
	bool _init_failed = false;
	mpz_t _Nz;
	engine * _engine = nullptr;
	ZP * _buf = nullptr;

private:
	void set_mpz(const size_t reg_y, const mpz_t & x)
	{
		const size_t n = size_t(1) << _ln;
		const mp_bitcnt_t lbase = mp_bitcnt_t(_lbase);
		ZP * const buf = _buf;

		mpz_t s, r; mpz_init_set(s, x); mpz_init(r);
		for (size_t i = 0; i < n; ++i)
		{
			mpz_mod_2exp(r, s, lbase);
			buf[i] = ZP(uint32(mpz_get_ui(r)));
			mpz_div_2exp(s, s, lbase);
		}
		mpz_clears(s, r, nullptr);

		engine * const eng = _engine;
		eng->write_memory(reg_y, buf);
		eng->write_memory(Reg_count + reg_y, buf);
	}

	void get_mpz(mpz_t & y, const size_t reg_x)
	{
		const size_t n = size_t(1) << _ln;
		const mp_bitcnt_t lbase = mp_bitcnt_t(_lbase);
		ZP * const buf = _buf;

		_engine->read_memory(reg_x, buf);

		mpz_set_ui(y, 0);
		for (size_t i = n; i > 0; --i)
		{
			mpz_mul_2exp(y, y, lbase);
			const int32 u = ZP1(buf[i - 1]).get_int();
			if (u > 0) mpz_add_ui(y, y, (unsigned long int)(u));
			else if (u < 0) mpz_sub_ui(y, y, (unsigned long int)(-u));
		}
	}

	void NTT_forward8(const size_t reg_x, const size_t reg_w)
	{
		const int ln = _ln;
		engine * const eng = _engine;

		if      (ln <= 6) { eng->forward8_0(reg_x, reg_w); }
		else if (ln <= 9) { eng->forward64_0(reg_x, reg_w); }
		else 
		{
			int lm = _ln - 3; size_t s = 1;
			eng->forward8_0(reg_x, reg_w);
			lm -= 3, s *= 8;
			for (int lm0 = _lm0; lm >= lm0; lm -= 3, s *= 8) eng->forward8(reg_x, reg_w, lm, s);
		}
	}

	void NTT_inverse8(const size_t reg_x, const size_t reg_w)
	{
		const int ln = _ln;
		engine * const eng = _engine;

		if      (ln <= 6) { eng->backward8_0(reg_x, reg_w); }
		else if (ln <= 9) { eng->backward64_0(reg_x, reg_w); }
		else
		{
			size_t s = _s0;
			for (int lm = _lm0; lm <= _ln - 3; lm += 3, s /= 8) eng->backward8(reg_x, reg_w, lm, s);
		}
	}

	void NTT_forward(const size_t reg_x, const size_t reg_w)
	{
		const int ln = _ln;
		engine * const eng = _engine;

		NTT_forward8(reg_x, reg_w);

		if      (ln == 4) {}
		else if (ln == 5) eng->fwd4x2(reg_x, reg_w);
		else if (ln == 6) eng->fwd8(reg_x, reg_w);
		else if (ln == 7) {}
		else if (ln == 8) eng->fwd4x2(reg_x, reg_w);
		else if (ln == 9) eng->fwd8(reg_x, reg_w);
		else
		{
			const int lm0 = _lm0;
			if      (lm0 == 9) eng->fwd512(reg_x, reg_w);
			else if (lm0 == 8) eng->fwd256(reg_x, reg_w);
			else if (lm0 == 7) eng->fwd128(reg_x, reg_w);
			else if (lm0 == 6) eng->fwd64(reg_x, reg_w);
			else if (lm0 == 5) eng->fwd32(reg_x, reg_w);
			else if (lm0 == 4) eng->fwd16(reg_x, reg_w);
			else if (lm0 == 3) eng->fwd8(reg_x, reg_w);
			else if (lm0 == 2) eng->fwd4x2(reg_x, reg_w);
		}
	}

	void square(const size_t reg_x, const size_t reg_w)
	{
		const int ln = _ln;
		engine * const eng = _engine;

		NTT_forward8(reg_x, reg_w);

		if      (ln == 4) eng->square2x4(reg_x, reg_w);
		else if (ln == 5) eng->square4x2(reg_x, reg_w);
		else if (ln == 6) eng->square8(reg_x, reg_w);
		else if (ln == 7) eng->square2x4(reg_x, reg_w);
		else if (ln == 8) eng->square4x2(reg_x, reg_w);
		else if (ln == 9) eng->square8(reg_x, reg_w);
		else
		{
			const int lm0 = _lm0;
			if      (lm0 == 9) eng->square512(reg_x, reg_w);
			else if (lm0 == 8) eng->square256(reg_x, reg_w);
			else if (lm0 == 7) eng->square128(reg_x, reg_w);
			else if (lm0 == 6) eng->square64(reg_x, reg_w);
			else if (lm0 == 5) eng->square32(reg_x, reg_w);
			else if (lm0 == 4) eng->square16(reg_x, reg_w);
			else if (lm0 == 3) eng->square8(reg_x, reg_w);
			else if (lm0 == 2) eng->square4x2(reg_x, reg_w);
			else if (lm0 == 1) eng->square2x4(reg_x, reg_w);
		}

		NTT_inverse8(reg_x, reg_w);
	}

	void mul_const(const size_t reg_y, const size_t reg_x, const size_t reg_w)
	{
		const int ln = _ln;
		engine * const eng = _engine;

		NTT_forward8(reg_y, reg_w);

		if      (ln == 4) eng->mul2x4(reg_y, reg_x, reg_w);
		else if (ln == 5) eng->mul4x2(reg_y, reg_x, reg_w);
		else if (ln == 6) eng->mul8(reg_y, reg_x, reg_w);
		else if (ln == 7) eng->mul2x4(reg_y, reg_x, reg_w);
		else if (ln == 8) eng->mul4x2(reg_y, reg_x, reg_w);
		else if (ln == 9) eng->mul8(reg_y, reg_x, reg_w);
		else
		{
			const int lm0 = _lm0;
			if      (lm0 == 9) eng->mul512(reg_y, reg_x, reg_w);
			else if (lm0 == 8) eng->mul256(reg_y, reg_x, reg_w);
			else if (lm0 == 7) eng->mul128(reg_y, reg_x, reg_w);
			else if (lm0 == 6) eng->mul64(reg_y, reg_x, reg_w);
			else if (lm0 == 5) eng->mul32(reg_y, reg_x, reg_w);
			else if (lm0 == 4) eng->mul16(reg_y, reg_x, reg_w);
			else if (lm0 == 3) eng->mul8(reg_y, reg_x, reg_w);
			else if (lm0 == 2) eng->mul4x2(reg_y, reg_x, reg_w);
			else if (lm0 == 1) eng->mul2x4(reg_y, reg_x, reg_w);
		}

		NTT_inverse8(reg_y, reg_w);
	}

	void mul(const size_t reg_y, const size_t reg_x, const size_t reg_w)
	{
		NTT_forward(reg_x, reg_w);
		mul_const(reg_y, reg_x, reg_w);
	}

	void carry_mul(const size_t reg_x, const int32 a, const bool nega)
	{
		engine * const eng = _engine;
		eng->carry_mul(reg_x, a);
		eng->carry_complete(reg_x, nega);
	}

	void carry_sub(const size_t reg_x, const size_t reg_y, const bool nega)
	{
		engine * const eng = _engine;
		eng->carry_sub(reg_x, reg_y);
		eng->carry_complete(reg_x, nega);
	}

	void mod_Q(const size_t reg_x) { _engine->mod_Q(reg_x); }
	void mod_R_add2(const size_t reg_x) { _engine->mod_R_add2(reg_x); }

	void set(const size_t reg_y, const size_t reg_x) { _engine->set(reg_y, reg_x); }

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
	transformGPU(const mpz_t & N, const int32_t a, const size_t device_id, const bool verbose) : transform()
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

		int lm = ln - 3; size_t s = 1;
		for (; lm > 9; lm -= 3, s *= 8);
		_lm0 = lm; _s0 = s;

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

		std::cout << "ln = " << ln << ": ";
		if      (ln <= 6) std::cout << "forward8_0";
		else if (ln <= 9) std::cout << "forward64_0";
		if      (ln == 4) std::cout << " square2x4";
		else if (ln == 5) std::cout << " square4x2";
		else if (ln == 6) std::cout << " square8";
		else if (ln == 7) std::cout << " square2x4";
		else if (ln == 8) std::cout << " square4x2";
		else if (ln == 9) std::cout << " square8";
		else
		{
			lm = ln - 3; s = 1;
			std::cout << "forward8_0";
			lm -= 3, s *= 8;
			for (int lm0 = _lm0; lm >= lm0; lm -= 3, s *= 8) std::cout << " forward8(" << lm << ", " << s << ")";
			if      (_lm0 == 9) std::cout << " square512";
			else if (_lm0 == 8) std::cout << " square256";
			else if (_lm0 == 7) std::cout << " square128";
			else if (_lm0 == 6) std::cout << " square64 " << BLK64 << ", " << eng->get_blk64();
			else if (_lm0 == 5) std::cout << " square32 " << BLK32 << ", " << eng->get_blk32();
			else if (_lm0 == 4) std::cout << " square16 " << BLK16 << ", " << eng->get_blk16();
			else if (_lm0 == 3) std::cout << " square8";
			else if (_lm0 == 2) std::cout << " square4x2";
			else if (_lm0 == 1) std::cout << " square2x4";
		}
		std::cout << std::endl;

		std::ostringstream src;

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

		src << "#define BLK16\t" << eng->get_blk16() << std::endl;	// TODO remove
		src << "#define BLK32\t" << eng->get_blk32() << std::endl;
		src << "#define BLK64\t" << eng->get_blk64() << std::endl;
		src << "#define BLK128\t" << BLK128 << std::endl;
		src << "#define BLK256\t" << BLK256 << std::endl;
		src << "#define BLK512\t" << BLK512 << std::endl;

		src << "#define CHUNK64\t" << CHUNK64 << std::endl;

// std::cout << src.str() << std::endl;
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

		if (verbose) std::cout << "Montgomery-McLaughlin-Gallot-Woltman reduction of length " << n << "x" << lbase << " bits" << std::endl;
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
};
