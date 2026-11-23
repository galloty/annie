/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>

#include <gmp.h>

#include "transform.h"
#include "engine.h"
#include "ocl/kernel.h"

#define finline	__attribute__((always_inline))

static constexpr uint_32 mul_hi(const uint_32 x, const uint_32 y) { return uint_32((x * uint_64(y)) >> 32); }

// p * q = 1 (mod 2^32) (Newton's method)
static constexpr uint_32 _invert(const uint_32 p)
{
	uint_32 q = 1, prev = 0;
	while (q != prev) { prev = q; q *= 2 - p * q; }
	return q;
}

// 2^32 mod p = Montgomery form of 1
static constexpr uint_32 _one(const uint_32 p) { return (uint_32)(((uint_64)(1) << 32) % p); }

// (2^32)^2 mod p
static constexpr uint_32 _rsq(const uint_32 one, const uint_32 p) { return (uint_32)((one * (uint_64)(one)) % p); }

static constexpr size_t _bitrev(const size_t i, const size_t n)
{
	size_t r = 0;
	for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
	return r;
}

// The prime finite field Z/PZ
class Mod
{
public:
	finline static uint_32 _add(const uint_32 lhs, const uint_32 rhs, const uint_32 p) { return lhs + rhs - ((lhs >= p - rhs) ? p : 0); }
	finline static uint_32 _sub(const uint_32 lhs, const uint_32 rhs, const uint_32 p) { return lhs - rhs + ((lhs < rhs) ? p : 0); }

	// Montgomery multiplication by a constant: 1 mul + 2 mul_hi
	finline static uint_32 _mul_const(const uint_32 lhs, const uint_32 rhs, const uint_32 rhs_Q, const uint_32 p)
	{
		const uint_32 mp = mul_hi(lhs * rhs_Q, p);
		return _sub(mul_hi(lhs, rhs), mp, p);
	}

	// Montgomery multiplication: 2 mul + 2 mul_hi
	finline static uint_32 _mul(const uint_32 lhs, const uint_32 rhs, const uint_32 p, const uint_32 q) { return _mul_const(lhs, rhs, rhs * q, p); }

	static ZP _pow(const ZP & lhs, const uint_32 e, const size_t k)
	{
		if (e == 0) return ZP(one[k]);
		const uint_32 p = P[k], q = Q[k];
		uint_32 r = one[k], y = lhs;
		for (uint_32 i = e; i > 1; i /= 2)
		{
			if (i % 2 != 0) r = _mul(r, y, p, q);
			y = _mul(y, y, p, q);
		}
		r = _mul(r, y, p, q);
		return ZP(r);
	}

	static ZP _primroot_n(const int ln, const size_t k) { return ZP(_pow(Mod::toMonty(primroot[k], k), (P[k] - 1) >> ln, k)); }

public:
	static const size_t PSIZE = 4;
	static constexpr uint_32 P[PSIZE] = { (125u << 25) + 1, (243u << 24) + 1, (235u << 24) + 1, (29u << 27) + 1 };
	static constexpr uint_32 Q[PSIZE] = { _invert(P[0]), _invert(P[1]), _invert(P[2]), _invert(P[3]) };
	static constexpr uint_32 primroot[PSIZE] = { 3, 7, 3, 3 };
	static constexpr uint_32 one[PSIZE] = { _one(P[0]), _one(P[1]), _one(P[2]), _one(P[3]) };
	static constexpr uint_32 rsq[PSIZE] = { _rsq(one[0], P[0]), _rsq(one[1], P[1]), _rsq(one[2], P[2]), _rsq(one[3], P[3]) };

	static int_32 get_i32(const ZP & lhs, const size_t k) { return int_32(lhs - ((lhs >= P[k] / 2) ? P[k] : 0)); }

	static ZP add(const ZP & lhs, const ZP & rhs, const size_t k) { return ZP(_add(lhs, rhs, P[k])); }
	static ZP sub(const ZP & lhs, const ZP & rhs, const size_t k) { return ZP(_sub(lhs, rhs, P[k])); }
	static ZP mul(const ZP & lhs, const ZP & rhs, const size_t k) { return ZP(_mul(lhs, rhs, P[k], Q[k])); }

	static ZP submul(const ZP & lhs, const ZP & rhs, const ZP & m, const size_t k) { return ZP(mul(sub(lhs, rhs, k), m, k)); }

	// Conversion into / out of Montgomery form
	static ZP toMonty(const ZP & lhs, const size_t k) { return ZP(mul(lhs, rsq[k], k)); }
	// static ZP fromMonty(const ZP & lhs, const size_t k) { return ZP(mul(lhs, 1, k)); }

	static ZP invert(const ZP & lhs, const size_t k) { return _pow(lhs, P[k] - 2, k); }
	static ZP norm(const int ln, const size_t k) { return ZP(P[k] - ((P[k] - 1) >> ln)); }

	// modulo x^n - 1
	static void set_roots_R(ZP * const w_R, ZP * const wi_R, const int ln, const size_t k)
	{
		const size_t n = size_t(1) << ln;

		ZP sr_s = _primroot_n(ln - 1, k), sr_si = invert(sr_s, k);	// (-1)^{1/s}
		for (size_t s = n / 4; s >= 1; s /= 2)
		{
			ZP t_j = one[k], ti_j = one[k];
			for (size_t j = 0; j < s; ++j)
			{
				const size_t jr = _bitrev(j, s);
				w_R[s + jr] = t_j; wi_R[s + jr] = ti_j;
				t_j = mul(t_j, sr_s, k); ti_j = mul(ti_j, sr_si, k);
			}

			sr_s = mul(sr_s, sr_s, k); sr_si = mul(sr_si, sr_si, k);
		}

		w_R[0] = Mod::toMonty(w_R[1], k);
	}

	// modulo x^n + 1
	static void set_roots_Q(ZP * const w_Q, ZP * const wi_Q, const int ln, const size_t k)
	{
		const size_t n = size_t(1) << ln;

		ZP sr_2s = _primroot_n(ln, k), sr_2si = invert(sr_2s, k);	// (-1)^{1/2s}
		for (size_t s = n / 4; s >= 1; s /= 2)
		{
			const ZP sr_s = mul(sr_2s, sr_2s, k), sr_si = mul(sr_2si, sr_2si, k);	// (-1)^{1/s}

			ZP t_j = sr_2s, ti_j = sr_2si;
			for (size_t j = 0; j < s; ++j)
			{
				const size_t jr = _bitrev(j, s);
				w_Q[s + jr] = t_j; wi_Q[s + jr] = ti_j;
				t_j = mul(t_j, sr_s, k); ti_j = mul(ti_j, sr_si, k);
			}

			sr_2s = sr_s; sr_2si = sr_si;
		}

		w_Q[0] = Mod::toMonty(w_Q[1], k);
	}
};

template<size_t SIZE>
class RNS
{
protected:
	ZP _r[SIZE];

	RNS & _add(const RNS & rhs) { for (size_t k = 0; k < SIZE; ++k) _r[k] = Mod::add(_r[k], rhs._r[k], k); return *this; }
	RNS & _sub(const RNS & rhs) { for (size_t k = 0; k < SIZE; ++k) _r[k] = Mod::sub(_r[k], rhs._r[k], k); return *this; }
	RNS & _mul(const RNS & rhs) { for (size_t k = 0; k < SIZE; ++k) _r[k] = Mod::mul(_r[k], rhs._r[k], k); return *this; }

	RNS & _toMonty() { for (size_t k = 0; k < SIZE; ++k) _r[k] = Mod::toMonty(_r[k], k); return *this; }
	RNS & _norm(const int ln) { for (size_t k = 0; k < SIZE; ++k) _r[k] = Mod::norm(ln, k); return *this; }

public:
	RNS() {}
	explicit RNS(const uint_32 n) { for (size_t k = 0; k < SIZE; ++k) _r[k] = ZP(n); }
	explicit RNS(const uint_32 * const n) { for (size_t k = 0; k < SIZE; ++k) _r[k] = ZP(n[k]); }

	const ZP & get_r(const size_t k) const { return _r[k]; }

};

class RNS1 : public RNS<1>
{
public:
	typedef int_32 big_type;

private:
	// Chinese remainder theorem
	finline int_32 garner1() const { return Mod::get_i32(_r[0], 0); }

public:
	RNS1() {}
	explicit RNS1(const uint_32 n) : RNS(n) {}
	explicit RNS1(const uint_32 * const n) : RNS(n) {}

	big_type getS_i() const { return big_type(garner1()); }
};

class RNS2 : public RNS<2>
{
public:
	typedef int_64 big_type;

private:
	// Chinese remainder theorem
	finline int_64 garner2() const
	{
		static const uint_32 P0 = Mod::P[0], P1 = Mod::P[1];
		static const ZP INVP1_P0 = Mod::invert(Mod::toMonty(P1, 0), 0);	// P2 < P1

		// convert to mixed-radix representation
		const uint_32 u1 = _r[1];
		const uint_32 u0 = Mod::submul(_r[0], u1, INVP1_P0, 0);
		const int_32 i0 = (u0 > P0 / 2) ? (int_32)(u0 - P0) : (int_32)(u0);

		return (int_64)(u1) + (int_64)(P1) * i0;
	}

public:
	RNS2() {}
	explicit RNS2(const uint_32 n) : RNS(n) {}
	explicit RNS2(const uint_32 * const n) : RNS(n) {}

	big_type getS_i() const { return garner2(); }
};

class RNS3 : public RNS<3>
{
public:
	// typedef int_96 big_type;

public:
	RNS3() {}
	explicit RNS3(const uint_32 n) : RNS(n) {}
	explicit RNS3(const uint_32 * const n) : RNS(n) {}

	RNS3 & operator+=(const RNS3 & rhs) { _add(rhs); return *this; }
	RNS3 & operator-=(const RNS3 & rhs) { _sub(rhs); return *this; }
	RNS3 & operator*=(const RNS3 & rhs) { _mul(rhs); return *this; }

	RNS3 operator+(const RNS3 & rhs) const { RNS3 r = *this; r += rhs; return r; }
	RNS3 operator-(const RNS3 & rhs) const { RNS3 r = *this; r -= rhs; return r; }
	RNS3 operator*(const RNS3 & rhs) const { RNS3 r = *this; r *= rhs; return r; }

	RNS3 toMonty() const { RNS3 r = *this; r._toMonty(); return r; }
	static RNS3 norm(const int ln) { RNS3 r; r._norm(ln); return r; }
};

class RNS4 : public RNS<4>
{
public:
	// typedef int_128 big_type;

public:
	RNS4() {}
	explicit RNS4(const uint_32 n) : RNS(n) {}
	explicit RNS4(const uint_32 * const n) : RNS(n) {}

	RNS4 & operator+=(const RNS4 & rhs) { _add(rhs); return *this; }
	RNS4 & operator-=(const RNS4 & rhs) { _sub(rhs); return *this; }
	RNS4 & operator*=(const RNS4 & rhs) { _mul(rhs); return *this; }

	RNS4 operator+(const RNS4 & rhs) const { RNS4 r = *this; r += rhs; return r; }
	RNS4 operator-(const RNS4 & rhs) const { RNS4 r = *this; r -= rhs; return r; }
	RNS4 operator*(const RNS4 & rhs) const { RNS4 r = *this; r *= rhs; return r; }

	RNS4 toMonty() const { RNS4 r = *this; r._toMonty(); return r; }
	static RNS4 norm(const int ln) { RNS4 r; r._norm(ln); return r; }

	void encode(const ZP & rhs, const RNS4 & base) { *this *= base; *this += RNS4(rhs); }
};

template<int LN>
class transformGPU : public transform
{
private:
	static const size_t Reg_w_R = 3, Reg_w_Q = 4, Reg_N_Q = 5, Reg_Ninv_R = 6, Reg_t_Q = 7, Reg_t = 8, Reg_one = 9, Reg_count = 10;
	engine<LN> _engine;
	const int _lbase;
	const size_t _prm_count, _pck_factor;
	bool _init_failed = false;
	mpz_t _Nz;
	ZP * _buf = nullptr;
	std::string _hinfo;

private:
	finline static constexpr mp_limb_t read_limb(const mp_limb_t * const x_ptr, const size_t x_size, const size_t i)
	{
		return (i < x_size) ? x_ptr[i] : 0;
	}

	finline static constexpr uint_32 read_word(const mp_limb_t * const x_ptr, const size_t x_size, const size_t bit_index)
	{
		const size_t limb_index = bit_index / 64;
		const int limb_shift = int(bit_index % 64);
		const mp_limb_t rl = read_limb(x_ptr, x_size, limb_index), rh = read_limb(x_ptr, x_size, limb_index + 1);
		mp_limb_t r = rl >> limb_shift;
		if (limb_shift != 0) r |= rh << (64 - limb_shift);
		return uint_32(r);
	}

	finline static constexpr void write_word(mp_limb_t * const x_ptr, const size_t bit_index, const uint_64 r)
	{
		const size_t limb_index = bit_index / 64;
		const int limb_shift = int(bit_index % 64);
		x_ptr[limb_index] |= mp_limb_t(r) << limb_shift;
		if (limb_shift != 0) x_ptr[limb_index + 1] |= mp_limb_t(r) >> (64 - limb_shift);
	}

	void set_mpz(const size_t reg_y, const mpz_t & x)
	{
		const size_t n = size_t(1) << LN;
		const int lbase = _lbase;
		const uint_32 base = 1u << lbase;
		const size_t prm_count = _prm_count;
		const RNS4 rbase4 = RNS4(1u << lbase).toMonty();
		ZP * const buf = _buf;

		const size_t x_size = mpz_size(x);
		const mp_limb_t * const x_ptr = mpz_limbs_read(x);

		if (_pck_factor == 1)
		{
			size_t bit_index = 0;
			for (size_t i = 0; i < n; ++i)
			{
				buf[0 * n + i] = read_word(x_ptr, x_size, bit_index) & (base - 1);
				bit_index += size_t(lbase);
			}
		}
		else if (_pck_factor == 2)
		{
			size_t bit_index = 0;
			for (size_t i = 0; i < n; ++i)
			{
				buf[0 * n + i] = read_word(x_ptr, x_size, bit_index) & (base - 1);
				bit_index += size_t(lbase);
				buf[1 * n + i] = read_word(x_ptr, x_size, bit_index) & (base - 1);
				bit_index += size_t(lbase);
			}
		}
		else {}	// TODO

		for (size_t i = 0; i < n; ++i)
		{
			if (prm_count == 2)
			{
				const RNS2 r = RNS2(buf[0 * n + i]);
				for (size_t k = 0; k < prm_count; ++k) buf[k * n + i] = r.get_r(k);
			}
			else if (prm_count == 4)
			{
				RNS4 r = RNS4(buf[1 * n + i]);
				r.encode(buf[0 * n + i], rbase4);

				for (size_t k = 0; k < prm_count; ++k) buf[k * n + i] = r.get_r(k);
			}
			else {}	// TODO
		}

		for (size_t k = 0; k < prm_count; ++k) _engine.write_memory(k * Reg_count + reg_y, &buf[k * n]);
	}

	void get_mpz(mpz_t & y, const size_t reg_x)
	{
		const size_t n = size_t(1) << LN;
		const int lbase = _lbase, lbase_pck = int(_pck_factor) * lbase;
		const uint_32 base = 1u << lbase;
		ZP * const buf = _buf;

		_engine.read_memory(0 * Reg_count + reg_x, &buf[0 * n]);
		if (_pck_factor > 1) _engine.read_memory(1 * Reg_count + reg_x, &buf[1 * n]);

		const size_t y_size = (size_t(lbase_pck) << LN) / 64 + 2;
		mp_limb_t * const y_ptr = mpz_limbs_write(y, mp_size_t(y_size));
		for (size_t i = 0; i < y_size; ++i) y_ptr[i] = 0;

		long fr = 0;
		if (_pck_factor == 1)
		{
			RNS1::big_type f = 0;
			size_t bit_index = 0;
			for (size_t i = 0; i < n; ++i)
			{
				f += RNS1(buf[i]).getS_i();
				const uint_32 r = uint_32(f) & (base - 1);
				f >>= lbase;

				write_word(y_ptr, bit_index, r);
				bit_index += size_t(lbase);
			}
			fr = long(f);
		}
		else if (_pck_factor == 2)
		{
			const uint_64 base2 = base * uint_64(base);
			RNS2::big_type f = 0;
			size_t bit_index = 0;
			for (size_t i = 0; i < n; ++i)
			{
				uint_32 u[_pck_factor]; for (size_t l = 0; l < _pck_factor; ++l) u[l] = buf[l * n + i];
				f += RNS2(u).getS_i();
				const uint_64 r = (uint_64)(f) & (base2 - 1);
				f >>= 2 * lbase;

				write_word(y_ptr, bit_index, r);
				bit_index += size_t(2 * lbase);
			}
			fr = long(f);
		}
		else {}	// TODO

		mp_size_t s = mp_size_t(y_size);
		while ((s > 0) && (y_ptr[s - 1] == 0)) --s;
 		mpz_limbs_finish(y, s);

		if (fr != 0)
		{
			mpz_t t; mpz_init_set_si(t, fr);
			mpz_mul_2exp(t, t, mp_bitcnt_t(lbase_pck) << LN);
			mpz_add(y, y, t);
			mpz_clear(t);
		}
	}

	finline void NTT_forward8(const size_t reg_x, const size_t reg_w)
	{
		if (LN <= 12) _engine.forward8_0(reg_x, reg_w);
		else if (LN <= 16) _engine.forward64_0(reg_x, reg_w);
		else _engine.forward512_0(reg_x, reg_w);

		if (LN >= 21) _engine.forward8(reg_x, reg_w, LN - 3 - 9, 512);
	}

	finline void NTT_inverse8(const size_t reg_x, const size_t reg_w)
	{
		if (LN >= 21) _engine.backward8(reg_x, reg_w, LN - 3 - 9, 512);

		if (LN <= 12) _engine.backward8_0(reg_x, reg_w);
		else if (LN <= 16) _engine.backward64_0(reg_x, reg_w);
		else _engine.backward512_0(reg_x, reg_w);
	}

	finline void NTT_forward(const size_t reg_x, const size_t reg_w)
	{
		NTT_forward8(reg_x, reg_w);
		_engine.fwd(reg_x, reg_w);
	}

	finline void square(const size_t reg_x, const size_t reg_w)
	{
		NTT_forward8(reg_x, reg_w);
		_engine.square(reg_x, reg_w);
		NTT_inverse8(reg_x, reg_w);
	}

	finline void mul_const(const size_t reg_y, const size_t reg_x, const size_t reg_w)
	{
		NTT_forward8(reg_y, reg_w);
		_engine.mul(reg_y, reg_x, reg_w);
		NTT_inverse8(reg_y, reg_w);
	}

	finline void mul(const size_t reg_y, const size_t reg_x, const size_t reg_w)
	{
		NTT_forward(reg_x, reg_w);
		mul_const(reg_y, reg_x, reg_w);
	}

	finline void carry_mul(const size_t reg_x, const uint_32 a, const bool nega)
	{
		_engine.carry_mul(reg_x, a);
		_engine.carry_complete(reg_x, nega);
	}

	finline void carry_sub(const size_t reg_x, const size_t reg_y, const bool nega)
	{
		_engine.carry_sub(reg_x, reg_y);
		_engine.carry_complete(reg_x, nega);
	}

	finline void mod_Q(const size_t reg_x) { _engine.mod_Q(reg_x); }
	finline void mod_R_add2(const size_t reg_x) { _engine.mod_R_add2(reg_x); }

	finline void set(const size_t reg_y, const size_t reg_x) { _engine.set(reg_y, reg_x); }

	// Montgomery-McLaughlin-Gallot-Woltman multiplication

	// void display(const size_t reg, const char * const str)
	// {
	// 	const size_t n = size_t(1) << LN;
	// 	ZP * const buf = _buf;
	// 	for (size_t i = 0; i < 4; ++i) _engine.read_memory(i * Reg_count + reg, buf, 1);
	// 	std::cout << str << ": ";
	// 	for (size_t i = 0; i < n; ++i)
	// 	{
	// 		std::cout << "(" << buf[0 * n + i].get() << ", " << buf[1 * n + i].get() << ", " << buf[2 * n + i].get() << ", " << buf[3 * n + i].get() << ")";
	// 	}
	// 	std::cout << std::endl;
	// }

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

	void squareMMGW(const uint_32 a)
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
		_engine.set(Reg0, Reg_one);
	}

	void get(mpz_t & z, const size_t src) override
	{
		// Conversion out of Montgomery-McLaughlin-Gallot-Woltman form
		_engine.set(Reg_t_Q, src);
		_engine.set(Reg_t, src);
		reduceMMGW(Reg_t);
		get_mpz(z, Reg_t);

		// 0 <= z < N
		mpz_mod(z, z, _Nz);
	}

	void copy(const size_t dst, const size_t src) override
	{
		_engine.set(dst, src);
	}

	void square_mul(const int32_t a) override
	{
		squareMMGW(uint_32(a));
	}

	void mul(const size_t dst, const size_t src) override
	{
		mulMMGW(dst, src);
	}

	bool init_failed() const override { return _init_failed; }
	void get_factor(mpz_t & z) const override { mpz_set(z, _Nz); }

	bool has_checkpoint() const override { return true; }

	bool read_checkpoint(file & cfile) override
	{
		const size_t n = size_t(1) << LN;
		ZP * const buf = _buf;

		uint32_t prm_count; if (!cfile.read(reinterpret_cast<char *>(&prm_count), sizeof(prm_count))) return false;
		if (prm_count != _prm_count) return false;
		uint32_t n32; if (!cfile.read(reinterpret_cast<char *>(&n32), sizeof(n32))) return false;
		if (n32 != n) return false;

		for (size_t k = 0; k < _prm_count; ++k)
		{
			if (!cfile.read(reinterpret_cast<char *>(buf), 2 * n * sizeof(ZP))) return false;
			_engine.write_memory(k * Reg_count + Reg0, buf, 2);
		}
		return true;
	}

	void save_checkpoint(file & cfile) override
	{
		const size_t n = size_t(1) << LN;
		ZP * const buf = _buf;

		const uint32_t prm_count = uint32_t(_prm_count);
		if (!cfile.write(reinterpret_cast<const char *>(&prm_count), sizeof(prm_count))) return;
		const uint32_t n32 = uint32_t(n);
		if (!cfile.write(reinterpret_cast<const char *>(&n32), sizeof(n32))) return;

		for (size_t k = 0; k < _prm_count; ++k)
		{
			_engine.read_memory(k * Reg_count + Reg0, buf, 2);
			if (!cfile.write(reinterpret_cast<const char *>(buf), 2 * n * sizeof(ZP))) return;
		}
	}

public:
	transformGPU(const mpz_t & N, const size_t device_id, const int lbase, const size_t prm_count, const size_t pck_factor)
		: transform(), _engine(device_id, prm_count, pck_factor), _lbase(lbase), _prm_count(prm_count), _pck_factor(pck_factor)
	{
		mpz_t R, t; mpz_inits(_Nz, R, t, nullptr);

		const size_t n = size_t(1) << LN;
		const mp_bitcnt_t lbase_pck = mp_bitcnt_t(pck_factor) * mp_bitcnt_t(lbase);

		_buf = new ZP[prm_count * n];

		mpz_set_ui(R, 1); mpz_mul_2exp(R, R, lbase_pck << LN); mpz_sub_ui(R, R, 1);
		if (mpz_invert(t, N, R) == 0)
		{
			mpz_gcd(_Nz, N, R);
			_init_failed = true;
			mpz_clears(R, t, nullptr);
			return;
		}

		std::ostringstream src;

		src << "#define LN_SZ\t" << LN << std::endl;
		src << "#define N_SZ\t" << n << "u" << std::endl;
		src << "#define LBASE\t" << lbase << std::endl;
		src << "#define PRM_COUNT\t" << prm_count << std::endl;
		src << "#define PCK_FACTOR\t" << pck_factor << std::endl;

		for (size_t k = 0; k < Mod::PSIZE; ++k)
		{
			src << "#define P" << k << "\t" << Mod::P[k] << "u" << std::endl;
			src << "#define Q" << k << "\t" << Mod::Q[k] << "u" << std::endl;
			src << "#define RSQ" << k << "\t" << Mod::rsq[k] << "u" << std::endl;
		}

		// Not converted into Montgomery form such that output is converted out of MF
		const RNS4 norm = RNS4::norm(LN - 1);
		src << "__constant uint g_norm[" << prm_count << "] = { " << norm.get_r(0) << "u";
		for (size_t l = 1; l < prm_count; ++l) src << ", " << norm.get_r(l) << "u";
		src << " };" << std::endl;

		const RNS4 rbase = RNS4(1u << lbase).toMonty();
		src << "__constant uint g_base[" << prm_count << "] = { " << rbase.get_r(0) << "u";
		for (size_t l = 1; l < prm_count; ++l) src << ", " << rbase.get_r(l) << "u";
		src << " };" << std::endl;

		RNS4 rbase_S = RNS4(1u << lbase); for (size_t l = 1; l < pck_factor; ++l) rbase_S *= rbase;
		src << "__constant uint g_baseS[" << prm_count << "] = { " << rbase_S.get_r(0) << "u";
		for (size_t l = 1; l < prm_count; ++l) src << ", " << rbase_S.get_r(l) << "u";
		src << " };" << std::endl;

		src << "#define INVP1_P0\t" << Mod::invert(Mod::toMonty(Mod::P[1], 0), 0) << "u" << std::endl;

		if (prm_count >= 3)
		{
			src << "#define INVP2_P0\t" << Mod::invert(Mod::toMonty(Mod::P[2], 0), 0) << "u" << std::endl;
			src << "#define INVP2_P1\t" << Mod::invert(Mod::toMonty(Mod::P[2], 1), 1) << "u" << std::endl;
		}
		if (prm_count >= 4)
		{
			src << "#define INVP3_P0\t" << Mod::invert(Mod::toMonty(Mod::P[3], 0), 0) << "u" << std::endl;
			src << "#define INVP3_P1\t" << Mod::invert(Mod::toMonty(Mod::P[3], 1), 1) << "u" << std::endl;
			src << "#define INVP3_P2\t" << Mod::invert(Mod::toMonty(Mod::P[3], 2), 2) << "u" << std::endl;
		}

		src << "#define CARRY_LENGTH\t" << CARRY_LENGTH << "u"  << std::endl;
		src << "#define CARRY_WG_SZ\t" << _engine.get_carry_workgroup_size() << "u" << std::endl;

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

		if (!_engine.readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_engine.loadProgram(src.str());
		_engine.alloc_memory(Reg_count);
		_engine.create_kernels();

		for (size_t k = 0; k < prm_count; ++k)
		{
			Mod::set_roots_R(&_buf[0 * (n / 2)], &_buf[1 * (n / 2)], LN, k);
			Mod::set_roots_Q(&_buf[2 * (n / 2)], &_buf[3 * (n / 2)], LN, k);
			_engine.write_memory(k * Reg_count + Reg_w_R, _buf, 2);
		}

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

		_engine.release_kernels();
		_engine.release_memory();
		_engine.clearProgram();

		delete[] _buf;
	}

	std::string get_type() const override { return _engine.getType(); }

	std::string get_data_size() const override
	{
		const size_t data_size = _engine.get_data_size();
		std::ostringstream ss; ss << "data size: ";
		if (data_size < (10u << 20)) ss << data_size / (1u << 10) << " KB"; else ss << data_size / (1u << 20) << " MB";
		return ss.str();
	}

	std::string get_cache_size() const override
	{
		const size_t cache_size = _engine.get_cache_size();
		std::ostringstream ss; ss << "cache size: ";
		if (cache_size < (10u << 20)) ss << cache_size / (1u << 10) << " KB"; else ss << cache_size / (1u << 20) << " MB";
		return ss.str();
	}

	std::string get_info_short() const override
	{
		const size_t n = size_t(1) << LN;
		std::ostringstream ss; ss << _prm_count << "-prime NTT of length " << n << "x" << _pck_factor << "x" << _lbase << " bits";
		return (ss.str());
	}

	std::string get_info() const override
	{
		std::ostringstream ss; ss << "Montgomery-McLaughlin-Gallot-Woltman reduction, " << get_info_short() << std::endl << _hinfo << std::endl;
		return (ss.str());
	}
};

inline void get_n_base(int & rln, int & rlbase, const int pck_factor, const double log2_16aN, const double p_2a)
{
	for (int lbase = 31; lbase >= 2; --lbase)
	{
		// log2(base^n) = lbase * n
		int ln = 4; while (double(lbase * pck_factor) * (1 << ln) < log2_16aN) ++ln;

		// A necessary condition is a * n * (base - 1)^2 < Prod_i(Pi) / 2
		const double dbase = std::pow(2.0, lbase * pck_factor);
		if ((1 << ln) * (dbase - 1) * (dbase - 1) < p_2a)
		{
			rln = ln;
			rlbase = lbase;
			break;
		}
	}
}

inline transform * create_transformGPU(const mpz_t & N, const int32_t a, const size_t device_id)
{
	// R = base^n - 1, Q = base^n + 1 such that base^n > 16 * a * N and n is a power of two
	long e_n; const double m_n = mpz_get_d_2exp(&e_n, N);
	const double log2_16aN = std::log2(16.0 * a) + std::log2(m_n) + e_n;

	const double p_2a2 = Mod::P[0] * double(Mod::P[1]) / 2 / a;
	int ln2 = 0, lbase2 = 0; get_n_base(ln2, lbase2, 1, log2_16aN, p_2a2);
	const double p_2a3 = p_2a2 * double(Mod::P[2]);
	int ln3 = 0, lbase3 = 0; get_n_base(ln3, lbase3, 2, log2_16aN, p_2a3);
	const double p_2a4 = p_2a3 * double(Mod::P[3]);
	int ln4 = 0, lbase4 = 0; get_n_base(ln4, lbase4, 2, log2_16aN, p_2a4);

	// const double c2 = 2 * double(1u << ln2) * ln2;
	// const double c3 = 3 * double(1u << ln3) * ln3;
	// const double c4 = 4 * double(1u << ln4) * ln4;
	// std::cout << "K = " << 2 << ": ln = " << ln2 << ", lbase = " << lbase2 << ", r = " << 1 << ", K*n*ln = " << c2 << std::endl;
	// std::cout << "K = " << 3 << ": ln = " << ln3 << ", lbase = " << lbase3 << ", r = " << 2 << ", K*n*ln = " << c3 << std::endl;
	// std::cout << "K = " << 4 << ": ln = " << ln4 << ", lbase = " << lbase4 << ", r = " << 2 << ", K*n*ln = " << c4 << std::endl;
	// return nullptr;

	const size_t prm_count = 2, pck_factor = 1;
	int ln = ln2, lbase = lbase2;

	// const size_t prm_count = 4, pck_factor = 2;
	// int ln = ln4, lbase = lbase4;

#if defined(QUICK_TEST)
	if (ln < 4) throw std::runtime_error("NTT size < 2^4 is not supported");
#else
	if (ln < 10) throw std::runtime_error("NTT size < 2^10 is not supported");
#endif
	if (ln > 23) throw std::runtime_error("NTT size > 2^23 is not supported");

	transform * p_transform = nullptr;

#if defined(QUICK_TEST)
	if      (ln ==  4) p_transform = new transformGPU<4>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln ==  5) p_transform = new transformGPU<5>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln ==  6) p_transform = new transformGPU<6>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln ==  7) p_transform = new transformGPU<7>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln ==  8) p_transform = new transformGPU<8>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln ==  9) p_transform = new transformGPU<9>(N, device_id, lbase, prm_count, pck_factor);
#endif
	if      (ln == 10) p_transform = new transformGPU<10>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 11) p_transform = new transformGPU<11>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 12) p_transform = new transformGPU<12>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 13) p_transform = new transformGPU<13>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 14) p_transform = new transformGPU<14>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 15) p_transform = new transformGPU<15>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 16) p_transform = new transformGPU<16>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 17) p_transform = new transformGPU<17>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 18) p_transform = new transformGPU<18>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 19) p_transform = new transformGPU<19>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 20) p_transform = new transformGPU<20>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 21) p_transform = new transformGPU<21>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 22) p_transform = new transformGPU<22>(N, device_id, lbase, prm_count, pck_factor);
	else if (ln == 23) p_transform = new transformGPU<23>(N, device_id, lbase, prm_count, pck_factor);

	return p_transform;
}
