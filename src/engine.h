/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <vector>

#include "ocl.h"

typedef cl_uint		uint32;
typedef cl_int		int32;
typedef cl_ulong	uint64;
typedef cl_long		int64;

class ZP
{
protected:
	uint32 _n;

public:
	ZP() {}
	explicit ZP(const uint32 n) : _n(n) {}
};

#define	CARRY_LENGTH	4u

#define BLK16		8	// local size = 16 * BLK16 * sizeof(ZP) = 512, workgroup size = 16 / 8 * BLK16 = 16
#define BLK32		8	// local size = 16 * BLK32 * sizeof(ZP) = 1KB, workgroup size = 16 / 8 * BLK16 = 32
#define BLK64		8	// local size = 16 * BLK16 * sizeof(ZP) = 2KB, workgroup size = 16 / 8 * BLK16 = 64
#define BLK128		4
#define BLK256		2
#define BLK512		1
#define BLK1024		1	// local size = 1024 * BLK1024 * sizeof(ZP) = 4KB, workgroup size = 1024 / 8 * BLK1024 = 128
#define BLK2048		1	// local size = 2048 * BLK2048 * sizeof(ZP) = 8KB, workgroup size = 2048 / 8 * BLK2048 = 256

#define CHUNK64		32	// local size = 64 * CHUNK64 * sizeof(ZP) = 8KB, workgroup size = 64 / 8 * CHUNK64 = 256	// TODO bench
#define CHUNK512	 4	// local size = 512 * CHUNK512 * sizeof(ZP) = 8KB, workgroup size = 512 / 8 * CHUNK512 = 256	// TODO bench

#define CREATE_KERNEL(name) _##name = create_kernel(#name);
#define CREATE_KERNEL_CARRY(name) _##name = create_kernel_carry(#name);

class engine : public device
{
private:
	const int _ln;
	const int _carry_shift;

	cl_mem _reg, _carry;

	cl_kernel _forward8 = nullptr, _backward8 = nullptr, _forward8_0 = nullptr, _backward8_0 = nullptr;
	cl_kernel _forward64_0 = nullptr, _backward64_0 = nullptr, _forward512_0 = nullptr, _backward512_0 = nullptr;;
	cl_kernel _square2x4 = nullptr, _square4x2 = nullptr, _square8 = nullptr;
	cl_kernel _square16 = nullptr, _square32 = nullptr, _square64 = nullptr, _square128 = nullptr;
	cl_kernel _square256 = nullptr, _square512 = nullptr, _square1024 = nullptr, _square2048 = nullptr;
	cl_kernel _mul2x4 = nullptr, _mul4x2 = nullptr, _mul8 = nullptr;
	cl_kernel _mul16 = nullptr, _mul32 = nullptr, _mul64 = nullptr, _mul128 = nullptr;
	cl_kernel _mul256 = nullptr, _mul512 = nullptr, _mul1024 = nullptr, _mul2048 = nullptr;
	cl_kernel _fwd4x2 = nullptr, _fwd8 = nullptr;
	cl_kernel _fwd16 = nullptr, _fwd32 = nullptr, _fwd64 = nullptr, _fwd128 = nullptr;
	cl_kernel _fwd256 = nullptr, _fwd512 = nullptr, _fwd1024 = nullptr, _fwd2048 = nullptr;
	cl_kernel _carry_mul = nullptr, _carry_sub = nullptr, _carry_complete = nullptr;
	cl_kernel _mod_Q = nullptr, _mod_R_add2 = nullptr;
	cl_kernel _set = nullptr, _extend = nullptr;

	std::vector<cl_kernel> _kernels;

	static constexpr int ilog2_32(const uint32_t n) { return (n == 0) ? -1 : (31 - __builtin_clz(n)); }

public:
	engine(const int ln, const size_t device_id) : device(platform(), device_id), _ln(ln),
		_carry_shift(ilog2_32(uint32_t(std::min((size_t(1) << ln) / CARRY_LENGTH, std::min(size_t(256), getMaxWorkGroupSize()))))) {}
	virtual ~engine() {}

	int get_carry_shift() const { return _carry_shift; }
	size_t get_carry_workgroup_size() const { return (size_t(1) << _carry_shift); }

///////////////////////////////

public:
	void alloc_memory(const size_t reg_count)
	{
#if defined(ocl_debug)
		std::cout << "Alloc gpu memory." << std::endl;
#endif
		const size_t n = size_t(1) << _ln;
		if (_ln != 0)
		{
			const size_t size = 2 * n * sizeof(ZP);
			const size_t mem_size = reg_count * size;
			if (getMaxMemAllocSize() < mem_size)
			{
				std::cerr << "Warning: trying to allocate " << mem_size / (1u << 20)
					<< " MB and the maximum OpenCL size is " << getMaxMemAllocSize() / (1u << 20) << " MB." << std::endl;
			}
			_reg = _createBuffer(CL_MEM_READ_WRITE, mem_size);
			_carry = _createBuffer(CL_MEM_READ_WRITE, n / CARRY_LENGTH * sizeof(int64));
		}
	}

	void release_memory()
	{
#if defined(ocl_debug)
		std::cout << "Free gpu memory." << std::endl;
#endif
		if (_ln != 0)
		{
			_releaseBuffer(_reg);
			_releaseBuffer(_carry);
		}
	}

///////////////////////////////

private:
	cl_kernel create_kernel(const char * const kernel_name)
	{
		cl_kernel kernel = _createKernel(kernel_name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_reg);
		_kernels.push_back(kernel);
		return kernel;
	}

	cl_kernel create_kernel_carry(const char * const kernel_name)
	{
		cl_kernel kernel = create_kernel(kernel_name);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_carry);
		return kernel;
	}

public:
	void create_kernels()
	{
#if defined(ocl_debug)
		std::cout << "Create ocl kernels." << std::endl;
#endif
		const int ln = _ln;
		if (ln >= 21)
		{
			CREATE_KERNEL(forward8);
			CREATE_KERNEL(backward8);
		}
		if (ln <= 12)
		{
			CREATE_KERNEL(forward8_0);
			CREATE_KERNEL(backward8_0);
		}
		if ((ln > 12) && (ln <= 16))
		{
			CREATE_KERNEL(forward64_0);
			CREATE_KERNEL(backward64_0);
		}
		if (ln > 16)
		{
			CREATE_KERNEL(forward512_0);
			CREATE_KERNEL(backward512_0);
		}

		if (ln == 4)
		{
			CREATE_KERNEL(square2x4);
			CREATE_KERNEL(mul2x4);
		}
		else if (ln == 5)
		{
			CREATE_KERNEL(square4x2);
			CREATE_KERNEL(mul4x2);
			CREATE_KERNEL(fwd4x2);
		}
		else if (ln == 6)
		{
			CREATE_KERNEL(square8);
			CREATE_KERNEL(mul8);
			CREATE_KERNEL(fwd8);
		}
		else if (ln == 7)
		{
			CREATE_KERNEL(square16);
			CREATE_KERNEL(mul16);
			CREATE_KERNEL(fwd16);
		}
		else if (ln == 8)
		{
			CREATE_KERNEL(square32);
			CREATE_KERNEL(mul32);
			CREATE_KERNEL(fwd32);
		}
		else if (ln == 9)
		{
			CREATE_KERNEL(square64);
			CREATE_KERNEL(mul64);
			CREATE_KERNEL(fwd64);
		}
		else if ((ln == 10) || (ln == 13))
		{
			CREATE_KERNEL(square128);
			CREATE_KERNEL(mul128);
			CREATE_KERNEL(fwd128);
		}
		else if ((ln == 11) || (ln == 14) || (ln == 17))
		{
			CREATE_KERNEL(square256);
			CREATE_KERNEL(mul256);
			CREATE_KERNEL(fwd256);
		}
		else if ((ln == 12) || (ln == 15) || (ln == 18) || (ln == 21))
		{
			CREATE_KERNEL(square512);
			CREATE_KERNEL(mul512);
			CREATE_KERNEL(fwd512);
		}
		else if ((ln == 16) || (ln == 19) || (ln == 22))
		{
			CREATE_KERNEL(square1024);
			CREATE_KERNEL(mul1024);
			CREATE_KERNEL(fwd1024);
		}
		else if ((ln == 20) || (ln == 23))
		{
			CREATE_KERNEL(square2048);
			CREATE_KERNEL(mul2048);
			CREATE_KERNEL(fwd2048);
		}

		CREATE_KERNEL_CARRY(carry_mul);
		CREATE_KERNEL_CARRY(carry_sub);
		CREATE_KERNEL_CARRY(carry_complete);

		CREATE_KERNEL(mod_Q);
		CREATE_KERNEL(mod_R_add2);

		CREATE_KERNEL(set);
		CREATE_KERNEL(extend);
	}

	void release_kernels()
	{
#if defined(ocl_debug)
		std::cout << "Release ocl kernels." << std::endl;
#endif
		for (cl_kernel & kernel : _kernels) _releaseKernel(kernel);
		_kernels.clear();
	}

///////////////////////////////

public:
	void read_memory(const size_t reg_i, ZP * const ptr, const size_t count = 1)
	{
		const size_t n = size_t(1) << _ln;
		const size_t size = n * sizeof(ZP);
		_readBuffer(_reg, ptr, count * size, reg_i * size);
	}

	void write_memory(const size_t reg_i, const ZP * const ptr, const size_t count = 1)
	{
		const size_t n = size_t(1) << _ln;
		const size_t size = n * sizeof(ZP);
		_writeBuffer(_reg, ptr, count * size, reg_i * size);
	}

///////////////////////////////

private:
	void execute_kernel(cl_kernel & kernel, const size_t reg_x, const size_t reg_w, const size_t wg_size = 0)
	{
		const uint32 rx = uint32(reg_x), rw = uint32(reg_w);
		_setKernelArg(kernel, 1, sizeof(uint32), &rx);
		_setKernelArg(kernel, 2, sizeof(uint32), &rw);
		_executeKernel(kernel, 2u << (_ln - 3), wg_size);
	}

	void execute_kernel_fb(cl_kernel & kernel, const size_t reg_x, const size_t reg_w, const int lm, const size_t s, const size_t wg_size = 0)
	{
		const int32 lm32 = int32(lm); const uint32 s32 = uint32(s);
		_setKernelArg(kernel, 3, sizeof(int32), &lm32);
		_setKernelArg(kernel, 4, sizeof(uint32), &s32);
		execute_kernel(kernel, reg_x, reg_w, wg_size);
	}

	void execute_kernel_mul(cl_kernel & kernel, const size_t reg_x, const size_t reg_xp, const size_t reg_w, const size_t wg_size = 0)
	{
		const uint32 rxp = uint32(reg_xp);
		_setKernelArg(kernel, 3, sizeof(uint32), &rxp);
		execute_kernel(kernel, reg_x, reg_w, wg_size);
	}

public:
 	void forward8(const size_t reg_x, const size_t reg_w, const int lm, const size_t s) { execute_kernel_fb(_forward8, reg_x, reg_w, lm, s); }
	void backward8(const size_t reg_x, const size_t reg_w, const int lm, const size_t s) { execute_kernel_fb(_backward8, reg_x, reg_w, lm, s); }

	void forward8_0(const size_t reg_x, const size_t reg_w) { execute_kernel(_forward8_0, reg_x, reg_w); }
	void backward8_0(const size_t reg_x, const size_t reg_w) { execute_kernel(_backward8_0, reg_x, reg_w); }
	void forward64_0(const size_t reg_x, const size_t reg_w) { execute_kernel(_forward64_0, reg_x, reg_w, 64 / 8 * CHUNK64); }
	void backward64_0(const size_t reg_x, const size_t reg_w) { execute_kernel(_backward64_0, reg_x, reg_w, 64 / 8 * CHUNK64); }
	void forward512_0(const size_t reg_x, const size_t reg_w) { execute_kernel(_forward512_0, reg_x, reg_w, 512 / 8 * CHUNK512); }
	void backward512_0(const size_t reg_x, const size_t reg_w) { execute_kernel(_backward512_0, reg_x, reg_w, 512 / 8 * CHUNK512); }

	void square(const size_t reg_x, const size_t reg_w)
	{
		const int ln = _ln;
		if      (ln == 4) execute_kernel(_square2x4, reg_x, reg_w);
		else if (ln == 5) execute_kernel(_square4x2, reg_x, reg_w);
		else if (ln == 6) execute_kernel(_square8, reg_x, reg_w);
		else if (ln == 7) execute_kernel(_square16, reg_x, reg_w, 16 / 8 * BLK16);
		else if (ln == 8) execute_kernel(_square32, reg_x, reg_w, 32 / 8 * BLK32);
		else if (ln == 9) execute_kernel(_square64, reg_x, reg_w, 64 / 8 * BLK64);
		else if ((ln == 10) || (ln == 13)) execute_kernel(_square128, reg_x, reg_w, 128 / 8 * BLK128);
		else if ((ln == 11) || (ln == 14) || (ln == 17)) execute_kernel(_square256, reg_x, reg_w, 256 / 8 * BLK256);
		else if ((ln == 12) || (ln == 15) || (ln == 18) || (ln == 21)) execute_kernel(_square512, reg_x, reg_w, 512 / 8 * BLK512);
		else if ((ln == 16) || (ln == 19) || (ln == 22)) execute_kernel(_square1024, reg_x, reg_w, 1024 / 8 * BLK1024);
		else if ((ln == 20) || (ln == 23)) execute_kernel(_square2048, reg_x, reg_w, 2048 / 8 * BLK2048);
	}

	void mul(const size_t reg_x, const size_t reg_xp, const size_t reg_w)
	{
		const int ln = _ln;

		if      (ln == 4) execute_kernel_mul(_mul2x4, reg_x, reg_xp, reg_w);
		else if (ln == 5) execute_kernel_mul(_mul4x2, reg_x, reg_xp, reg_w);
		else if (ln == 6) execute_kernel_mul(_mul8, reg_x, reg_xp, reg_w);
		else if (ln == 7) execute_kernel_mul(_mul16, reg_x, reg_xp, reg_w, 16 / 8 * BLK16);
		else if (ln == 8) execute_kernel_mul(_mul32, reg_x, reg_xp, reg_w, 32 / 8 * BLK32);
		else if (ln == 9) execute_kernel_mul(_mul64, reg_x, reg_xp, reg_w, 64 / 8 * BLK64);
		else if ((ln == 10) || (ln == 13)) execute_kernel_mul(_mul128, reg_x, reg_xp, reg_w, 128 / 8 * BLK128);
		else if ((ln == 11) || (ln == 14) || (ln == 17)) execute_kernel_mul(_mul256, reg_x, reg_xp, reg_w, 256 / 8 * BLK256);
		else if ((ln == 12) || (ln == 15) || (ln == 18) || (ln == 21)) execute_kernel_mul(_mul512, reg_x, reg_xp, reg_w, 512 / 8 * BLK512);
		else if ((ln == 16) || (ln == 19) || (ln == 22)) execute_kernel_mul(_mul1024, reg_x, reg_xp, reg_w, 1024 / 8 * BLK1024);
		else if ((ln == 20) || (ln == 23)) execute_kernel_mul(_mul2048, reg_x, reg_xp, reg_w, 2048 / 8 * BLK2048);
	}

	void fwd(const size_t reg_x, const size_t reg_w)
	{
		const int ln = _ln;
		if      (ln == 4) {} // 2x4
		else if (ln == 5) execute_kernel(_fwd4x2, reg_x, reg_w);
		else if (ln == 6) execute_kernel(_fwd8, reg_x, reg_w);
		else if (ln == 7) execute_kernel(_fwd16, reg_x, reg_w);
		else if (ln == 8) execute_kernel(_fwd32, reg_x, reg_w, 32 / 8 * BLK32);
		else if (ln == 9) execute_kernel(_fwd64, reg_x, reg_w, 64 / 8 * BLK64);
		else if ((ln == 10) || (ln == 13)) execute_kernel(_fwd128, reg_x, reg_w, 128 / 8 * BLK128);
		else if ((ln == 11) || (ln == 14) || (ln == 17)) execute_kernel(_fwd256, reg_x, reg_w, 256 / 8 * BLK256);
		else if ((ln == 12) || (ln == 15) || (ln == 18) || (ln == 21)) execute_kernel(_fwd512, reg_x, reg_w, 512 / 8 * BLK512);
		else if ((ln == 16) || (ln == 19) || (ln == 22)) execute_kernel(_fwd1024, reg_x, reg_w, 1024 / 8 * BLK1024);
		else if ((ln == 20) || (ln == 23)) execute_kernel(_fwd2048, reg_x, reg_w, 2048 / 8 * BLK2048);
	}

	void carry_mul(const size_t reg_x, const int32 a)
	{
		const uint32 rx = uint32(reg_x);
		_setKernelArg(_carry_mul, 2, sizeof(uint32), &rx);
		_setKernelArg(_carry_mul, 3, sizeof(int32), &a);
		_executeKernel(_carry_mul, (1u << _ln) / CARRY_LENGTH, 1u << _carry_shift);
	}

	void carry_sub(const size_t reg_x, const size_t reg_y)
	{
		const uint32 rx = uint32(reg_x), ry = uint32(reg_y);
		_setKernelArg(_carry_sub, 2, sizeof(uint32), &rx);
		_setKernelArg(_carry_sub, 3, sizeof(uint32), &ry);
		_executeKernel(_carry_sub, (1u << _ln) / CARRY_LENGTH, 1u << _carry_shift);
	}

	void carry_complete(const size_t reg_x, const bool nega)
	{
		const uint32 rx = uint32(reg_x); const int32 is_nega = nega ? 1 : 0;
		_setKernelArg(_carry_complete, 2, sizeof(uint32), &rx);
		_setKernelArg(_carry_complete, 3, sizeof(int32), &is_nega);
		_executeKernel(_carry_complete, ((1u << _ln) / CARRY_LENGTH) >> _carry_shift);
	}

	void mod_Q(const size_t reg_x)
	{
		const uint32 rx = uint32(reg_x);
		_setKernelArg(_mod_Q, 1, sizeof(uint32), &rx);
		_executeKernel(_mod_Q, 1);
	}

	void mod_R_add2(const size_t reg_x)
	{
		const uint32 rx = uint32(reg_x);
		_setKernelArg(_mod_R_add2, 1, sizeof(uint32), &rx);
		_executeKernel(_mod_R_add2, 1);
	}

	void set(const size_t reg_y, const size_t reg_x)
	{
		if (reg_x == reg_y) return;
		const uint32 ry = uint32(reg_y), rx = uint32(reg_x);
		_setKernelArg(_set, 1, sizeof(uint32), &ry);
		_setKernelArg(_set, 2, sizeof(uint32), &rx);
		_executeKernel(_set, 1u << (_ln - 2));
	}

	void extend01()
	{
		_executeKernel(_extend, 2u << (_ln - 2));	// Reg0 & Reg1
	}
};
