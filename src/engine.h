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

#define	CARRY_LENGTH	4u

#define CREATE_KERNEL(name) _##name = create_kernel(#name);

class engine : public device
{
private:
	const int _ln;
	const int _carry_shift;

	cl_mem _reg, _carry;

	cl_kernel _forward8 = nullptr, _backward8 = nullptr, _forward8_0 = nullptr;
	cl_kernel _square2x4 = nullptr, _square4x2 = nullptr, _square8 = nullptr;
	cl_kernel _mul2x4 = nullptr, _mul4x2 = nullptr, _mul8 = nullptr;
	cl_kernel _fwd4x2 = nullptr, _fwd8 = nullptr;
	cl_kernel _carry_mul = nullptr, _carry_sub = nullptr, _carry_complete = nullptr;
	cl_kernel _mod_Q = nullptr, _mod_R_add2 = nullptr;
	cl_kernel _set = nullptr;

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
			const size_t size = 2 * n * sizeof(uint32);
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

public:
	void create_kernels()
	{
#if defined(ocl_debug)
		std::cout << "Create ocl kernels." << std::endl;
#endif
		CREATE_KERNEL(forward8);
		CREATE_KERNEL(backward8);
		CREATE_KERNEL(forward8_0);

		CREATE_KERNEL(square2x4);
		CREATE_KERNEL(square4x2);
		CREATE_KERNEL(square8);

		CREATE_KERNEL(mul2x4);
		CREATE_KERNEL(mul4x2);
		CREATE_KERNEL(mul8);

		CREATE_KERNEL(fwd4x2);
		CREATE_KERNEL(fwd8);

		CREATE_KERNEL(carry_mul);
		CREATE_KERNEL(carry_sub);
		CREATE_KERNEL(carry_complete);

		CREATE_KERNEL(mod_Q);
		CREATE_KERNEL(mod_R_add2);

		CREATE_KERNEL(set);
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
	void read_memory(const size_t reg_i, void * const ptr, const size_t count = 1)
	{
		const size_t n = size_t(1) << _ln;
		const size_t size = 2 * n * sizeof(uint32);
		_readBuffer(_reg, ptr, count * size, reg_i * size);
	}

	void write_memory(const size_t reg_i, const void * const ptr, const size_t count = 1)
	{
		const size_t n = size_t(1) << _ln;
		const size_t size = 2 * n * sizeof(uint32);
		_writeBuffer(_reg, ptr, count * size, reg_i * size);
	}

///////////////////////////////

public:
 	void forward8(const size_t reg_x, const size_t reg_w, const int lm, const size_t s)
	{
		const uint32 rx = uint32(reg_x), rw = uint32(reg_w);
		const int32 lm32 = int32(lm); const uint32 s32 = uint32(s);
		_setKernelArg(_forward8, 1, sizeof(uint32), &rx);
		_setKernelArg(_forward8, 2, sizeof(uint32), &rw);
		_setKernelArg(_forward8, 3, sizeof(int32), &lm32);
		_setKernelArg(_forward8, 4, sizeof(uint32), &s32);
		_executeKernel(_forward8, 1u << (_ln - 3));
	}

	void backward8(const size_t reg_x, const size_t reg_w, const int lm, const size_t s)
	{
		const uint32 rx = uint32(reg_x), rw = uint32(reg_w);
		const int32 lm32 = int32(lm); const uint32 s32 = uint32(s);
		_setKernelArg(_backward8, 1, sizeof(uint32), &rx);
		_setKernelArg(_backward8, 2, sizeof(uint32), &rw);
		_setKernelArg(_backward8, 3, sizeof(int32), &lm32);
		_setKernelArg(_backward8, 4, sizeof(uint32), &s32);
		_executeKernel(_backward8, 1u << (_ln - 3));
	}

	void forward8_0(const size_t reg_x, const size_t reg_w)
	{
		const uint32 rx = uint32(reg_x), rw = uint32(reg_w);
		_setKernelArg(_forward8_0, 1, sizeof(uint32), &rx);
		_setKernelArg(_forward8_0, 2, sizeof(uint32), &rw);
		_executeKernel(_forward8_0, 1u << (_ln - 3));
	}

	void square2x4(const size_t reg_x, const size_t reg_w)
	{
		const uint32 rx = uint32(reg_x), rw = uint32(reg_w);
		_setKernelArg(_square2x4, 1, sizeof(uint32), &rx);
		_setKernelArg(_square2x4, 2, sizeof(uint32), &rw);
		_executeKernel(_square2x4, 1u << (_ln - 3));
	}

	void square4x2(const size_t reg_x, const size_t reg_w)
	{
		const uint32 rx = uint32(reg_x), rw = uint32(reg_w);
		_setKernelArg(_square4x2, 1, sizeof(uint32), &rx);
		_setKernelArg(_square4x2, 2, sizeof(uint32), &rw);
		_executeKernel(_square4x2, 1u << (_ln - 3));
	}

	void square8(const size_t reg_x, const size_t reg_w)
	{
		const uint32 rx = uint32(reg_x), rw = uint32(reg_w);
		_setKernelArg(_square8, 1, sizeof(uint32), &rx);
		_setKernelArg(_square8, 2, sizeof(uint32), &rw);
		_executeKernel(_square8, 1u << (_ln - 3));
	}

	void mul2x4(const size_t reg_x, const size_t reg_xp, const size_t reg_w)
	{
		const uint32 rx = uint32(reg_x), rxp = uint32(reg_xp), rw = uint32(reg_w);
		_setKernelArg(_mul2x4, 1, sizeof(uint32), &rx);
		_setKernelArg(_mul2x4, 2, sizeof(uint32), &rxp);
		_setKernelArg(_mul2x4, 3, sizeof(uint32), &rw);
		_executeKernel(_mul2x4, 1u << (_ln - 3));
	}

	void mul4x2(const size_t reg_x, const size_t reg_xp, const size_t reg_w)
	{
		const uint32 rx = uint32(reg_x), rxp = uint32(reg_xp), rw = uint32(reg_w);
		_setKernelArg(_mul4x2, 1, sizeof(uint32), &rx);
		_setKernelArg(_mul4x2, 2, sizeof(uint32), &rxp);
		_setKernelArg(_mul4x2, 3, sizeof(uint32), &rw);
		_executeKernel(_mul4x2, 1u << (_ln - 3));
	}

	void mul8(const size_t reg_x, const size_t reg_xp, const size_t reg_w)
	{
		const uint32 rx = uint32(reg_x), rxp = uint32(reg_xp), rw = uint32(reg_w);
		_setKernelArg(_mul8, 1, sizeof(uint32), &rx);
		_setKernelArg(_mul8, 2, sizeof(uint32), &rxp);
		_setKernelArg(_mul8, 3, sizeof(uint32), &rw);
		_executeKernel(_mul8, 1u << (_ln - 3));
	}

	void fwd4x2(const size_t reg_x, const size_t reg_w)
	{
		const uint32 rx = uint32(reg_x), rw = uint32(reg_w);
		_setKernelArg(_fwd4x2, 1, sizeof(uint32), &rx);
		_setKernelArg(_fwd4x2, 2, sizeof(uint32), &rw);
		_executeKernel(_fwd4x2, 1u << (_ln - 3));
 	}

	void fwd8(const size_t reg_x, const size_t reg_w)
	{
		const uint32 rx = uint32(reg_x), rw = uint32(reg_w);
		_setKernelArg(_fwd8, 1, sizeof(uint32), &rx);
		_setKernelArg(_fwd8, 2, sizeof(uint32), &rw);
		_executeKernel(_fwd8, 1u << (_ln - 3));
	}

	void carry_mul(const size_t reg_x, const int32 a)
	{
		const uint32 rx = uint32(reg_x);
		_setKernelArg(_carry_mul, 1, sizeof(cl_mem), &_carry);
		_setKernelArg(_carry_mul, 2, sizeof(uint32), &rx);
		_setKernelArg(_carry_mul, 3, sizeof(int32), &a);
		_executeKernel(_carry_mul, (1u << _ln) / CARRY_LENGTH, 1u << _carry_shift);
	}

	void carry_sub(const size_t reg_x, const size_t reg_y)
	{
		const uint32 rx = uint32(reg_x), ry = uint32(reg_y);
		_setKernelArg(_carry_sub, 1, sizeof(cl_mem), &_carry);
		_setKernelArg(_carry_sub, 2, sizeof(uint32), &rx);
		_setKernelArg(_carry_sub, 3, sizeof(uint32), &ry);
		_executeKernel(_carry_sub, (1u << _ln) / CARRY_LENGTH, 1u << _carry_shift);
	}

	void carry_complete(const size_t reg_x, const bool nega)
	{
		const uint32 rx = uint32(reg_x); const int32 is_nega = nega ? 1 : 0;
		_setKernelArg(_carry_complete, 1, sizeof(cl_mem), &_carry);
		_setKernelArg(_carry_complete, 2, sizeof(uint32), &rx);
		_setKernelArg(_carry_complete, 3, sizeof(int32), &is_nega);
		_executeKernel(_carry_complete, ((1u << _ln) / CARRY_LENGTH) >> _carry_shift);	// TODO log2
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
		_executeKernel(_set, 1 << _ln);
	}
};
