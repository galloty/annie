/*
Copyright 2026, Yves Gallot

annie is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <string>
#include <memory>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(_WIN64)
#include <Windows.h>
#else
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif

#include <gmp.h>

#include "annie.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wuninitialized"
extern "C" {
#include <pari.h>
}
#pragma GCC diagnostic pop

#define	QUICK_TEST	true

class application
{
private:
	struct deleter { void operator()(const application * const p) { delete p; } };

private:
	static void quit(int)
	{
		get_instance()._instance->quit();
	}

private:
#if defined(_WIN64)
	static BOOL WINAPI HandlerRoutine(DWORD)
	{
		quit(1);
		return TRUE;
	}
#endif

private:
	annie * _instance = nullptr;

public:
	application()
	{
#if defined(_WIN64)	
		SetConsoleCtrlHandler(HandlerRoutine, TRUE);
#else
		signal(SIGTERM, quit);
		signal(SIGINT, quit);
#endif
	}

	virtual ~application() { if (_instance != nullptr) delete _instance; }

	static application & get_instance()
	{
		static std::unique_ptr<application, deleter> p_instance(new application());
		return *p_instance;
	}

private:
	static std::string header(const std::vector<std::string> & args)
	{
		const char * const sysver =
#if defined(_WIN64)
#if defined(__aarch64__)
			"win arm64";
#else
			"win x64";
#endif
#elif defined(__linux__)
#if defined(__aarch64__)
			"linux arm64";
#else
			"linux x64";
#endif
#elif defined(__APPLE__)
#if defined(__aarch64__)
			"macOS arm64";
#else
			"macOS x64";
#endif
#else
			"unknown";
#endif

		std::ostringstream ssc;
#if defined(__clang__)
		ssc << ", clang-" << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(__GNUC__)
		ssc << ", gcc-" << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#endif

		std::ostringstream ss;
		ss << "annie version 26.10.0 (" << sysver << ssc.str() << ")" << std::endl;
		ss << "Copyright (c) 2026, Yves Gallot" << std::endl;
		ss << "annie is free source code, under the MIT license." << std::endl;
		ss << std::endl << "Command line: '";
		bool first = true;
		for (const std::string & arg : args)
		{
			if (first) first = false; else ss << " ";
			ss << arg;
		}
		ss << "'" << std::endl;
		return ss.str();
	}

private:
	static std::string usage()
	{
		std::ostringstream ss;
		ss << "Usage: annie -q <expression> or -f <filename> [options]" << std::endl;
		ss << "  -d <n>    set the OpenCL device number (default 0)" << std::endl << std::endl;
		// ss << "  -cpu      execute the implementation for CPU" << std::endl;
		ss << " <expression> is a Pari/GP math expression. It must be an integer." << std::endl;
		ss << " If the expression is too long, it can be written to the file <filename>." << std::endl;
		ss << " annie checks if the integer n is an Euler probable prime to base a," << std::endl;
		ss << " where a is the smallest integer such that the Jacobi symbol (a/n) = -1." << std::endl;
		ss << " If n is a Proth number then the test is a primality proof (Proth's theorem)." << std::endl;
		ss << std::endl;
		return ss.str();
	}

public:
	void run(int argc, char * argv[])
	{
		std::vector<std::string> args;
		for (int i = 1; i < argc; ++i) args.push_back(argv[i]);

		std::cout << header(args) << std::endl;

#ifdef QUICK_TEST
		_instance = new annie();

		mpz_t n; mpz_init(n);
		// 3, 4, 5, 7, 11, 13, 17, 23, 29, 43, 47, 83, 131, 137, 359, 431, 433, 449, 509, 569, 571, 2971, 4723, 5387,
		// 9311, 9677, 14431, 25561, 30757, 35999, 37511, 50833, 81839, 104911, 130021, 148091, 201107
		for (unsigned long j = 3; j <= 5388; ++j)
		{
			const unsigned long i = (j == 5388) ? 30757 : j;
			const std::string expression = "fibonacci(" + std::to_string(i) + ")";
			mpz_fib_ui(n, i);

			const annie::EReturn ret = _instance->check(n, expression, (j < 5) || (j == 5388));
			if (ret != annie::EReturn::Success)
			{
				if (ret == annie::EReturn::Failed) std::cout << expression << ": validation failed!";
				else if (ret == annie::EReturn::Aborted) std::cout << expression << ": test was aborted.";
				std::cout << std::endl;
				break;
			}
		}
		mpz_clear(n);
#else
		std::string expression = "167!-1", filename;	// DC2EBAF7065DD7D2, 166!-1 is prime
		size_t device = 0;

		// parse args
		for (size_t i = 0, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if (arg.substr(0, 2) == "-q")
			{
				if (!filename.empty()) throw std::runtime_error("-q used with an incompatible option (-f)");
				expression = ((arg == "-q") && (i + 1 < size)) ? args[++i] : arg.substr(2);
			}
			if (arg.substr(0, 2) == "-f")
			{
				if (!expression.empty()) throw std::runtime_error("-f used with an incompatible option (-q)");
				filename = ((arg == "-f") && (i + 1 < size)) ? args[++i] : arg.substr(2);
			}
			if (arg.substr(0, 2) == "-d")
			{
				const std::string dstr = ((arg == "-d") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				device = size_t(std::atoi(dstr.c_str()));
			}
		}

		if (!filename.empty())
		{
			std::ifstream file(filename);
			if (!file.is_open()) throw std::runtime_error("cannot open input file");
			std::getline(file, expression);
			file.close();
		}

		if (expression.empty())
		{
			std::cout << usage();
			return;
		}

		std::string n_str;
		// Initialize the PARI stack (100 MB)
		pari_init(100000000, 0);
		try
		{
			// Parse and evaluate the string using the GP interpreter
			GEN result = gp_read_str(expression.c_str());
			// Convert the resulting GEN type to a string
			char * const cstr = GENtostr(result);
			n_str = cstr;
			pari_free(cstr);
		} 
		catch (...)
		{
			throw std::runtime_error("an error occurred during Pari/GP evaluation");
		}
		pari_close();

		mpz_t n; mpz_init_set_str(n, n_str.c_str(), 10);

		_instance = new annie();
		const annie::EReturn ret = _instance->check(n, expression);
		mpz_clear(n);

		if (ret == annie::EReturn::Failed) std::cout << "Validation failed!";
		else if (ret == annie::EReturn::Aborted) std::cout << "Test was aborted.";
		std::cout << std::endl;
#endif
	}
};

int main(int argc, char * argv[])
{
	try
	{
		application & app = application::get_instance();
		app.run(argc, argv);
	}
	catch (const std::runtime_error & e)
	{
		std::cerr << "Error: " << e.what() << "." << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
