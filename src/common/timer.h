/*
 * timer.h - Portable High-Resolution Timer for HPC Benchmarking
 * ==============================================================
 *
 * This header provides a single function get_time() that returns
 * the current wall-clock time in seconds (as a double).
 *
 * Usage:
 *   double start = get_time();
 *   // ... do some work ...
 *   double end = get_time();
 *   double elapsed = end - start;
 *   printf("Elapsed: %.6f seconds\n", elapsed);
 *
 * The timer automatically picks the best available method:
 *   1. omp_get_wtime()          - if compiled with OpenMP (-fopenmp)
 *   2. QueryPerformanceCounter   - on Windows without OpenMP
 *   3. gettimeofday()           - on Linux/Unix without OpenMP
 *
 * All methods give microsecond or better resolution.
 *
 * Authors: Team HPC
 * Date: March 2026
 */

#ifndef TIMER_H
#define TIMER_H

#ifdef _OPENMP
    /* 
     * OpenMP timer - the gold standard for HPC timing.
     * Available whenever code is compiled with -fopenmp flag.
     * Returns wall-clock time in seconds with high resolution.
     */
    #include <omp.h>
    static inline double get_time(void) {
        return omp_get_wtime();
    }

#elif defined(_WIN32)
    /* 
     * Windows high-resolution timer using QueryPerformanceCounter.
     * Gives nanosecond-level resolution on modern Windows.
     */
    #include <windows.h>
    static inline double get_time(void) {
        LARGE_INTEGER freq, count;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&count);
        return (double)count.QuadPart / (double)freq.QuadPart;
    }

#else
    /* 
     * Linux/Unix fallback timer using gettimeofday.
     * Gives microsecond resolution - good enough for most benchmarks.
     */
    #include <sys/time.h>
    static inline double get_time(void) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
    }

#endif

#endif /* TIMER_H */
