#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__

#include <cstdlib>
#include <mpi.h>
#include <assert.h>
#include <complex>

typedef std::complex<double> double_complex;

enum mpi_op_t {op_sum, op_max};

/// Wrapper for data types
template <typename T> 
class type_wrapper;

template<> 
class type_wrapper<double>
{
    public:
        typedef std::complex<double> complex_t;
        typedef double real_t;
        
        static inline double conjugate(double const& v)
        {
            return v;
        }

        static inline double sift(double_complex const& v)
        {
            return std::real(v);
        }
        
        static MPI_Datatype mpi_type_id()
        {
            return MPI_DOUBLE;
        }

        static bool is_complex()
        {
            return false;
        }
        
        static bool is_real()
        {
            return true;
        }

        static inline double random()
        {
            return double(rand()) / RAND_MAX;
        }
};

template<> 
class type_wrapper<long double>
{
    public:
        typedef std::complex<long double> complex_t;
        typedef long double real_t;
        
        static MPI_Datatype mpi_type_id()
        {
            return MPI_LONG_DOUBLE;
        }

        static bool is_complex()
        {
            return false;
        }
        
        static bool is_real()
        {
            return true;
        }
};

template<> 
class type_wrapper<float>
{
    public:
        typedef std::complex<float> complex_t;
        typedef float real_t;
};

template<> 
class type_wrapper< std::complex<double> >
{
    public:
        typedef std::complex<double> complex_t;
        typedef double real_t;
        
        static inline std::complex<double> conjugate(double_complex const& v)
        {
            return conj(v);
        }
        
        static inline std::complex<double> sift(double_complex const& v)
        {
            return v;
        }
        
        static MPI_Datatype mpi_type_id()
        {
            return MPI_COMPLEX16;
        }

        static bool is_complex()
        {
            return true;
        }
        
        static bool is_real()
        {
            return false;
        }
        
        static inline std::complex<double> random()
        {
            return std::complex<double>(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
        }
};

template<> 
class type_wrapper<int>
{
    public:

        static MPI_Datatype mpi_type_id()
        {
            return MPI_INT;
        }
};

template<> 
class type_wrapper<char>
{
    public:

        static MPI_Datatype mpi_type_id()
        {
            return MPI_CHAR;
        }

        static inline char random()
        {
            return static_cast<char>(255 * (double(rand()) / RAND_MAX));
        }
};

#endif // __TYPEDEFS_H__
