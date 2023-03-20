#include <iostream>

enum class distr_type { flat, rejection, weighted };

class distribution
{
    public:
    distribution( const distr_type & type ) : m_type( type ) {}

    private:
    distr_type   m_type; // Type of distribution
    std::int64_t m_N;    // Total number of samples
    std::int64_t m_M;    // Number of bins
    long double  m_dx;   // Bin width
    /* The domain & range min/max. */
    long double m_dmin;
    long double m_dmax;
    long double m_rmin;
    long double m_rmax;
};

int
main() {}
