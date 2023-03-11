#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

long long
MSG_i64( const int64_t s, const int64_t a, const int64_t c, const int64_t m ) {
    static bool    first_time = true;
    static int64_t seed = 0;
    if ( first_time ) {
        seed = s;
        first_time = false;
    }
    seed = ( a * seed + c ) % m;
    return seed;
}

float
MSG_f( const int64_t s, const int64_t a, const int64_t c, const int64_t m ) {
    static bool    first_time = true;
    static int64_t seed = 0;
    if ( first_time ) {
        seed = s;
        first_time = false;
    }
    seed = ( a * seed + c ) % m;
    return ( float ) seed / m;
}

float
uni_binned( const int64_t n_bins, const int64_t n_samples ) {
    float distr[n_samples];
    for ( int64_t i = 0; i < n_samples; ++i ) { distr[i] = 0; }

    float bins[n_bins];
    return 0.0;
}

int
main() {
    printf( "MSG:\n" );
    for ( int i = 0; i < 10; ++i ) {
        printf( "%lld\n", MSG_i64( 1, 16807, 0, 2147483647 ) );
    }

    putchar( '\n' );

    printf( "Uniform probability distribution:\n" );
    for ( int i = 0; i < 10; ++i ) {
        printf( "%.3f\n", MSG_f( 1, 16807, 0, 2147483647 ) );
    }

    putchar( '\n' );
}

