#include <math.h>
#include <stdio.h>

int
main() {
    // this is a simplistic implementation, dumping everything in main
    // (its a very small problem anyway, so that is forgivable)

    int    N = 101;
    double I[N]; // using array because we want to store intermediate values too
    double s[N];
    double s_max = 100.;
    double ds = s_max / ( N - 1 );
    double alpha, S;
    double eta_1 = 3.;
    double eta_2 = 2.;

    double tau[N]; // an extra, cumulative optical depth
    double dtau;

    // first two iterations manually

    // no incoming light at s = 0. Set this first entry as the boundary
    // condition
    I[0] = 0.;
    s[0] = 0.;
    tau[0] = 0;

    I[1] = pow( s[1] / s_max, eta_2 - eta_1 );
    tau[1] = 0;
    s[1] = ds;

    // already set alpha and S for use in the next iteration
    alpha = pow( s[1] / s_max, eta_1 );
    S = pow( s[1] / s_max, eta_2 - eta_1 );

    for ( int i_s = 1; i_s < N; i_s++ ) {
        s[i_s] = s[i_s - 1] + ds;

        dtau = alpha * ds;
        // set cumulative optical depth
        tau[i_s] = tau[i_s - 1] + dtau;

        // this is equation (3), the part depending on the previous iteration
        I[i_s] = I[i_s - 1] * exp( -dtau )
                 + S * ( ( 1. - exp( -dtau ) ) / dtau - exp( -dtau ) );

        // again, set alpha and S based on current position
        alpha = pow( s[i_s] / s_max, eta_1 );
        S = pow( s[i_s] / s_max, eta_2 - eta_1 );

        // finish implementing equation (3)
        I[i_s] += S * ( 1. - ( 1. - exp( -dtau ) ) / dtau );

        // temp: ignore radiation generated deeper inside
        // if (i_s < 60) I[i_s] = 0.;
    }

    for ( int i_s = 0; i_s < N; i_s++ )
        printf( "%e, %e, %e\n", s[i_s], I[i_s], tau[N - 1] - tau[i_s] );

    return 0;
}
