#include <sys/time.h>
#include <cstdlib>
#include <numeric>
#include <vector>
#include <algorithm>
#include <iostream>

inline double wtime()
{
    timeval t;
    gettimeofday(&t, NULL);
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

inline double rnd()
{
    return double(std::rand()) / RAND_MAX;
}

int main(int argn, char** argv)
{
    int n{10000000};
    double t = -wtime();
    std::vector<double> v(n);
    std::generate(v.begin(), v.end(), []{return rnd();});
    std::sort(v.begin(), v.end());
    t += wtime();
    std::cout << "time: " << t << std::endl;
    return 0;
}
