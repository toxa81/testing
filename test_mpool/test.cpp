#include "include/timer.hpp"
#include "include/stream_id.hpp"
#include "new/memory.hpp"
//#include "old/mdarray.hpp"

// gcc-mp-7 -std=c++11  -O3 -I./include test.cpp  -lstdc++

using namespace sddk;

double test1()
{
    std::vector<mdarray<double, 1>> v;
    for (int i = 0; i < 20; i++) {
        v.push_back(std::move(mdarray<double, 1>(1 << 20)));
        v.back().zero();
    }
    return v[0](0);
}

int main(int argn, char** argv)
{
    memory_pool mp(memory_t::host);

    double t{0};
    utils::timer t1("test");
    for (int k = 0; k < (1 << 9); k++) {
        t += test1();
    }
    t += t1.stop();
    std::cout << "time : " << t << "\n";

    return 0;
}
