#include <sched.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <omp.h>
#include <stdio.h>

int main(int argn, char** argv)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        cpu_set_t cpuset;
        
        pthread_t thread = pthread_self();

        int s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        if (s != 0) {
            printf("error in pthread_getaffinity_np\n");
        }
        
        #pragma omp critical
        {
            printf("thread: %i\n", tid);
            for (int j = 0; j < CPU_SETSIZE; j++) {
                if (CPU_ISSET(j, &cpuset)) {
                    printf(" %i", j);
                }
            }
            printf("\n");
        }
    }
    return 0;
}
