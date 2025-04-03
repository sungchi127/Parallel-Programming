#pragma GCC optimize("Ofast", "unroll-loops")
#include <smmintrin.h>
#include <immintrin.h>
#include <bits/stdc++.h>
#include <sys/time.h>
#include <pthread.h>
using namespace std;

struct Job {
        int n;
        uint32_t seed;
        long long result;
}; 

uint32_t rand(uint32_t &pre) {
        return (((pre = pre * 214013L + 2531011L)));
}

void *thread_job(void *job) {

        Job *data = (Job*)job;
        uint32_t seed = data -> seed;
        long long res = 0;
        int n = data -> n;

        for(int i = 0; i < n; ++i) {
                uint32_t R = rand(seed); 
                double x = (double)(R & 0x7fff) / 32767.0;
                double y = (double)((R >> 16) & 0x7fff)  / 32767.0;
                if(x * x + y * y < 1) res++;
        }

        data -> result = res;

        pthread_exit(NULL);
}

signed main(int argc, char **argv) {

        int thread_count = atoi(argv[1]);
        int N = atoi(argv[2]);

        int thread_tosses = N / thread_count;

        vector<pthread_t> threads(thread_count);
        vector<Job> jobs(thread_count);

        mt19937 seed_generator(time(nullptr));

        for(int i = 0; i < thread_count; ++i) {
                jobs[i].seed = seed_generator();
                jobs[i].n = (i == thread_count - 1) ? (N - i * thread_tosses) : thread_tosses;
        }

        for(int i = 0; i < thread_count; ++i) {
                pthread_create(&threads[i], NULL, thread_job, &jobs[i]);
        }

        long long sum = 0;
        for(int i = 0; i < thread_count; ++i) {
                pthread_join(threads[i], NULL);
                sum += jobs[i].result;
        }

        cout << setprecision(100) << (double)(4LL * sum) / N << endl;

        return 0;
}