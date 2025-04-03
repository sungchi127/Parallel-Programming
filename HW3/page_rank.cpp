#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"
using namespace std;

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  const int MaxThreads = omp_get_max_threads();

  vector<int> no_out;
  for (int i = 0; i < numNodes; i++) {
        if(!outgoing_size(g, i)) {
            no_out.emplace_back(i);
        }
  }

  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i) {
      solution[i] = equal_prob;
  }
  
  bool converged = false;
  double *temp = new double [numNodes];

  while (!converged) {

      #pragma omp parallel for
      for (int i = 0; i < numNodes; i++) 
          temp[i] = 0.0;
    
      const int block_size = max(1, min(numNodes / 10, 10000));
      auto process = [&](int i) -> void {
          int l = i, r = min(numNodes, i + block_size);
          for(int i = l; i < r; i++) {
              const Vertex *start = incoming_begin(g, i);
              const Vertex *end = incoming_end(g, i);
              for(const Vertex *v = start; v != end; v++) {
                  int j = *v;
                  temp[i] += solution[j] / (double)outgoing_size(g, j);
              }
          }
      };

      #pragma omp parallel 
      {
          #pragma omp single 
          {
              for (int i = 0; i < numNodes; i += block_size) {
                  #pragma omp task firstprivate(i)
                  process(i);
              }
          }
      }


      #pragma omp parallel for
      for (int i = 0; i <  numNodes; i++) {
          temp[i] = (damping * temp[i]) + (1.0 - damping) / (double)numNodes;
      }

      double sum = 0.0;

      double ThreadSum[MaxThreads];
      for (int i = 0; i < MaxThreads; i++)
          ThreadSum[i] = 0.0;

      
      #pragma omp parallel
      {
          int id, nthrds;
          id = omp_get_thread_num();
          nthrds = omp_get_num_threads();
          int N = no_out.size();
          int L = id * (N / nthrds);
          int R = (id == nthrds - 1) ? N : (L + N / nthrds);
          for (int i = L; i < R; i++) {
              if (outgoing_size(g, no_out[i]) == 0) {
                  ThreadSum[id] += damping * solution[no_out[i]] / (double)numNodes;
              }
          }
      }

      for (int i = 0; i < MaxThreads; i++)
          sum += ThreadSum[i];

      #pragma omp parallel for 
      for (int i = 0; i < numNodes; i++) {
          temp[i] += sum;
      }

      double global_diff = 0.0;

      for (int i = 0; i < MaxThreads; i++)
          ThreadSum[i] = 0.0;
      
      #pragma omp parallel
      {
          int id, nthrds;
          id = omp_get_thread_num();
          nthrds = omp_get_num_threads();
          int L = id * (numNodes / nthrds);
          int R = (id == nthrds - 1) ? numNodes : (L + numNodes / nthrds);
          for (int i = L; i < R; i++) {
              ThreadSum[id] += abs(temp[i] - solution[i]);
              solution[i] = temp[i];
          }
      }

      for (int i = 0; i < MaxThreads; i++)
          global_diff += ThreadSum[i];

      converged = (global_diff < convergence);
  }
  
  delete temp;
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
