// #pragma GCC optimize("Ofast", "unroll-loops")
#include "bfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <bitset>
#include <vector>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

int top_down_step(Graph g, solution *sol, int step, int numNodes, int *old_frontier, int frontier_size, int *new_frontier, int *m_f) {
    int n_f = 0;
    if(m_f != NULL)
        *m_f = 0;
    #pragma omp parallel for
    for(int i = 0; i < frontier_size; ++i) {
        int node = old_frontier[i];
        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1) 
                        ? g->num_edges 
                        : g->outgoing_starts[node + 1];
        for(int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
            int outgoing = g->outgoing_edges[neighbor];
            if(sol->distances[outgoing] == NOT_VISITED_MARKER) sol->distances[outgoing] = step;
        }
    }
    for(int i = 0; i < numNodes; ++i)
        if(sol->distances[i] == step) new_frontier[n_f++] = i;
    if(m_f != NULL) {
        for(int i = 0; i < n_f; ++i) 
            *m_f += outgoing_size(g, new_frontier[i]);
    }
    return n_f;
} 

int bottom_up_step(Graph g, solution *sol, int step, int numNodes, int *old_frontier, int frontier_size, int *new_frontier, int *m_f) {
    int n_f = 0, new_step = step + 1;
    if(m_f != NULL)
        *m_f = 0;
    #pragma omp parallel for
    for(int node = 0; node < numNodes; ++node) {
        if(sol->distances[node] == NOT_VISITED_MARKER) {
            int start_edge = g->incoming_starts[node];
            int end_edge = (node == g->num_nodes - 1) 
                            ? g->num_edges 
                            : g->incoming_starts[node + 1];
            for(int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
                int incoming = g->incoming_edges[neighbor];
                if(sol->distances[incoming] == step) {
                    sol->distances[node] = new_step; break;
                }
            }
        }    
    }
    for(int node = 0; node < numNodes; ++node) {
        if(sol->distances[node] == new_step) {
            new_frontier[n_f++] = node;
        } 
    }
    if(m_f != NULL) {
        for(int i = 0; i < n_f; ++i) 
            *m_f += outgoing_size(g, new_frontier[i]);
    }
    return n_f;
}

void bfs_top_down(Graph g, solution *sol) {
    
    #pragma omp parallel for
    for (int i = 0; i < g->num_nodes; ++i)
        sol->distances[i] = NOT_VISITED_MARKER;

    int count = 0, step = 0;
    int numNodes = num_nodes(g);
    int *new_frontier, *old_frontier, *frontier = new int[numNodes];
    
    frontier[count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    old_frontier = frontier; new_frontier = frontier + count;

    while(count > 0) {
        count = top_down_step(g, sol, ++step, numNodes, old_frontier, count, new_frontier, NULL);
        old_frontier = new_frontier; new_frontier += count;
    }

    delete frontier;
}

void bfs_bottom_up(Graph g, solution *sol) {

    #pragma omp parallel for
    for (int i = 0; i < g->num_nodes; ++i)
        sol->distances[i] = NOT_VISITED_MARKER;

    int not_done = 1;
    int numNodes = g->num_nodes;
    sol->distances[ROOT_NODE_ID] = 0;
    
    int count = numNodes;
    int *nodes = new int [numNodes];

    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) 
        nodes[i] = i; 

    for(int step = 0; not_done; step++) {

        not_done = 0;
        
        #pragma omp parallel for
        for(int i = 0; i < count; ++i) {
            int node = nodes[i];
            if(sol->distances[node] == NOT_VISITED_MARKER) {
                int start_edge = g->incoming_starts[node];
                int end_edge = (node == numNodes - 1) 
                                ? g->num_edges 
                                : g->incoming_starts[node + 1];
                for(int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
                    int incoming = g->incoming_edges[neighbor];
                    if(sol->distances[incoming] == step) {
                        sol->distances[node] = step + 1; 
                        #pragma omp atomic
                        not_done++;
                        break;
                    }
                }
            }
        }

        int new_count = 0;
        for(int i = 0; i < count; ++i) {
            if(sol->distances[nodes[i]] == NOT_VISITED_MARKER) {
                nodes[new_count++] = nodes[i];
            }
        }
        count = new_count;
    }
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    // delete nodes;
}

void bfs_hybrid(Graph g, solution *sol) {

    #pragma omp parallel for
    for (int i = 0; i < g->num_nodes; ++i)
        sol->distances[i] = NOT_VISITED_MARKER;

    int n_f = 0, step = 0;
    int numNodes = num_nodes(g);
    int *new_frontier, *old_frontier, *frontier = new int[numNodes];
    
    frontier[n_f++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
 
    old_frontier = frontier; new_frontier = frontier + n_f;

    // parameters for hybrid  bfs
    double alpha = 14.0, beta = 24.0;
    bool top_down = true; // start at top down

    int m_f = outgoing_size(g, ROOT_NODE_ID), m_u = g->num_edges - m_f; 
    int new_m_f, new_n_f, new_m_u;


    while(n_f > 0) {
        if(top_down) {
            new_n_f = top_down_step(g, sol, ++step, numNodes, old_frontier, n_f, new_frontier, &new_m_f);
            new_m_u = m_u - new_m_f;
        }
        else {
            new_n_f = bottom_up_step(g, sol, step++, numNodes, old_frontier, n_f, new_frontier, &new_m_f);
            new_m_u = m_u - new_m_f;
        }
        double C_TB = (double)new_m_u  / alpha;
        double C_BT = (double)numNodes / beta;
        if(top_down) {
            if(new_m_f > C_TB && new_n_f > n_f) 
                top_down = false;
        }
        else {
            if(n_f < C_BT && new_n_f < n_f)
                top_down = true;
        }
        n_f = new_n_f; m_f = new_m_f; m_u = new_m_u;
        old_frontier = new_frontier; new_frontier += n_f;
    }
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
