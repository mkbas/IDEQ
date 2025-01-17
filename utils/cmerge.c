#include <stdio.h>
#include <stdlib.h>
#include <math.h>


//double * global_dists;

int find_begin(int *route_begin, int i) {
    int begin_i = route_begin[i];
    if (begin_i !=i){
        begin_i = find_begin(route_begin, begin_i);
        route_begin[i] = begin_i;
    }
    return begin_i ;
}

int find_end(int *route_end, int i) {
    int end_i = route_end[i];
    if (end_i !=i){
        end_i = find_end(route_end, end_i);
        route_end[i] = end_i;
    }
    return end_i;
}


int merge(int *sorted_edges,int N, int *A) {
    int i, j, edge, begin_i, end_i, begin_j, end_j;
    int *route_begin = (int *)malloc(N * sizeof(int));
    int *route_end = (int *)malloc(N * sizeof(int));
    int merge_count = 0;
    int  merge_iterations =0;
    // Initialize route_begin and route_end
    for (i = 0; i < N; i++) {
        route_begin[i] = i;
        route_end[i] = i;
    }
    for (int k = 0; k < N * N; k++) {
        edge = (int) sorted_edges[k];
        merge_iterations++;
        i = edge / N;
        j = edge % N;
        
        begin_i = find_begin(route_begin, i);
        end_i = find_end(route_end, i);
        begin_j=find_begin(route_begin, j);
        end_j=find_end(route_end, j);

        if (begin_i == begin_j) continue;
        if (i != begin_i && i != end_i) continue;
        if (j != begin_j && j != end_j) continue;

        A[j * N + i] = 1;
        A[i * N + j] = 1;
        merge_count++;

        if (i == begin_i && j == end_j) {
            route_begin[begin_i] = begin_j;
            route_end[end_j] = end_i;
        } else if (i == end_i && j == begin_j) {
            route_begin[begin_j] = begin_i;
            route_end[end_i] = end_j;
        } else if (i == begin_i && j == begin_j) {
            route_begin[begin_i] = end_j;
            route_begin[begin_j] = end_j;
            route_begin[end_j] = end_j;
            route_end[end_j] = end_i;
            route_end[begin_j] = end_i;
        } else if (i == end_i && j == end_j) {
            route_end[end_i] = begin_j;
            route_begin[begin_j] = begin_i;
            route_begin[end_j] = begin_i;
            route_end[end_j] = begin_j;
            route_end[begin_j] = begin_j;
        }
        if (merge_count == N - 1) break;
    }
    begin_i=find_begin(route_begin, 0);
    end_i=find_end(route_end, 0);
    A[end_i * N + begin_i] = 1;
    A[begin_i * N + end_i] = 1;
 
    free(route_begin);
    free(route_end);
    return merge_iterations;
}
