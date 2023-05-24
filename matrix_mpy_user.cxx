#include "matrix.hxx"

using namespace std;
#include <thread>

////////////////////////////////////////////////////////////////
// One thread, blocked. Loop order rB, kB, cB, r, k, c.
// This function is for you to write.
//
void Matrix::mpy1(const Matrix& A, const Matrix& B, int BS) {
    int NBLK = this->N() / BS;     // An NBLKxNBLK grid of blocks
    assert(this->N() >= BS);
    for (int rB = 0; rB < NBLK; rB++) {
        for (int kB = 0; kB < NBLK; kB++) {
            for (int cB = 0; cB < NBLK; cB++) {
                for (int r_offset = 0; r_offset < BS; r_offset++) {
                    for (int k_offset= 0; k_offset < BS; k_offset++) {
                        for (int c_offset = 0; c_offset < BS; c_offset++) {
                            
                           
                            int r = rB * BS + r_offset;
                            int k = kB * BS + k_offset;
                            int c = cB * BS + c_offset;
                            if (k == 0) {
                                this->data[this->index(r, c)] = A(r, k) * B(k, c);
                            }
                            else {
                                this->data[this->index(r, c)] += A(r, k) * B(k, c);
                            }
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////
// Multithreaded, blocked version.
//
// This function, th_func2(), does the per-thread work of multithreaded, blocked
// matrix multiplication.
static void th_func2( const Matrix& A, const Matrix& B, Matrix &C, int thread_Id, int NBLK, int num_of_threads, int BS) {
    int NBL = A.N() / BS;// A.size=B.size
    int num_of_block=NBL*NBL;
    int threadblock=num_of_block/num_of_threads;//how many blcoks one thread has
    
    int id=thread_Id;
    //int threadsize=A.N()/num_of_threads;//how many lines one thread contains matrix A= matrix B
    int lowbound=id*threadblock; // the initial block of nth thread
    for (int h=lowbound ; h<(lowbound+threadblock); h++) {//allocate the data to each thread   traverse  block
        
        int rB = id / NBLK;
        int cB = id % NBLK;
        for (int kB = 0; kB < NBLK; kB++) {
            for (int r_offset = 0; r_offset < BS; r_offset++) {
                for (int k_offset= 0; k_offset < BS; k_offset++) {
                    for (int c_offset = 0; c_offset < BS; c_offset++) {
                        int r = rB * BS + r_offset;
                        int k = kB * BS + k_offset;
                        int c = cB * BS + c_offset;
                        if (k == 0) {
                            C(r,c) = A(r, k) * B(k, c);
                        }
                        else {
                            C(r,c) += A(r, k) * B(k, c);
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////
// This function does multithreaded, blocked matrix multiplication. It is for
// you to write. The parameters:
//	A, B: the input matrices
//	BS: block size; i.e., you should use blocks of BSxBS.
//	n_procs: how many processors to use.
// You must store the output in (*this), which already has its .data array
// allocated (but not necessarily cleared).
// Note that you can find out the size of the A, B and (*this) matrices by
// either looking at the _N member variable, or calling Matrix.N().
void Matrix::mpy2(const Matrix& A, const Matrix& B, int BS, int n_threads) {
    int NBLK = this->N() / BS; // An NBLKxNBLK grid of blocks
    vector<thread> threads;
    for (int i = 0; i < n_threads; i++) {
        threads.push_back(thread(th_func2, ref(A), ref(B), ref(*this),i, NBLK, n_threads, BS));
    }
    for (auto& th : threads) {
        th.join();
    }
}
