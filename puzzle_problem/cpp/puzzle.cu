#include <image_helpers.hpp>
#include "GeneticAlgorithm.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/swap.h>
#include <thrust/sequence.h>
#include <thrust/copy.h> 
#include <cuda/std/cassert>
__managed__ float** dissimilarity_measures_LR;
__managed__ float** dissimilarity_measures_TD;

__managed__ int rows,cols,len,piece_size;
vector<vector<vector<pair<int,float>>>> best_match_table;
#include <crossover.hpp>

template<>//特化染色体的初始化操作
__device__ void Chromosome<int>::initial(unsigned int tid){
    thrust::sequence(thrust::device,genes,genes+this->lengthOfGenes);
    curandState_t state;
    curand_init(clock(),tid,0,&state);
    for(unsigned int i=0; i<2*this->lengthOfGenes;i++){
        int id1 = curand(&state)%this->lengthOfGenes;
        int id2 = curand(&state)%this->lengthOfGenes;
        thrust::swap(genes[id1],genes[id2]);
    }
}

template<>//特化cpu串行染色体交叉操作
__host__ void Chromosome<int>::crossover(Chromosome<int>* mate){
    Crossover crossover_operator(this->genes, mate->genes);
    crossover_operator.run();
    int child[len];
    crossover_operator.child(child);
    for(int i=0;i<len;i++){
        this->genes[i]=child[i];
    }
}
template<>//特化GPU并行染色体的交叉操作
__device__ void Chromosome<int>::crossover(Chromosome *mate,unsigned int tid){
    //thrust::device_vector<int> childGenes(this->lengthOfGenes,-1);//初始化子代基因，每一元素代表一个位置，数值代表应该放置的碎片id

    // curandState_t state;
    // curand_init(clock(),tid,0,&state);//初始化随机数生成器
    
    // int initIdx = curand(&state)%lengthOfGenes;
    // for(int i=0;i<rows;i++){
    //     for(int j=0;j<cols;j++){
    //     }
    // }
    // for (unsigned int i = 0; i < this->lengthOfGenes; i++) {
    //     int idx=(position.row - _min_row) * cols + (position.col - _min_column);
    //     this->genes[idx] = ;
    // }
}
template<>//特化染色体的变异操作
__device__ void Chromosome<int>::mutation(){
    unsigned int tid= blockIdx.x * blockDim.x + threadIdx.x;//线程编号
    curandState_t state;
    curand_init(clock(),tid,0,&state);//初始化随机数生成器
    
    for (unsigned int i = 0; i < this->lengthOfGenes; i++) {
        int id1 = curand(&state)%this->lengthOfGenes;
        int id2 = curand(&state)%this->lengthOfGenes;
        thrust::swap(genes[id1],genes[id2]);
    }
}

template<>//特化cpu串行计算适应度
__host__ void Chromosome<int>::get_fitness(){
    float fitness_value = 0;
    // For each two adjacent pieces in rows 在同一行
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols-1;j++){
            fitness_value += dissimilarity_measures_LR[ genes[i*cols+j] ][ genes[i*cols+j+1] ];
        }
    }
    // For each two adjacent pieces in columns 在同一列
    for(int i=0;i<rows-1;i++){
        for(int j=0;j<cols;j++){
            fitness_value += dissimilarity_measures_TD[ genes[i*cols+j] ][ genes[(i+1)*cols+j] ];
        }
    }
    fitness = -fitness_value;
}
template<>//特化GPU并行计算适应度
__device__ void Chromosome<int>::get_fitness(unsigned int tid){
    float fitness_value = 0;
    // For each two adjacent pieces in rows 在同一行
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols-1;j++){
            fitness_value += dissimilarity_measures_LR[ genes[i*cols+j] ][ genes[i*cols+j+1] ];
        }
    }
    // For each two adjacent pieces in columns 在同一列
    for(int i=0;i<rows-1;i++){
        for(int j=0;j<cols;j++){
            fitness_value += dissimilarity_measures_TD[ genes[i*cols+j] ][ genes[(i+1)*cols+j] ];
        }
    }
    fitness = -fitness_value;
}

int main(int agrc, char** agrv){
    Mat img=imread("./puzzle.jpg");
    piece_size = 48;
    rows = img.rows / piece_size; 
    cols = img.cols / piece_size;
    len = rows*cols;
    // vector<vector<float>> dissimilarity_measures_LR(len,vector<float>(len,0));
    // vector<vector<float>> dissimilarity_measures_TD(len,vector<float>(len,0));
    cudaMallocManaged(&dissimilarity_measures_LR,sizeof(float*)*len);
    cudaMallocManaged(&dissimilarity_measures_TD,sizeof(float*)*len);
    for(int i=0;i<len;i++){
        cudaMallocManaged(&(dissimilarity_measures_LR[i]),sizeof(float)*len);
        memset(dissimilarity_measures_LR[i],0,sizeof(float)*len);
        cudaMallocManaged(&(dissimilarity_measures_TD[i]),sizeof(float)*len);
        memset(dissimilarity_measures_TD[i],0,sizeof(float)*len);
    }
    best_match_table.resize(4);
    for(int i=0;i<4;i++){
        best_match_table[i].resize(len);
    }

    vector<Mat> pieces = flatten_image(img,piece_size);
    analyze_image(len, pieces, dissimilarity_measures_LR, dissimilarity_measures_TD, best_match_table);
    for(int i=0;i<len;i++){//排序
        for(int j=0;j<4;j++){
            sort(best_match_table[j][i].begin(),best_match_table[j][i].end(),cmp_value);
        }
    }

    int genes1[len]={0};
    int genes2[len]={0};
    unsigned seed = clock();
    std::default_random_engine generator(seed);
    for(int i=0;i<len;i++){
        swap(genes1[generator()%len] , genes1[generator()%len]);
        swap(genes2[generator()%len] , genes2[generator()%len]);
    }
    
    //遗传算法
    Chromosome<int> best(len);
    printf("hello!Genetic Algorithm begin!\n");
    GA<int>(&best,1024,len,0.8,1,0,0,10,false);
    printf("MaxFitness=%f\n",best.fitness);

    Mat mergeImg = assemble_image(pieces,best.genes,rows,cols);
    imshow("originImg",img);
    imshow("mergeImg",mergeImg);
    waitKey(0);

    //测试交叉算子
    // Crossover crossover_operator(genes1,genes2);
    // crossover_operator.run();
    // int child[len];
    // crossover_operator.child(child);
    // for(int i=0;i<len;i++){
    //     printf("%d ",child[i]);
    // }
    // printf("\n");
    return 0;
}
