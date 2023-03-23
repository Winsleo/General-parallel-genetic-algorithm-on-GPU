#ifndef GENETIC_ALGORITHM
#define GENETIC_ALGORITHM
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <stdlib.h>
#include <random>
#include "error.cuh"
#define BLOCK_SIZE 256
const int  _Neg_Inf=0xFF800000;
#define FLOAT_NEG_INF (*((float *)&(_Neg_Inf))) //float型的负无穷

template <typename T>
class Chromosome{//染色体模板类
public:
    //variables
    T* genes;//基因指针
    unsigned int lengthOfGenes;//基因个数
    float fitness;//适应度

    //functions
    __device__ void initial(unsigned int tid);//初始化函数
    __host__ void crossover(Chromosome<T>* mate);//交叉函数
    __device__ void crossover(Chromosome<T>* mate,unsigned int tid);//交叉函数
    __device__ void mutation();//变异函数
    __host__ void get_fitness();//计算适应度
    __device__ void get_fitness(unsigned int tid);//计算适应度
    __host__ __device__ Chromosome(){//无参构造函数
        genes=NULL;
        lengthOfGenes=0;
        fitness=FLOAT_NEG_INF;
    }
    __host__  Chromosome(unsigned int length_gen){//构造函数
        lengthOfGenes=length_gen;
        cudaMallocManaged(&genes,sizeof(T)*lengthOfGenes);
        memset(genes,0,sizeof(T)*lengthOfGenes);
        fitness=FLOAT_NEG_INF;
    }
    __host__  Chromosome(Chromosome<T> &c){//复制构造函数
        lengthOfGenes=c.lengthOfGenes;
        cudaMallocManaged(&genes,sizeof(T)*lengthOfGenes);
        fitness=c.fitness;
        for(unsigned int i=0;i<lengthOfGenes;i++){
            genes[i]=c.genes[i];
        }
    }
    __host__  void setup(unsigned int length_gen)//初始化函数
    {
        lengthOfGenes=length_gen;
        cudaMallocManaged(&genes,sizeof(T)*lengthOfGenes);
        memset(genes,0,sizeof(T)*lengthOfGenes);
        fitness=FLOAT_NEG_INF;
    }
    __host__  ~Chromosome()//析构函数
    {
        cudaFree(this->genes);
    }

    __host__ __device__ void operator=(Chromosome<T> &c){
        if(this->lengthOfGenes==c.lengthOfGenes){
            this->fitness=c.fitness;
            for (int i = 0; i < this->lengthOfGenes; i++)
            {
                this->genes[i]=c.genes[i];
            }
        }
    }
};
template <typename T>
__host__ __device__ bool cmp_fitness( Chromosome<T>* lhs, Chromosome<T>* rhs){
    return lhs->fitness < rhs->fitness;//适应度小的在前
}

template <typename T>
class Population{//种群模板类
public: 
    //variables
    Chromosome<T> **individuals;//种群个体
    Chromosome<T> **sons;//子代
    Chromosome<T> *maxFitnessIdv;//最大适应度个体
    unsigned int numOfIndiv;//种群个体数目
    unsigned int numOfGenes;//个体基因数
    float tournamentThre;//锦标赛选择的概率阈值
    float matRate;//交叉比例
    float mutRate;//变异比例
    //functions
    __host__ __device__ Population(){//无参构造函数
        numOfIndiv=0;
        numOfGenes=0;
        individuals =NULL;
        sons =NULL;
        tournamentThre=0;
        matRate=0;
        mutRate=0;
        maxFitnessIdv=NULL;
    }
    __host__ void setup(unsigned int numIndiv,unsigned int numGenes,float tourThre,float matrate,float mutrate){//初始化函数
        numOfIndiv=numIndiv;
        numOfGenes=numGenes;
        cudaMallocManaged(&individuals,sizeof(Chromosome<T>*)*numOfIndiv);
        cudaMallocManaged(&sons,sizeof(Chromosome<T>*)*numOfIndiv);
        cudaMallocManaged(&maxFitnessIdv,sizeof(Chromosome<T>));

        tournamentThre=tourThre;
        matRate=matrate;
        mutRate=mutrate;
        maxFitnessIdv->setup(numOfGenes);
        for(unsigned int i=0;i<numOfIndiv;i++)
        {
            cudaMallocManaged(&(individuals[i]),sizeof(Chromosome<T>));
            individuals[i]->setup(numOfGenes);
            cudaMallocManaged(&(sons[i]),sizeof(Chromosome<T>));
            sons[i]->setup(numOfGenes);
        }
    }

    __host__ ~Population(){//析构函数
        for(unsigned int i=0;i<this->numOfIndiv;i++)
        {
            cudaFree(this->individuals[i]);
        }
        cudaFree(this->individuals);
        cudaFree(this->sons);
        cudaFree(this->fitness);
    }    
    __device__ void initial(unsigned int tid){
        if(tid<this->numOfIndiv){
            this->individuals[tid]->initial(tid);                            
        }
        __threadfence();//对于全局内存的读写同步
    }
};

template<class T>//种群随机初始化
__global__ void pop_init(Population<T> *p){
    unsigned int tid= blockIdx.x * blockDim.x + threadIdx.x;//线程编号,*2为个体编号 
    p->initial(tid);
    if(tid==0){
        *(p->maxFitnessIdv)=*((p->individuals)[tid]);
    }
}
template<class T>//种群选择
__global__ void pop_select(Population<T> *p){
    unsigned int tid= blockIdx.x * blockDim.x + threadIdx.x;//线程编号
    
    curandState_t state;
    curand_init(clock(),tid,0,&state);//初始化随机数生成器
    if(tid<p->numOfIndiv){
        Chromosome<T>* choosen[2];
        //rank_selection选择
        // for(int i=0;i<2;i++){
        //     float randNum = fabs(curand_normal(&state))/2;
        //     if( randNum <1 ){//按照rank selection选择
        //         int idx=floor(randNum*p->numOfIndiv);
        //         if(idx>=p->numOfIndiv) idx=p->numOfIndiv-1;
        //         choosen[i]=p->individuals[idx];
        //     }
        //     else{//随机选
        //         choosen[i]=p->individuals[curand(&state)%(p->numOfIndiv)];
        //     }
        // }
        //锦标赛选择
        choosen[0]=p->individuals[curand(&state)%p->numOfIndiv];//随机选
        choosen[1]=p->individuals[curand(&state)%p->numOfIndiv];//随机选
        if(choosen[0]->fitness>choosen[1]->fitness){
            if(curand_uniform(&state)<p->tournamentThre){
                *(p->sons[tid])=*(choosen[0]);
            }
            else{
                *(p->sons[tid])=*(choosen[1]);
            }
        }
        else{
            if(curand_uniform(&state)>p->tournamentThre){
                *(p->sons[tid])=*(choosen[0]);
            }
            else{
                *(p->sons[tid])=*(choosen[1]);
            }
        }

    }
}

template<class T>//并行交叉操作
__global__ void para_crossover(Population<T> *p){
    unsigned int tid= blockIdx.x * blockDim.x + threadIdx.x;//线程编号
    
    curandState_t state;
    curand_init(clock(),tid,0,&state);//初始化随机数生成器

    if(tid<(p->matRate*p->numOfIndiv) && tid<p->numOfIndiv){//交叉
        Chromosome<T>* choosen[2];
        //rank_selection选择
        // for(int i=0;i<2;i++){
        //     float randNum = fabs(curand_normal(&state))/2;
        //     if( randNum <1 ){//按照rank selection选择
        //         int idx=floor(randNum*p->numOfIndiv);
        //         if(idx>=p->numOfIndiv) idx=p->numOfIndiv-1;
        //         choosen[i]=p->individuals[idx];
        //     }
        //     else{//随机选
        //         choosen[i]=p->individuals[curand(&state)%(p->numOfIndiv)];
        //     }
        // }
        //锦标赛选择      
        choosen[0]=p->individuals[curand(&state)%p->numOfIndiv];//随机选
        choosen[1]=p->individuals[curand(&state)%p->numOfIndiv];//随机选
        if(choosen[0]->fitness>choosen[1]->fitness){
            if(curand_uniform(&state)<p->tournamentThre){
                p->sons[tid]->crossover(choosen[0],tid);
            }
            else{
                p->sons[tid]->crossover(choosen[1],tid);
            }
        }
        else{
            if(curand_uniform(&state)>p->tournamentThre){
                p->sons[tid]->crossover(choosen[0],tid);
            }
            else{
                p->sons[tid]->crossover(choosen[1],tid);
            }
        }
    }
}

template<class T>//串行交叉操作
__host__ void crossover(Population<T>* p){
    unsigned int mateNum = floor(p->matRate*p->numOfIndiv);
    unsigned seed = clock();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> gaussDistr(0, 1);
    std::normal_distribution<float> uniDistr(0, 1);
    
    
    for(int k=0;k<mateNum;k++){//交叉
        Chromosome<T>* choosen[2];
        //rank_selection选择
        // for(int i=0;i<2;i++){
        //     float randNum=fabs( gaussDistr(generator) )/2;
        //     if( randNum <1 ){//按照rank selection选择
        //         int idx=floor(randNum*p->numOfIndiv);
        //         if(idx>=p->numOfIndiv) idx=p->numOfIndiv-1;
        //         choosen[i]=p->individuals[idx];
        //     }
        //     else{//随机选
        //         choosen[i]=p->individuals[generator()%(p->numOfIndiv)];
        //     }
        // }
        //锦标赛选择
        choosen[0]=p->individuals[generator()%p->numOfIndiv];//随机选
        choosen[1]=p->individuals[generator()%p->numOfIndiv];//随机选
        if(choosen[0]->fitness>choosen[1]->fitness){
            if(uniDistr(generator)<p->tournamentThre){
                p->sons[k]->crossover(choosen[0]);
            }
            else{
                p->sons[k]->crossover(choosen[1]);
            }
        }
        else{
            if(uniDistr(generator)>p->tournamentThre){
                p->sons[k]->crossover(choosen[0]);
            }
            else{
                p->sons[k]->crossover(choosen[1]);
            }
        }
    }
}

template<class T>//种群交叉
void pop_crossover(Population<T> *p,bool parallel){
    if(parallel){//cuda并行
        unsigned int mateNum = floor(p->matRate*p->numOfIndiv);
        para_crossover<<<(mateNum+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(p);
    }
    else {//cpu串行
        crossover(p);
    }
    
}

template<class T>//种群变异
__global__ void pop_mutate(Population<T> *p){
    unsigned int tid= blockIdx.x * blockDim.x + threadIdx.x;//线程编号
    curandState_t state;
    curand_init(clock(),tid,0,&state);//初始化随机数生成器
    if(tid < p->numOfIndiv){
        if(curand_uniform(&state)<p->mutRate){//变异
            p->sons[tid]->mutation();
        }
        *(p->individuals[tid])=*(p->sons[tid]);
    }
}

template<class T>//并行计算适应度
__global__ void para_get_fitness(Population<T>* p){
    unsigned int tid= blockIdx.x * blockDim.x + threadIdx.x;//线程编号
    if(tid<p->numOfIndiv){
        p->individuals[tid]->get_fitness(tid);
    }
}

template<class T>//串行计算适应度
__host__ void get_fitness(Population<T>* p){
    for(int i=0; i < p->numOfIndiv; i++){
        p->individuals[i]->get_fitness();
    }
}

template<class T>//评估种群中每个个体的适应度函数声明
void pop_evaluate(Population<T> *p,bool parallel){
    if(parallel){//cuda并行
        para_get_fitness<<<(p->numOfIndiv+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(p);
    }  
    else {//cpu串行
        get_fitness(p);
    }
}

template<class T>//寻找当前种群最优个体，并更新历史最优
__global__ void find_best(Population<T> *p, float* max){
    __shared__ float mTop[BLOCK_SIZE];
    __shared__ unsigned int mIdx[BLOCK_SIZE];
    unsigned int tid=threadIdx.x + blockDim.x * blockIdx.x;
    float top = p->individuals[tid]->fitness;
    unsigned int topIdx=tid;
    
    for(unsigned int idx = tid; idx<p->numOfIndiv; idx+= gridDim.x*blockDim.x)
    {
        if(p->individuals[idx]->fitness>top)
        {
            top = p->individuals[idx]->fitness ;
            topIdx=idx;
        }
    }
    mTop[threadIdx.x]=top;
    mIdx[threadIdx.x]=topIdx;
    __syncthreads();
    
    for (unsigned int i = BLOCK_SIZE / 2; i; i /=2 )
    {
        if(threadIdx.x < i)
        {
            if(mTop[threadIdx.x]<mTop[threadIdx.x+i])
            {
                mTop[threadIdx.x] = mTop[threadIdx.x+i];
                mIdx[threadIdx.x] = mIdx[threadIdx.x+i];
            }
        }
        __syncthreads();//线程块内同步
    }
    if(blockIdx.x*blockDim.x < p->numOfIndiv)
    {
        if(tid == 0)
        {
            if(p->maxFitnessIdv->fitness<mTop[0]){
                *(p->maxFitnessIdv)=*(p->individuals[mIdx[0]]);  
            }
            max[blockIdx.x] = mTop[0];
        }
    }
}

template<class T>
__host__ void GA(
    Chromosome<T>*best,
    unsigned int numIndiv,
    unsigned int numGenes,
    float tourThreshold,
    float matrate,
    float mutrate,
    float fitnessThreshold,
    unsigned long maxIteration,
    bool crossoverParallel=true,
    bool evaluateParallel=true)
{  
    Population<T> *Pop;//指针声明
    float *devMax;//指针声明
    CHECK(cudaMallocManaged(&devMax, sizeof(float)));//申请设备内存
    CHECK(cudaMallocManaged(&Pop, sizeof(Population<T>)));//申请设备内存
    Pop->setup(numIndiv,numGenes,tourThreshold,matrate,mutrate);
    
    cudaEvent_t start, stop;//声明事件,用来记录时间
    CHECK(cudaEventCreate(&start));//事件创建
    CHECK(cudaEventCreate(&stop));//事件创建
    CHECK(cudaEventRecord(start));//记录开始的时间

    pop_init<T><<<(numIndiv+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(Pop);//随机初始化种群
    CHECK(cudaDeviceSynchronize());//CPU与GPU同步
    pop_evaluate<T>(Pop,evaluateParallel);//评估整个种群每个个体的适应度
    CHECK(cudaDeviceSynchronize());//CPU与GPU同步
    find_best<T><<<1,BLOCK_SIZE>>>(Pop,devMax);//寻找最优个体
    CHECK(cudaDeviceSynchronize());//CPU与GPU同步

    for(unsigned int it=0;it<maxIteration && *devMax<fitnessThreshold;it++){
        //种群选择
        pop_select<T><<<(numIndiv+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(Pop);
        CHECK(cudaDeviceSynchronize());//CPU与GPU同步
        //种群交叉
        pop_crossover<T>(Pop,crossoverParallel);
        CHECK(cudaDeviceSynchronize());//CPU与GPU同步
        //种群变异
        pop_mutate<T><<<(numIndiv+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(Pop);
        CHECK(cudaDeviceSynchronize());//CPU与GPU同步
        //评估整个种群每个个体的适应度
        pop_evaluate<T>(Pop,evaluateParallel);
        CHECK(cudaDeviceSynchronize());//CPU与GPU同步
        find_best<T><<<1,BLOCK_SIZE>>>(Pop,devMax);//寻找最优个体,如果是历史最优则更新
        CHECK(cudaDeviceSynchronize());//CPU与GPU同步
        printf("The max fitness of %uth generation is %f\n",it,*devMax);
    }
    //记录算法结束时间
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));//记录算法消耗的时间
    printf("Genetic algorithm spends time of %f ms.\n", elapsed_time);
    
    *best=*(Pop->maxFitnessIdv);//将最优个体提取出来

    CHECK(cudaEventDestroy(start));//销毁事件
    CHECK(cudaEventDestroy(stop));//销毁事件
    CHECK(cudaFree(devMax));//释放设备内存
    CHECK(cudaFree(Pop));//释放设备内存
}
#endif

