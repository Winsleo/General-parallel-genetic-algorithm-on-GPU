#include "GeneticAlgorithm.hpp"

template<>//特化染色体的初始化操作
__device__ void Chromosome<float>::initial(){
    unsigned int tid= blockIdx.x * blockDim.x + threadIdx.x;//线程编号
    curandState_t state;
    curand_init(clock(),tid,0,&state);
    for(unsigned int i=0; i<this->lengthOfGenes;i++){
        this->genes[i]=curand(&state)%1000;
    }
}
template<>//特化染色体的交叉操作
__device__ void Chromosome<float>::crossover(Chromosome *c){
    for (unsigned int i = 0; i < this->lengthOfGenes; i++) {
        this->genes[i] = (this->genes[i] + c->genes[i]) / 2.0;
    }
}
template<>//特化染色体的变异操作
__device__ void Chromosome<float>::mutation(){
    unsigned int tid= blockIdx.x * blockDim.x + threadIdx.x;//线程编号
    curandState_t state;
    curand_init(clock(),tid,0,&state);//初始化随机数生成器
    for (unsigned int i = 0; i < this->lengthOfGenes; i++) {
        this->genes[i] = this->genes[i]+(curand_uniform(&state)-0.5)*100;
    }
}
//定义染色体的适应度函数

__global__ void get_fitness(Population<float>* p){
    unsigned int tid= blockIdx.x * blockDim.x + threadIdx.x;//线程编号
    if(tid<p->numOfIndiv){
         p->individuals[tid]->fitness=20-powf(p->individuals[tid]->genes[0],2)-powf(p->individuals[tid]->genes[1],2)+10*(cosf(2*M_PI*p->individuals[tid]->genes[0])+cosf(2*M_PI*p->individuals[tid]->genes[1]));
    }
}
template<>//特化评估函数,评估种群中每个个体的适应度
void pop_evaluate(Population<float> *p,unsigned int numIndiv){
    get_fitness<<<(numIndiv+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(p);
}
int main(int argc, char const *argv[])
{
    Chromosome<float> best(2);
    printf("hello!Genetic Algorithm begin!\n");
    GA<float>(&best,1024,2,0.8,0.5,0.5,38,200);
    printf("Best genes are\t%f %f\tMaxFitness=%f\n",best.genes[0],best.genes[1],best.fitness);
    
    return 0;
}
