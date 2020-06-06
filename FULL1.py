import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
elitesize = 4
generations = 10 
N=10
jobs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
population = []
nextGenerationsarri = []
time = []
parents = []
bestest = [401]
bestpopulation = [5]
#generations = 100
elitesize = 2
best = 1
def randomize(jobs):
    sequence = random.sample(jobs,len(jobs))
    return sequence
a = pd.read_excel (r'C:\Users\agiom\Desktop\Manufacturing Systems\Coursework\job.xlsx')

def timecalc(individual):
    time = []
    bob = individual
    #print('here:'+str(individual))
    for p in range (0,30):
         bob[p]=bob[p]-1
    #print('bob'+str(bob))
    df = pd.DataFrame(index=[0,1, 2 , 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],columns=[0,1, 2, 3, 4, 5, 6, 7])
    for i in range (0,30):
        df.iloc[i] = a.iloc[bob[i]]
    #print('a'+str(a))
    #print('bobi'+str(bob[i]))
    for p in range (0,30):
        individual[p]=individual[p]+1
    #print('dfn'+str(df))
    #print('here')
###############################################################################################
   # print(population)
###############################################################################################
    #print('df2'+str(df))
    #print('here2')
    align1 = pd.DataFrame(index=[1, 2 , 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],columns=[1, 2, 3, 4, 5, 6, 7])
    for n in range (0,7):
        align1.iloc[0,n] = 0
    for i in range (1,30):
        align1.iloc[i,0] = (4+(df.iloc[i,1]-df.iloc[i-1,1]))
    for i in range (1,30):
        align1.iloc[i,1] = (7+(df.iloc[i,2]-df.iloc[i-1,2]))
    for i in range (1,30):
        align1.iloc[i,2] = (2+(df.iloc[i,3]-df.iloc[i-1,3]))
    for i in range (1,30):
        align1.iloc[i,3] = (5+(df.iloc[i,4]-df.iloc[i-1,4]))
    for i in range (1,30):
        align1.iloc[i,4] = (8+(df.iloc[i,5]-df.iloc[i-1,5]))
    for i in range (1,30):
        align1.iloc[i,5] = (3+(df.iloc[i,6]-df.iloc[i-1,6]))
    for i in range (1,30):
        align1.iloc[i,6] = (6+(df.iloc[i,7]-df.iloc[i-1,7]))
    align1
###############################################################################################

###############################################################################################
    align2 = pd.DataFrame(index=[1, 2 , 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],columns=[1, 2, 3, 4, 5, 6, 7])
    for n in range (0,7):
        align2.iloc[0,n] = 0
    for j in range (0,7):
        for i in range (1,30):
                if(align1.iloc[i,j]<1):
                    align2.iloc[i,j] = align1.iloc[i,j]+8
                elif(align1.iloc[i,j]> 8):
                    align2.iloc[i,j] = align1.iloc[i,j]-8
                else:
                    align2.iloc[i,j] = align1.iloc[i,j]
    align2
###############################################################################################

###############################################################################################
    align3 = pd.DataFrame(index=[1, 2 , 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],columns=[1, 2, 3, 4, 5, 6, 7])
    for n in range (0,7):
        align3.iloc[0,n] = 0
    for i in range (1,30):
        align3.iloc[i,0] = abs(1-(align2.iloc[i,0]))
    for i in range (1,30):
        align3.iloc[i,1] = abs(4-(align2.iloc[i,1]))
    for i in range (1,30):
        align3.iloc[i,2] = abs(7-(align2.iloc[i,2]))
    for i in range (1,30):
        align3.iloc[i,3] = abs(2-(align2.iloc[i,3]))
    for i in range (1,30):
        align3.iloc[i,4] = abs(5-(align2.iloc[i,4]))
    for i in range (1,30):
        align3.iloc[i,5] = abs(8-(align2.iloc[i,5]))
    for i in range (1,30):
        align3.iloc[i,6] = abs(3-(align2.iloc[i,6]))
    align3
###############################################################################################

###############################################################################################
    align4 = pd.DataFrame(index=[1, 2 , 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],columns=[1, 2, 3, 4, 5, 6, 7,8])
    for m in range (0,30):
        align4.iloc[m,7] = 0
    for n in range (0,7):
        align4.iloc[0,n] = 0
    for j in range (0,7):
        for i in range (1,30):
                if(align3.iloc[i,j]>4):
                    align4.iloc[i,j] = abs(align3.iloc[i,j]-8)
                else:
                    align4.iloc[i,j] = align3.iloc[i,j]
    align4
###############################################################################################

###############################################################################################
    process = pd.DataFrame(index=[1, 2 , 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],columns=[1, 2, 3, 4, 5, 6, 7,8])
    for m in range (0,30):
        process.iloc[m,7] = 0
    for i in range (0,30):
        process.iloc[i,0] = align4.iloc[i,0]+1+ max(3,df.iloc[i,1])
    for j in range (0,7):
        process.iloc[0,j] = align4.iloc[0,j]+1+ max(3,df.iloc[0,j+1])
    for n in range (1,7):
        for m in range (1,30):
            process.iloc[m,n] =  max(3,df.iloc[m,n+1])
    process
###############################################################################################

###############################################################################################
    currentringoutput = pd.DataFrame(index=[1, 2 , 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],columns=[1, 2, 3, 4, 5, 6, 7,8])
    nextringinput = pd.DataFrame(index=[1, 2 , 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],columns=[1, 2, 3, 4, 5, 6, 7,8])
    wait = pd.DataFrame(index=[1, 2 , 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],columns=[1, 2, 3, 4, 5, 6, 7,8])
    for m in range (0,30):
        wait.iloc[m,7] = 0
    for m in range (0,30):
        nextringinput.iloc[m,7] = 0
    for m in range (0,30):
        currentringoutput.iloc[m,7] = 0

    #create first row of current ring output values
    currentringoutput.iloc[0,0] = process.iloc[0,0]
    for i in range (1,7):
        currentringoutput.iloc[0,i] = currentringoutput.iloc[0,i-1] + process.iloc[0,i]

    # create first row of next ring input values
    for j in range (0,6):
        nextringinput.iloc[0,j] = currentringoutput.iloc[0,j]+1+align4.iloc[0,j+1]
    nextringinput.iloc[0,6] = currentringoutput.iloc[0,6]+1

    #create  waits  
    #for i in range (1,2):
    #    currentringoutput.iloc[i,0] = nextringinput.iloc[i-1,0] + process.iloc[i,0]
    #
    #    if (align4.iloc[i,1] - (currentringoutput.iloc[i,0]-nextringinput.iloc[i-1,1])>0 ):
    #        wait.iloc[i,0] = align4.iloc[i,1] - (currentringoutput.iloc[i,0]-nextringinput.iloc[i-1,1])
    #    else :
    #        wait.iloc[i,0] = 0

    #lol
    for b in range (1,30):
        currentringoutput.iloc[b,0] = nextringinput.iloc[b-1,0] + process.iloc[b,0] 
    #    
        if (align4.iloc[b,1] - (currentringoutput.iloc[b,0]-nextringinput.iloc[b-1,1])>0 ):
            wait.iloc[b,0] = align4.iloc[b,1] - (currentringoutput.iloc[b,0]-nextringinput.iloc[b-1,1])
        else :
            wait.iloc[b,0] = 0
        for p in range (0,6):    

            if (nextringinput.iloc[b-1,p+1]>currentringoutput.iloc[b,p]):
                nextringinput.iloc[b,p]=nextringinput.iloc[b-1,p+1] + align4.iloc[b,p+1] + 1
            else:
                nextringinput.iloc[b,p]=currentringoutput.iloc[b,p] + wait.iloc[b,p] + 1
           # print(currentringoutput.iloc[b,p])
           # print(wait.iloc[b,p])
           # print(currentringoutput.iloc[b,p] + wait.iloc[b,p] + 1)
           # print("next:"+ str(nextringinput.iloc[b,p]))

            #print(process.iloc[b,p+1])
            currentringoutput.iloc[b,p+1] = nextringinput.iloc[b,p]+process.iloc[b,p+1]

           # print(currentringoutput.iloc[b,p+1])

            if (align4.iloc[b,p+2] - (currentringoutput.iloc[b,p+1]-nextringinput.iloc[b-1,p+2])>0 ):
                    wait.iloc[b,p+1] = align4.iloc[b,p+2] - (currentringoutput.iloc[b,p+1]-nextringinput.iloc[b-1,p+2])
            else :
                wait.iloc[b,p+1] = 0
        nextringinput.iloc[b,6]=currentringoutput.iloc[b,6]+1
    #print('next'+str(nextringinput))
    pp = nextringinput.iloc[29,6]
    return pp

def fitnesscalc(population,time): #Fitness is a list for us and it is the output of this function
    populationp = []
    fitness = []
    for i in range(0,N-1):
        populationp.append(population[i])
        #time_i = timecalc(populationp[i])
        fitness.append(round(1000/time[i],2))
    return fitness

def rank(fitness): # RANK is a list of lists, with the lists ranked based on fitness
    # need to run fitness above before this code block for correct result evertytime
    fitnesspap = fitness
    indexfitnesspap = []
    ranked = []
    for n in range(0,len(fitness)):
        k = max(fitness)
        #print(k)
        j =fitnesspap.index(k)
        #print(j)
        indexfitnesspap.append(j)
        #print(indexfitnesspapari)
        fitnesspap[j]=0
        #print(fitnesspapari)
    for voley in range (0,len(fitness)-1):
        #print('indexfitnesspapari'+str(indexfitnesspapari))
        #print('voley:'+str(voley))
        index_voley = indexfitnesspap[voley]
        ranked.append(population[index_voley])
    #for voley1 in range (0,9):   
    #    ranked.append(population[index_voley1])
    ranked.append(population[voley+1])
    return ranked

#MATING POOL : A list of lists, effectivelyh rank remixed rn 
def matingpool(ranked):
    parents = []
    for i in range(0,len(population)):#len(popRanked)):
        r = random.randint(1,len(population)-1)
       # dictlist = list(population.keys())
        parent_i = population[r]
        #parent_i = population[]
        parents.append(parent_i)
    return parents

#BREED: CHILD IS A SINGLE LIST made with cross over
def breed(parent1,parent2):
    child = []
    childP1 = []
    chiledP2 = []

    geneA = int(random.random()*len(parent1))
    geneB = int(random.random()*len(parent2))
    #print("length:"+str(len(parent1)))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    #print(startGene)
    #print(endGene)
    #print("sequence is:" + str(parent1))
    for i in range(startGene,endGene):
        #print("parent is:" + str(parent1[0][i]))
        #print("parent type is:")
        #print(type(parent1[0][i]))
        childP1.append(parent1[i])
        #print("child is:" + str(childP1))

    childP2 = [item for item in parent2 if item not in childP1]
    child = childP1 + childP2
    return child

#Breed Population: Children is now list of lists with 10 childs
def breedpopulation(parents):
    children = []
    length = len(parents)
    #print("parents:" + str(parents))
    pool = random.sample(parents,len(parents))
    for j in range(0,length):
#print("parent1:"+str(pool[j]))
#print("parent2:"+str(pool[len(parents)-j-1])
        child = breed(pool[j],pool[len(parents)-j-1])
        children.append(child)
#print("child:" + str(child))
#print("children:" + str(children))
    return children

## Define elite sequences that are kept as they are
def elitecalc(ranked): # Rank is a list of 4 lists
#elitesize = 4
    elites = []
    for l in range (0,elitesize):
        elites.append(ranked[l])
    return elites

#Mutate        
def mutate(individual, mutationRate):
    for swapped in range(0,30):
        if(random.random() < mutationRate):
            #print("individual:"+str(individual))
            #print("lenindividual:"+str(len(individual)))
            swapWith = int(random.random() * len(individual))
            
            bob1 = individual[swapped]
            bob2 = individual[swapWith]
            
            individual[swapped] = bob2
            individual[swapWith] = bob1
    return individual

  #MUTATE roll: MutatedPopulation is a list of 10 lists
def mutatePopulation(pop, mutationRate):
    mutatedPop=[]

    for ind in range(0, len(pop)):
        mutate(pop[ind],mutationRate)
        mutatedPop.append(pop[ind])
    return mutatedPop

def nextGeneration():
    nextGenerationsarri = []
    time = []
    parents = []
    for t in range (0,N-1):
        #print('t:'+str(t))
        #print('population(t):'+str(population[t]))
        #print('t:'+str(t))
        time_t = timecalc(population[t])
        time.append(time_t)
        best = min(time)
        if best < bestest[-1]:
            bestpopulation[0] = population[t]
    #print('out')
    if best < bestest[-1]:
        bestest.append(best)
##    print('time:'+str(time))
##    print('best:'+str(best))
    print('bestest:'+str(bestest))
##    print('bestpopulation:'+str(bestpopulation))
    fitness = fitnesscalc(population,time)
    #print('fitness:'+str(fitness))
    ranked = rank(fitness)
    #print('ranked:'+str(ranked))
    matingpools = matingpool(ranked)
    #print('matingpools:'+str(matingpools))
    children = breedpopulation(matingpools)
    #print('children:'+str(children))
    elites = elitecalc(ranked)
    #print('elites:'+str(elites))
    for kaka in range (0,len(elites)):
        nextGenerationsarri.append(elites[kaka])
    for kako in range (0, len(ranked)-len(elites)):
        nextGenerationsarri.append(children[kako])
    #print('nextGenerationsarri:'+str(nextGenerationsarri))
    mutatedpopulation = mutatePopulation(nextGenerationsarri,0.15)
    #print('mutatedpopulation:'+str(mutatedpopulation))
    return mutatedpopulation

if __name__ == "__main__":
    ##import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
    ##elitesize = 4
    ##generations = 10 
    ##N=10
    ##jobs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    ##population = []
    ##nextGenerationsarri = []
    ##time = []
    ##parents = []
    ################################################################################################################
    ####a = pd.read_excel (r'C:\Users\agiom\Desktop\Manufacturing Systems\Coursework\job.xlsx')
    ##def randomize(jobs):
    ##    sequence = random.sample(jobs,len(jobs))
    ##    return sequence
    ##sequence = randomize(jobs)
    ##sequence
    
    for i in range(N):
        jobs_N = randomize(jobs)
        population.append(jobs_N)
    #############################################################################################################
    for iterations in range (0,100):
        #print('iterations:'+str(iterations))
        #print('population from previous itr:'+ str(population))
        population = nextGeneration()
        nextGenerationsarri = []
        parents = []
##    print('best:'+str(best))
    print('bestest:'+str(bestest))
    print('bestpopulation:'+str(bestpopulation))
    plt.plot(bestest)
    plt.ylabel('End Time')
    plt.xlabel('Improvements')
    plt.show()
        
