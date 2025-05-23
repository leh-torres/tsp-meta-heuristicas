import numpy as np
import random

class AlgoritmoGenetico:
    def __init__(self, grafo, tamanho=100, geracoes=500, taxa_de_mutacao=0.01):
        self.grafo = grafo
        self.tamanho_pop = tamanho
        self.geracoes = geracoes
        self.taxa_de_mutacao = taxa_de_mutacao

    def iniciar(self):
        cidades = list(self.grafo.keys())
        populacao = [list(np.random.permutation(cidades)) for _ in range(self.tamanho_pop)]
        
        for i in range(self.geracoes):

            custos = list(self.__custo_da_rota(p, self.grafo) for p in populacao)
            nova_pop = []

            for _ in range(self.tamanho_pop):
                p1 = self.__selecionar_pais(populacao, custos)
                p2 = self.__selecionar_pais(populacao, custos)

                filho = self.__crossover(p1,p2)
                filho = self.__mutacao(filho, self.taxa_de_mutacao)
                nova_pop.append(filho)

        custos = [self.__custo_da_rota(ind, self.grafo) for ind in populacao]
        melhor_ind = np.min(custos)
        print(f'Melhor caminho: {populacao[melhor_ind]} | Custo: {custos[melhor_ind]}')
        return populacao[melhor_ind], custos[melhor_ind], custos
    
    def iniciar_continuo(self, dim=5, intervalo=(-500, 500)):
        # Gera população contínua com valores aleatórios entre -500 e 500
        populacao = [np.random.uniform(intervalo[0], intervalo[1], size=dim) for _ in range(self.tamanho_pop)]
        melhores = []

        for _ in range(self.geracoes):
            custos = [self.__schwefel(ind) for ind in populacao]
            nova_pop = []

            for _ in range(self.tamanho_pop):
                p1 = self.__selecionar_pais(populacao, custos)
                p2 = self.__selecionar_pais(populacao, custos)

                filho = self.__crossover_continuo(p1, p2)
                filho = self.__mutacao_continua(filho, self.taxa_de_mutacao, intervalo)
                nova_pop.append(filho)

            populacao = nova_pop
            melhores.append(np.min(custos))

        custos = [self.__schwefel(ind) for ind in populacao]
        melhor_idx = np.argmin(custos)
        melhor_solucao = populacao[melhor_idx]
        melhor_custo = custos[melhor_idx]
        print(f'Melhor vetor: {melhor_solucao} | Custo: {melhor_custo}')
        return melhor_solucao, melhor_custo, melhores


    def __custo_da_rota(self, caminho, grafo):
        custo = 0
        for i in range(len(caminho)):
            origem = caminho[i]
            destino = caminho[(i+1)%len(caminho)]

            for vizinho, peso in grafo[origem].items():
                if vizinho == destino:
                    custo += peso
            
        return custo
    
    def __selecionar_pais(self, populacao, custos):
        candidatos = random.sample(list(enumerate(custos)), 5)
        return populacao[min(candidatos, key=lambda x:x[1])[0]]
    
    def __crossover(self, p1, p2):
        len_pais = len(p1)
        a, b = sorted(random.sample(range(len_pais), 2))

        filho = [None] * len_pais
        filho[a:b] = p1[a:b]
        pos = b

        for p in p2:
            if p not in filho:
                if pos == len_pais:
                    pos = 0
                filho[pos] = p
                pos += 1

        return filho

    def __mutacao(self, filho, taxa):
        novo_filho = filho[:]

        if random.random() < taxa:
            i, j = random.sample(range(len(filho)), 2)
            novo_filho[i], novo_filho[j] = novo_filho[j], novo_filho[i]

        return novo_filho
    
    def __schwefel(self, x):
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    def __crossover_continuo(self, p1, p2):
        alpha = np.random.rand()
        return alpha * np.array(p1) + (1 - alpha) * np.array(p2)

    def __mutacao_continua(self, individuo, taxa, intervalo):
        novo = np.array(individuo)
        if np.random.rand() < taxa:
            i = np.random.randint(len(novo))
            perturbacao = np.random.uniform(-20, 20)
            novo[i] = np.clip(novo[i] + perturbacao, intervalo[0], intervalo[1])
        return novo
            
        