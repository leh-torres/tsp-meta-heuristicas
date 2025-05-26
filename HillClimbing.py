import random
import time
import math
import itertools
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class HillClimbing:
    """
    Implementação do algoritmo Hill Climbing para resolver problemas de otimização.
    Suporta tanto o TSP (Traveling Salesman Problem) quanto otimização contínua.
    """
    
    def __init__(self, grafo: Optional[Dict] = None, 
                 max_iter_sem_melhora: int = 100,
                 cidade_inicial: int = 1):
        """
        Inicializa o Hill Climbing.
        
        Args:
            grafo: Dicionário representando o grafo para TSP
            max_iter_sem_melhora: Número máximo de iterações sem melhoria
            cidade_inicial: Cidade inicial para o TSP
        """
        self.grafo = grafo
        self.max_iter_sem_melhora = max_iter_sem_melhora
        self.cidade_inicial = cidade_inicial
        self.cidades = list(grafo.keys()) if grafo else []
        
        # Estatísticas de execução
        self.tempo_execucao = 0
        self.total_iteracoes = 0
        self.historico_custos = []
        
    def calcular_distancia_rota(self, rota: List[int]) -> float:
        """
        Calcula a distância total de uma rota no TSP.
        
        Args:
            rota: Lista de cidades representando a rota
            
        Returns:
            Distância total da rota
        """
        distancia_total = 0
        for i in range(len(rota) - 1):
            cidade_atual = rota[i]
            proxima_cidade = rota[i + 1]
            if proxima_cidade in self.grafo.get(cidade_atual, {}):
                distancia_total += self.grafo[cidade_atual][proxima_cidade]
            else:
                return float("inf")
        return distancia_total
    
    def gerar_rota_inicial(self) -> List[int]:
        """
        Gera uma rota inicial aleatória para o TSP.
        
        Returns:
            Rota inicial com cidade de partida e chegada iguais
        """
        cidades_a_visitar = [c for c in self.cidades if c != self.cidade_inicial]
        random.shuffle(cidades_a_visitar)
        return [self.cidade_inicial] + cidades_a_visitar + [self.cidade_inicial]
    
    def gerar_vizinhos_permutacao(self, rota: List[int]) -> List[List[int]]:
        """
        Gera vizinhos através de permutação de duas cidades.
        
        Args:
            rota: Rota atual
            
        Returns:
            Lista de rotas vizinhas
        """
        vizinhos = []
        indices_permutaveis = list(range(1, len(rota) - 1))
        
        for i, j in itertools.combinations(indices_permutaveis, 2):
            vizinho = rota[:]
            vizinho[i], vizinho[j] = vizinho[j], vizinho[i]
            vizinhos.append(vizinho)
            
        return vizinhos
    
    def iniciar_tsp(self) -> Tuple[List[int], float, List[float]]:
        """
        Executa o Hill Climbing para o problema do TSP.
        
        Returns:
            Tupla contendo (melhor_rota, menor_distancia, historico_custos)
        """
        if not self.grafo:
            raise ValueError("Grafo não foi definido para resolver TSP")
            
        tempo_inicio = time.time()
        
        # Inicialização
        rota_atual = self.gerar_rota_inicial()
        distancia_atual = self.calcular_distancia_rota(rota_atual)
        self.historico_custos = [distancia_atual]
        
        iter_sem_melhora = 0
        self.total_iteracoes = 0
        
        print(f"Rota inicial: {' -> '.join(map(str, rota_atual))}")
        print(f"Distância inicial: {distancia_atual:.2f}")
        
        # Loop principal do Hill Climbing
        while iter_sem_melhora < self.max_iter_sem_melhora:
            self.total_iteracoes += 1
            melhor_vizinho = None
            melhor_distancia_vizinho = distancia_atual
            
            # Gera e avalia vizinhos
            vizinhos = self.gerar_vizinhos_permutacao(rota_atual)
            random.shuffle(vizinhos)
            
            encontrou_melhor = False
            for vizinho in vizinhos:
                distancia_vizinho = self.calcular_distancia_rota(vizinho)
                if distancia_vizinho < melhor_distancia_vizinho:
                    melhor_vizinho = vizinho
                    melhor_distancia_vizinho = distancia_vizinho
                    encontrou_melhor = True
            
            # Atualiza se encontrou melhoria
            if encontrou_melhor:
                rota_atual = melhor_vizinho
                distancia_atual = melhor_distancia_vizinho
                self.historico_custos.append(distancia_atual)
                iter_sem_melhora = 0
                print(f"Iteração {self.total_iteracoes}: Nova melhor distância = {distancia_atual:.2f}")
            else:
                iter_sem_melhora += 1
                self.historico_custos.append(distancia_atual)
        
        tempo_fim = time.time()
        self.tempo_execucao = tempo_fim - tempo_inicio
        
        print(f"\nAlgoritmo convergiu após {self.total_iteracoes} iterações")
        print(f"Melhor rota: {' -> '.join(map(str, rota_atual))}")
        print(f"Menor distância: {distancia_atual:.2f}")
        
        return rota_atual, distancia_atual, self.historico_custos
    
    def schwefel(self, x: np.ndarray) -> float:
        """
        Implementa a função de Schwefel para otimização contínua.
        
        Args:
            x: Vetor de entrada
            
        Returns:
            Valor da função Schwefel
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        x = np.clip(x, -500, 500)  # Limites da função Schwefel
        termo1 = 418.9829 * len(x)
        termo2 = np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return termo1 - termo2
    
    def gerar_solucao_aleatoria(self, dimensoes: int, limite_inf: float, limite_sup: float) -> np.ndarray:
        """
        Gera uma solução inicial aleatória para otimização contínua.
        
        Args:
            dimensoes: Número de dimensões
            limite_inf: Limite inferior
            limite_sup: Limite superior
            
        Returns:
            Vetor solução inicial
        """
        return np.random.uniform(limite_inf, limite_sup, dimensoes)
    
    def gerar_vizinho_continuo(self, solucao_atual: np.ndarray, 
                              passo_maximo: float = 5.0,
                              limite_inf: float = -500,
                              limite_sup: float = 500) -> np.ndarray:
        """
        Gera um vizinho para otimização contínua.
        
        Args:
            solucao_atual: Solução atual
            passo_maximo: Tamanho máximo do passo
            limite_inf: Limite inferior
            limite_sup: Limite superior
            
        Returns:
            Solução vizinha
        """
        vizinho = solucao_atual + np.random.normal(0, passo_maximo, len(solucao_atual))
        return np.clip(vizinho, limite_inf, limite_sup)
    
    def iniciar_continuo(self, dimensoes: int, intervalo: Tuple[float, float],
                        max_reinicios: int = 10, 
                        num_vizinhos_por_iter: int = 20) -> Tuple[np.ndarray, float, List[float]]:
        """
        Executa o Hill Climbing com reinício aleatório para otimização contínua.
        
        Args:
            dimensoes: Número de dimensões do problema
            intervalo: Tupla (limite_inferior, limite_superior)
            max_reinicios: Número máximo de reinícios
            num_vizinhos_por_iter: Número de vizinhos avaliados por iteração
            
        Returns:
            Tupla contendo (melhor_solucao, melhor_valor, historico_custos)
        """
        tempo_inicio = time.time()
        limite_inf, limite_sup = intervalo
        
        melhor_solucao_global = None
        melhor_valor_global = float("inf")
        self.historico_custos = []
        self.total_iteracoes = 0
        
        print(f"Iniciando Hill Climbing contínuo com {max_reinicios} reinícios")
        print(f"Dimensões: {dimensoes}, Intervalo: {intervalo}")
        
        for reinicio in range(max_reinicios):
            print(f"\n--- Reinício {reinicio + 1}/{max_reinicios} ---")
            
            # Solução inicial para este reinício
            solucao_atual = self.gerar_solucao_aleatoria(dimensoes, limite_inf, limite_sup)
            valor_atual = self.schwefel(solucao_atual)
            print(f"Valor inicial: {valor_atual:.4f}")
            
            iter_sem_melhora = 0
            while iter_sem_melhora < self.max_iter_sem_melhora:
                self.total_iteracoes += 1
                melhor_vizinho = None
                melhor_valor_vizinho = valor_atual
                encontrou_melhor_local = False
                
                # Gera e avalia múltiplos vizinhos
                for _ in range(num_vizinhos_por_iter):
                    vizinho = self.gerar_vizinho_continuo(solucao_atual, 
                                                        limite_inf=limite_inf, 
                                                        limite_sup=limite_sup)
                    valor_vizinho = self.schwefel(vizinho)
                    
                    if valor_vizinho < melhor_valor_vizinho:
                        melhor_vizinho = vizinho
                        melhor_valor_vizinho = valor_vizinho
                        encontrou_melhor_local = True
                
                # Atualiza se encontrou melhoria local
                if encontrou_melhor_local:
                    solucao_atual = melhor_vizinho
                    valor_atual = melhor_valor_vizinho
                    iter_sem_melhora = 0
                else:
                    iter_sem_melhora += 1
                
                # Atualiza melhor solução global
                if valor_atual < melhor_valor_global:
                    melhor_solucao_global = solucao_atual.copy()
                    melhor_valor_global = valor_atual
                    print(f"*** Novo melhor global: {melhor_valor_global:.4f} ***")
                
                self.historico_custos.append(melhor_valor_global)
        
        tempo_fim = time.time()
        self.tempo_execucao = tempo_fim - tempo_inicio
        
        # Calcula precisão (distância ao mínimo global conhecido)
        minimo_global = np.array([420.968746] * dimensoes)
        precisao = np.linalg.norm(melhor_solucao_global - minimo_global) if melhor_solucao_global is not None else float("inf")
        
        print(f"\nAlgoritmo concluído após {self.total_iteracoes} iterações totais")
        print(f"Melhor valor encontrado: {melhor_valor_global:.4f}")
        print(f"Precisão (distância euclidiana): {precisao:.4f}")
        
        return melhor_solucao_global, melhor_valor_global, self.historico_custos
    
    def get_estatisticas(self) -> Dict[str, Any]:
        """
        Retorna estatísticas da última execução.
        
        Returns:
            Dicionário com estatísticas de execução
        """
        return {
            "tempo_execucao": self.tempo_execucao,
            "total_iteracoes": self.total_iteracoes,
            "historico_custos": self.historico_custos,
            "max_iter_sem_melhora": self.max_iter_sem_melhora
        }