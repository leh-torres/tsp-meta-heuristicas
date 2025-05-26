import random
import math
import time
import matplotlib.pyplot as plt

class ACO_TSP:
    def __init__(self, grafo_adj, num_formigas=None, num_iteracoes=100, 
                 alfa=1.0, beta=2.0, taxa_evaporacao=0.5, Q_constante=100.0):
        """
        Inicializa o algoritmo ACO para TSP
        
        Args:
            grafo_adj: Dicionário de adjacência no formato {cidade: {vizinho: distancia, ...}}
            num_formigas: Número de formigas (padrão: 10 * número de cidades)
            num_iteracoes: Número de iterações
            alfa: Parâmetro de influência do feromônio
            beta: Parâmetro de influência da heurística
            taxa_evaporacao: Taxa de evaporação do feromônio
            Q_constante: Constante para deposição de feromônio
        """
        self.grafo_adj = grafo_adj
        self.num_cidades = len(grafo_adj)
        self.cidades = list(grafo_adj.keys())
        
        # Parâmetros do algoritmo
        self.num_formigas = num_formigas if num_formigas else 10 * self.num_cidades
        self.num_iteracoes = num_iteracoes
        self.alfa = alfa
        self.beta = beta
        self.taxa_evaporacao = taxa_evaporacao
        self.Q_constante = Q_constante
        
        # Converte lista de adjacência para matriz de distâncias para facilitar cálculos
        self.matriz_distancias = self._converter_para_matriz()
        
        # Inicializa feromônios
        self._inicializar_feromonios()
        
        # Variáveis para armazenar resultados
        self.melhor_rota = None
        self.menor_distancia = float('inf')
        self.historico_convergencia = []
    
    def _converter_para_matriz(self):
        """Converte dicionário de adjacência para matriz de distâncias"""
        matriz = [[float('inf') for _ in range(self.num_cidades)] for _ in range(self.num_cidades)]
        
        # Mapeia cidades para índices
        self.cidade_para_indice = {cidade: i for i, cidade in enumerate(self.cidades)}
        self.indice_para_cidade = {i: cidade for i, cidade in enumerate(self.cidades)}
        
        # Preenche a matriz
        for i in range(self.num_cidades):
            matriz[i][i] = 0  # Distância para si mesmo é 0
        
        for cidade, vizinhos in self.grafo_adj.items():
            i = self.cidade_para_indice[cidade]
            for vizinho, distancia in vizinhos.items():
                j = self.cidade_para_indice[vizinho]
                matriz[i][j] = distancia
                
        return matriz
    
    def _inicializar_feromonios(self):
        """Inicializa matriz de feromônios"""
        feromonio_inicial = 1.0 / (self.num_cidades * self.num_cidades)
        self.feromonios = [[feromonio_inicial for _ in range(self.num_cidades)] 
                          for _ in range(self.num_cidades)]
    
    def calcular_distancia_total(self, rota):
        """Calcula a distância total de uma rota"""
        distancia = 0
        num_cidades_rota = len(rota)
        
        for i in range(num_cidades_rota):
            cidade_atual = rota[i]
            proxima_cidade = rota[(i + 1) % num_cidades_rota]
            distancia += self.matriz_distancias[cidade_atual][proxima_cidade]
            
        return distancia
    
    def _construir_solucao_formiga(self, cidade_inicial_idx):
        """Constrói uma solução (rota) para uma formiga"""
        rota = [cidade_inicial_idx]
        cidades_disponiveis = list(range(self.num_cidades))
        cidades_disponiveis.remove(cidade_inicial_idx)
        
        cidade_atual = cidade_inicial_idx
        
        while cidades_disponiveis:
            probabilidades = []
            soma_denominador = 0.0
            
            # Calcula probabilidades para cada cidade disponível
            for proxima_cidade in cidades_disponiveis:
                if self.matriz_distancias[cidade_atual][proxima_cidade] > 0:
                    fator_feromonio = math.pow(self.feromonios[cidade_atual][proxima_cidade], self.alfa)
                    fator_heuristico = math.pow(1.0 / self.matriz_distancias[cidade_atual][proxima_cidade], self.beta)
                    valor_prob = fator_feromonio * fator_heuristico
                    probabilidades.append({'cidade': proxima_cidade, 'prob': valor_prob})
                    soma_denominador += valor_prob
                else:
                    probabilidades.append({'cidade': proxima_cidade, 'prob': 0})
            
            # Seleciona próxima cidade
            proxima_cidade = None
            if soma_denominador > 0:
                # Normaliza probabilidades
                for item in probabilidades:
                    item['prob'] /= soma_denominador
                
                # Seleção por roleta
                rand_val = random.random()
                soma_acumulada = 0.0
                probabilidades.sort(key=lambda x: x['prob'], reverse=True)
                
                for item in probabilidades:
                    soma_acumulada += item['prob']
                    if rand_val <= soma_acumulada:
                        proxima_cidade = item['cidade']
                        break
            
            # Fallback: escolha aleatória se necessário
            if proxima_cidade is None and cidades_disponiveis:
                proxima_cidade = random.choice(cidades_disponiveis)
            
            if proxima_cidade is not None:
                rota.append(proxima_cidade)
                cidades_disponiveis.remove(proxima_cidade)
                cidade_atual = proxima_cidade
            else:
                break
                
        return rota
    
    def _atualizar_feromonios(self, todas_rotas, custos_rotas):
        """Atualiza os níveis de feromônio"""
        # Evaporação
        for i in range(self.num_cidades):
            for j in range(self.num_cidades):
                self.feromonios[i][j] *= (1.0 - self.taxa_evaporacao)
        
        # Deposição de novo feromônio
        for k in range(len(todas_rotas)):
            rota = todas_rotas[k]
            custo = custos_rotas[k]
            
            if custo == 0:
                continue
                
            deposito = self.Q_constante / custo
            
            for i in range(len(rota)):
                cidade_origem = rota[i]
                cidade_destino = rota[(i + 1) % len(rota)]
                
                self.feromonios[cidade_origem][cidade_destino] += deposito
                self.feromonios[cidade_destino][cidade_origem] += deposito
    
    def resolver(self, cidade_inicial=None, verbose=True):
        """
        Executa o algoritmo ACO para resolver o TSP
        
        Args:
            cidade_inicial: Cidade inicial (nome ou None para usar a primeira)
            verbose: Se deve imprimir progresso
            
        Returns:
            tuple: (melhor_rota_nomes, menor_distancia, historico_convergencia)
        """
        # Define cidade inicial
        if cidade_inicial is None:
            cidade_inicial_idx = 0
        else:
            cidade_inicial_idx = self.cidade_para_indice.get(cidade_inicial, 0)
        
        # Reinicia variáveis de resultado
        self.melhor_rota = None
        self.menor_distancia = float('inf')
        self.historico_convergencia = []
        
        if verbose:
            print(f"Resolvendo TSP para {self.num_cidades} cidades.")
            print(f"Parâmetros: Formigas={self.num_formigas}, Iterações={self.num_iteracoes}")
            print(f"Alfa={self.alfa}, Beta={self.beta}, Evaporação={self.taxa_evaporacao}, Q={self.Q_constante}")
        
        tempo_inicio = time.time()
        
        for iteracao in range(self.num_iteracoes):
            rotas_iteracao = []
            custos_iteracao = []
            
            # Cada formiga constrói uma rota
            for _ in range(self.num_formigas):
                rota = self._construir_solucao_formiga(cidade_inicial_idx)
                
                # Verifica se a rota é válida
                if len(set(rota)) == self.num_cidades:
                    custo = self.calcular_distancia_total(rota)
                    rotas_iteracao.append(rota)
                    custos_iteracao.append(custo)
                    
                    # Atualiza melhor solução global
                    if custo < self.menor_distancia:
                        self.menor_distancia = custo
                        self.melhor_rota = list(rota)
            
            # Atualiza feromônios
            if rotas_iteracao:
                self._atualizar_feromonios(rotas_iteracao, custos_iteracao)
            
            self.historico_convergencia.append(self.menor_distancia)
            
            if verbose:
                print(f"Iteração {iteracao+1}/{self.num_iteracoes} | Melhor Distância: {self.menor_distancia:.2f}")
        
        tempo_fim = time.time()
        tempo_execucao = tempo_fim - tempo_inicio
        
        # Converte rota de índices para nomes das cidades
        if self.melhor_rota:
            melhor_rota_nomes = [self.indice_para_cidade[idx] for idx in self.melhor_rota]
            
            if verbose:
                print(f"\n--- Resultados do ACO para TSP ---")
                print(f"Melhor rota encontrada: {melhor_rota_nomes} -> {melhor_rota_nomes[0]}")
                print(f"Menor distância total: {self.menor_distancia:.2f}")
                print(f"Tempo de execução: {tempo_execucao:.4f} segundos")
                self._imprimir_convergencia()
            
            return melhor_rota_nomes, self.menor_distancia, self.historico_convergencia
        else:
            if verbose:
                print("Nenhuma rota válida foi encontrada.")
            return None, float('inf'), self.historico_convergencia
    
    def _imprimir_convergencia(self):
        """Imprime resumo da convergência"""
        if not self.historico_convergencia:
            return
            
        print("\n--- Convergência (Resumo) ---")
        print(f"Distância inicial: {self.historico_convergencia[0]:.2f}")
        print(f"Distância final:   {self.historico_convergencia[-1]:.2f}")
        
        melhoria = self.historico_convergencia[0] - self.historico_convergencia[-1]
        print(f"Melhoria total:    {melhoria:.2f}")
        
        # Mostra distância a cada 10 iterações
        passo = max(1, len(self.historico_convergencia) // 10)
        for i in range(0, len(self.historico_convergencia), passo):
            print(f"Iteração {i+1:>3}: {self.historico_convergencia[i]:.2f}")
        
        # Melhor iteração
        melhor_valor = min(self.historico_convergencia)
        melhor_iteracao = self.historico_convergencia.index(melhor_valor) + 1
        print(f"\nMelhor distância atingida na iteração {melhor_iteracao}: {melhor_valor:.2f}")
    
    def plotar_convergencia(self):
        """Plota gráfico de convergência"""
        if not self.historico_convergencia:
            print("Nenhum histórico de convergência disponível.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.historico_convergencia) + 1), 
                self.historico_convergencia, marker='o', linestyle='-', markersize=4)
        plt.title(f"Convergência do ACO para TSP ({self.num_cidades} Cidades)")
        plt.xlabel("Iteração")
        plt.ylabel("Menor Distância Encontrada")
        plt.xticks(range(0, len(self.historico_convergencia) + 1, 
                        max(1, len(self.historico_convergencia)//10)))
        plt.grid(True)
        plt.show()

