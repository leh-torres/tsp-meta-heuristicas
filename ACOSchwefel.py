import random
import math
import time
import matplotlib.pyplot as plt

class ACO_Schwefel:
    def __init__(self, dimensoes, num_formigas_por_iter=20, num_iteracoes=100, 
                 tamanho_arquivo_solucoes=10, q_seletividade=0.1, xi_exploracao=0.85,
                 limite_inferior=-500, limite_superior=500):
        """
        Inicializa o algoritmo ACO para otimização da função Schwefel
        
        Args:
            dimensoes: Número de dimensões do problema
            num_formigas_por_iter: Número de formigas por iteração
            num_iteracoes: Número de iterações do algoritmo
            tamanho_arquivo_solucoes: Tamanho do arquivo de soluções
            q_seletividade: Parâmetro de seletividade (menor = mais seletivo)
            xi_exploracao: Parâmetro de exploração (controla largura da busca)
            limite_inferior: Limite inferior das variáveis
            limite_superior: Limite superior das variáveis
        """
        self.dimensoes = dimensoes
        self.num_formigas_por_iter = num_formigas_por_iter
        self.num_iteracoes = num_iteracoes
        self.tamanho_arquivo_solucoes = tamanho_arquivo_solucoes
        self.q_seletividade = q_seletividade
        self.xi_exploracao = xi_exploracao
        self.limite_inferior = limite_inferior
        self.limite_superior = limite_superior
        
        # Variáveis para armazenar resultados
        self.melhor_solucao = None
        self.melhor_custo = float('inf')
        self.historico_convergencia = []
        self.arquivo_solucoes = []
    
    def funcao_schwefel(self, x_vetor):
        """Calcula o valor da função Schwefel para um dado vetor x"""
        d = len(x_vetor)
        termo_constante = 418.9829 * d
        soma_senos = 0
        for xi in x_vetor:
            soma_senos += xi * math.sin(math.sqrt(abs(xi)))
        return termo_constante - soma_senos
    
    def _inicializar_arquivo_solucoes(self):
        """Inicializa o arquivo de soluções com amostras aleatórias"""
        arquivo = []
        for _ in range(self.tamanho_arquivo_solucoes):
            solucao_aleatoria = [random.uniform(self.limite_inferior, self.limite_superior) 
                               for _ in range(self.dimensoes)]
            custo_solucao = self.funcao_schwefel(solucao_aleatoria)
            arquivo.append({'vetor': solucao_aleatoria, 'custo': custo_solucao})
        
        # Ordena o arquivo: melhor custo (menor valor da função) primeiro
        arquivo.sort(key=lambda s: s['custo'])
        return arquivo
    
    def _calcular_pesos_roleta(self, arquivo_solucoes):
        """Calcula pesos para seleção por roleta (favorece melhores soluções no arquivo)"""
        pesos = []
        soma_pesos_nao_norm = 0
        rank_max = len(arquivo_solucoes)

        for i_rank in range(rank_max):  # i_rank=0 é a melhor solução
            # Função de peso Gaussiana baseada no rank
            expoente = -(i_rank * i_rank) / (2 * self.q_seletividade * self.q_seletividade * rank_max * rank_max)
            peso = (1 / (self.q_seletividade * rank_max * math.sqrt(2 * math.pi))) * math.exp(expoente)
            pesos.append(peso)
            soma_pesos_nao_norm += peso

        # Normaliza os pesos para somarem 1
        if soma_pesos_nao_norm > 0:
            pesos_normalizados = [p / soma_pesos_nao_norm for p in pesos]
        else:  # fallback se todos os pesos forem zero (improvável com a Gaussiana)
            pesos_normalizados = [1.0 / rank_max] * rank_max
        return pesos_normalizados
    
    def _selecionar_solucao_guia(self, arquivo_solucoes, pesos_roleta):
        """Seleciona uma solução 'guia' do arquivo usando o método da roleta"""
        rand_val = random.random()
        soma_acumulada_pesos = 0.0
        for i, solucao_candidata in enumerate(arquivo_solucoes):
            soma_acumulada_pesos += pesos_roleta[i]
            if rand_val <= soma_acumulada_pesos:
                return solucao_candidata
        return arquivo_solucoes[-1]  # fallback: retorna a última (pior) do arquivo ordenado
    
    def resolver(self, verbose=True):
        """
        Executa o algoritmo ACO para otimização da função Schwefel
        
        Args:
            verbose: Se deve imprimir progresso
            
        Returns:
            tuple: (melhor_solucao_vetor, melhor_custo, historico_convergencia)
        """
        if verbose:
            print(f"Resolvendo otimização da função Schwefel para {self.dimensoes} dimensões.")
            print(f"Parâmetros: Formigas={self.num_formigas_por_iter}, Iterações={self.num_iteracoes}")
            print(f"Arquivo={self.tamanho_arquivo_solucoes}, q={self.q_seletividade}, xi={self.xi_exploracao}")
            print(f"Limites: [{self.limite_inferior}, {self.limite_superior}]")
        
        tempo_inicio = time.time()
        
        # 1. Inicialização
        self.arquivo_solucoes = self._inicializar_arquivo_solucoes()
        
        # Melhor solução global inicial é a melhor do arquivo inicial
        self.melhor_solucao = list(self.arquivo_solucoes[0]['vetor'])
        self.melhor_custo = self.arquivo_solucoes[0]['custo']
        self.historico_convergencia = [self.melhor_custo]
        
        # 2. Loop principal de iterações
        for iteracao_idx in range(self.num_iteracoes):
            novas_solucoes_geradas_nesta_iteracao = []

            # Calcula os pesos para seleção das guias (uma vez por iteração)
            pesos_para_roleta = self._calcular_pesos_roleta(self.arquivo_solucoes)

            # Cada formiga gera uma nova solução
            for _ in range(self.num_formigas_por_iter):
                # a. Seleciona uma solução guia do arquivo_solucoes
                solucao_guia_escolhida = self._selecionar_solucao_guia(self.arquivo_solucoes, pesos_para_roleta)
                vetor_solucao_guia = solucao_guia_escolhida['vetor']

                novo_vetor_solucao_formiga = []
                # b. Para cada dimensão, amostra um novo valor
                for d_idx in range(self.dimensoes):
                    # i. Calcula o desvio padrão (sigma_d) para a dimensão 'd_idx'
                    soma_diferencas_abs_dim = 0
                    for outra_sol_no_arquivo in self.arquivo_solucoes:
                        # Evita comparar com a própria solução guia se ela for a única no arquivo
                        if outra_sol_no_arquivo['vetor'] != vetor_solucao_guia or len(self.arquivo_solucoes) == 1:
                             soma_diferencas_abs_dim += abs(vetor_solucao_guia[d_idx] - outra_sol_no_arquivo['vetor'][d_idx])

                    # Média das distâncias para as outras k-1 soluções no arquivo
                    if len(self.arquivo_solucoes) > 1:
                        sigma_d = self.xi_exploracao * (soma_diferencas_abs_dim / (len(self.arquivo_solucoes) - 1))
                    else:  # Se o arquivo tem apenas uma solução
                        sigma_d = self.xi_exploracao * abs(self.limite_superior - self.limite_inferior) / 10.0

                    if sigma_d < 1e-5: 
                        sigma_d = 1e-5  # Evita sigma muito pequeno ou zero

                    # ii. Amostra novo valor de uma Gaussiana
                    novo_valor_dimensao = random.gauss(vetor_solucao_guia[d_idx], sigma_d)

                    # iii. Garante que o novo valor esteja dentro dos limites
                    novo_valor_dimensao = max(self.limite_inferior, min(novo_valor_dimensao, self.limite_superior))
                    novo_vetor_solucao_formiga.append(novo_valor_dimensao)

                # Avalia a nova solução gerada pela formiga
                custo_nova_solucao = self.funcao_schwefel(novo_vetor_solucao_formiga)
                novas_solucoes_geradas_nesta_iteracao.append({'vetor': novo_vetor_solucao_formiga, 'custo': custo_nova_solucao})

                # Atualiza a melhor solução global se a nova for melhor
                if custo_nova_solucao < self.melhor_custo:
                    self.melhor_custo = custo_nova_solucao
                    self.melhor_solucao = list(novo_vetor_solucao_formiga)

            # 3. Adiciona as novas soluções geradas ao arquivo
            self.arquivo_solucoes.extend(novas_solucoes_geradas_nesta_iteracao)

            # 4. Ordena o arquivo pelo custo e mantém o tamanho fixo
            self.arquivo_solucoes.sort(key=lambda s: s['custo'])
            self.arquivo_solucoes = self.arquivo_solucoes[:self.tamanho_arquivo_solucoes]

            # Atualiza melhor global caso uma solução promovida no arquivo seja melhor
            if self.arquivo_solucoes[0]['custo'] < self.melhor_custo:
                self.melhor_custo = self.arquivo_solucoes[0]['custo']
                self.melhor_solucao = list(self.arquivo_solucoes[0]['vetor'])

            self.historico_convergencia.append(self.melhor_custo)
            
            if verbose:
                print(f"Iteração {iteracao_idx+1}/{self.num_iteracoes} | Melhor Custo: {self.melhor_custo:.6f}")
        
        tempo_fim = time.time()
        tempo_execucao = tempo_fim - tempo_inicio
        
        if verbose:
            print(f"\n--- Resultados do ACO para Função Schwefel ---")
            print(f"Melhor solução encontrada: {[round(x, 4) for x in self.melhor_solucao]}")
            print(f"Melhor custo: {self.melhor_custo:.6f}")
            print(f"Valor ótimo teórico: {418.9829 * self.dimensoes}")
            print(f"Tempo de execução: {tempo_execucao:.4f} segundos")
            self._imprimir_convergencia()
        
        return self.melhor_solucao, self.melhor_custo, self.historico_convergencia
    
    def _imprimir_convergencia(self):
        """Imprime resumo da convergência"""
        if not self.historico_convergencia:
            return
            
        print("\n--- Convergência (Resumo) ---")
        print(f"Custo inicial: {self.historico_convergencia[0]:.6f}")
        print(f"Custo final:   {self.historico_convergencia[-1]:.6f}")
        
        melhoria = self.historico_convergencia[0] - self.historico_convergencia[-1]
        print(f"Melhoria total: {melhoria:.6f}")
        
        # Mostra custo a cada 10 iterações
        passo = max(1, len(self.historico_convergencia) // 10)
        for i in range(0, len(self.historico_convergencia), passo):
            print(f"Iteração {i+1:>3}: {self.historico_convergencia[i]:.6f}")
        
        # Melhor iteração
        melhor_valor = min(self.historico_convergencia)
        melhor_iteracao = self.historico_convergencia.index(melhor_valor) + 1
        print(f"\nMelhor custo atingido na iteração {melhor_iteracao}: {melhor_valor:.6f}")
    
    def plotar_convergencia(self):
        """Plota gráfico de convergência"""
        if not self.historico_convergencia:
            print("Nenhum histórico de convergência disponível.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.historico_convergencia) + 1), 
                self.historico_convergencia, marker='o', linestyle='-', markersize=4)
        plt.title(f"Convergência ACO - Função Schwefel ({self.dimensoes}D)")
        plt.xlabel("Iteração")
        plt.ylabel("Melhor Valor da Função Schwefel")
        plt.xticks(range(0, len(self.historico_convergencia) + 1, 
                        max(1, len(self.historico_convergencia)//10)))
        plt.grid(True)
        plt.show()

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
         
