import random
import time
import math
import numpy as np
import json

# Definição da função Schwefel
DIMENSOES = 5
LIMITE_INFERIOR = -500
LIMITE_SUPERIOR = 500
MINIMO_GLOBAL_X = np.array([420.968746] * DIMENSOES)
MINIMO_GLOBAL_F = 0.0 # O valor exato é 0 para x_i = 420.968746

def schwefel(x):
    """Calcula o valor da função Schwefel para um vetor x."""
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    # Garante que x está dentro dos limites (embora a geração de vizinhos deva cuidar disso)
    x = np.clip(x, LIMITE_INFERIOR, LIMITE_SUPERIOR)
    
    termo1 = 418.9829 * DIMENSOES
    termo2 = np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return termo1 - termo2

def gerar_solucao_aleatoria(dimensoes, limite_inf, limite_sup):
    """Gera uma solução inicial aleatória dentro dos limites."""
    return np.random.uniform(limite_inf, limite_sup, dimensoes)

def gerar_vizinho_continuo(solucao_atual, passo_maximo=5.0):
    """Gera um vizinho adicionando um pequeno ruído gaussiano, respeitando os limites."""
    vizinho = solucao_atual + np.random.normal(0, passo_maximo, len(solucao_atual))
    # Garante que o vizinho permaneça dentro dos limites
    vizinho = np.clip(vizinho, LIMITE_INFERIOR, LIMITE_SUPERIOR)
    return vizinho

def hill_climbing_random_restart(func, dimensoes, limite_inf, limite_sup, 
                                 max_iter_sem_melhora_local=50, 
                                 max_reinicios=10, 
                                 num_vizinhos_por_iter=20):
    """Executa o Hill Climbing com Reinício Aleatório."""
    tempo_inicio = time.time()
    
    melhor_solucao_global = None
    melhor_valor_global = float("inf")
    historico_melhor_valor_global = []
    total_iteracoes_efetivas = 0 # Conta iterações que exploram vizinhos

    for reinicio in range(max_reinicios):
        # print(f"\n--- Reinício {reinicio + 1}/{max_reinicios} ---")
        # Gera uma solução inicial aleatória para este reinício
        solucao_atual = gerar_solucao_aleatoria(dimensoes, limite_inf, limite_sup)
        valor_atual = func(solucao_atual)
        # print(f"Solução Inicial: {valor_atual:.4f}")

        iter_sem_melhora = 0
        while iter_sem_melhora < max_iter_sem_melhora_local:
            total_iteracoes_efetivas += 1
            melhor_vizinho = None
            melhor_valor_vizinho = valor_atual
            encontrou_melhor_local = False

            # Gera e avalia vários vizinhos
            for _ in range(num_vizinhos_por_iter):
                vizinho = gerar_vizinho_continuo(solucao_atual)
                valor_vizinho = func(vizinho)

                if valor_vizinho < melhor_valor_vizinho:
                    melhor_vizinho = vizinho
                    melhor_valor_vizinho = valor_vizinho
                    encontrou_melhor_local = True
                    # Estratégia First-Choice local (dentro da geração de vizinhos)
                    # break # Descomente se quiser parar no primeiro vizinho melhor da amostra
            
            # Atualiza se um vizinho melhor foi encontrado na amostra
            if encontrou_melhor_local:
                solucao_atual = melhor_vizinho
                valor_atual = melhor_valor_vizinho
                iter_sem_melhora = 0
                # print(f"Iter {total_iteracoes_efetivas}: Melhoria local encontrada, Valor = {valor_atual:.4f}")
            else:
                iter_sem_melhora += 1
                # print(f"Iter {total_iteracoes_efetivas}: Sem melhoria local ({iter_sem_melhora}/{max_iter_sem_melhora_local})")

            # Atualiza a melhor solução global encontrada até agora
            if valor_atual < melhor_valor_global:
                melhor_solucao_global = solucao_atual.copy()
                melhor_valor_global = valor_atual
                # print(f"*** Novo Melhor Global Encontrado: {melhor_valor_global:.4f} ***")
            
            # Registra o melhor valor global atual para o gráfico de convergência
            historico_melhor_valor_global.append(melhor_valor_global)

    tempo_fim = time.time()
    tempo_execucao = tempo_fim - tempo_inicio

    # Calcula a precisão (distância euclidiana ao mínimo global conhecido)
    if melhor_solucao_global is not None:
        distancia_ao_minimo = np.linalg.norm(melhor_solucao_global - MINIMO_GLOBAL_X)
    else: # Caso nenhum reinício tenha produzido uma solução válida (improvável)
        distancia_ao_minimo = float("inf")
        melhor_solucao_global = np.array([np.nan] * dimensoes) # Indica que não foi encontrada

    resultado = {
        "melhor_solucao_encontrada": melhor_solucao_global.tolist(),
        "melhor_valor_encontrado": melhor_valor_global,
        "precisao_dist_euclidiana": distancia_ao_minimo,
        "tempo_execucao_s": tempo_execucao,
        "total_iteracoes_efetivas": total_iteracoes_efetivas,
        "num_reinicios": max_reinicios,
        "historico_melhor_valor_global": historico_melhor_valor_global
    }
    return resultado

# --- Execução e Análise de Robustez ---
if __name__ == "__main__":
    print("Executando Hill Climbing com Reinício Aleatório para a função Schwefel...")
    
    NUM_EXECUCOES = 5 # Para análise de robustez
    resultados_execucoes = []

    print(f"Realizando {NUM_EXECUCOES} execuções para análise de robustez...")
    for i in range(NUM_EXECUCOES):
        print(f"Execução {i+1}/{NUM_EXECUCOES}...")
        resultado_hc_rr = hill_climbing_random_restart(
            schwefel, 
            DIMENSOES, 
            LIMITE_INFERIOR, 
            LIMITE_SUPERIOR, 
            max_iter_sem_melhora_local=100, # Aumentado para dar mais chance local
            max_reinicios=20, # Aumentado para explorar mais o espaço
            num_vizinhos_por_iter=50 # Aumentado para explorar mais vizinhos
        )
        resultados_execucoes.append(resultado_hc_rr)
        print(f"  Melhor Valor Encontrado: {resultado_hc_rr['melhor_valor_encontrado']:.4f}")
        print(f"  Precisão (Dist. Euclid.): {resultado_hc_rr['precisao_dist_euclidiana']:.4f}")
        print(f"  Tempo: {resultado_hc_rr["tempo_execucao_s"]:.2f}s")

    # Análise de Robustez
    valores_finais = [r["melhor_valor_encontrado"] for r in resultados_execucoes]
    precisoes = [r["precisao_dist_euclidiana"] for r in resultados_execucoes]
    tempos = [r["tempo_execucao_s"] for r in resultados_execucoes]

    media_valor = np.mean(valores_finais)
    std_valor = np.std(valores_finais)
    media_precisao = np.mean(precisoes)
    std_precisao = np.std(precisoes)
    media_tempo = np.mean(tempos)

    print("\n--- Resultados Hill Climbing com Reinício Aleatório (Schwefel) --- Média de {} execuções ---".format(NUM_EXECUCOES))
    print(f"Melhor Valor Médio: {media_valor:.4f} (Desvio Padrão: {std_valor:.4f})")
    print(f"Precisão Média (Dist. Euclid.): {media_precisao:.4f} (Desvio Padrão: {std_precisao:.4f})")
    print(f"Tempo Médio de Execução: {media_tempo:.4f} segundos")
    
    # Salvar o resultado da primeira execução (ou a melhor) e as estatísticas
    resultado_primeira_exec = resultados_execucoes[0]
    estatisticas = {
        "num_execucoes": NUM_EXECUCOES,
        "media_melhor_valor": media_valor,
        "std_melhor_valor": std_valor,
        "media_precisao_dist_euclidiana": media_precisao,
        "std_precisao_dist_euclidiana": std_precisao,
        "media_tempo_execucao_s": media_tempo
    }

    resultados_json = {
        "algoritmo": "Hill Climbing (Reinício Aleatório)",
        "problema": "Otimização Contínua (Schwefel)",
        "dimensoes": DIMENSOES,
        "limites": [LIMITE_INFERIOR, LIMITE_SUPERIOR],
        "minimo_global_conhecido": {"x": MINIMO_GLOBAL_X.tolist(), "f(x)": MINIMO_GLOBAL_F},
        "resultado_exemplo": resultado_primeira_exec, # Salva uma execução como exemplo
        "estatisticas_robustez": estatisticas
    }

    with open("results/resultado_hc_schwefel.json", "w") as f:
        json.dump(resultados_json, f, indent=4)
    
    print("\nResultados e estatísticas salvos em results/resultado_hc_schwefel.json")

    # Gerar gráfico de convergência da primeira execução
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        # O histórico guarda o melhor valor global *até* aquela iteração
        plt.plot(resultado_primeira_exec["historico_melhor_valor_global"], marker=".", linestyle="-")
        plt.title(f"Convergência do Hill Climbing c/ Reinício (Schwefel) - Exemplo Execução 1")
        plt.xlabel("Iteração Efetiva (Total)")
        plt.ylabel("Melhor Valor Global Encontrado (f(x))")
        plt.axhline(y=MINIMO_GLOBAL_F, color='r', linestyle='--', label=f"Mínimo Global ({MINIMO_GLOBAL_F:.2f})")
        plt.legend()
        plt.grid(True)
        plt.yscale('symlog') # Escala logarítmica simétrica para visualizar perto de zero
        plt.savefig("results/convergencia_hc_schwefel.png")
        print("Gráfico de convergência (exemplo) salvo em results/convergencia_hc_schwefel.png")
        plt.close()

    except ImportError:
        print("\nAviso: Matplotlib não instalado. Não foi possível gerar o gráfico de convergência.")
        print("Para instalar: pip install matplotlib")