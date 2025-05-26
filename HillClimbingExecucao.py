import HillClimbing as hc
import time
import matplotlib.pyplot as plt
import json
import os

GRAFO = {
    1: {2: 20, 8: 29, 12: 29, 13: 37},
    2: {1: 20, 3: 25, 8: 28, 12: 39},
    3: {2: 25, 4: 25, 8: 30, 13: 54},
    4: {3: 25, 6: 32, 5: 39, 9: 23, 10: 33, 7: 42, 14: 56},
    5: {4: 39, 6: 12, 7: 26, 10: 19},
    6: {4: 32, 5: 12, 7: 17, 10: 35, 11: 30},
    7: {4: 42, 5: 26, 6: 17, 11: 38},
    8: {1: 29, 2: 28, 3: 30, 12: 25, 13: 22},
    9: {4: 23, 10: 26, 13: 34, 14: 34, 16: 43},
    10: {4: 33, 5: 19, 6: 35, 9: 26, 11: 24, 14: 30, 15: 19},
    11: {6: 30, 7: 38, 10: 24, 15: 26, 18: 36},
    12: {1: 29, 2: 39, 8: 25, 13: 27, 16: 43},
    13: {1: 37, 3: 54, 8: 22, 9: 34, 12: 27, 14: 24, 16: 19},
    14: {4: 56, 9: 34, 10: 30, 13: 24, 15: 20, 16: 19, 17: 17},
    15: {10: 19, 11: 26, 14: 20, 17: 18, 18: 21},
    16: {9: 43, 12: 43, 13: 19, 14: 19, 17: 26},
    17: {14: 17, 15: 18, 16: 26, 18: 15},
    18: {11: 36, 15: 21, 17: 15}
}

if __name__ == "__main__":
    print('----------------- Hill Climbing TSP -----------------')

    # Configurações para TSP
    max_iter_sem_melhora = 200
    cidade_inicial = 1

    # Criar instância do Hill Climbing
    hill_climbing = hc.HillClimbing(GRAFO, max_iter_sem_melhora, cidade_inicial)

    # Executar para TSP
    inicio_hc = time.time()
    melhor_rota, menor_distancia, historico_custos_tsp = hill_climbing.iniciar_tsp()
    fim_hc = time.time()

    tempo_hc = fim_hc - inicio_hc
    print(f'Tempo de execução Hill Climbing TSP: {tempo_hc:.2f} segundos')

    # Plotar convergência TSP
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(historico_custos_tsp, label="Melhor custo por iteração", color="red", marker='o', markersize=3)
    plt.title("Convergência do Hill Climbing (TSP)")
    plt.xlabel("Iteração")
    plt.ylabel("Custo da melhor rota")
    plt.legend()
    plt.grid(True, alpha=0.3)

    print('-----------------------------------------------------')

    print('------------- Hill Climbing Schwefel ---------------')

    # Configurações para função Schwefel
    dimensoes = 5
    intervalo = (-500, 500)
    max_reinicios = 20
    max_iter_sem_melhora_schwefel = 100

    # Criar nova instância para Schwefel
    hill_climbing_schwefel = hc.HillClimbing(max_iter_sem_melhora=max_iter_sem_melhora_schwefel)

    # Executar para Schwefel
    inicio_hc_schwefel = time.time()
    melhor_solucao, melhor_valor, historico_custos_schwefel = hill_climbing_schwefel.iniciar_continuo(
        dimensoes, intervalo, max_reinicios
    )
    fim_hc_schwefel = time.time()

    tempo_hc_schwefel = fim_hc_schwefel - inicio_hc_schwefel
    print(f'Tempo de execução Hill Climbing Schwefel: {tempo_hc_schwefel:.2f} segundos')

    # Plotar convergência Schwefel
    plt.subplot(1, 2, 2)
    plt.plot(historico_custos_schwefel, label="Melhor valor por iteração", color="green", marker='.')
    plt.title("Convergência do Hill Climbing (Schwefel)")
    plt.xlabel("Iteração")
    plt.ylabel("Valor da função")
    plt.yscale('symlog')  # Escala logarítmica para melhor visualização
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print('-----------------------------------------------------')

    # Comparação de estatísticas
    print('\n=============== RESUMO COMPARATIVO ================')
    print(f'TSP - Melhor distância: {menor_distancia:.2f}')
    print(f'TSP - Tempo de execução: {tempo_hc:.2f}s')
    print(f'TSP - Iterações: {hill_climbing.total_iteracoes}')

    print(f'\nSchwefel - Melhor valor: {melhor_valor:.4f}')
    print(f'Schwefel - Tempo de execução: {tempo_hc_schwefel:.2f}s')
    print(f'Schwefel - Iterações: {hill_climbing_schwefel.total_iteracoes}')

    # Salvar resultados em arquivo (opcional)
    os.makedirs("results", exist_ok=True)

    resultados = {
        "tsp": {
            "melhor_rota": melhor_rota,
            "menor_distancia": menor_distancia,
            "tempo_execucao": tempo_hc,
            "iteracoes": hill_climbing.total_iteracoes,
            "historico_custos": historico_custos_tsp
        },
        "schwefel": {
            "melhor_solucao": melhor_solucao.tolist() if melhor_solucao is not None else None,
            "melhor_valor": melhor_valor,
            "tempo_execucao": tempo_hc_schwefel,
            "iteracoes": hill_climbing_schwefel.total_iteracoes,
            "historico_custos": historico_custos_schwefel[:100]  # Primeiros 100 valores para não sobrecarregar
        }
    }

    with open("results/resultados_hill_climbing.json", "w") as f:
        json.dump(resultados, f, indent=4)

    print("\nResultados salvos em results/resultados_hill_climbing.json")
    print('=====================================================')