import random
import time
import math
import itertools
import json
import os  # <- Adicionado

# Grafo fornecido pelo usuário
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

CIDADES = list(GRAFO.keys())
CIDADE_INICIAL = 1

def calcular_distancia_rota(rota):
    distancia_total = 0
    for i in range(len(rota) - 1):
        cidade_atual = rota[i]
        proxima_cidade = rota[i+1]
        if proxima_cidade in GRAFO.get(cidade_atual, {}):
            distancia_total += GRAFO[cidade_atual][proxima_cidade]
        else:
            return float("inf")
    return distancia_total

def gerar_rota_inicial(cidades, cidade_inicial):
    cidades_a_visitar = [c for c in cidades if c != cidade_inicial]
    random.shuffle(cidades_a_visitar)
    return [cidade_inicial] + cidades_a_visitar + [cidade_inicial]

def gerar_vizinhos_permutacao(rota):
    vizinhos = []
    indices_permutaveis = list(range(1, len(rota) - 1))
    for i, j in itertools.combinations(indices_permutaveis, 2):
        vizinho = rota[:]
        vizinho[i], vizinho[j] = vizinho[j], vizinho[i]
        vizinhos.append(vizinho)
    return vizinhos

def hill_climbing_tsp(cidades, cidade_inicial, max_iter_sem_melhora=100):
    tempo_inicio = time.time()
    rota_atual = gerar_rota_inicial(cidades, cidade_inicial)
    distancia_atual = calcular_distancia_rota(rota_atual)
    historico_distancias = [distancia_atual]
    iter_sem_melhora = 0
    iteracoes = 0

    while iter_sem_melhora < max_iter_sem_melhora:
        iteracoes += 1
        melhor_vizinho = None
        melhor_distancia_vizinho = distancia_atual
        vizinhos = gerar_vizinhos_permutacao(rota_atual)
        random.shuffle(vizinhos)

        encontrou_melhor = False
        for vizinho in vizinhos:
            distancia_vizinho = calcular_distancia_rota(vizinho)
            if distancia_vizinho < melhor_distancia_vizinho:
                melhor_vizinho = vizinho
                melhor_distancia_vizinho = distancia_vizinho
                encontrou_melhor = True

        if encontrou_melhor:
            rota_atual = melhor_vizinho
            distancia_atual = melhor_distancia_vizinho
            historico_distancias.append(distancia_atual)
            iter_sem_melhora = 0
        else:
            iter_sem_melhora += 1
            historico_distancias.append(distancia_atual)

    tempo_fim = time.time()
    return {
        "melhor_rota": rota_atual,
        "menor_distancia": distancia_atual,
        "tempo_execucao_s": tempo_fim - tempo_inicio,
        "total_iteracoes": iteracoes,
        "historico_distancias": historico_distancias
    }

# --- Execução ---
if __name__ == "__main__":
    print("Executando Hill Climbing para o Problema do Caixeiro Viajante...")

    resultado_hc = hill_climbing_tsp(CIDADES, CIDADE_INICIAL, max_iter_sem_melhora=200)

    print("\n--- Resultados Hill Climbing (TSP) ---")
    print(f"Melhor Rota Encontrada: {' -> '.join(map(str, resultado_hc['melhor_rota']))}")
    if math.isinf(resultado_hc['menor_distancia']):
        print("Menor Distância: Infinita (rota inválida encontrada devido a conexões ausentes no grafo)")
    else:
        print(f"Menor Distância: {resultado_hc['menor_distancia']:.2f}")
    print(f"Tempo de Execução: {resultado_hc['tempo_execucao_s']:.4f} segundos")
    print(f"Total de Iterações: {resultado_hc['total_iteracoes']}")

    # Garante que a pasta 'results' exista
    os.makedirs("results", exist_ok=True)

    resultados_json = {
        "algoritmo": "Hill Climbing (Permutação)",
        "problema": "TSP",
        "grafo": GRAFO,
        "melhor_rota": resultado_hc['melhor_rota'],
        "menor_distancia": resultado_hc['menor_distancia'] if not math.isinf(resultado_hc['menor_distancia']) else 'infinito',
        "tempo_execucao_s": resultado_hc['tempo_execucao_s'],
        "total_iteracoes": resultado_hc['total_iteracoes'],
        "historico_distancias": resultado_hc['historico_distancias'] if not math.isinf(resultado_hc['menor_distancia']) else []
    }

    with open("results/resultado_hc_tsp.json", "w") as f:
        json.dump(resultados_json, f, indent=4)

    print("\nResultados salvos em results/resultado_hc_tsp.json")

    try:
      import networkx as nx
      import matplotlib.pyplot as plt

      # Criar grafo com NetworkX
      G = nx.Graph()

      # Adicionar arestas com pesos
      for origem, vizinhos in GRAFO.items():
          for destino, peso in vizinhos.items():
              G.add_edge(origem, destino, weight=peso)

      # Posicionar os nós automaticamente (ou você pode definir posições fixas)
      pos = nx.spring_layout(G, seed=42)  # Layout com aparência agradável e consistente

      # Plotar o grafo base
      plt.figure(figsize=(12, 8))
      nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue')
      nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
      nx.draw_networkx_edges(G, pos, alpha=0.3)  # Arestas cinza claras
      nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)}, font_size=8)

      # Desenhar a melhor rota com setas e cor destacada
      rota = resultado_hc['melhor_rota']
      edges_rota = list(zip(rota, rota[1:]))

      nx.draw_networkx_edges(G, pos, edgelist=edges_rota, edge_color='red', width=2.5, arrows=True, style='solid')

      plt.title("Melhor Rota Encontrada - Hill Climbing (TSP)")
      plt.axis('off')
      plt.tight_layout()
      plt.savefig('results/melhor_rota_grafo.png')
      plt.show()
      print("Gráfico de rota salvo em results/melhor_rota_grafo.png")

    except ImportError:
      print("\nAviso: É necessário instalar networkx e matplotlib.")
      print("Instale com: pip install networkx matplotlib")

