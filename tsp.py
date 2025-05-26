import AlgoritmoGenetico as ag
import time
import matplotlib.pyplot as plt

GRAFO = {
  1: {2: 20, 8: 29, 12: 29, 13: 37},
  2: {1: 20, 3: 25, 8: 28, 12: 39},
  3: {2: 25, 4: 25, 8: 30, 13: 54},
  4: {3: 25, 6: 32, 5: 39, 9: 23, 10: 33, 7: 42, 14:56},
  5: {4: 39, 6: 12, 7:26, 10: 19},
  6: {4: 32, 5: 12, 7: 17, 10: 35, 11: 30},
  7: {4: 42, 5:26, 6: 17, 11: 38},
  8: {1: 29, 2: 28, 3: 30, 12: 25, 13: 22},
  9: {4: 23, 10: 26, 13: 34, 14: 34, 16: 43},
  10: {4: 33, 5: 19, 6: 35, 9: 26, 11: 24, 14: 30, 15: 19},
  11: {6: 30, 7: 38, 10: 24, 15: 26, 18:36},
  12: {1: 29, 2: 39, 8: 25, 13: 27, 16: 43},
  13: {1: 37, 3: 54, 8: 22, 9: 34, 12: 27, 14: 24, 16: 19},
  14: {4: 56, 9: 34, 10: 30, 13: 24, 15: 20, 16: 19, 17: 17},
  15: {10: 19, 11: 26, 14: 20, 17: 18, 18: 21},
  16: {9: 43, 12: 43, 13: 19, 14: 19, 17: 26},
  17: {14: 17, 15: 18, 16: 26, 18: 15},
  18: {11: 36, 15: 21, 17: 15}
}

'''for i in GRAFO:
  print(f'Vertice: {i}')
  print('\tVisinhos')
  for j in GRAFO[i]:
    print(f'\t Vertice {j} - Distancia {GRAFO[i][j]}')
  print()'''

print('---------------- Algoritmo Genético ----------------')

tamanho_da_populacao = 200
geracoes = 500
taxa_de_mutacao = 0.001

algoritmo_genetico = ag.AlgoritmoGenetico(GRAFO, tamanho_da_populacao, geracoes, taxa_de_mutacao)

inicio_ag = time.time()
_, _, melhores_custos= algoritmo_genetico.iniciar()
fim_ag = time.time()

tempo_ag = fim_ag - inicio_ag
print(f'Tempo de execução algoritmo genético: {tempo_ag:.2f} segundos')

plt.figure(figsize=(10, 5))
plt.plot(melhores_custos, label="Melhor custo por geração", color="blue")
plt.title("Convergência do Algoritmo Genético (TSP)")
plt.xlabel("Geração")
plt.ylabel("Custo da melhor rota")
plt.legend()
plt.tight_layout()
plt.show()

print('-----------------------------------------------------')

dim = 5
intervalo = (-500, 500)
inicio_ag = time.time()
_, _, melhores_custos= algoritmo_genetico.iniciar_continuo(dim, intervalo)
fim_ag = time.time()

tempo_ag = fim_ag - inicio_ag
print(f'Tempo de execução algoritmo genético com schwefel: {tempo_ag:.2f} segundos')

plt.figure(figsize=(10, 5))
plt.plot(melhores_custos, label="Melhor custo por geração", color="blue")
plt.title("Convergência do Algoritmo Genético (TSP) com schwefel")
plt.xlabel("Geração")
plt.ylabel("Custo da melhor rota")
plt.legend()
plt.tight_layout()
plt.show()
      