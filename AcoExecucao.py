import ACO as aco
import ACOSchwefel as acos

if __name__ == '__main__':

    # Exemplo de grafo em formato dicionário de dicionários
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
    
    print("=== Exemplo ACO para Função TSP ===")

    # Cria instância do ACO
    aco = aco.ACO_TSP(
        grafo_adj=GRAFO,
        num_formigas=50,
        num_iteracoes=100,
        alfa=1.0,
        beta=2.0,
        taxa_evaporacao=0.5,
        Q_constante=100.0
    )
    
    # Resolve o problema
    melhor_rota, menor_distancia, historico = aco.resolver(cidade_inicial=1)
    
    # Plota convergência
    aco.plotar_convergencia()

    print("=== Exemplo ACO para Função Schwefel ===")
    
    # Cria instância do ACO para Schwefel
    aco_schwefel = acos.ACO_Schwefel(
        dimensoes=5,
        num_formigas_por_iter=20,
        num_iteracoes=50,
        tamanho_arquivo_solucoes=10,
        q_seletividade=0.1,
        xi_exploracao=0.85,
        limite_inferior=-500,
        limite_superior=500
    )
    
    # Resolve o problema
    melhor_solucao, melhor_custo, historico = aco_schwefel.resolver()
    
    # Plota convergência
    aco_schwefel.plotar_convergencia()