#!/bin/bash

# Script para executar experimentos multi-parâmetros
# Atualizado: 29/07/2025 - Correção de conflitos MLflow

echo "=== Multi-Parameter Experiment Runner ==="
echo "Versão: 2.0 - Com correções de MLflow"

# Verificar se MLflow está rodando
echo "Verificando servidor MLflow..."
if curl -s http://127.0.0.1:5001 > /dev/null; then
    echo "✅ MLflow server está rodando"
else
    echo "⚠️  MLflow server não encontrado. Iniciando..."
    source .venv/bin/activate
    mlflow ui --host 127.0.0.1 --port 5001 &
    sleep 5
    echo "✅ MLflow server iniciado"
fi

# Ativar ambiente virtual
source .venv/bin/activate

# Teste rápido do sistema
echo "0. Executando teste de verificação..."
python test_fixed_version.py
if [ $? -ne 0 ]; then
    echo "❌ Teste de verificação falhou. Abortando."
    exit 1
fi
echo "✅ Sistema verificado com sucesso!"

# Opção 1: Executar experimento simples
echo "1. Running simple multi-parameter experiment..."
export MULTI_PARAM_MODE=true
export EXPERIMENT_NAME="simple_grid_search_$(date +%Y%m%d_%H%M%S)"
python main.py

# Opção 2: Executar experimento de demonstração (interativo)
echo "2. Running demonstration experiment..."
python run_demo_experiment.py

# Opção 3: Executar experimento avançado
echo "3. Running advanced multi-parameter experiment..."
export EXPERIMENT_NAME="advanced_experiment_$(date +%Y%m%d_%H%M%S)"
python main_multi_params.py

# Opção 4: Executar pipeline único (modo normal)
echo "4. Running single pipeline..."
export MULTI_PARAM_MODE=false
python main.py

echo "=== Experiments completed! ==="
echo "🔗 Visualize resultados em: http://127.0.0.1:5001"
