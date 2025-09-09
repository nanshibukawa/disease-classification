# Multi-Parameter Experiment System

Este sistema permite executar experimentos com múltiplas combinações de parâmetros para otimizar seu modelo de classificação de doenças.

## Arquivos Criados

1. **main_multi_params.py** - Sistema avançado de experimentos multi-parâmetros
2. **main.py** - Versão atualizada com suporte básico a multi-parâmetros
3. **experiment_configs.py** - Configurações de exemplo para experimentos
4. **run_experiments.sh** - Script bash para executar experimentos

## Como Usar

### Método 1: Experimento Simples (main.py)

```bash
# Executar experimento multi-parâmetros
export MULTI_PARAM_MODE=true
export EXPERIMENT_NAME="my_experiment"
python main.py

# Executar pipeline único (modo normal)
export MULTI_PARAM_MODE=false
python main.py
```

### Método 2: Experimento Avançado (main_multi_params.py)

```python
# Edite o arquivo main_multi_params.py para definir suas variações de parâmetros
param_variations = {
    'LEARNING_RATE': [0.01, 0.003, 0.001],
    'MODEL_NAME': ['VGG16', 'VGG19', 'MobileNetV3Large'],
    'BATCH_SIZE': [16, 32],
    'EPOCHS': [20, 30]
}

# Execute o experimento
python main_multi_params.py
```

### Método 3: Script Automatizado

```bash
# Execute todos os tipos de experimentos
./run_experiments.sh
```

## Configurações de Exemplo

### Experimento de Taxa de Aprendizado
```python
param_variations = {
    'LEARNING_RATE': [0.1, 0.01, 0.001, 0.0001],
    'MODEL_NAME': ['MobileNetV3Large'],
    'BATCH_SIZE': [16]
}
```

### Comparação de Modelos
```python
param_variations = {
    'LEARNING_RATE': [0.01, 0.001],
    'MODEL_NAME': ['VGG16', 'VGG19', 'MobileNetV3Large'],
    'BATCH_SIZE': [16, 32]
}
```

### Otimização de Batch Size
```python
param_variations = {
    'LEARNING_RATE': [0.001],
    'MODEL_NAME': ['MobileNetV3Large'],
    'BATCH_SIZE': [8, 16, 32, 64]
}
```

## Status do Sistema

✅ **SISTEMA FUNCIONANDO CORRETAMENTE** (Atualizado: 29/07/2025 21:27)

### 🔧 **Correções Aplicadas:**
- ✅ **Conflito de MLflow Runs**: Resolvido - sistema não tenta criar run aninhado
- ✅ **Logging de Modelo TensorFlow**: Corrigido - usa artifacts ao invés de model logging
- ✅ **Tratamento de Erros**: Melhorado - logs detalhados e recuperação robusta
- ✅ **MLflow Server**: Funcionando na porta 5001 (http://127.0.0.1:5001)

### 📊 **Última Verificação:**
- ✅ Conexão MLflow: OK (porta 5001)
- ✅ Logging de parâmetros: OK
- ✅ Logging de métricas: OK
- ✅ Extração de scores.json: OK
- ✅ Logging de artefatos: OK
- ✅ Prevenção de conflitos: OK
- ✅ Tratamento de erros: Robusto

### 🎯 **Sistema Pronto Para Uso**

## Recursos

### MLflow Integration
- Todos os experimentos são automaticamente logados no MLflow
- Acesse http://127.0.0.1:5000 para visualizar os resultados
- Parâmetros e métricas são rastreados automaticamente

### Resultados Salvos
- `experiment_results_*.json` - Resultados detalhados de cada experimento
- `experiment_summary_*.json` - Resumo dos experimentos
- `params_used_*.yaml` - Parâmetros usados em cada experimento

### Gerenciamento de Memória
- Limpeza automática de memória GPU entre experimentos
- Backup e restauração automática dos parâmetros originais

## Parâmetros Suportados

- `LEARNING_RATE` - Taxa de aprendizado
- `MODEL_NAME` - Nome do modelo (VGG16, VGG19, MobileNetV3Large)
- `BATCH_SIZE` - Tamanho do lote
- `EPOCHS` - Número de épocas
- `IMAGE_SIZE` - Tamanho da imagem
- `AUGMENTATION` - Usar aumentação de dados
- `CLASSES` - Número de classes
- `WEIGHTS` - Pesos pré-treinados

## Exemplo de Uso Prático

### 1. Verificação Rápida do Sistema
```bash
# Verificar se tudo está funcionando
python test_fixed_version.py
```

### 2. Experimento de Demonstração
```bash
# Executar experimento interativo com poucos parâmetros
python run_demo_experiment.py
```

### 3. Experimento Completo
```bash
# Editar main_multi_params.py para definir parâmetros desejados
# Depois executar:
python main_multi_params.py
```

### 4. Monitoramento em Tempo Real
```bash
# Em um terminal separado, manter MLflow rodando:
source .venv/bin/activate
mlflow ui --host 127.0.0.1 --port 5001

# Acessar no navegador: http://127.0.0.1:5001
```

## Exemplo de Uso Prático (Antigo)

```bash
# 1. Teste rápido para debug
export MULTI_PARAM_MODE=true
export EXPERIMENT_NAME="debug_test"
# Edite main.py para usar debug_params
python main.py

# 2. Experimento completo
python main_multi_params.py

# 3. Visualizar resultados no MLflow
mlflow ui --host 127.0.0.1 --port 5000
```

## Monitoramento

- Verifique os logs para acompanhar o progresso
- Use MLflow UI para visualizar métricas em tempo real
- Arquivos JSON fornecem backup dos resultados

## Troubleshooting

### Problema: MLflow não conecta
```bash
# Solução: Iniciar o servidor MLflow na porta 5001
source .venv/bin/activate
mlflow ui --host 127.0.0.1 --port 5001
```

### Problema: Conflito de MLflow Runs
**Erro**: `Run with UUID ... is already active`
**Status**: ✅ **CORRIGIDO** - Sistema detecta run ativo e evita criar runs aninhados

### Problema: Erro de logging de modelo TensorFlow
**Erro**: `Invalid filepath extension for saving`
**Status**: ✅ **CORRIGIDO** - O sistema agora salva modelos como artefatos ao invés de usar mlflow.tensorflow.log_model

### Problema: Scores.json não encontrado
**Solução**: Execute pelo menos uma avaliação completa antes de rodar experimentos multi-parâmetros
```bash
python main.py  # Para gerar scores.json inicial
```

### Verificação Rápida do Sistema
```bash
# Teste básico
python test_simple_mlflow.py

# Teste completo de métricas
python test_metrics_extraction.py

# Teste da versão corrigida
python test_fixed_version.py
```

## Recuperação de Falhas

- Em caso de falha, os parâmetros originais são restaurados
- Experimentos individuais não afetam outros experimentos
- Resultados são salvos incrementalmente
