# Trabalho IA – Desafio Enchentes RS (Etapa 1)

Implementação da Etapa 1 do desafio de IA para resposta às enchentes do Rio Grande do Sul.  
Treina e avalia dois modelos de classificação com **Pandas + scikit-learn**:

| Agente | Dataset | Alvo | Classes |
|--------|---------|------|---------|
| Monitoramento (reativo) | `dataset_m.csv` | `risco` | 0 – 5 |
| Triagem (reativo) | `dataset_t.csv` | `prioridade` | 0 – 8 |

---

## Pré-requisitos

- Python 3.9+
- pip

## Instalação

```bash
pip install -r requirements.txt
```

## Datasets

Os arquivos CSV **não estão incluídos no repositório** por serem dados sensíveis/grandes.  
Coloque-os em qualquer pasta acessível e passe os caminhos via argumento (ou na raiz do projeto com os nomes padrão):

```
dataset_m.csv   # dataset do agente de monitoramento
dataset_t.csv   # dataset do agente de triagem
```

### Colunas esperadas

**dataset_m.csv**

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `data` | datetime | Data/hora da observação |
| `local` | int | Zona mapeada (1–50) |
| `velocidade_vento` | float | 0–100 |
| `terreno` | categórico | navegavel, lama, seco, asfalto molhado, telhado |
| `correnteza` | float | 0–100 |
| `visibilidade` | float | percentual |
| `obstaculos` | bool | presença de obstáculos |
| `risco` (**alvo**) | int | 0 (sem risco) a 5 |

**dataset_t.csv**

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `data` | datetime | Data/hora da observação |
| `local` | int | Zona mapeada (1–50) |
| `risco` | int | 0–5, importado do agente de monitoramento |
| `origem` | categórico | whatsapp, 190, instagram |
| `espera` | float | minutos desde o primeiro chamado (0–120) |
| `n_pessoas` | int | número estimado de vítimas |
| `vulneraveis` | bool | idosos, crianças ou PCD |
| `nivel_agua` | categórico | tornozelo, cintura, teto |
| `necessidade_medica` | bool | feridos graves ou dependentes de aparelhos |
| `sentimento` | categórico | calmo, nervoso, panico |
| `confiavel` | bool | chamado plausível |
| `prioridade` (**alvo**) | int | 0 a 8 (8 = máxima prioridade) |

## Execução

```bash
# Caminhos padrão (dataset_m.csv e dataset_t.csv na pasta atual)
python desafio1_etapa1.py

# Caminhos customizados
python desafio1_etapa1.py --monitoring data/dataset_m.csv --triage data/dataset_t.csv

# Salvar resultados em arquivo
python desafio1_etapa1.py --output results/
```

## Saídas

Para cada modelo, o script imprime:

- **Acurácia**
- **F1 score (weighted)**
- **Matriz de confusão**
- *Classification report* completo (precision, recall, F1 por classe)

Se `--output DIR` for fornecido, os resultados são salvos como arquivos `.txt` na pasta `DIR`.

## Pré-processamento aplicado

1. Parse da coluna `data` → extração de `hora` e `dia_semana`; remoção do datetime bruto.
2. Remoção de colunas 100% nulas (artefatos de cabeçalho).
3. Conversão de booleanos textuais (`"True"/"False"`) para `0/1`.
4. Variáveis numéricas: imputação pela mediana.
5. Variáveis categóricas: imputação pela moda + OneHotEncoding (com `handle_unknown="ignore"`).

## Modelo

Utiliza **RandomForestClassifier** do scikit-learn com `class_weight="balanced"` para lidar com possível desbalanceamento de classes.
