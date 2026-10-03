# Meu Modelo GGUF

Treina um modelo leve (Qwen2.5-0.5B-Instruct por padrão) com LoRA a partir do
dataset `train.jsonl` e exporta o resultado em formato **GGUF**, pronto para
rodar no llama.cpp, Ollama ou LM Studio.

## Como treinar

1. Abra a aba **Actions** do repositório.
2. Selecione o workflow **"Treinar Modelo e Exportar GGUF"**.
3. Clique em **Run workflow**, ajuste os parâmetros (ou deixe os padrões) e confirme.
4. Aguarde a conclusão (~3 a 4 horas com os valores padrão, em CPU).
5. Baixe o artefato **`modelo-gguf`** (contém `modelo_final.gguf`) na seção
   **Artifacts** da execução.

> **Dica:** antes de uma rodada longa, faça um teste de 10 minutos com
> `max_exemplos = 20` e `tamanho_maximo = 512`. Isso percorre o pipeline
> inteiro (treino → merge → GGUF) e confirma que tudo está funcionando.

### Parâmetros do workflow

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `modelo_base` | `Qwen/Qwen2.5-0.5B-Instruct` | Modelo base no Hugging Face |
| `tamanho_maximo` | `1024` | Tokens por exemplo (prompt + resposta) |
| `epocas` | `1` | Épocas de treino |
| `max_exemplos` | `0` | Limita o dataset (0 = todos). Use `20` para testar |
| `quantizacao` | `q8_0` | Formato do GGUF (`q8_0`, `f16`, `bf16`, `f32`) |
| `limite_horas` | `4` | Encerra o treino com elegância e exporta mesmo assim |
| `llama_cpp_ref` | `v0.5.0` | Tag do llama.cpp usada na conversão |

### Artefatos gerados

- **`modelo-gguf`** → `modelo_final.gguf`: modelo completo (base + LoRA já fundido).
- **`adaptador-lora`** → pasta `lora/` + `adaptador_lora.gguf`: só o adaptador,
  para aplicar sobre o modelo base original. É enviado **mesmo se a conversão
  falhar**, para que horas de treino nunca sejam perdidas.

### Usando o modelo

```bash
# llama.cpp
llama-cli -m modelo_final.gguf -p "Explain a race condition"

# Ollama
printf 'FROM ./modelo_final.gguf\n' > Modelfile
ollama create meu-modelo -f Modelfile && ollama run meu-modelo
```

## Rodando localmente

```bash
pip install torch transformers peft accelerate
git clone --depth 1 --branch v0.5.0 https://github.com/ggml-org/llama.cpp.git
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

python scripts/treinar.py --apenas-validar        # valida o dataset (segundos)
python scripts/treinar.py --max-exemplos 20       # rodada curta de teste
python scripts/treinar.py                         # rodada completa
python scripts/treinar.py --help                  # todas as opções
```

## Estrutura

- `.github/workflows/treinar.yml` — workflow de treino + exportação GGUF (manual)
- `.github/workflows/validar.yml` — checagem rápida do dataset a cada push
- `scripts/treinar.py` — treino LoRA, merge e conversão para GGUF
- `scripts/testes.py` — testes das funções de dados (rodam sem rede)
- `train.jsonl` — dataset de treino (997 conversas no formato `messages`)

## Formato do dataset

Uma conversa por linha; a última mensagem precisa ser do `assistant`.
Campos extras como `domain` e `meta` são ignorados automaticamente.

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "domain": "web_security"}
```

## Histórico de correções

### 1. `FileNotFoundError: 'Qwen/Qwen2.5-0.5B-Instruct'` no fim do job

O job treinava por ~51 minutos e só então quebrava na conversão:

```
File "llama.cpp/conversion/base.py", line 1244, in get_model_part_names
    for filename in os.listdir(dir_model):
FileNotFoundError: [Errno 2] No such file or directory: 'Qwen/Qwen2.5-0.5B-Instruct'
subprocess.CalledProcessError: Command '['python', 'llama.cpp/convert_lora_to_gguf.py',
  '--base', 'Qwen/Qwen2.5-0.5B-Instruct', ...]' returned non-zero exit status 1.
```

**Causa:** o parâmetro `--base` do `convert_lora_to_gguf.py` espera um
**diretório local** com o `config.json` do modelo base — não um ID do Hugging
Face. O script tentou listar o diretório `Qwen/Qwen2.5-0.5B-Instruct`, que não
existe no runner. (Para um ID remoto, o parâmetro correto seria `--base-model-id`.)

**Correção:** o pipeline agora funde o LoRA no modelo base e converte com
`convert_hf_to_gguf.py`, gerando um GGUF autossuficiente. O adaptador isolado
também é exportado, aí sim com um diretório local de configuração em `--base`.

### 2. O treino não via as respostas (`max_length=256`)

Os exemplos têm, em média, ~10.300 caracteres, sendo ~9.300 só na resposta do
assistant. Com truncamento em 256 tokens, o modelo via apenas o *system prompt*
(que é idêntico nas 997 linhas) — ou seja, o treino decorava o cabeçalho e
nunca chegava na resposta.

**Correção:** `tamanho_maximo` passou a 1024 tokens e virou parâmetro do
workflow; além disso o prompt é mascarado (`-100`), de modo que a loss é
calculada **apenas sobre os tokens da resposta**.

### 3. Outras melhorias de robustez

- **Falha rápido:** o dataset é validado em segundos, antes de instalar torch
  e treinar.
- **Nada se perde:** o adaptador LoRA é enviado como artefato mesmo quando um
  passo posterior falha (`if: always()`).
- **Sem estouro de tempo:** `limite_horas` interrompe o treino com elegância e
  ainda exporta o GGUF, em vez de o job ser morto pelo timeout sem produzir nada.
- **Instalação enxuta:** torch é instalado na build de CPU (evita ~5 GB de
  pacotes CUDA) e as versões seguem os pinos do llama.cpp, sem reinstalações.
- **Versões fixadas:** o llama.cpp é clonado por tag (`v0.5.0`), então uma
  mudança no `master` não quebra mais a conversão.
- **Actions atualizadas:** `checkout@v5` / `setup-python@v6` / `upload-artifact@v5`
  (fim do aviso de depreciação do Node 20).
