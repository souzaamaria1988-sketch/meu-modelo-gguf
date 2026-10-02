# Meu Modelo GGUF

Repositório criado automaticamente para treinar um modelo leve (Qwen2.5-0.5B-Instruct) com LoRA a partir do dataset `train.jsonl` e exportar em formato GGUF.

## Como treinar

1. Abra a aba **Actions** do repositório.
2. Selecione o workflow **"Treinar Modelo e Exportar GGUF"**.
3. Clique em **Run workflow** → **Run workflow**.
4. Aguarde a conclusão (pode levar de 20 a 60 minutos).
5. Baixe o arquivo `modelo_final.gguf` na seção **Artifacts** da execução.

## Estrutura

- `.github/workflows/treinar.yml` — workflow do GitHub Actions
- `train.jsonl` — dataset de treinamento no formato de mensagens
