#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Treina um adaptador LoRA e exporta o modelo final em formato GGUF.

Fluxo:
  1. valida e carrega o dataset `train.jsonl` (formato de mensagens de chat);
  2. treina um adaptador LoRA sobre o modelo base (CPU);
  3. salva o adaptador (`lora/`);
  4. funde (merge) o adaptador no modelo base e converte para GGUF com o
     `convert_hf_to_gguf.py` do llama.cpp;
  5. opcionalmente também exporta o adaptador isolado em GGUF.

Exemplos:
  python scripts/treinar.py --apenas-validar
  python scripts/treinar.py --max-exemplos 20 --tamanho-maximo 512
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("treinar")

PAPEIS_VALIDOS = {"system", "user", "assistant", "tool"}


# --------------------------------------------------------------------------- #
# Dataset (funções puras: não dependem de torch/transformers)
# --------------------------------------------------------------------------- #
def validar_mensagens(dado: Any, linha: int) -> list[dict[str, str]]:
    """Valida uma linha do JSONL e devolve apenas o campo `messages`.

    Colunas extras (`domain`, `meta`, ...) são descartadas aqui — é por isso
    que não é preciso remover colunas depois, no dataset.
    """
    if not isinstance(dado, dict):
        raise ValueError(f"linha {linha}: esperado um objeto JSON, veio {type(dado).__name__}")
    if "messages" not in dado:
        raise ValueError(f"linha {linha}: campo 'messages' ausente")

    mensagens = dado["messages"]
    if not isinstance(mensagens, list) or not mensagens:
        raise ValueError(f"linha {linha}: 'messages' precisa ser uma lista não vazia")

    limpas: list[dict[str, str]] = []
    for i, msg in enumerate(mensagens):
        if not isinstance(msg, dict):
            raise ValueError(f"linha {linha}: mensagem {i} não é um objeto")
        papel = msg.get("role")
        conteudo = msg.get("content")
        if papel not in PAPEIS_VALIDOS:
            raise ValueError(f"linha {linha}: mensagem {i} tem role inválido: {papel!r}")
        if not isinstance(conteudo, str):
            raise ValueError(f"linha {linha}: mensagem {i} tem 'content' que não é texto")
        limpas.append({"role": papel, "content": conteudo})

    if limpas[-1]["role"] != "assistant":
        raise ValueError(f"linha {linha}: a última mensagem precisa ser do 'assistant'")
    return limpas


def ler_exemplos(caminho: Path, max_exemplos: int = 0) -> list[list[dict[str, str]]]:
    """Lê o JSONL e devolve a lista de conversas válidas."""
    if not caminho.is_file():
        raise FileNotFoundError(f"dataset não encontrado: {caminho}")

    exemplos: list[list[dict[str, str]]] = []
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                dado = json.loads(linha)
            except json.JSONDecodeError as erro:
                raise ValueError(f"linha {numero}: JSON inválido ({erro})") from erro
            exemplos.append(validar_mensagens(dado, numero))
            if max_exemplos and len(exemplos) >= max_exemplos:
                break

    if not exemplos:
        raise ValueError(f"nenhum exemplo utilizável em {caminho}")
    return exemplos


def resumir_dataset(exemplos: list[list[dict[str, str]]]) -> dict[str, Any]:
    """Estatísticas simples usadas no modo --apenas-validar."""
    tamanhos = [sum(len(m["content"]) for m in conversa) for conversa in exemplos]
    respostas = [len(conversa[-1]["content"]) for conversa in exemplos]
    return {
        "exemplos": len(exemplos),
        "chars_por_exemplo_media": round(sum(tamanhos) / len(tamanhos)),
        "chars_por_exemplo_max": max(tamanhos),
        "chars_resposta_media": round(sum(respostas) / len(respostas)),
    }


# --------------------------------------------------------------------------- #
# Tokenização
# --------------------------------------------------------------------------- #
def tokenizar_exemplo(tok, mensagens: list[dict[str, str]], tamanho_maximo: int) -> dict[str, list[int]] | None:
    """Aplica o chat template e mascara o prompt (loss só na resposta).

    Devolve None quando o prompt sozinho já ocupa todo o `tamanho_maximo`
    (nesse caso não sobraria nenhum token de resposta para aprender).
    """
    texto_completo = tok.apply_chat_template(mensagens, tokenize=False)
    texto_prompt = tok.apply_chat_template(mensagens[:-1], tokenize=False, add_generation_prompt=True)

    ids = tok(texto_completo, truncation=True, max_length=tamanho_maximo, add_special_tokens=False)["input_ids"]
    ids_prompt = tok(texto_prompt, add_special_tokens=False)["input_ids"]

    n_prompt = min(len(ids_prompt), len(ids))
    if n_prompt >= len(ids):
        return None

    rotulos = [-100] * n_prompt + list(ids[n_prompt:])
    return {"input_ids": list(ids), "attention_mask": [1] * len(ids), "labels": rotulos}


def montar_lote(exemplos: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Padding dinâmico: completa até o maior item do batch (sem torch)."""
    maior = max(len(e["input_ids"]) for e in exemplos)
    entradas, mascaras, rotulos = [], [], []
    for exemplo in exemplos:
        faltam = maior - len(exemplo["input_ids"])
        entradas.append(list(exemplo["input_ids"]) + [pad_token_id] * faltam)
        mascaras.append(list(exemplo["attention_mask"]) + [0] * faltam)
        rotulos.append(list(exemplo["labels"]) + [-100] * faltam)
    return {"input_ids": entradas, "attention_mask": mascaras, "labels": rotulos}


class ColetorCausal:
    """Coletor do Trainer: converte o lote montado em tensores."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, exemplos: list[dict[str, list[int]]]):
        import torch

        lote = montar_lote(exemplos, self.pad_token_id)
        return {chave: torch.tensor(valor, dtype=torch.long) for chave, valor in lote.items()}


# --------------------------------------------------------------------------- #
# Conversão para GGUF
# --------------------------------------------------------------------------- #
def executar(comando: list[str]) -> None:
    log.info("$ %s", " ".join(comando))
    subprocess.run(comando, check=True)


def converter_modelo_para_gguf(llama_cpp: Path, dir_modelo: Path, saida: Path, quantizacao: str) -> None:
    script = llama_cpp / "convert_hf_to_gguf.py"
    if not script.is_file():
        raise FileNotFoundError(f"script de conversão não encontrado: {script}")
    executar([sys.executable, str(script), str(dir_modelo),
              "--outfile", str(saida), "--outtype", quantizacao])


def converter_adaptador_para_gguf(llama_cpp: Path, dir_base: Path, dir_lora: Path, saida: Path) -> None:
    """Exporta o adaptador LoRA isolado.

    IMPORTANTE: `--base` precisa ser um DIRETÓRIO LOCAL com o config.json do
    modelo base. Passar um ID do Hugging Face aqui (ex.: 'Qwen/Qwen2.5-0.5B-Instruct')
    causa `FileNotFoundError` — para IDs remotos o parâmetro correto é
    `--base-model-id`.
    """
    script = llama_cpp / "convert_lora_to_gguf.py"
    if not script.is_file():
        raise FileNotFoundError(f"script de conversão não encontrado: {script}")
    executar([sys.executable, str(script), "--base", str(dir_base), str(dir_lora),
              "--outfile", str(saida), "--outtype", "f16"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina LoRA e exporta GGUF")
    p.add_argument("--dataset", type=Path, default=Path("train.jsonl"))
    p.add_argument("--modelo-base", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--llama-cpp", type=Path, default=Path("llama.cpp"))
    p.add_argument("--saida", type=Path, default=Path("modelo_final.gguf"))
    p.add_argument("--saida-adaptador", type=Path, default=Path("adaptador_lora.gguf"))
    p.add_argument("--dir-lora", type=Path, default=Path("lora"))
    p.add_argument("--dir-merged", type=Path, default=Path("modelo_merged"))
    p.add_argument("--dir-base-config", type=Path, default=Path("base_config"))
    p.add_argument("--tamanho-maximo", type=int, default=1024,
                   help="tokens por exemplo (prompt + resposta). Padrão: 1024")
    p.add_argument("--epocas", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--acumulo", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-exemplos", type=int, default=0, help="0 = todos")
    p.add_argument("--quantizacao", default="q8_0", choices=["q8_0", "f16", "bf16", "f32"])
    p.add_argument("--limite-horas", type=float, default=4.0,
                   help="interrompe o treino com elegância após N horas e exporta mesmo assim")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--apenas-validar", action="store_true",
                   help="só valida o dataset (rápido, não precisa de torch)")
    p.add_argument("--sem-adaptador-gguf", action="store_true",
                   help="não exportar o adaptador LoRA isolado em GGUF")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    log.info("Lendo dataset: %s", args.dataset)
    exemplos = ler_exemplos(args.dataset, args.max_exemplos)
    resumo = resumir_dataset(exemplos)
    log.info("Dataset OK: %s", json.dumps(resumo, ensure_ascii=False))

    if args.apenas_validar:
        log.info("Modo --apenas-validar: nada mais a fazer.")
        return 0

    # Imports pesados só depois da validação (falha rápido e barato).
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model

    set_seed(args.semente)
    torch.set_num_threads(torch.get_num_threads())
    log.info("torch %s | threads=%d", torch.__version__, torch.get_num_threads())

    log.info("Baixando tokenizer e modelo base: %s", args.modelo_base)
    tok = AutoTokenizer.from_pretrained(args.modelo_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    try:  # transformers >= 4.56 usa `dtype`; versões antigas usam `torch_dtype`
        modelo = AutoModelForCausalLM.from_pretrained(
            args.modelo_base, dtype=torch.float32, low_cpu_mem_usage=True)
    except TypeError:
        modelo = AutoModelForCausalLM.from_pretrained(
            args.modelo_base, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    modelo.config.use_cache = False

    # Guarda config + tokenizer do modelo base: é o diretório local exigido
    # pelo `convert_lora_to_gguf.py --base`.
    if not getattr(modelo.config, "architectures", None):
        modelo.config.architectures = [modelo.__class__.__name__]
    shutil.rmtree(args.dir_base_config, ignore_errors=True)
    args.dir_base_config.mkdir(parents=True, exist_ok=True)
    modelo.config.save_pretrained(args.dir_base_config)
    tok.save_pretrained(args.dir_base_config)

    # ---------------------------------------------------------------- dados --
    log.info("Tokenizando (tamanho máximo = %d tokens, loss só na resposta)...", args.tamanho_maximo)
    dados: list[dict[str, list[int]]] = []
    descartados = 0
    for conversa in exemplos:
        item = tokenizar_exemplo(tok, conversa, args.tamanho_maximo)
        if item is None:
            descartados += 1
            continue
        dados.append(item)

    if not dados:
        log.error("Todos os exemplos foram descartados: o prompt sozinho já ocupa "
                  "%d tokens. Aumente --tamanho-maximo.", args.tamanho_maximo)
        return 1
    if descartados:
        log.warning("%d exemplo(s) descartado(s): prompt maior que --tamanho-maximo.", descartados)

    tokens_treinados = sum(sum(1 for r in d["labels"] if r != -100) for d in dados)
    log.info("Exemplos de treino: %d | tokens supervisionados: %d (média %d por exemplo)",
             len(dados), tokens_treinados, tokens_treinados // len(dados))

    # --------------------------------------------------------------- treino --
    modelo = get_peft_model(modelo, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    modelo.print_trainable_parameters()

    class LimiteDeTempo(TrainerCallback):
        """Encerra o treino antes do timeout do runner para garantir o export."""

        def __init__(self, segundos: float) -> None:
            self.segundos = segundos
            self.inicio = time.time()
            self.avisou = False

        def on_step_end(self, args_, state, control, **kwargs):  # noqa: D401
            if not self.avisou and time.time() - self.inicio > self.segundos:
                self.avisou = True
                log.warning("Limite de %.2f h atingido no passo %d: encerrando o treino "
                            "para exportar o modelo.", self.segundos / 3600, state.global_step)
                control.should_training_stop = True
            return control

    treinador = Trainer(
        model=modelo,
        args=TrainingArguments(
            output_dir="out",
            num_train_epochs=args.epocas,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.acumulo,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=args.semente,
            remove_unused_columns=False,
            dataloader_num_workers=0,
        ),
        train_dataset=dados,
        data_collator=ColetorCausal(tok.pad_token_id),
        callbacks=[LimiteDeTempo(args.limite_horas * 3600)],
    )

    inicio = time.time()
    treinador.train()
    log.info("Treino concluído em %.1f min", (time.time() - inicio) / 60)

    # --------------------------------------------------------------- export --
    shutil.rmtree(args.dir_lora, ignore_errors=True)
    modelo.save_pretrained(args.dir_lora)
    tok.save_pretrained(args.dir_lora)
    log.info("Adaptador LoRA salvo em %s/", args.dir_lora)

    log.info("Fundindo o adaptador no modelo base...")
    modelo_final = modelo.merge_and_unload()
    if args.quantizacao != "f32":
        modelo_final = modelo_final.to(torch.bfloat16)
    modelo_final.config.use_cache = True

    shutil.rmtree(args.dir_merged, ignore_errors=True)
    modelo_final.save_pretrained(args.dir_merged, safe_serialization=True)
    tok.save_pretrained(args.dir_merged)
    log.info("Modelo fundido salvo em %s/", args.dir_merged)

    log.info("Convertendo para GGUF (%s)...", args.quantizacao)
    converter_modelo_para_gguf(args.llama_cpp, args.dir_merged, args.saida, args.quantizacao)
    tamanho_mb = args.saida.stat().st_size / (1024 * 1024)
    log.info("OK: %s (%.1f MB)", args.saida, tamanho_mb)

    if not args.sem_adaptador_gguf:
        try:
            converter_adaptador_para_gguf(args.llama_cpp, args.dir_base_config,
                                          args.dir_lora, args.saida_adaptador)
            log.info("OK: %s", args.saida_adaptador)
        except (subprocess.CalledProcessError, FileNotFoundError) as erro:
            log.warning("Não foi possível exportar o adaptador isolado (%s). "
                        "O modelo completo em %s não é afetado.", erro, args.saida)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
