#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testes rápidos das funções de dados de `treinar.py` (não precisam de rede).

Rode com: python scripts/testes.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from treinar import (  # noqa: E402
    ColetorCausal,
    ler_exemplos,
    montar_lote,
    resumir_dataset,
    tokenizar_exemplo,
    validar_mensagens,
)

falhas: list[str] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  ok   {descricao}")
    else:
        print(f"  FALHA {descricao}")
        falhas.append(descricao)


def espera_erro(funcao, descricao: str) -> None:
    try:
        funcao()
    except (ValueError, FileNotFoundError):
        print(f"  ok   {descricao}")
    else:
        print(f"  FALHA {descricao} (não levantou erro)")
        falhas.append(descricao)


class TokenizadorFalso:
    """Tokenizador mínimo: 1 token por palavra, sem downloads."""

    pad_token_id = 0

    def apply_chat_template(self, mensagens, tokenize=False, add_generation_prompt=False):
        partes = [f"<{m['role']}> {m['content']}" for m in mensagens]
        if add_generation_prompt:
            partes.append("<assistant>")
        return " ".join(partes)

    def __call__(self, texto, truncation=False, max_length=None, add_special_tokens=True):
        ids = [len(p) + 1 for p in texto.split()]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_validar_mensagens() -> None:
    print("validar_mensagens")
    msgs = validar_mensagens(
        {"messages": [{"role": "user", "content": "oi"},
                      {"role": "assistant", "content": "olá"}],
         "domain": "x", "meta": {"a": 1}},
        1,
    )
    checar(msgs == [{"role": "user", "content": "oi"},
                    {"role": "assistant", "content": "olá"}],
           "mantém só 'messages' e descarta colunas extras (domain/meta)")
    espera_erro(lambda: validar_mensagens({"texto": "oi"}, 1), "rejeita linha sem 'messages'")
    espera_erro(lambda: validar_mensagens({"messages": []}, 1), "rejeita 'messages' vazio")
    espera_erro(lambda: validar_mensagens(
        {"messages": [{"role": "robo", "content": "x"},
                      {"role": "assistant", "content": "y"}]}, 1),
        "rejeita role desconhecido")
    espera_erro(lambda: validar_mensagens(
        {"messages": [{"role": "user", "content": "x"}]}, 1),
        "rejeita conversa que não termina em 'assistant'")


def test_ler_exemplos() -> None:
    print("ler_exemplos")
    linhas = [
        {"messages": [{"role": "system", "content": "s"},
                      {"role": "user", "content": "u"},
                      {"role": "assistant", "content": "a" * 10}], "domain": "d"},
        {"messages": [{"role": "user", "content": "u2"},
                      {"role": "assistant", "content": "a2"}]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "dados.jsonl"
        caminho.write_text("\n".join(json.dumps(x) for x in linhas) + "\n\n", encoding="utf-8")
        exemplos = ler_exemplos(caminho)
        checar(len(exemplos) == 2, "lê todas as linhas e ignora linhas em branco")
        checar(len(ler_exemplos(caminho, max_exemplos=1)) == 1, "respeita --max-exemplos")
        resumo = resumir_dataset(exemplos)
        checar(resumo["exemplos"] == 2 and resumo["chars_resposta_media"] == 6,
               "resumo do dataset com estatísticas")

        quebrado = Path(tmp) / "quebrado.jsonl"
        quebrado.write_text('{"messages": [}\n', encoding="utf-8")
        espera_erro(lambda: ler_exemplos(quebrado), "aponta JSON inválido")
        espera_erro(lambda: ler_exemplos(Path(tmp) / "nao_existe.jsonl"), "aponta arquivo ausente")


def test_tokenizar() -> None:
    print("tokenizar_exemplo")
    tok = TokenizadorFalso()
    mensagens = [{"role": "system", "content": "regra"},
                 {"role": "user", "content": "pergunta"},
                 {"role": "assistant", "content": "uma resposta longa aqui"}]

    item = tokenizar_exemplo(tok, mensagens, 100)
    assert item is not None
    n_mascarados = sum(1 for r in item["labels"] if r == -100)
    n_treinados = sum(1 for r in item["labels"] if r != -100)
    checar(len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"]),
           "input_ids, labels e attention_mask têm o mesmo tamanho")
    checar(n_mascarados > 0 and n_treinados == 4,
           "prompt mascarado (-100) e loss só nos tokens da resposta")
    checar(item["labels"][n_mascarados:] == item["input_ids"][n_mascarados:],
           "labels da resposta repetem os input_ids")

    # o prompt sozinho ocupa 5 tokens no tokenizador falso
    curto = tokenizar_exemplo(tok, mensagens, 5)
    checar(curto is None,
           "descarta exemplo cujo prompt já estoura o tamanho máximo (evita loss vazia)")

    truncado = tokenizar_exemplo(tok, mensagens, 8)
    assert truncado is not None
    checar(len(truncado["input_ids"]) == 8, "respeita o tamanho máximo de tokens")


EXEMPLOS_LOTE = [
    {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
    {"input_ids": [4], "attention_mask": [1], "labels": [4]},
]


def test_montar_lote() -> None:
    print("montar_lote")
    lote = montar_lote(EXEMPLOS_LOTE, pad_token_id=7)
    checar(lote["input_ids"] == [[1, 2, 3], [4, 7, 7]], "padding com pad_token_id")
    checar(lote["attention_mask"] == [[1, 1, 1], [1, 0, 0]], "attention_mask zera o padding")
    checar(lote["labels"] == [[-100, 2, 3], [4, -100, -100]], "labels do padding viram -100")


def test_coletor() -> None:
    print("ColetorCausal")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  pulado (torch não instalado)")
        return

    lote = ColetorCausal(pad_token_id=7)(EXEMPLOS_LOTE)
    checar(lote["input_ids"].tolist() == [[1, 2, 3], [4, 7, 7]], "tensores com o lote correto")
    checar(all(t.dtype == torch.long for t in lote.values()), "tensores em int64")


def main() -> int:
    for teste in (test_validar_mensagens, test_ler_exemplos, test_tokenizar, test_montar_lote, test_coletor):
        teste()
    if falhas:
        print(f"\n{len(falhas)} teste(s) falharam")
        return 1
    print("\nTodos os testes passaram")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
