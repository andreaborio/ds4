# R2 sbloccata sul verificatore batched — e perché non paga (ancora)

2026-08-05, Apple M5 Pro 64 GiB, DeepSeek V4 Flash 0731, ctx 2048, 256 token.

## Sbloccata

Il round n-gram ora verifica sul percorso **batched**
(`metal_graph_verify_suffix_tops`: un passaggio layer-major su entrambe le
righe più la testa batched), non sul verificatore esatto riga-per-riga.

Bug che lo bloccava: `g->spec_logits` era allocato **solo dentro il blocco
MTP**, quindi il percorso n-gram trovava NULL e il verificatore rifiutava
subito. Ora è allocato anche quando la frontiera n-gram è richiesta.

### Il verificatore esatto era davvero rotto

| | esatto (prima) | batched (ora) |
|---|---|---|
| run identiche tra loro | **NO** (nondeterministico) | **SÌ** |
| prima divergenza dal sequenziale | token **8** | token **112** |
| acceptance sui round drafted | 48% (31/64) | **84%** (71/85) |

Il batched è deterministico e segue il greedy sequenziale per 111 token prima
di divergere su un quasi-pareggio — il comportamento noto delle riduzioni
batched, che è la stessa aritmetica che il prefill usa per l'intero prompt.
L'esatto divergeva all'ottavo token e non era nemmeno riproducibile: era
corruzione, e l'acceptance del 48% era misurata su stato già rotto.

## Ma end-to-end non guadagna: 0.94×

| | decode |
|---|---:|
| sequenziale | 10.97 t/s |
| n-gram batched (2.56 token/round) | 10.92 t/s |

### Perché — misurato

Round = 1 passaggio sequenziale (valuta `first_token`) + 1 verifica batched a
2 righe. Tempi mediani su 40 round:

- passaggio sequenziale: **77.3 ms**
- verifica batched a 2 righe: **141.1 ms = 1.83×** un passaggio

Con la cattura prefix-1 disattivata la verifica scende a 1.74×: **la cattura
costa solo il 9%**, non è lei il problema.

Quindi la seconda riga costa **~0.83 di un passaggio intero**, non ~0.19 come
predice il conto dei soli byte (routed 1.70 su 8.90 GiB). Il profilo per stage
indica `routed_moe` come voce dominante della verifica (33% del tempo per il
19% dei byte), ma il profiler sincronizza a ogni stage e gonfia gli assoluti:
**l'attribuzione precisa del costo marginale di riga va rifatta con una misura
pulita**.

Conto del round: 1 + 1.83 = 2.83 passaggi per 2.56 token = 1.11 passaggi/token
contro 1.00 sequenziale → 0.90×, coerente col 0.94× misurato.

## Cosa servirebbe

1. **Togliere il passaggio sequenziale sprecato.** Oggi ogni round valuta
   `first_token` da solo e poi ne verifica altri due. Se anche `t1` venisse
   dal draft n-gram, il round sarebbe **un solo batch a 3 righe**: 1 + 2×0.83
   = 2.66 passaggi per fino a 3 token. È la modifica con il rapporto
   valore/rischio migliore.
2. **Capire perché la riga marginale costa 0.83 e non 0.19.** Se il costo è
   davvero il MoE routed che gira lontano dal picco di banda, è una leva a sé
   che vale più della speculazione: il routed è ~19% dei byte ma sembra
   pesare molto di più in tempo.

Finché la riga marginale costa 0.83 di un passaggio, la speculazione su questo
modello ha un tetto basso qualunque sia l'acceptance.

## Stato

Opt-in (`--ngram-spec`), spento di default. `DS4_NGRAM_SPEC_EXACT=1` seleziona
il vecchio verificatore esatto (ancora fail-closed sotto streaming).
Fallimento del verificatore batched = **errore duro**: lo stato KV è già
avanzato su entrambe le righe e non c'è snapshot per tornare indietro, quindi
proseguire produrrebbe esattamente la corruzione silenziosa da cui partivamo.
