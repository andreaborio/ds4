# La leva dei byte densi — dove sta davvero il decode

Misurato il 2026-08-05 su Apple M5 Pro 64 GiB, DeepSeek V4 Flash 0731.
Nasce dal verdetto L1 (`l1-decode-overlap-plan-20260804.md`): la direzione
overlap è esaurita a ~1.095×, e `gpu_total` è l'83% dello span e invariante a
qualsiasi riordino. Quindi: **meno lavoro GPU o meno byte**.

## Inventario: dove sono i byte per token

Dal GGUF (1200 tensori):

| categoria | GiB | % denso | tipi |
|---|---:|---:|---|
| attention | 5.40 | 65.9% | F16, F32, Q8_0 |
| embed/output | 1.51 | 18.4% | F16, Q8_0 |
| shared expert | 1.07 | 13.1% | Q8_0 |
| norme/router | 0.21 | 2.5% | F16, F32 |
| **denso totale** | **8.19** | | |

Il `token_embd` (0.99 GiB, F16) è una lookup di una riga, non uno stream:
**denso realmente letto per token = 7.20 GiB**. I routed sono 6/256 × 72.56 =
**1.70 GiB**. Totale **8.90 GiB/token** — che a 51 ms di device time misurati
fa ~174 GiB/s, coerente col fatto che il decode è banda-limitato.

**Il denso è l'82% dei byte per token. Gli esperti sono il 18%.** Tutta la
campagna precedente (cache, prefetch, prune, streaming) lavorava sul 18%.

## Il kernel: lo speedup è esattamente il rapporto dei byte

Sonda `ds4_gpu_internal_dense_matvec_bandwidth_test` (shape reale
4096×8192, `DS4_METAL_DENSE_MV_BENCH=1` in `ds4_test --metal-kernels`):

| formato | ms | GiB/s | speedup |
|---|---:|---:|---:|
| Q8_0 (oggi) | 0.134 | 247.4 | — |
| Q4_K | 0.071 | 248.5 | **1.89×** (byte 1.89×) |

Stabile su nsg ∈ {1,2,4,8}. **Entrambi girano al picco di banda della
macchina**: il costo di dequantizzazione del Q4_K è completamente nascosto,
quindi il guadagno è esattamente il rapporto dei byte. Niente da ottimizzare
nel kernel — bisogna solo dargli meno byte da leggere.

### Due trappole che hanno prodotto numeri falsi prima

1. **Residenza in cache**: una matrice da 34 MiB riletta 200 volte sta nella
   SLC e riporta 382 GiB/s, sopra la banda DRAM. Il probe ora legge da un pool
   da 1 GiB con slice diversa a ogni iterazione — come fa il decode vero, che
   attraversa 43 layer di matrici distinte.
2. **Partizionamento righe diverso tra i due kernel**: Q8_0 dà a un
   threadgroup NR0 righe e divide le colonne tra i simdgroup; Q4_K dà a ogni
   simdgroup righe proprie (`first_row = (r0*NSG + sgitg)*nr0`), quindi vuole
   NSG volte meno threadgroup. Con il dispatch alla Q8_0 il Q4_K leggeva oltre
   il buffer e sembrava **2.8× più LENTO**. Il probe ora verifica anche la
   copertura (nessuna riga di output non scritta).

## Proiezione end-to-end

Requantizzare i soli tensori densi **Q8_0 → Q4_K** (attn_output_a/b,
attn_q_b, attn_q_a, attn_kv, i tre shexp, output):

- risparmio 2.89 GiB/token → 8.90 → **6.01 GiB/token = 1.48×** sulla parte
  banda-limitata;
- sul TPOT misurato a 32K (73.3 ms, di cui ~51 ms device): **~1.29×**
  end-to-end, cioè **12.1 → ~15.5 t/s**. A contesto corto il guadagno è
  maggiore (i pesi densi pesano di più rispetto al KV).

Aggiungendo i densi F16 (indexer.attn_q_b, compressor gate/kv: 0.84 GiB) si
arriva vicino a 1.6×, ma con un salto di qualità più marcato.

Questo **si somma** a L1 (+4%) e non interferisce: L1 toglie tempo host, questo
toglie byte.

## Cosa serve per incassarlo (non fatto)

1. **Qualificare numericamente il kernel Q4_K denso.** È committato ma
   **nessun percorso di produzione lo chiama**: la sonda verifica solo banda e
   copertura, non i valori. Serve un test contro riferimento CPU.
2. **Percorso denso nel motore**: `ds4_gpu_matmul_q4_k_tensor` che rispecchia
   `ds4_gpu_matmul_q8_0_tensor`, più il dispatch nell'encode dell'attenzione.
3. **Converter**: emettere i tensori scelti a Q4_K. Serve un **nuovo
   artefatto GGUF** (~78 GiB) e quindi una decisione di pubblicazione.
4. **Valutazione di qualità.** Il modello è stato costruito deliberatamente
   con `AProjQ8` (proiezioni di attenzione a Q8): scendere a Q4_K su MLA è il
   rischio vero. Gradini intermedi: **Q6_K** (−23% byte) o **Q5_K** (−35%),
   entrambi con lo stesso ragionamento di banda.

## Raccomandazione

È la leva più grande rimasta, di gran lunga, ed è l'unica che punta al target
1.66× dichiarato all'inizio della campagna. Il passo successivo a costo basso
è **Q6_K sui tre tensori di attenzione grandi** (4.29 GiB → 3.3 GiB, ~1.12×
end-to-end) come prova di qualità prima di considerare Q4_K.

---

## Aggiornamento: gli stessi byte SENZA toccare la precisione

Obiezione dell'utente, corretta: i tre tensori grandi sono `attn_output_a/b` e
`attn_q_b`, cioè proprio quelli su cui una quantizzazione più aggressiva si
paga in qualità. Il modello è stato costruito apposta con `AProjQ8`.

Ma i byte densi si leggono **una volta per passaggio, non per token**. Misura
sulla stessa sonda (Q8_0, shape 4096×8192, tre ripetizioni concordi):

| righe nel passaggio | ms | costo per token |
|---|---:|---:|
| n=1 | 0.130 | 1.00× |
| n=2 | 0.134 | **1.95-2.15×** |
| n=4 | 0.138 | **3.78-4.03×** |

**Un passaggio a 4 token costa quanto un passaggio a 1 token.** Il matvec
denso è interamente banda-limitato sui pesi: aggiungere righe è gratis fino
ad almeno 4. Quindi ogni token speculativo verificato è quasi gratuito in
byte — lo stesso guadagno della requantizzazione, **senza cambiare un solo
peso**.

### Economia della speculazione su questo modello

Passaggio batched a 2 righe: denso 7.20 (una volta) + routed 2×1.70 = **10.60
GiB**, contro 8.90 GiB/token sequenziale. **Pareggio a solo il 19% di
accettazione.** Con i token/round già misurati su R2 (1.59):
10.60/1.59 = 6.67 GiB/token = **1.33×** — stessa taglia della requantizzazione
Q4_K (1.29×), a precisione invariata.

La vecchia soglia del gate MTP (75% di accettazione) era giusta per un modello
compute-bound. Qui il pareggio è al 19%: **la soglia va rifatta**.

### Ma il verificatore attuale NON batcha

`metal_graph_verify_decode_exact` fa `for il { for row { encode_decode_layer
(n_tok=1) } }`: due passaggi separati a una riga per layer, quindi legge i pesi
densi **due volte** e non incassa nulla. Questo spiega da solo perché MTP fu
misurato NO-GO e perché R2 dava −10%: su un decode banda-limitato, una
speculazione che non batcha non può vincere, qualunque sia l'accettazione.

L'infrastruttura però c'è già: `metal_graph_encode_layer_batch(g, model,
layer, il, pos0, n_tokens)` è l'encoder del prefill, e il verificatore alloca
già `batch_cur_hc`/`batch_next_hc` con le row view che quell'encoder usa.

**Avvertenza onesta**: le riduzioni batched possono scegliere un token diverso
sui quasi-pareggi — è esattamente il motivo per cui il verificatore "esatto"
fu scritto riga-per-riga. Quindi non è bit-identico al greedy sequenziale. Ma
è la stessa aritmetica che il prefill usa già per l'intero prompt: è un ordine
di riduzione diverso, non una perdita di precisione nei pesi.

### Raccomandazione aggiornata

Prima **la speculazione batched** (nessun artefatto nuovo, nessuna perdita di
precisione, ~1.3×), poi eventualmente la requantizzazione come leva
indipendente che si somma. Il modello 0731 **non ha teste MTP** (zero tensori,
malgrado `nextn_predict_layers=1` nei metadati), quindi il drafter dev'essere
quello n-gram: R2, che è già costruito e va sbloccato.

---

## Il MoE routed NON è il collo — misurato (2026-08-05, sera)

Il profiler per stage indicava `routed_moe` come voce dominante della verifica
batched. **Falsificato con una misura pulita.** Sonda
`ds4_gpu_internal_routed_moe_bandwidth_test`: kernel veri del motore
(`kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32` + `..._q2_k_sum6_f32`), shape di
decode reali, pool da 1 GiB fuori cache, una tabella indirizzi **per slice**.

| | GiB/s | ms/token |
|---|---:|---:|
| denso Q8_0 | 244-250 | 29.4 (7.20 GiB) |
| **routed MoE** (6 esperti/layer) | **186** | **9.1** (1.70 GiB) |

Il routed gira al **76% del ritmo denso**, non a 1/5. Con 43 layer costa 9.1
ms/token: c'è margine per ~2 ms, non per 25.

### Due trappole in più, oltre a quelle del probe denso

3. **Riempimento uniforme**: IQ2_XXS e Q2_K derivano le scale dal payload, e
   un fill costante decodifica a zero esatto — il controllo di copertura non
   distingueva un kernel che funziona da uno che non parte. Ora il riempimento
   è pseudo-casuale.
4. **Una sola tabella indirizzi riscritta per iterazione**: l'encoder registra
   buffer+offset, non i byte, quindi tutti i dispatch leggevano l'ultima
   tabella scritta (e fuori range). Ora c'è una tabella per slice.

## Il budget del token, ricostruito con numeri puliti

streaming pesi = 29.4 (denso) + 9.1 (routed) = **38.5 ms/token**, cioè il
**43-53% del TPOT** a seconda del contesto. Il resto è attenzione/KV,
attivazioni, lanci e tempo host — non lettura di pesi.

Conseguenza sulla requantizzazione: portare il denso Q8_0 a Q4_K risparmia
**11.8 ms/token**, quindi **1.15× a 8K e 1.19× a 32K** — non l'1.29× stimato
prima assumendo che tutto il tempo GPU scalasse coi byte. Resta la leva
singola più grande, ma la taglia onesta è quella.

## E la riga speculativa marginale resta inspiegata

La seconda riga costa 63.8 ms misurati (141.1 verifica − 77.3 sequenziale). I
pesi routed della seconda riga sono **9.1 ms, il 14%**. Restano **54.7 ms non
spiegati dai pesi**: sono i costi fissi e la granularità di tile del percorso
batched (quello del prefill) usato con sole 2 righe. È lì che va guardato se
si vuole far pagare la speculazione — non nel MoE.
