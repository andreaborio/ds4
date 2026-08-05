# Piano L1 — sovrapporre host e GPU nel decode DeepSeek AUTO→SSD

Data: 2026-08-04. Untracked, non committare. Obiettivo: portare il TPOT del
decode al tempo GPU misurato, eliminando il 40% di idle device.

**Bersaglio quantificato (misurato, non stimato):**

| ctx | TPOT oggi | GPU device | tetto | fattore |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 74,9 ms (13,4 t/s) | 45,1 ms | 22,2 t/s | 1,66× |
| 8192 | 91,9 ms (10,9 t/s) | 54,5 ms | 18,4 t/s | 1,69× |

Fonte: `DS4_METAL_SYNC_PROFILE=1`, differenza gen512−gen0, budget chiuso al
4,5% (`decode-bottleneck-register-20260804.md`, §3-bis e 3-ter). L'idle 40% è
identico a ctx 512 e 8192: strutturale, non legato al contesto.

Chi implementa deve leggere prima: il registro bottleneck (per il modello
mentale) e questo piano per intero. Regola non negoziabile, imparata due
volte in questa campagna: **ogni run di performance verifica l'identità
dell'output** (`--dump-decode-evidence-dir` + confronto con la baseline
AUTO). Un numero senza output verificato non è una misura.

---

## 1. Perché la GPU è ferma il 40% del tempo (stato attuale, misurato)

Per layer, oggi (percorso `overlap_selected_shared` + `async_selected_load`,
tutto già ON di default):

```
CPU:  encode[attn→router→signal]  flush₁  encode[sharedFFN]  flush₂
      └─ start worker ─┐                                        │
WORKER:                └─ wait(router event) → read ids → pread → done ─┐
CPU:  ────────────── BLOCCATA in async_load_finish ────────────────────┘
CPU:  encode[MoE + hc_post]          ← GPU FERMA QUI (≈0,7 ms/layer)
      (poi layer successivo)
```

Numeri per token (8K): main thread bloccata sul worker 72,0 ms (di cui 57,6
è l'attesa evento = GPU-mascherata, 14,4 è pread+bookkeeping), encode vero
16,8 ms, 83,5 commit. La GPU finisce lo shared FFN e resta ferma finché la
CPU non ha codificato il MoE del layer corrente e l'attention del successivo.

La dipendenza che serializza: **la CPU deve conoscere gli slot cache degli
esperti selezionati prima di poter codificare il MoE**. È questa che il
redesign rimuove.

## 2. Il redesign: attesa lato GPU + tabella indirizzi

```
CPU:  encode[attn→router→signal(router_ready,L)]
      encode[sharedFFN]
      encode[waitEvent(experts_ready,L)]        ← attesa IN CODA, non su CPU
      encode[MoE via tabella indirizzi]
      encode[hc_post]
      commit (UNO per layer)                    → passa SUBITO al layer L+1
WORKER: wait(router_ready,L) → read ids → pread miss → scrive 6 indirizzi
        nella tabella GPU-visibile → setSignaledValue(experts_ready,L)
GPU:  attn/router L → sharedFFN L → [stallo solo se worker in ritardo] → MoE L
```

La CPU non si blocca mai; l'unico stallo possibile è la GPU al `waitEvent`
per la latenza worker (media attesa: ~0,1 ms bookkeeping + ~0,27 ms pread,
in parte già coperta dallo shared FFN). Idle atteso: dal 40% a <10%.

### I mattoni esistono già nel repo

| pezzo | dove | stato |
| --- | --- | --- |
| worker persistente con job | `ds4.c:26168` (`metal_graph_selected_async_load_worker_main`) | attivo |
| evento condiviso router→CPU | `ds4_metal.m:9244` (`signal_batch_and_wait_event`) e 9169/9223 | attivo |
| tabella 6 indirizzi GPU per layer, ×4 rotazione | `ds4_metal.m:13745` (`stream_compact_addr_ensure_buffers`), `gpuAddress` | esiste, spenta sul decode DeepSeek |
| kernel MoE che leggono indirizzi | `metal/moe.metal` famiglia `mul_mv_addr_*` (iq2 pair swiglu, q2_K) | release, usati da GLM router-ahead e DSpark |
| protezione vittime dall'evizione | `ds4_metal.m:14876` (victim scan con `protect_ids`) + seq `last_use_seq` | attivo |
| segnale CPU→GPU | `MTLSharedEvent setSignaledValue:` dal worker | API standard, da cablare |

Il lavoro è **plumbing di orchestrazione**, non kernel nuovi.

### L'invariante di sicurezza (e perché il deadlock NON può accadere a profondità 1)

Rischio nominato nel registro: la vittima LRU scelta dal worker per i miss di
L+1 potrebbe essere uno slot referenziato da un buffer in volo, e il percorso
di evizione oggi può "attendere i buffer in volo" — attendere il buffer che
sta aspettando il TUO segnale = deadlock.

Con pipeline a profondità 1 l'ordine di coda lo rende impossibile:

1. Il worker carica i miss di L+1 solo DOPO il segnale `router_ready(L+1)`.
2. Quel segnale è codificato DOPO il MoE di L nella stessa coda in-order.
3. Quindi quando il worker sceglie vittime per L+1, il MoE di L è GIÀ
   eseguito: nessuno slot di L è più referenziato da lavoro non completato.
4. Gli slot referenziati dal buffer di L+1 stesso sono esattamente i 6
   selezionati di L+1 — che sono i target del load, mai vittime
   (`protect_ids`).

Obblighi di implementazione derivati:
- **Nel nuovo modo il percorso di evizione non deve mai attendere buffer in
  volo** (la garanzia d'ordine lo rende superfluo); sostituire l'attesa con
  un'asserzione: `slot.last_use_seq <= completed_seq` per ogni vittima.
  Se l'asserzione scatta, il modo si disattiva fail-closed per la sessione.
- La profondità resta 1 (un solo layer con `waitEvent` pendente). Multi-layer
  in volo è un'estensione futura con analisi nuova — NON in questo piano.

### Percorsi di fallimento (da progettare, non da scoprire)

- **Pread fallito nel worker**: il worker DEVE comunque segnalare l'evento
  (la GPU è in attesa!) e marcare uno stato d'errore; il main thread controlla
  lo stato a fine token e fallisce la generazione in modo pulito. Mai lasciare
  la GPU appesa: il timeout 60 s esistente è il backstop, non il piano.
- **Interrupt/CTRL-C/rewind/context-full**: ogni percorso di annullamento che
  oggi scarta il batch deve prima segnalare gli eventi pendenti.
- **`decode_stage_profile` e i debug path**: forzano il modo vecchio (già
  fanno opt-out dall'overlap; estendere il gating).
- **Modelli non-DeepSeek/famiglie diverse**: fuori scope, gating esplicito su
  DeepSeek ExpertMajor v2 + M5.

## 3. Fasi, tempi, gate

Tutto dietro UN flag opt-in: `DS4_METAL_ENABLE_DEEPSEEK_DECODE_GPU_WAIT`
(default OFF finché ogni gate passa). Convenzione repo rispettata: A/B
appaiato, output identico, suite cross-model per file condivisi.

| fase | contenuto | tempo | gate di uscita |
| --- | --- | --- | --- |
| **P0** | flag + plumbing modo + telemetria: latenza worker (istogramma), stallo al waitEvent (GPUStartTime gap), contatore commit | 0,5 g | build pulita; telemetria visibile con flag ON in modo legacy |
| **P1** | worker esteso: wait router → ids → `begin_selected_load` → scrittura tabella compact addr → `setSignaledValue`. Gestione errore-con-segnale | 1 g | unit: worker segnala SEMPRE (successo, pread fallito, shutdown); tabella coerente con gli ids |
| **P2** | encode ristrutturato: un buffer per layer con `waitEvent` + MoE via `mul_mv_addr_*`; rimozione del blocco `async_load_finish`; primo/ultimo layer, testa logits, cancellazioni | 1-1,5 g | genera testo IDENTICO ad AUTO su 3 prompt × ctx {512, 8192}; interrupt e context-full puliti |
| **P3** | invariante evizione: niente attese in-volo nel nuovo modo + asserzione seq fail-closed; stress con cache al floor (259) per forzare evizioni massime | 1 g | 1.000 token al floor senza asserzioni né mismatch |
| **P4** | suite completa: `--metal-kernels`, `--metal-expert-pack`, oracle, admission, Qwen session, GLM smoke (file condivisi toccati) | 0,5-1 g | tutto verde con flag OFF **e** ON |
| **P5** | A/B/B/A ×3 cicli, prosa+codice, 8K/512 + punto ctx512, con identità output in OGNI run; aggiornare registro | 0,5 g | vedi criteri sotto |
| riserva | imprevisti (precedente GLM M2: −15%) | 1-2 g | — |

**Totale: 4,5-7 giorni.**

### P0 ESEGUITO 2026-08-04 — risultati e revisione al ribasso del tetto

Consegnato: flag `DS4_METAL_ENABLE_DEEPSEEK_DECODE_GPU_WAIT` (visibile nella
riga `effective profile=… decode-gpu-wait=…`, percorso legacy attivo),
istogramma della latenza worker post-segnale nel sync profile, e microbench
`ds4_gpu_internal_shared_event_resume_latency_test` in `--metal-kernels`.

**Gate di rischio A — latenza segnale CPU→ripresa GPU: PASSA.**
32 round: min 0,119, **media 0,302**, max 1,003 ms. Soglia del piano 0,5 ms.
Il meccanismo `waitEvent`/`setSignaledValue` è abbastanza reattivo.

**Gate di rischio B — distribuzione della latenza worker: PASSA CON RISERVA.**
20.480 job (40/token × 512), latenza misurata DOPO il segnale del router
(scrittura tabella + bookkeeping + pread), quindi esattamente lo stallo che
la GPU subirebbe nel nuovo modo:

| bucket | job | quota |
| --- | ---: | ---: |
| < 0,2 ms | 9.838 | 48,0% |
| < 0,5 ms | 4.395 | 21,5% |
| < 2 ms | 6.157 | 30,1% |
| ≥ 2 ms | 90 | 0,4% |

media 0,324 ms, max 8,4 ms → **13,9 ms per token** di latenza worker totale.

**Conseguenza sul tetto: va rivisto al ribasso.** Il tetto 1,66-1,69×
assumeva host completamente sovrapposto, cioè TPOT = tempo GPU. Ma il 30,5%
dei job supera 0,5 ms e lo shared FFN non li copre: quella coda diventa
stallo GPU al `waitEvent`. Stima con i numeri misurati:

- caso pessimo (nessuna copertura): TPOT 54,5 + 13,9 = 68,4 ms → **1,34×**
- caso realistico (mediana coperta dallo shared FFN, stalla la coda):
  ~70 ms → **~1,31×**

**Il tetto realistico di L1 è quindi ~1,3×, non 1,66×.** L'1,66× resta il
limite teorico assoluto (GPU pura) ma richiederebbe latenza worker ≈ 0.

Questo cambia i criteri di accettazione (sotto) e apre una domanda che P1
deve affrontare: il 30% di job sopra 0,5 ms è dominato dai pread di miss.
**Prefetch del layer L+1 sugli id del token precedente** (i router adiacenti
hanno 22-29% di overlap, misurato) sposterebbe quei pread fuori dal percorso
critico. Era "fuori scope"; con questi dati diventa la parte che determina
se L1 vale 1,3× o si avvicina a 1,6×. Da valutare all'inizio di P1.

### Criteri di accettazione (RIVISTI dopo P0)

Il tetto realistico misurato è ~1,3×, quindi le soglie originali (successo
≥1,40×, kill <1,15×) non sono più coerenti: 1,40× sarebbe sopra il tetto.
Nuove soglie:

- **Successo**: ≥1,25× su 8K/512 (≥13,6 t/s), output byte-identico ad AUTO,
  TPOT p95 non peggiore di +5%.
- **Successo pieno** (giustifica il prefetch L+1 come follow-up): ≥1,45×.
- **Kill**: <1,10× a fine P3, o mismatch di output irrisolti in P2, o
  asserzione di evizione non eliminabile senza attese in-volo.

### Criteri di accettazione (originali, superati da P0)

- **Successo**: ≥1,40× su 8K/512 (≥15,3 t/s) con output byte-identico ad AUTO
  e TPOT p95 non peggiore di +5%. Il tetto è 1,69×; sotto 1,40× il redesign
  non sta pagando il suo rischio.
- **Kill**: <1,15× a fine P3, o mismatch di output non risolti in P2, o
  asserzione di evizione non eliminabile senza attese in-volo. In tal caso:
  flag resta OFF, registro aggiornato con la causa, si valuta il piano B.
- **Promozione a default**: solo dopo P5 + rerun real-model DeepSeek/GLM/Qwen
  (regola del repo per i file condivisi) in una tranche separata.

### Metriche di avanzamento per fase (dal profiler già committato)

- P2: commit/token 83→43; `host_block_total` decode → ~0.
- P3: GPU idle (span−gpu_total)/span decode → obiettivo <15%.
- P5: t/s e TPOT p50/p95, due workload, cicli appaiati.

## 4. Rischi principali con mitigazione

| rischio | probabilità | mitigazione |
| --- | --- | --- |
| latenza `setSignaledValue`→ripresa GPU maggiore del previsto | media | misurarla in P0 con un microbench evento; se >0,5 ms, valutare segnale anticipato per-componente (gate/up prima, down dopo) |
| il victim scan ha attese in-volo non rimovibili in punti profondi | media | P3 dedicata; l'asserzione seq rende il problema visibile subito, fail-closed |
| p95 peggiora per stalli worker su layer con molti miss | media | telemetria P0 (istogramma); eventuale prefetch del layer L+1 sugli id del token precedente come follow-up, NON in questo piano |
| interazione con growth/shrink della cache a runtime | bassa | il nuovo modo parte solo a cache stabile post-prefill (stessa condizione della phase schedule) |
| effetto GLM M2 (−15% inatteso) | media | i presupposti qui sono diversi e misurati (idle 40% reale); comunque: criterio kill a 1,15× e riserva tempo |

## 5. Fuori scope, esplicitamente

- Multi-layer in volo (profondità >1): analisi deadlock nuova richiesta.
- Batching di 2 richieste server (amortizza i pesi non-routed senza
  speculazione): candidato L2 separato, dopo L1.
- Hotlist preload (+17% storico sulla rampa): ortogonale, 30 minuti, si può
  fare in qualunque momento; non tocca il regime.
- Fix vero di `--ssd-streaming-cache-experts` su DeepSeek (oggi fail-closed,
  commit 2266bb1): tracciato nel registro, non blocca L1.
- DSpark: chiuso, NO-GO. Dossier completo parcheggiato sul branch `experiment/dspark-128gb` (commit 66b3dcc); artefatti combined/support cancellati (~99 GiB liberati), catena di rigenerazione nel dossier.

---

## Stato P2 — 2026-08-05

P1 committata (`d82ab4b`). P2 implementata e funzionante: primo decode GPU-wait
**byte-identico** al legacy (ctx 512 e 8K, evidence `cmp` su
`frontier_*.decode.json`).

### Come è fatta (differenze dal disegno originale)

1. **Niente rotazione tabella**: la tabella compatta è un solo set di buffer
   per layer (sovrascritti). Con profondità-1 e ordine di coda è sicuro:
   il worker riempie la tabella del layer L solo dopo il router L, e il MoE
   del token/layer precedente che la referenziava è già completato.
2. **Residenza slab-wide**: con gli indirizzi pubblicati dopo l'encode non si
   possono dichiarare i buffer per-entry; i dispatch `mul_mv_addr` in modalità
   pending dichiarano TUTTI gli slab della cache (`use_all_slabs`). Regge su
   due invarianti: gli slab non vengono mai liberati durante il decode, e la
   crescita slab è CONGELATA sul worker (rifiutata in `alloc_slab_slot`) —
   cresce solo il main al confine token (`gpu_wait_token_grow`), dove nessun
   buffer è parcheggiato. Stessa finestra per il trim al ribasso
   (`gpu_wait_token_trim`, sostituisce le prune del main per il tier anti-swap).
3. **Scoperta chiave — semantica pending-load**: `begin_selected_load` NON
   installa in cache: mette in staging il pool pread e parcheggia i risultati
   nella pending-load struct; nel legacy era la risoluzione sul main a
   consumarla. Il worker GPU-wait ora chiama `pending_load_finish` (attesa
   pool + install) prima del peek: senza, "expert non residente" al primo
   layer routed.
4. **Commit della shared-FFN prima del wait**: senza il flush tra shared e
   `encodeWaitForEvent`, la parte densa resta non-committata durante il pread
   e la GPU idla per l'intero load (misurato: p95 144→125 ms col fix).
5. **Fail-closed ovunque**: wait_inflight sul worker → fallimento pulito del
   job (mai flush/wait dal worker); slab nuovo richiesto dal worker →
   eviction; signal-always garantito anche su reserve orfano.

### Prime misure (M5 Pro 64 GiB, 0731)

| Punto | OFF | ON | Note |
|---|---|---|---|
| ctx 512, gen 96 (ramp corta) | 10.95 t/s | **11.48 t/s (+4.8%)** | identico |
| 8K, gen 128–256 (RAMPA, hit 0.844) | 9.86 | 9.52→9.72 dopo fix flush | identico |

Sync profile 8K/128: encode main **dimezzato** (2950→1281 ms, il design
funziona); la catena worker si allunga (load 0.431→0.891 ms/job, ≥2 ms:
39→388) perché pool-wait+install+fill ora stanno sul percorso che gata il
MoE. A hit 0.844 (rampa) il worker è il critical path in ENTRAMBE le
modalità: il verdetto va emesso a regime (gen 512), dove l'hit sale e la
catena si accorcia. `gpu_total` ON è gonfiato dall'artefatto del buffer
parcheggiato (il wall time del buffer include il park).

A/B/B/A 8K gen 512 in corso.

### P2.5 — publish-then-install (2026-08-05, sera)

Il profilo mostrava la catena worker (pool-wait + install + fill) come gate
del MoE. Riordino: `wait_io` (solo attesa pread) → `fill_gpu_wait` (tabella
da hit-peek + buffer pending, mappati per **load_i via source_slots→load_slots**,
NON per indice di selezione — bug trovato al primo colpo: mask 0x3c con bufs
compattati in [0..3]) → **signal** → `pending_load_commit` (install fuori dal
percorso gated). Sicurezza: slot pending lockati fino all'install; il job
successivo (e quindi ogni eviction) parte solo dopo il done di questo.
Identità 512 ✓. A/B/B/A steady 8K in corso (ordine OFF/ON/ON/OFF).

### P3 da fare (disegno)

1. Assertion d'invariante: in modalità pending, al fill, verificare che ogni
   buffer pubblicato appartenga a uno slab dichiarato o ai pending lockati
   (DS4_TEST_HOOKS only, costo O(6×n_slab)).
2. Stress al floor: forzare budget≈259 (env cache-experts è rifiutato; usare
   prefill lungo + decode corto ripetuto) e verificare niente deadlock con
   worker freeze + grow/trim al confine.
3. Cancellazione: kill del bench a metà decode ripetuto (SIGINT loop) — nessun
   hang, nessun timeout command buffer nel log.
