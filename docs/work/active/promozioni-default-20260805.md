# Promozioni a default — cosa è passato e cosa no

2026-08-05, Apple M5 Pro 64 GiB, DeepSeek V4 Flash 0731.

## R4 (cache calda tenuta tra i turni): PROMOSSO

`DS4_DEEPSEEK_PREFILL_CACHE_KEEP_MAX_TOKENS` ora vale **4096 di default**
(prima 0 = teardown sempre). Impostare 0 ripristina il comportamento storico.

Perché era sbagliato lasciarlo spento: senza, un secondo turno di
conversazione da ~2,4K token demolisce la cache esperti calda fino al floor e
la ricostruisce — **~27 GiB riletti dall'SSD per uno stato caldo un attimo
prima**.

| | TTFT turno 2 | decode a regime |
|---|---:|---:|
| default (keep) | **17,8 s** | **20,9 char/s** |
| disattivato (storico) | 21,5 s | 10,9 char/s |
| | | **1,92×** |

**Bordo dei 4096 token**: turno da ~4k con la cache tenuta → decode 33,6
char/s, nessun messaggio di pressione, swap o reclaim. Sopra i 4096 il
comportamento è invariato per costruzione (`prefill_tokens <= keep_max`),
quindi i prompt grossi prendono ancora il teardown, che è dove il headroom
serve davvero.

## L1 (decode GPU-wait): NON promosso, resta opt-in

Provato a promuoverlo e **rimesso opt-in dopo la misura**.

| tornata | default/ON | legacy/OFF |
|---|---:|---:|
| mattina, 2 coppie | 12,08 | 11,63 (+3,9%) |
| sera, 4 run A/B/B/A | 11,40 | 11,25 (**+1,3%**) |

Il punto non è la media: è che nella tornata serale lo spread **dentro lo
stesso braccio** è 10,77-12,03 t/s, cioè **11,7%**. Il rumore della macchina
è tre volte l'effetto cercato. La prima run della serie dava L1 in perdita del
2,7%; la quarta in guadagno del 5%. Con quel rumore, +3,9% non è una misura,
è un campione.

**Regola da qui in poi: su questo host, a 32K, niente sotto il ~10% può
essere dichiarato senza molte più ripetizioni.** Vale retroattivamente su
tutto ciò che oggi è stato misurato in quella fascia (L1 stesso, la parità di
R2).

Restano validi e non toccati da questo: identità dell'output byte-a-byte,
determinismo run-to-run, invariante di residenza, gate sugli slab,
cancellazione pulita. È solo la rivendicazione di velocità a non reggere la
ripetizione — e il tetto dell'intera direzione overlap è comunque 1,095×.
