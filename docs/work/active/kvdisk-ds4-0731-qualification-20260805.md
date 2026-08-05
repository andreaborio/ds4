# KV-disk su DeepSeek V4 Flash 0731 — qualifica

Host: Apple M5 Pro 64 GiB, AUTO→SSD streaming, `hebrus-server`, 2026-08-05.
Modello: `DeepSeek-V4-Flash-0731-...-DS4-ExpertMajor-v2.gguf`.
Nessuna modifica al codice: la funzione esisteva già (validata su GLM il
2026-07-11, vedi memoria `kvdisk-glm-validated`), qui viene qualificata su DS.

## Risultato

| lane | freddo | caldo (dopo restart) | fattore | restore |
|---|---:|---:|---:|---:|
| ~1.7k token (ctx 4096) | 17.8 s | **2.1 s** | **8.5×** | 7.7 ms |
| ~26.6k token (ctx 32768) | 185.4 s | **23.0 s** | **8.1×** | 74.1 ms |

A 32K si risparmiano **162 secondi** sul prompt ripetuto.

### Perché il caldo a 32K è 23 s e non 2 s

Scomposizione onesta: il hit copre 26.624 token di un prompt da ~28k, quindi
restano ~1.600 token di **suffisso da prefillare** (~9 s a 170 t/s), più la
generazione con **cache esperti fredda** (processo nuovo: la cache riparte dal
floor e risale). Il restore in sé è 74 ms. Il costo residuo è dominato dal
suffisso: allineare il prompt al confine salvato lo riduce ulteriormente.

## Correttezza dello stato — verificata, non assunta

La risposta a freddo e quella a caldo **differiscono**, ma non è colpa del
KV-disk:

1. Due run **a freddo** dello stesso prompt danno risposte diverse tra loro →
   il nondeterminismo è nel **sampler del server**, non nel percorso KV.
2. Prova diretta sullo stato: i due payload salvati da due run a freddo dello
   stesso prompt differiscono in **3 byte, tutti nell'header** (offset 13-33).
   I **47,8 MB di stato KV sono byte-identici**. Il prefill è deterministico e
   il salvataggio è esatto.

## Costo su disco

~14 KB/token a 32K → **372 MiB per 26,6k token**. Molto più economico di GLM
(la memoria riportava 5,8 GiB per 32k). Un budget da 4-8 GiB regge diverse
conversazioni lunghe. Attenzione: il server salva anche entry intermedie
(`reason=continued`) e allo shutdown (`reason=shutdown`), quindi la directory
cresce più del singolo snapshot (1,4 GB dopo le prove).

## Trappole verificate

- **ctx più piccolo che punta a uno store più grande**: nessun crash. Il
  server da 8192 non usa l'entry da 32K, scrive le proprie entry più piccole,
  e l'entry grande resta **integra** (verificata dopo il test).
- Prefisso **byte-esatto**: la chiave è `token-text`; basta un carattere di
  differenza per mancare il hit.
- La cache esperti **non** viene persistita (è il modello stesso su disco, la
  page cache dell'OS fa il resto): il primo turno dopo un restart paga sempre
  la risalita della cache.

## Ricetta

```
hebrus-server -m <0731.gguf> --ctx 32768 \
  --kv-disk-dir ~/.hebrus/server-kv --kv-disk-space-mb 8192
```

Il restore è automatico al primo prompt con prefisso testuale coincidente.
