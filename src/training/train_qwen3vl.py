#!/usr/bin/env python
import os
import sys
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch import amp
from tqdm import tqdm

# ---------------------------------------------------------------------
# PYTHONPATH: aggiungo la root del progetto
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../src/training
PROJECT_ROOT = THIS_DIR.parent.parent               # .../ (root del repo)
SRC_ROOT = PROJECT_ROOT / "src"                     # .../src

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets.how2sign_loader import How2SignDataset, how2sign_collate_fn
from src.models.qwen3vl_lora import load_qwen3vl_lora


# ---------------------------------------------------------------------
# Config di base
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
DATASET_JSON = PROJECT_ROOT / "data/How2Sign/how2sign_dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs/qwen3vl_lora_how2sign"

BATCH_SIZE = 1              # poi si prova ad alzare
NUM_EPOCHS = 1              # per ora smoke test
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_ACCUM_STEPS = 8        # gradient accumulation (effettivo batch = BATCH_SIZE * GRAD_ACCUM_STEPS)
LOG_EVERY = 50              # step di logging
MAX_VAL_BATCHES = 50        # quante batch usare in val (per velocità)
MAX_GEN_TOKENS = 64         # max token generati per valutazione
MAX_STEPS = None            # se voglio fermare dopo N step globali, metto un int

RESUME_FROM_LATEST = True       # se True, prova a riprendere dall’ultimo checkpoint
EARLY_STOPPING_PATIENCE = 3     # numero di epoche senza miglioramento prima di fermarsi
EARLY_STOPPING_MIN_DELTA = 0.0  # quanto deve migliorare almeno la val_loss per essere considerato "miglioramento"


# ---------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------
def build_instruction_prompt() -> str:
    """
    Prompt 'istruzione' senza la risposta.
    Questo testo è quello che forniamo in input in fase di generazione.
    """
    
    N_FRAMES_PER_CLIP = 16  # deve coincidere con n_frames_to_take del dataset
    IMAGE_TOKEN_STR = "<|image_pad|>"
    
    # Ripeti il token immagine una volta per ogni frame del video
    image_tokens = " ".join([IMAGE_TOKEN_STR] * N_FRAMES_PER_CLIP)
    
    return (
        "You are a sign language translation model. "
        "Given the following sign language video frames {image_tokens}, "
        "translate it into English.\n\n"
        "Answer with the English sentence only.\n\n"
        "Translation:"
    )


def build_training_text(target_text: str) -> str:
    """
    Testo che usiamo in training come target.
    Qui, per semplicità, il modello deve rigenerare:
        [istruzione + spazio + frase_target]
    In seguito possiamo raffinarsi (loss solo sulla frase, ecc.).
    """
    instr = build_instruction_prompt()
    return instr + " " + target_text.strip()

def mask_prompt_tokens(labels: torch.Tensor, input_ids: torch.Tensor, processor) -> torch.Tensor:
    """
    Imposta a -100 (ignore_index) i token della parte di prompt
    (build_instruction_prompt()) nelle labels, così la loss viene
    calcolata solo sulla frase target.

    labels e input_ids hanno shape [B, T] ed inizialmente sono identici.
    """
    # Ottengo gli ID del prompt (senza special tokens)
    prompt_text = build_instruction_prompt()
    prompt_tok = processor.tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors=None,
    )
    prompt_ids = prompt_tok["input_ids"] # Ritorna il tensore dalla parte del prompt (solo la parte aggiunta da noi)
    prompt_len = len(prompt_ids)

    if prompt_len == 0:
        return labels

    # Lavoro su una copia, per sicurezza
    labels = labels.clone()
    ignore_index = -100

    batch_size, seq_len = labels.shape

    for i in range(batch_size): # Per ogni batch
        seq = input_ids[i].tolist() # tiro fuori la sequenza di token del prompt del batch i-esimo

        # Cerco dove inizia la sottosequenza prompt_ids dentro seq
        start_idx = -1
        max_j = seq_len - prompt_len + 1
        for j in range(max_j):
            if seq[j : j + prompt_len] == prompt_ids: # prende una finestra lunga quanto il prompt che inizia in posizione j e la confronto con i token del prompt pre-fatto
                start_idx = j                         # se matcha so che il prompt inizia lì
                break

        if start_idx == -1:
            # Non ho trovato esattamente il prompt: in questo caso per sicurezza non maschero nulla.
            continue

        end_idx = start_idx + prompt_len
        # Maschero tutto fino alla fine del prompt
        labels[i, :end_idx] = ignore_index

    return labels


# ---------------------------------------------------------------------
# DDP (Distributed Data Parallel) helpers
# ---------------------------------------------------------------------
def setup_ddp_if_needed() -> Dict[str, Any]:
    """
    Setup DistributedDataParallel se WORLD_SIZE > 1.
    Restituisce un dict con:
        world_size -> numero totale di processi coinvolti nel training
        rank -> ID globale del processo (univoco fra tutti)
        local_rank -> ID della GPU locale che deve usare quel processo
        device -> la GPU fisica dove mettere il modello (cuda:0, cuda:1, cuda:2…)
        is_main_process -> True solo per rank 0, cioè il processo “capo”
    """
    if not torch.cuda.is_available():
        # solo CPU -> niente DDP
        device = "cpu"
        return {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "device": device,
            "is_main_process": True,
        }
    # le variabili di ambiente e il parallelismo sono gestiti da torchrun (quando lancio torchrun --nproc_per_node=4 train.py, 4 processi distinti su 4 GPU eseguiranno parallelamente il train)
    world_size = int(os.environ.get("WORLD_SIZE", "1")) # ritorna quanti processi totali sto usando per addestrare il modello
    if world_size == 1:
        # single GPU
        device = "cuda:0"
        torch.cuda.set_device(device) # Tutti i tensori creati andranno automaticamente su "device"
        return {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "device": device,
            "is_main_process": True,
        }

    # Multi-GPU DDP (via torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    
    dist.init_process_group(backend="nccl") # Crea il gruppo di comunicazione tra tutti i processi/GPU (nccl ottimizzato per NVIDIA)

    is_main_process = (rank == 0)

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "is_main_process": is_main_process,
    }


def cleanup_ddp():
    """Chiude il process group se DDP è inizializzato."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

# ---------------------------------------------------------------------
# Checkpoint utils & logging
# ---------------------------------------------------------------------
def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """
    Restituisce il path dell'ultimo checkpoint in OUTPUT_DIR/checkpoints,
    oppure None se non ne trova.
    """
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    ckpts = sorted(ckpt_dir.glob("epoch_*_step_*.pt"))
    if not ckpts:
        return None

    return ckpts[-1]  # l'ultimo in ordine alfabetico (grazie a zeri padding)

def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    epochs_no_improve: int,
    output_dir: Path,
    is_main_process: bool,
):
    """
    Salva un checkpoint completo (model+optimizer+scaler+metadati).
    Salva solo dal processo principale (rank 0).
    """
    if not is_main_process:
        return

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"epoch_{epoch:02d}_step_{global_step:06d}.pt"
    print(f"[INFO] Saving checkpoint to {ckpt_path}")

    # Se il modello è DDP, vogliamo i pesi del 'module' interno (model è un oggetto DDP dove model.module è il vero modello)
    model_to_save = model.module if hasattr(model, "module") else model

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(state, ckpt_path)


def log_epoch_metrics(epoch: int, global_step: int, metrics: Dict[str, float], output_dir: Path, is_main_process: bool):
    """
    Logga le metriche per epoca in un file JSONL:
        outputs/.../logs/metrics.jsonl
    Ogni riga = dict con epoch, global_step, metriche...
    """
    if not is_main_process:
        return

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "metrics.jsonl"
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        **metrics,
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

    print(f"[LOG] Epoch {epoch} metrics: {payload}")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, processor, device: str, val_loader, max_batches: int, is_main_process: bool) -> Dict[str, float]:
    """
    Valutazione:
        - val_loss (teacher forcing)
        - BLEU/ROUGE generando testo dal modello

    Per semplicità, valutiamo solo su un sottoinsieme di batch (max_batches).
    """
    # In DDP, facciamo la valutazione solo sul rank 0, per semplicità.
    # Se volessimo fare average tra rank, dovremmo fare all_reduce.
    if not is_main_process:
        return {}

    model.eval()
    losses = []

    for b_idx, batch in enumerate(val_loader):  # per ogni batch, batch = {"images": ..., "texts": ..., "meta": ...}
        if b_idx >= max_batches:
            break

        images_batch = batch["images"]     # List[List[PIL.Image]] 
        texts_batch = batch["texts"]       # List[str] (ground truth)

        # ====== 1) Loss in teacher forcing ======
        train_texts = [build_training_text(t) for t in texts_batch]
        # Processor converte immagini + testo in tensori. input è un dict di tensori input = {"input_ids": token del testo, "pixel_values": tensori dei frame}
        inputs = processor( 
            images=images_batch,
            text=train_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()} # sposto tutti i tensori sulla GPU corretta
        labels = inputs["input_ids"].clone() # clono il tensore dei testi da predirre (in modo da tenerle come label)
        labels = mask_prompt_tokens(labels, inputs["input_ids"], processor)

        outputs = model(**inputs, labels=labels) # Forward
        losses.append(outputs.loss.item()) # memorizzo la loss

    # aggrego risultati
    if not losses:
        avg_loss = float("nan")
    else:
        avg_loss = float(sum(losses) / len(losses))

    metrics = {
        "val_loss": avg_loss,
    }

    model.train()
    return metrics


# ---------------------------------------------------------------------
# MAIN TRAINING
# ---------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ddp_info = setup_ddp_if_needed()                            # Ritorna il dict con tutte le info su ddp ed eventualmente lo inizializza
    rank = ddp_info["rank"]
    world_size = ddp_info["world_size"]
    local_rank = ddp_info["local_rank"]
    device = ddp_info["device"]
    is_main_process = ddp_info["is_main_process"]

    if is_main_process:
        print("==============================================")
        print("   QWEN3-VL LoRA — HOW2SIGN TRAINING")
        print("==============================================\n")
        print(f"[INFO] Root dir:       {PROJECT_ROOT}")
        print(f"[INFO] Dataset JSON:   {DATASET_JSON}")
        print(f"[INFO] Output dir:     {OUTPUT_DIR}")
        print(f"[INFO] World size:     {world_size}")
        print(f"[INFO] Rank / L-Rank:  {rank} / {local_rank}")
        print(f"[INFO] Device:         {device}\n")

    # -----------------------
    # 1) Modello + processor
    # -----------------------
    model, processor = load_qwen3vl_lora(
        model_name=MODEL_NAME,
        r=16,
        alpha=32,
        dropout=0.05,
        device=device,
    )

    # In multi-GPU, wrappiamo in DDP (1 processo per GPU)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(      # orchestra i vari processi
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,                       # non tutti i parametri vengono utilizzati in forward
        )

    # GradScaler
    if device.startswith("cuda"):
        scaler = amp.GradScaler("cuda")                      # serve a stabilizzare il gradiente allenando in float != 32
    else:
        scaler = None  # su CPU non serve

    # -----------------------
    # 2) Dataset + Dataloader
    # -----------------------
    train_ds = How2SignDataset(             # Costruisco un oggetto Dataset per training
        json_path=str(DATASET_JSON),
        split="train",
        root_dir=str(PROJECT_ROOT),
    )
    val_ds = How2SignDataset(               # Costruisco un oggetto Dataset per validation
        json_path=str(DATASET_JSON),
        split="val",
        root_dir=str(PROJECT_ROOT),
    )

    if is_main_process:
        print(f"[INFO] Train examples: {len(train_ds)}")
        print(f"[INFO] Val   examples: {len(val_ds)}")

    # In DDP, utilizziamo DistributedSampler per dividere il dataset tra i processi
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)  # creo i distributori di sample su piu GPU
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)     
        shuffle_flag = False  # con sampler, shuffle=False nel DataLoader
    else:
        train_sampler = None
        val_sampler = None
        shuffle_flag = True

    # Ricordo che i DataLoader sono oggetti iterabili (su cui posso fare "for batch in loader") e ogni batch in questo caso è batch = {"images": ..., "texts": ..., "meta": ...}
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle_flag,       # mischia i sample prima di costruire il batch
        sampler=train_sampler,      # distribuisce i sample (eventualmente su più GPU)
        num_workers=4,
        collate_fn=how2sign_collate_fn,     # funzione di batching personalizzata (dizionari non tensori)
        pin_memory=True,            # accelera il trasferimento CPU->GPU
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        collate_fn=how2sign_collate_fn,
        pin_memory=True,
    )

    # -----------------------
    # 3) Optimizer (solo LoRA trainabile)
    # -----------------------
    optimizer = torch.optim.AdamW(          # crea optimizer per aggiornare pesi    
        filter(lambda p: p.requires_grad, model.parameters()),     # aggiorna solo i parametri con requires_grad true (solo LoRA, il resto è freezed)
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------
    # 4) Stato training (resume / early stopping)
    # -----------------------
    global_step = 0                     # conto quanti update l’optimizer ha già fatto (1 per ogni batch)
    best_val_loss = float("inf")        # starto la loss iniziale ad infinito
    epochs_no_improve = 0               # contatore per early stopping

    start_epoch = 1

    if RESUME_FROM_LATEST:
        ckpt_path = find_latest_checkpoint(OUTPUT_DIR)
        if ckpt_path is not None:
            if is_main_process:
                print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
            # tutti i rank caricano lo stesso checkpoint
            map_location = device if device == "cpu" else device
            ckpt = torch.load(ckpt_path, map_location=map_location)

            # i pesi del modello funzionano sia con DDP che senza
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if scaler is not None and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])

            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("global_step", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)

            if is_main_process:
                print(
                    f"[INFO] Resume state -> start_epoch={start_epoch}, "
                    f"global_step={global_step}, best_val_loss={best_val_loss:.4f}, "
                    f"epochs_no_improve={epochs_no_improve}"
                )
        else:
            if is_main_process:
                print("[INFO] No checkpoint found, starting from scratch.")

    # Se il checkpoint era già oltre NUM_EPOCHS, non ha senso ripartire
    if start_epoch > NUM_EPOCHS:
        if is_main_process:
            print(f"[INFO] start_epoch={start_epoch} > NUM_EPOCHS={NUM_EPOCHS}, nothing to do. Exiting.")
        cleanup_ddp()
        return

    # -----------------------
    # 5) Training loop
    # -----------------------
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        if world_size > 1:
            # importantissimo per DDP: shuffle diverso per epoca
            train_sampler.set_epoch(epoch)

        if is_main_process:
            print(f"\n[INFO] Starting epoch {epoch}/{NUM_EPOCHS}")

        running_loss = 0.0                  # per il log ogni LOG_EVERY step
        epoch_loss_sum = 0.0                # somma delle loss su tutta l'epoca
        epoch_loss_count = 0                # numero di step (batch) nell'epoca

        model.train()

        # tqdm solo sul rank 0, gli altri usano iteratore “normale”
        data_iter = train_loader
        if is_main_process:
            data_iter = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)   # creo progress bar

        # azzeriamo i grad all'inizio
        optimizer.zero_grad()

        for step_in_epoch, batch in enumerate(data_iter, start=1):
            global_step += 1

            images_batch = batch["images"]  # List[List[PIL.Image]]
            texts_batch = batch["texts"]    # List[str]

            # Testo di training = istruzione + frase target
            full_texts = [build_training_text(t) for t in texts_batch]

            # Processor converte immagini + testo in tensori. input è un dict di tensori input = {"input_ids": token del testo, "pixel_values": tensori dei frame}
            inputs = processor(
                images=images_batch,
                text=full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()} # sposto tutti i tensori sulla GPU corretta
            labels = inputs["input_ids"].clone() # clono il tensore dei testi da predirre (in modo da tenerle come label)
            labels = mask_prompt_tokens(labels, inputs["input_ids"], processor)

            # Attivo autocast (modalità di PyTorch che abilita il mixed precision, FP16 dove è sicuro, FP32 dove serve più precisione)
            if device.startswith("cuda"): # solo su GPU
                autocast_ctx = torch.amp.autocast("cuda")
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                outputs = model(**inputs, labels=labels)    # Forward
                loss = outputs.loss                         # loss
                # gradient accumulation: faccio media sui GRAD_ACCUM_STEPS
                loss = loss / GRAD_ACCUM_STEPS              # siccome il “batch effettivo” è più grande di quello che entra in GPU

            # backward con o senza scaler
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * GRAD_ACCUM_STEPS  # torniamo alla loss "vera"
            
            # Accumulo la loss per calcolare la media dell'epoca
            epoch_loss_sum += loss.item() * GRAD_ACCUM_STEPS
            epoch_loss_count += 1

            # Se sono arrivato al batch size effettivo ottimizzo il gradiente accumulato (sempre con o senza scaler)
            if global_step % GRAD_ACCUM_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)          
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # evita esplosioni di gradiente
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # evita esplosioni di gradiente
                    optimizer.step()
                optimizer.zero_grad()

            # Logging solo su main process
            if is_main_process and (global_step % LOG_EVERY == 0):
                avg_loss = running_loss / LOG_EVERY
                running_loss = 0.0
                if isinstance(data_iter, tqdm):
                    data_iter.set_postfix({"loss": f"{avg_loss:.4f}"})
                print(f"[STEP {global_step}] train_loss = {avg_loss:.4f}")

            if MAX_STEPS is not None and global_step >= MAX_STEPS:
                if is_main_process:
                    print(f"[INFO] Reached MAX_STEPS={MAX_STEPS}, stopping training after epoch {epoch}.")
                break

        # ===== Fine epoca: valutazione =====
        metrics = evaluate(
            model=model,
            processor=processor,
            device=device,
            val_loader=val_loader,
            max_batches=MAX_VAL_BATCHES,
            is_main_process=is_main_process,
        )
        
        # Loss media di training per l'epoca (sui batch visti da questo processo)
        if epoch_loss_count > 0:
            avg_train_loss_epoch = epoch_loss_sum / epoch_loss_count
        else:
            avg_train_loss_epoch = float("nan")

        if is_main_process:
            val_loss = metrics.get("val_loss", float("inf"))
            print(
                f"[INFO] Epoch {epoch} — "
                f"train_loss: {avg_train_loss_epoch:.4f} | val_loss: {val_loss:.4f}"
            )
            
        # Aggiungo la train_loss media per l'epoca alle metriche
        metrics["train_loss"] = avg_train_loss_epoch

        # Log per epoca (loss + metriche)
        log_epoch_metrics(
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            output_dir=OUTPUT_DIR,
            is_main_process=is_main_process,
        )

        # ===== Checkpoint + early stopping (solo main process) =====
        stop_training = False

        if is_main_process:
            val_loss = metrics.get("val_loss", float("inf"))

            # Miglioramento?
            if val_loss < (best_val_loss - EARLY_STOPPING_MIN_DELTA):
                print(f"[INFO] New best val_loss={val_loss:.4f} (prev={best_val_loss:.4f}) -> saving checkpoint")
                best_val_loss = val_loss
                epochs_no_improve = 0

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    epochs_no_improve=epochs_no_improve,
                    output_dir=OUTPUT_DIR,
                    is_main_process=is_main_process,
                )
            else:
                epochs_no_improve += 1
                print(
                    f"[INFO] val_loss did not improve (best={best_val_loss:.4f}), "
                    f"epochs_no_improve={epochs_no_improve}/{EARLY_STOPPING_PATIENCE}"
                )

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                stop_training = True

        # Sincronizza stop_training fra tutti i rank in DDP
        if world_size > 1:
            stop_tensor = torch.tensor(1 if stop_training else 0, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            break

        if MAX_STEPS is not None and global_step >= MAX_STEPS:
            if is_main_process:
                print(f"[INFO] Reached MAX_STEPS={MAX_STEPS}, stopping training after epoch {epoch}.")
            break

    if is_main_process:
        print("\n[INFO] Training finished!")

    cleanup_ddp()


if __name__ == "__main__":
    main()