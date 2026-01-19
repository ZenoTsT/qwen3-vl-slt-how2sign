#!/usr/bin/env python
import os
import shutil
import sys
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any, List
import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch import amp
from tqdm import tqdm
import time
import random

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
from src.models.qwen3vl_lora_singlestage import load_qwen3vl_full_lora


# ---------------------------------------------------------------------
# Config di base
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
STAGE = "singlestage"  # "stage1" | "stage2" | "singlestage"

DATASET_JSON = PROJECT_ROOT / "data/How2Sign_resized/how2sign_dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs/qwen3vl_lora_how2sign"

BATCH_SIZE = 2              # effettiva
NUM_EPOCHS = 10              # per ora smoke test
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_ACCUM_STEPS = 16        # gradient accumulation (effettivo batch = BATCH_SIZE * GRAD_ACCUM_STEPS)
LOG_EVERY = 512              # step di logging
MAX_VAL_BATCHES = 64        # quante batch usare in val (per velocità)
MAX_GEN_TOKENS = 128         # max token generati per valutazione
MAX_STEPS = None            # se voglio fermare dopo N step globali, metto un int

RESUME_FROM_LATEST = True       # se True, prova a riprendere dall’ultimo checkpoint
EARLY_STOPPING_PATIENCE = 100     # numero di epoche senza miglioramento prima di fermarsi
EARLY_STOPPING_MIN_DELTA = 0.0  # quanto deve migliorare almeno la val_loss per essere considerato "miglioramento"

INTRA_SAVE_EVERY_STEPS = 256    # salvo un intra step ogni 256 global steps         
GEN_EVERY_STEPS = 512
GEN_N_EXAMPLES = 8

# --- Overfit test ---
OVERFIT_TEST = False
OVERFIT_N_SAMPLES = 32
OVERFIT_SEED = 123

def make_overfit_subset(ds, n: int, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[: min(n, len(idx))]
    idx.sort()
    return torch.utils.data.Subset(ds, idx)

# ---------------------------------------------------------------------
# Prompt (chat template) + masking 
# ---------------------------------------------------------------------
SYSTEM_PROMPT = "You are a sign language translation model."
USER_INSTRUCTION = (
    "Translate the sign language video into English.\n\n"
    "Answer with the English sentence only.\n\n"
    "Translation:"
)
ANCHOR_TEXT = "Translation:"


def build_messages(video_path: str, target_text: str | None = None) -> List[Dict[str, Any]]:
    """
    Multimodal chat:
      system: text
      user:   [video + instruction text]
      assistant (optional): target text
    """
    user_content = [
        {"type": "video", "video": video_path},
        {"type": "text", "text": USER_INSTRUCTION},
    ]
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if target_text is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target_text}]})
    return msgs


def _apply_chat_template(processor, messages, add_generation_prompt: bool) -> str:
    """
    Compat:
    - alcuni processor hanno apply_chat_template
    - altrimenti usare tokenizer.apply_chat_template
    """
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def make_batch_inputs_and_labels(
    processor,
    videos_batch: List[str],
    texts_batch: List[str],
    device: str,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    - costruisce full_text (prompt + risposta) via chat template
    - processor(videos=..., text=...) -> input_ids con token multimodali inclusi
    - masking: ignora tutto fino a (e incluso) "Translation:" cercato in input_ids
    """
    full_texts: List[str] = []
    for vp, tgt in zip(videos_batch, texts_batch):
        msgs_full = build_messages(vp, target_text=tgt)
        full_texts.append(_apply_chat_template(processor, msgs_full, add_generation_prompt=False))

    inputs = processor(
        videos=videos_batch,
        text=full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    ignore_index = -100

    anchor_ids = processor.tokenizer(
        ANCHOR_TEXT,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]

    B, T = input_ids.shape
    for b in range(B):
        seq = input_ids[b].tolist()
        start_idx = -1
        for j in range(0, T - len(anchor_ids) + 1):
            if seq[j : j + len(anchor_ids)] == anchor_ids:
                start_idx = j
                break

        if start_idx != -1:
            end_idx = start_idx + len(anchor_ids)
            labels[b, :end_idx] = ignore_index
        # fallback: se non troviamo l’anchor, non mascheriamo (meglio che mascherare tutto)

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = ignore_index

    return inputs, labels


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
def _get_stage_ckpt_dir(
    output_dir: Path,
    stage: str,
    kind: str,  # "intra_latest" | "epoch_latest" | "epoch_best"
) -> Path:
    """
    Directory radice per i checkpoint di uno stage e di un certo tipo.
    Esempio:
        outputs/.../checkpoints/stage1/epoch_best/
    """
    return output_dir / "checkpoints" / stage / kind


def save_lora_checkpoint(
    model,
    optimizer,
    scaler,
    epoch: int,
    step_in_epoch: int | None,
    global_step: int,
    best_val_loss: float,
    epochs_no_improve: int,
    output_dir: Path,
    stage: str,
    kind: str,              # "intra_latest" | "epoch_latest" | "epoch_best"
    is_main_process: bool,
):
    """
    Salva un checkpoint *solo LoRA* + stato di training, in formato uniforme,
    per:
        - intra_latest  (ripartenza dentro epoca)
        - epoch_latest  (ripartenza pulita tra epoche)
        - epoch_best    (miglior modello)

    """
    if not is_main_process:
        return

    ckpt_dir = _get_stage_ckpt_dir(output_dir, stage, kind)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = ckpt_dir / "adapter.pt"
    state_path = ckpt_dir / "state.pt"

    # Se il modello è wrappato in DDP, prendo il .module interno (PeftModel)
    model_to_save = model.module if hasattr(model, "module") else model

    # 1) Salvo SOLO i pesi LoRA (filtrando i parametri che contengono "lora_")
    lora_state = {
        name: p.detach().cpu()
        for name, p in model_to_save.state_dict().items()
        if "lora_" in name
    }

    torch.save(lora_state, adapter_path)

    # 2) Stato di training (uguale per intra_latest / epoch_latest / epoch_best)
    state = {
        "stage": stage,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,  # None per epoch_*
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(state, state_path)

    print(f"[CKPT] Saved {kind} checkpoint for {stage} at {ckpt_dir}")

def load_lora_checkpoint(
    model,
    optimizer,
    scaler,
    output_dir: Path,
    stage: str,
    device: str,
    is_main_process: bool,
):
    """
    Carica un checkpoint LoRA seguendo questa priorità:

        1) checkpoints/{stage}/intra_latest/{adapter.pt,state.pt}
        2) checkpoints/{stage}/epoch_latest/{adapter.pt,state.pt}

    Se non trova nulla, torna il modello com'è e uno stato di training "fresh".

    Cosa viene caricato:
        - adapter.pt: SOLO i pesi LoRA (parametri con 'lora_' nel nome)
                      che vengono fusi nello state_dict del modello
        - state.pt:   stato di training, con chiavi come:
                        - epoch
                        - step_in_epoch
                        - global_step
                        - best_val_loss
                        - epochs_no_improve
                        - optimizer (state_dict)
                        - scaler    (state_dict opzionale)
                        - stage     (opzionale, per sicurezza)
    """
    ckpt_root = output_dir / "checkpoints" / stage

    # Ordine di priorità: prima intra_latest (crash dentro epoca),
    # poi epoch_latest (ripartenza pulita tra epoche).
    candidates = ["intra_latest", "epoch_latest"]

    chosen_kind = None
    adapter_path = None
    state_path = None

    for kind in candidates:
        kind_dir = ckpt_root / kind
        a_path = kind_dir / "adapter.pt"
        s_path = kind_dir / "state.pt"

        if a_path.exists() and s_path.exists():
            chosen_kind = kind
            adapter_path = a_path
            state_path = s_path
            break

    # Nessun checkpoint trovato: ritorno stato iniziale
    if chosen_kind is None:
        if is_main_process:
            print("[INFO] No LoRA checkpoints found (intra_latest / epoch_latest). Starting from scratch.")

        training_state = {
            "epoch": 1,
            "step_in_epoch": 0,           # verrà interpretato dal training loop
            "global_step": 0,
            "best_val_loss": float("inf"),
            "epochs_no_improve": 0,
            "kind": None,
        }
        return model, optimizer, scaler, training_state

    # ------------------------------------------------------------
    # 1) Log di debug
    # ------------------------------------------------------------
    if is_main_process:
        print(f"[INFO] Loading LoRA checkpoint kind='{chosen_kind}' from: {state_path.parent}")

    # ------------------------------------------------------------
    # 2) Carico SOLO i pesi LoRA (adapter.pt)
    # ------------------------------------------------------------
    # adapter_state contiene solo i parametri LoRA (quelli che salveremo noi),
    # quindi li fondiamo nello state_dict attuale del modello.
    adapter_state = torch.load(adapter_path, map_location="cpu")

    # Se DDP, lavoriamo su model.module, come nel save
    model_to_load = model.module if hasattr(model, "module") else model
    model_state = model_to_load.state_dict()

    for k, v in adapter_state.items():
        if k in model_state:
            model_state[k] = v.to(device)
        else:
            if is_main_process:
                print(f"[WARN] LoRA key '{k}' not found in model.state_dict() — skipping.")

    # carica sul "vero" modello, non sul wrapper DDP
    model_to_load.load_state_dict(model_state)

    # ------------------------------------------------------------
    # 3) Carico lo stato di training (state.pt)
    # ------------------------------------------------------------
    train_state = torch.load(state_path, map_location="cpu")

    # Ottimizzatore
    if optimizer is not None and "optimizer" in train_state and train_state["optimizer"] is not None:
        optimizer.load_state_dict(train_state["optimizer"])

    # GradScaler
    if scaler is not None and train_state.get("scaler") is not None:
        scaler.load_state_dict(train_state["scaler"])

    # Meta-dati di training
    epoch = train_state.get("epoch", 1)
    step_in_epoch = train_state.get("step_in_epoch", 0)
    global_step = train_state.get("global_step", 0)
    best_val_loss = train_state.get("best_val_loss", float("inf"))
    epochs_no_improve = train_state.get("epochs_no_improve", 0)

    # (opzionale) check stage
    ckpt_stage = train_state.get("stage", None)
    if ckpt_stage is not None and ckpt_stage != stage and is_main_process:
        print(
            f"[WARN] Checkpoint stage='{ckpt_stage}' "
            f"does not match current STAGE='{stage}'. "
            "Are you sure this is the right checkpoint?"
        )

    training_state = {
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
        "kind": chosen_kind,  # 'intra_latest' o 'epoch_latest'
    }

    if is_main_process:
        print(
            f"[INFO] Loaded training state: "
            f"epoch={epoch}, step_in_epoch={step_in_epoch}, "
            f"global_step={global_step}, best_val_loss={best_val_loss:.4f}, "
            f"epochs_no_improve={epochs_no_improve}, kind={chosen_kind}"
        )

    return model, optimizer, scaler, training_state

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
def evaluate(model, processor, device: str, val_loader, max_batches: int) -> Dict[str, float]:
    """
    Valutazione:
        - val_loss (teacher forcing)

    Per semplicità, valutiamo solo su un sottoinsieme di batch (max_batches).
    """
    model.eval()

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_max = None if max_batches is None else int(math.ceil(max_batches / world_size))

    loss_sum = 0.0
    count = 0

    for b_idx, batch in enumerate(val_loader):
        if local_max is not None and b_idx >= local_max:
            break

        videos_batch = batch["videos"]
        texts_batch = batch["texts"]

        skip = 0
        err_msg = None
        try:
            inputs, labels = make_batch_inputs_and_labels(
                processor=processor,
                videos_batch=videos_batch,
                texts_batch=texts_batch,
                device=device,
            )
        except ValueError as e:
            msg = str(e)
            if ("temporal_factor" in msg) or ("must be larger" in msg and "t:" in msg):
                skip = 1
                err_msg = msg
            else:
                raise

        if dist.is_available() and dist.is_initialized():
            skip_t = torch.tensor(skip, device=device, dtype=torch.int32)
            dist.all_reduce(skip_t, op=dist.ReduceOp.MAX)
            skip = int(skip_t.item())

        if skip:
            if dist.is_available() and dist.is_initialized():
                # (opzionale) allineamento: tutti fanno la stessa cosa
                pass
            continue

        outputs = model(**inputs, labels=labels)
        loss_sum += float(outputs.loss.item())
        count += 1

    # riduzione globale
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([loss_sum, count], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_sum_global = float(t[0].item())
        count_global = int(t[1].item())
    else:
        loss_sum_global = loss_sum
        count_global = count

    avg_loss = loss_sum_global / max(count_global, 1)

    model.train()
    return {"val_loss": avg_loss}


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
    if STAGE == "stage1":
        model, processor = load_qwen3vl_lora(
            model_name=MODEL_NAME,
            r=16, alpha=32, dropout=0.05,
            device=device,
            stage="stage1",
        )
    elif STAGE == "stage2":
        model, processor = load_qwen3vl_lora(
            model_name=MODEL_NAME,
            r=16, alpha=32, dropout=0.05,
            device=device,
            stage="stage2",
            stage1_adapter_dir=str(OUTPUT_DIR / "checkpoints" / "stage1" / "epoch_best"),
        )
    else:
        model, processor = load_qwen3vl_full_lora(
            model_name=MODEL_NAME,
            r=16, alpha=32, dropout=0.05,
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
    train_ds_full = How2SignDataset(
        json_path=str(DATASET_JSON),
        split="train",
        root_dir=str(PROJECT_ROOT),
        return_type="video",
    )

    if OVERFIT_TEST:
        train_ds = make_overfit_subset(train_ds_full, OVERFIT_N_SAMPLES, OVERFIT_SEED)
        val_ds = train_ds
    else:
        train_ds = train_ds_full
        val_ds = How2SignDataset(
            json_path=str(DATASET_JSON),
            split="val",
            root_dir=str(PROJECT_ROOT),
            return_type="video",
        )

    if is_main_process:
        print(f"[INFO] Train examples: {len(train_ds)}")
        print(f"[INFO] Val   examples: {len(val_ds)}")

    # In DDP, utilizziamo DistributedSampler per dividere il dataset tra i processi
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        shuffle_flag = False
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
        num_workers=2,
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
    # default: training "fresh"
    training_state = {
        "epoch": 1,
        "step_in_epoch": 0,
        "global_step": 0,
        "best_val_loss": float("inf"),
        "epochs_no_improve": 0,
        "kind": None,
    }

    if RESUME_FROM_LATEST:
        # carica (se esiste) intra_latest o epoch_latest
        model, optimizer, scaler, training_state = load_lora_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            output_dir=OUTPUT_DIR,
            stage=STAGE,
            device=device,
            is_main_process=is_main_process,
        )

    # estraggo lo stato
    ckpt_kind = training_state["kind"]           # None | "intra_latest" | "epoch_latest"
    ckpt_epoch = training_state["epoch"]
    step_resume = training_state["step_in_epoch"]
    global_step = training_state["global_step"]
    best_val_loss = training_state["best_val_loss"]
    epochs_no_improve = training_state["epochs_no_improve"]

    # log riassuntivo
    if is_main_process:
        print(
            f"[INFO] Initial training state -> "
            f"epoch={ckpt_epoch}, step_in_epoch={step_resume}, "
            f"global_step={global_step}, best_val_loss={best_val_loss:.4f}, "
            f"epochs_no_improve={epochs_no_improve}, kind={ckpt_kind}"
        )

    # se ho un epoch_latest, riparto dall'epoca successiva
    if ckpt_kind == "epoch_latest":
        start_epoch = ckpt_epoch + 1
        step_resume = 0  # ricomincio da inizio epoca
    else:
        # fresh oppure intra_latest: riparto dalla stessa epoca
        start_epoch = ckpt_epoch

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
        
        # >>> DEBUG: per misurare tempo data loading / step
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_prev = time.perf_counter()
        # <<< DEBUG
        
        # Flag: sto riprendendo da un intra_latest in QUESTA epoca?
        resuming_in_this_epoch = (
            ckpt_kind == "intra_latest"
            and epoch == ckpt_epoch
            and step_resume > 0
        )

        for step_in_epoch, batch in enumerate(data_iter, start=1):
            # --------- SKIP BATCHES GIÀ FATTI (intra_latest) ----------
            if resuming_in_this_epoch and step_in_epoch <= step_resume:
                if is_main_process and step_in_epoch == 1:
                    print(
                        f"[INFO] Resuming from intra_latest: "
                        f"skipping first {step_resume} steps of epoch {epoch}."
                    )
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_prev = time.perf_counter()
                continue
            
            global_step += 1
            
            # >>> DEBUG: tempo data loading
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t_batch_start = time.perf_counter()
            data_loading_time = t_batch_start - t_prev
            # <<< DEBUG

            # images_batch = batch["images"]  # List[List[PIL.Image]]
            videos_batch = batch["videos"]  # List[str] (path ai video)
            texts_batch = batch["texts"]    # List[str]

            # >>> DEBUG: tempo processor (decoding + preprocess)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # --- DDP-safe skip per video "rotto" (es. t=1 frame) ---
            skip = 0
            err_msg = None

            try:
                inputs, labels = make_batch_inputs_and_labels(
                    processor=processor,
                    videos_batch=videos_batch,
                    texts_batch=texts_batch,
                    device=device,
                )
            except ValueError as e:
                msg = str(e)
                # caso tipico: "t:1 must be larger than temporal_factor:2"
                if ("temporal_factor" in msg) or ("must be larger" in msg and "t:" in msg):
                    skip = 1
                    err_msg = msg
                else:
                    raise  # altri ValueError li vogliamo vedere subito

            # sincronizzo lo "skip" tra rank (se uno deve skip, skippano tutti)
            if dist.is_available() and dist.is_initialized():
                skip_t = torch.tensor(skip, device=device, dtype=torch.int32)
                dist.all_reduce(skip_t, op=dist.ReduceOp.MAX)
                skip = int(skip_t.item())

            if skip:
                if is_main_process:
                    print(f"[WARN] Skipping batch due to video preprocess error: {err_msg}")
                    # stampo anche i path (molto utile per blacklisting)
                    print(f"[WARN] videos_batch={videos_batch}")
                # IMPORTANTISSIMO: NON fare backward/step su alcuni rank e non su altri
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_prev = time.perf_counter()
                continue
            # --- fine DDP-safe skip ---
            
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            prep_time = t1 - t0

            # Attivo autocast (modalità di PyTorch che abilita il mixed precision, FP16 dove è sicuro, FP32 dove serve più precisione)
            if device.startswith("cuda"): # solo su GPU
                autocast_ctx = torch.amp.autocast("cuda")
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                outputs = model(**inputs, labels=labels)    # FORWARD !!!
                loss = outputs.loss                         # loss
                # gradient accumulation: faccio media sui GRAD_ACCUM_STEPS
                loss = loss / GRAD_ACCUM_STEPS              # siccome il “batch effettivo” è più grande di quello che entra in GPU
    

            # backward con o senza scaler
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # >>> DEBUG: tempo forward+backward
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            fwd_bwd_time = t2 - t1
            # <<< DEBUG

            running_loss += loss.item() * GRAD_ACCUM_STEPS  # torniamo alla loss "vera"
            
            # Accumulo la loss per calcolare la media dell'epoca
            epoch_loss_sum += loss.item() * GRAD_ACCUM_STEPS
            epoch_loss_count += 1

            # Se sono arrivato al batch size effettivo ottimizzo il gradiente accumulato (sempre con o senza scaler)
            if global_step % GRAD_ACCUM_STEPS == 0:
                
                if is_main_process:
                    print(f"[update {global_step // GRAD_ACCUM_STEPS}] last_micro_loss={loss.item() * GRAD_ACCUM_STEPS:.4f}")
                
                if scaler is not None:
                    scaler.unscale_(optimizer)          
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # evita esplosioni di gradiente
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # evita esplosioni di gradiente
                    optimizer.step()
                optimizer.zero_grad()
                
                if global_step % INTRA_SAVE_EVERY_STEPS == 0:
                    # --------- SALVA intra_latest DOPO UNO STEP PULITO ----------
                    save_lora_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        epochs_no_improve=epochs_no_improve,
                        output_dir=OUTPUT_DIR,
                        stage=STAGE,
                        kind="intra_latest",
                        is_main_process=is_main_process,
                    )
                    # ------------------------------------------------------------
                    
                # ---- quick qualitative check every GEN_EVERY_STEPS ----
                if global_step % GEN_EVERY_STEPS == 0:
                    if world_size > 1 and val_sampler is not None:
                        val_sampler.set_epoch(epoch)
                    
                    log_generation_examples(
                        model=model,
                        processor=processor,
                        device=device,
                        val_loader=val_loader,
                        n_examples_total=GEN_N_EXAMPLES,   
                        max_new_tokens=MAX_GEN_TOKENS,
                        is_main_process=is_main_process,
                    )
                    model.train()

            # >>> DEBUG: log tempi ogni LOG_EVERY step
            if is_main_process and (global_step % LOG_EVERY == 0):
                avg_loss = running_loss / LOG_EVERY
                running_loss = 0.0
                if isinstance(data_iter, tqdm):
                    data_iter.set_postfix({"loss": f"{avg_loss:.4f}"})
                print(
                    f"[STEP {global_step}] "
                    f"loss={avg_loss:.4f} | "
                    f"data={data_loading_time:.2f}s | "
                    f"prep={prep_time:.2f}s | "
                    f"fwd+bwd={fwd_bwd_time:.2f}s"
                )
            # <<< DEBUG

            # >>> DEBUG: tempo fine step per misura data loading prossimo step
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t_prev = time.perf_counter()
            # <<< DEBUG

            if MAX_STEPS is not None and global_step >= MAX_STEPS:
                if is_main_process:
                    print(f"[INFO] Reached MAX_STEPS={MAX_STEPS}, stopping training after epoch {epoch}.")
                break

        # ===== Fine epoca: valutazione =====
        if world_size > 1 and val_sampler is not None:
            val_sampler.set_epoch(epoch)
        
        metrics = evaluate(
            model=model,
            processor=processor,
            device=device,
            val_loader=val_loader,
            max_batches=MAX_VAL_BATCHES,
        )
        
        # ---- train_loss globale (somma su tutti i rank) ----
        train_sum = float(epoch_loss_sum)
        train_cnt = int(epoch_loss_count)

        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([train_sum, train_cnt], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            train_sum = float(t[0].item())
            train_cnt = int(t[1].item())

         # Aggiungo la train_loss media per l'epoca alle metriche
        avg_train_loss_epoch = train_sum / max(train_cnt, 1)
        metrics["train_loss"] = avg_train_loss_epoch

       

        if is_main_process:
            val_loss = metrics.get("val_loss", float("inf"))
            print(
                f"[INFO] Epoch {epoch} — "
                f"train_loss: {avg_train_loss_epoch:.4f} | val_loss: {val_loss:.4f}"
            )
            
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

                # ---- epoch_best (miglior modello) ----
                save_lora_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step_in_epoch=None,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    epochs_no_improve=epochs_no_improve,
                    output_dir=OUTPUT_DIR,
                    stage=STAGE,
                    kind="epoch_best",
                    is_main_process=is_main_process,
                )
            else:
                epochs_no_improve += 1
                print(
                    f"[INFO] val_loss did not improve (best={best_val_loss:.4f}), "
                    f"epochs_no_improve={epochs_no_improve}/{EARLY_STOPPING_PATIENCE}"
                )
                
            # ---- epoch_latest (ultima epoca COMPLETATA) ----
            save_lora_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step_in_epoch=None,
                global_step=global_step,
                best_val_loss=best_val_loss,
                epochs_no_improve=epochs_no_improve,
                output_dir=OUTPUT_DIR,
                stage=STAGE,
                kind="epoch_latest",
                is_main_process=is_main_process,
            )
            
            # Dopo aver salvato epoch_latest, l'intra_latest non serve più
            intra_dir = _get_stage_ckpt_dir(OUTPUT_DIR, STAGE, "intra_latest")
            if intra_dir.exists():
                shutil.rmtree(intra_dir, ignore_errors=True)

            # Early stopping?
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

@torch.no_grad()
def generate_translations(model, processor, device: str, videos: List[str], max_new_tokens: int) -> List[str]:
    m = model.module if hasattr(model, "module") else model
    m.eval()

    prompts: List[str] = []
    for vp in videos:
        msgs = build_messages(vp, target_text=None)
        prompts.append(_apply_chat_template(processor, msgs, add_generation_prompt=True))

    inputs = processor(
        videos=videos,
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    gen_ids = m.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    attn = inputs["attention_mask"]
    prompt_lens = attn.sum(dim=1).tolist()  # length reale per sample

    decoded = []
    for i, plen in enumerate(prompt_lens):
        gen_only_i = gen_ids[i, plen:]
        decoded.append(processor.tokenizer.decode(gen_only_i, skip_special_tokens=True).strip())

    cleaned = []
    for s in decoded:
        s2 = s.strip()
        if "\n" in s2:
            parts = [p.strip() for p in s2.split("\n") if p.strip()]
            if parts:
                s2 = parts[-1]
        cleaned.append(s2)
    return cleaned


@torch.no_grad()
def log_generation_examples(
    model,
    processor,
    device: str,
    val_loader,
    n_examples_total: int,
    max_new_tokens: int,
    is_main_process: bool,
):
    """
    Distributed generation:
      - ogni rank genera su ~n_examples_total/world_size esempi del proprio shard
      - poi all_gather_object e stampa su rank0
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    n_per_rank = int(math.ceil(n_examples_total / world_size))

    model.eval()

    local_pairs = []
    for batch in val_loader:
        videos_batch = batch["videos"]
        texts_batch = batch["texts"]

        skip = 0
        try:
            preds = generate_translations(
                model=model,
                processor=processor,
                device=device,
                videos=videos_batch,
                max_new_tokens=max_new_tokens,
            )
        except ValueError as e:
            msg = str(e)
            if ("temporal_factor" in msg) or ("must be larger" in msg and "t:" in msg):
                skip = 1
                preds = [""] * len(texts_batch)
            else:
                raise

        if dist.is_initialized():
            skip_t = torch.tensor(skip, device=device, dtype=torch.int32)
            dist.all_reduce(skip_t, op=dist.ReduceOp.MAX)
            skip = int(skip_t.item())

        if skip:
            # se uno skippa, skippano tutti quel batch e passano al prossimo
            continue

        for ref, pred in zip(texts_batch, preds):
            local_pairs.append((ref, pred))
            if len(local_pairs) >= n_per_rank:
                break

        if len(local_pairs) >= n_per_rank:
            break

    # gather su rank0
    if dist.is_initialized():
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_pairs)  # list of lists
    else:
        gathered = [local_pairs]

    if is_main_process:
        # flatten e taglia al totale richiesto
        flat = []
        for r, pairs in enumerate(gathered):
            for ref, pred in pairs:
                flat.append((r, ref, pred))

        flat = flat[:n_examples_total]

        for r, ref, pred in flat:
            print("----- GEN DEBUG -----")
            print(f"RANK {r}")
            print("REF :", ref)
            print("PRED:", pred)
            print("---------------------\n")

    model.train()

if __name__ == "__main__":
    main()