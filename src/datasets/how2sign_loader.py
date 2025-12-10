# src/datasets/how2sign_dataset.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.utils.video_io import extract_frames_from_video


class How2SignDataset(Dataset):
    """
    Dataset per How2Sign a livello di clip.

    Legge il file JSON (how2sign_dataset.json) e:
      - risolve il path del video
      - estrae i frame on-the-fly
      - restituisce frames + testo + metadati

    Il JSON è quello che abbiamo appena creato con build_how2sign_json.py.
    """

    def __init__(
        self,
        json_path: str = "data/How2Sign/how2sign_dataset.json",
        split: str = "train",                      # "train" | "val" | "test"
        n_frames_to_take: Optional[int] = 32,      # None = tutti i frame
        frame_sampling_strategy: str = "uniform",  # "uniform" | "consecutive" | "center" | "random" | "fps2_max32"
        root_dir: Optional[str] = None,
        return_type: str = "video",               # "images" | "video"
    ) -> None:
        super().__init__()                          # chiamo il costruttore della classe Dataset (per __getitem__ ecc ecc)

        # modalità di ritorno del dataset: "images" (frame estratti) o "video" (solo path al file video)
        self.return_type = return_type
        if self.return_type not in ("images", "video"):
            raise ValueError(
                f"return_type deve essere 'images' oppure 'video', trovato: {self.return_type}"
            )

        self.json_path = Path(json_path).resolve()
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON non trovato: {self.json_path}")

        with self.json_path.open("r", encoding="utf-8") as f: # apre il json in modalità read
            data = json.load(f)                               # il json diventa un dict

        self.meta: Dict[str, Any] = data.get("meta", {})                        # salvo i metadatati (campo meta del json)
        self.splits: Dict[str, List[Dict[str, Any]]] = data.get("splits", {})   # salvo le entry del dataset (splits meta del json)

        if split not in self.splits:
            raise ValueError(f"Split '{split}' non presente nel JSON. Split disponibili: {list(self.splits.keys())}")

        self.split = split                                          
        self.entries: List[Dict[str, Any]] = self.splits[split] # salvo se entry dello split selezionato

        self.n_frames_to_take = n_frames_to_take
        self.frame_sampling_strategy = frame_sampling_strategy
        
        # rimposto un eventuale root dir
        meta_root = self.meta.get("root_dir")
        if root_dir is not None:
            self.root_dir = Path(root_dir).resolve()
        elif meta_root is not None:
            self.root_dir = Path(meta_root).resolve()
        else:
            self.root_dir = self.json_path.parents[2]

        print(
            f"[How2SignDataset] split={self.split} | num_samples={len(self.entries)} | "
            f"n_frames_to_take={self.n_frames_to_take} | strategy={self.frame_sampling_strategy} | "
            f"return_type={self.return_type}"
        )
        print(f"[How2SignDataset] json_path={self.json_path}")
        print(f"[How2SignDataset] root_dir={self.root_dir}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]

        # Path relativo del video (es. "data/How2Sign/train_raw_videos/....mp4")
        video_rel = Path(entry["video_path"])
        video_abs = (self.root_dir / video_rel).resolve()

        if not video_abs.exists():
            raise FileNotFoundError(f"Video non trovato: {video_abs}")

        # Testo target (ground truth)
        sentence: str = entry["sentence"]

        if self.return_type == "images":
            # Estrae i frame on-the-fly
            frames = extract_frames_from_video(
                video_path=str(video_abs),
                n_frames_to_take=self.n_frames_to_take,
                strategy=self.frame_sampling_strategy,
            )

            sample: Dict[str, Any] = {
                "images": frames,                 # List[PIL.Image]
                "target_text": sentence,          # stringa inglese ground truth
                "split": entry["split"],
                "video_path": str(video_abs),
                "video_rel_path": entry["video_path"],
                "video_id": entry["video_id"],
                "video_name": entry["video_name"],
                "sentence_id": entry["sentence_id"],
                "sentence_name": entry["sentence_name"],
                "start_realigned": entry["start_realigned"],
                "end_realigned": entry["end_realigned"],
            }
        else:
            # modalità "video": non estraiamo frame, lasciamo che sia il processor (es. Qwen3VLVideoProcessor)
            # a gestire il video a partire dal path.
            sample: Dict[str, Any] = {
                # nessun campo "images" qui: il collate_fn gestirà questa modalità restituendo "videos"
                "target_text": sentence,          # stringa inglese ground truth
                "split": entry["split"],
                "video_path": str(video_abs),
                "video_rel_path": entry["video_path"],
                "video_id": entry["video_id"],
                "video_name": entry["video_name"],
                "sentence_id": entry["sentence_id"],
                "sentence_name": entry["sentence_name"],
                "start_realigned": entry["start_realigned"],
                "end_realigned": entry["end_realigned"],
            }

        return sample


def how2sign_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]: # batch è una lista di sample (come quelli sopra) data dal DataLoader di torch
    """
    Collate function semplice:
      - raggruppa le liste di frame
      - raggruppa i testi
      - tiene i metadati in una lista di dict

    Il batching "vero" per Qwen3-VL (processor -> tensori) lo faremo nello
    script di training, dove avremo accesso all'oggetto `processor`.
    """

    # Se i sample hanno il campo "images", siamo in modalità frame/immagini (comportamento originale).
    if "images" in batch[0]:
        images_batch = [sample["images"] for sample in batch]          # List[List[PIL.Image]]
        texts_batch = [sample["target_text"] for sample in batch]      # List[str]

        # meta = tutto tranne immagini e testo
        meta_batch: List[Dict[str, Any]] = []
        for sample in batch:
            meta = {
                k: v
                for k, v in sample.items()
                if k not in ("images", "target_text")
            }
            meta_batch.append(meta)

        return {
            "images": images_batch, 
            "texts": texts_batch,
            "meta": meta_batch,
        }

    # Altrimenti assumiamo modalità "video": nessun campo "images", ma abbiamo "video_path" + "target_text".
    videos_batch = [sample["video_path"] for sample in batch]         # List[str] (path assoluti dei video)
    texts_batch = [sample["target_text"] for sample in batch]         # List[str]

    # meta = tutto tranne immagini e testo (qui non ci sono immagini, ma manteniamo la stessa logica)
    meta_batch: List[Dict[str, Any]] = []
    for sample in batch:
        meta = {
            k: v
            for k, v in sample.items()
            if k not in ("images", "target_text")
        }
        meta_batch.append(meta)

    return {
        "videos": videos_batch, 
        "texts": texts_batch,
        "meta": meta_batch,
    }