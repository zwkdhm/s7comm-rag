"""Build the ENTITY-relation multi-hop knowledge base.

Counterpart to multihop_build_kb.py. The decisive knowledge is a per-tank ROLE
document: it names the tank/removal-step and states which physical quantity that
tank's normalized 0-100 % signal represents. These role docs are deliberately
written WITHOUT the statistics vocabulary ("0-100 / percent / normalized /
smooth"), so the bare statistics query cannot surface them; they are reached only
when the entity graph augments the query with the tank's entity terms
(copper_removal / cobalt_removal / concentration / potential).

Distractor notes carry the statistics vocabulary and occupy the top-k of the
statistics query, keeping the role docs off the first hop.
"""
import os
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "data/vector_kb_entity"
COLLECTION  = "entity_knowledge"
EMBED_MODEL = "/home/ind-bert/models/all-MiniLM-L6-v2"

DOCS = [
    # ---- per-tank ROLE docs (decisive; reachable only via entity context) ---
    {
        "id": "role_copper_removal",
        "source": "copper_removal_tank_role",
        # names the tank + the quantity; free of 0-100/percent/smooth wording
        "content": (
            "Copper-removal tank instrumentation role, zinc purification stage. "
            "In the copper-removal (cementation) tank the primary regulated "
            "analog signal reports the residual metal-ION CONCENTRATION in "
            "solution, tracked so operators can dose zinc powder to keep copper "
            "ions within spec. The controlled variable of this tank is a "
            "concentration, family = concentration."
        ),
    },
    {
        "id": "role_cobalt_removal",
        "source": "cobalt_removal_tank_role",
        "content": (
            "Cobalt-removal tank instrumentation role, zinc purification stage. "
            "Cobalt removal is redox-controlled: the primary regulated analog "
            "signal of the cobalt-removal tank reports an electrode POTENTIAL "
            "(redox potential) that governs the cementation of cobalt with "
            "activators. The controlled variable of this tank is a potential, "
            "family = potential."
        ),
    },
    # ---- distractors carrying the statistics vocabulary (occupy hop-1) ------
    {
        "id": "distractor_normalized_span",
        "source": "plant_note_normalized_span",
        "content": (
            "Purification-stage transmitter note. Many analog signals in the "
            "purification tanks are transmitted to the PLC as a smooth normalized "
            "0-100 percent span rather than in engineering units, so on the wire "
            "they are statistically indistinguishable non-negative values. This "
            "note gives no rule for which quantity a given 0-100 percent signal is."
        ),
    },
    {
        "id": "distractor_calibration",
        "source": "plant_note_calibration",
        "content": (
            "Instrument calibration note. Field transmitters are scaled to a "
            "smooth normalized 0-100 percent span before transmission to the PLC "
            "as non-negative analog signals. Describes calibration procedure only; "
            "does not state which physical quantity a signal represents."
        ),
    },
    {
        "id": "distractor_monitoring",
        "source": "plant_note_monitoring",
        "content": (
            "Process monitoring note. Operators watch smooth normalized 0-100 "
            "percent analog trends across the purification train for stability. "
            "Covers monitoring workflow only; gives no mapping from a normalized "
            "percent signal to a physical quantity or tank role."
        ),
    },
]


def main():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(name=COLLECTION, embedding_function=embed_fn)
    col.add(documents=[d["content"] for d in DOCS],
            metadatas=[{"source": d["source"]} for d in DOCS],
            ids=[d["id"] for d in DOCS])
    print(f"Indexed {len(DOCS)} docs -> {PERSIST_DIR}/{COLLECTION}")

    # Verify: bare stats query should surface distractors, NOT the role docs;
    # entity-augmented query (with tank terms) should surface the role doc.
    stat_q = ("industrial PLC process variable, normalized 0-100 percent signal, "
              "smooth non-negative analog signal")
    print(f"\n[bare stats query] top-3:")
    for m in col.query(query_texts=[stat_q], n_results=3)["metadatas"][0]:
        print(f"   -> {m['source']}")

    aug_q = stat_q + "; related process context: purification, cobalt, potential"
    print(f"\n[entity-augmented query: cobalt] top-3:")
    for m in col.query(query_texts=[aug_q], n_results=3)["metadatas"][0]:
        print(f"   -> {m['source']}")


if __name__ == "__main__":
    main()
