"""Build the multi-hop vector KB (separate collection, isolated from thesis KB).

The corpus encodes the decisive knowledge as a TWO-HOP chain so that a single
retrieval round cannot solve the ambiguous 0-100 signals, but a reflective
second hop can:

  hop-1  BRIDGE note — retrievable from the statistics query (mentions the
         electrowinning stage and 0-100 signals). It deliberately withholds the
         answer and instead points to a named reference
         ("cellhouse signal-role reference table"). This pointer phrase is what
         the assessor keys on to trigger a query rewrite.

  hop-2  ROLE reference — only retrievable once the query is rewritten with the
         bridge's pointer phrase. It gives the fine rule that maps the temporal
         signature to the family:
           very low cv + high autocorrelation  => normalized cell potential
           larger slow drift (integrating)      => tank level

Two distractor docs ensure hop-2 is NOT accidentally top-k on the hop-1 query.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "data/vector_kb_multihop"
COLLECTION  = "multihop_knowledge"
EMBED_MODEL = "/home/ind-bert/models/all-MiniLM-L6-v2"

# Pointer phrase shared by hop-1 (mentions it) and hop-2 (titled with it).
# A unique catalogue code makes hop-2 retrievable ONLY via this exact phrase,
# not via the statistics query — enforcing the two-hop structure.
BRIDGE_POINTER = "reference table ER-CH-07"

DOCS = [
    # ---- hop-1 : reachable from the statistics query, but answer withheld ----
    {
        "id": "bridge_electrowinning",
        "source": "plant_note_electrowinning",
        "content": (
            "Zinc electrowinning (cellhouse) stage instrumentation note, PLC data "
            "block DB1. Several analog signals in this stage are transmitted to "
            "the PLC as a "
            "normalized 0-100 percent span rather than in engineering units, so "
            "on the wire they are statistically indistinguishable: they share the "
            "same non-negative 0-100 range and smooth behaviour. This note does "
            "NOT resolve which physical quantity each 0-100 signal represents. "
            f"To disambiguate such a normalized cellhouse signal you MUST consult "
            f"{BRIDGE_POINTER} (the cellhouse signal-role catalogue), which gives "
            "the temporal-signature decision rule for each signal role. Do not "
            "guess from the 0-100 range alone."
        ),
    },
    # ---- hop-2 : reachable ONLY after rewrite with the pointer phrase ---------
    # Deliberately free of "0-100 / percent / signal / smooth / non-negative"
    # wording so the statistics query cannot surface it in round 1; only the
    # exact catalogue code "ER-CH-07" retrieves it.
    {
        "id": "role_reference",
        "source": "cellhouse_signal_role_reference",
        "content": (
            f"{BRIDGE_POINTER.capitalize()} — cellhouse signal-role catalogue, "
            "decision rule entry. Classification by temporal signature: "
            "(1) an item held very tightly by the rectifier control loop, showing "
            "coefficient of variation below 0.05 with lag-1 autocorrelation above "
            "0.9, is the cell POTENTIAL (electrode potential), family = potential. "
            "(2) an item that drifts more freely, with larger coefficient of "
            "variation and slower integrating ramps, is the tank LEVEL, family = "
            "level. Both entries occupy the identical engineering span, therefore "
            "only the coefficient of variation and the autocorrelation distinguish "
            "potential from level in catalogue ER-CH-07."
        ),
    },
    # ---- distractors : keep hop-2 off the hop-1 top-k ------------------------
    {
        "id": "distractor_leaching",
        "source": "plant_note_leaching",
        "content": (
            "Leaching stage note. Sulphuric acid leaching of roasted zinc "
            "calcine dissolves zinc oxide into zinc sulphate solution. Acid "
            "concentration and slurry density are monitored in g/L; these are "
            "unrelated to normalized percent signals and are not electrowinning "
            "cellhouse variables."
        ),
    },
    {
        "id": "distractor_purification",
        "source": "plant_note_purification",
        "content": (
            "Purification stage note. Copper and cobalt are removed from zinc "
            "sulphate solution by zinc-dust cementation. Ion concentrations are "
            "tracked in g/L and redox potential in millivolts (mV, negative "
            "range). This stage does not use normalized 0-100 percent cellhouse "
            "signals."
        ),
    },
    # ---- filler distractors : occupy the top-k of the statistics query so the
    #      hop-2 role reference does NOT surface until the query is rewritten ---
    {
        "id": "distractor_cellhouse_ops",
        "source": "plant_note_cellhouse_ops",
        "content": (
            "Electrowinning cellhouse operations note. The cellhouse runs banks "
            "of electrolytic cells drawing normalized 0-100 percent analog "
            "signals to the PLC for smooth non-negative process monitoring. "
            "Operators track stripping cycles, cathode harvesting schedules and "
            "busbar temperature. This note covers workflow only and gives no rule "
            "for identifying what any individual 0-100 percent signal represents."
        ),
    },
    {
        "id": "distractor_rectifier",
        "source": "plant_note_rectifier",
        "content": (
            "Rectifier and busbar note. The cellhouse rectifier supplies smooth "
            "regulated direct current to the cell banks; a normalized 0-100 "
            "percent analog value reports rectifier loading to the PLC. General "
            "background on power delivery; contains no decision rule mapping a "
            "0-100 percent signal to a physical quantity."
        ),
    },
    {
        "id": "distractor_acidmist",
        "source": "plant_note_acidmist",
        "content": (
            "Acid-mist and ventilation note. Cellhouse acid-mist extraction fans "
            "are monitored as smooth normalized 0-100 percent analog signals for "
            "non-negative process values. Purely environmental context; no rule "
            "for classifying the physical role of a 0-100 percent signal."
        ),
    },
    {
        "id": "distractor_calibration",
        "source": "plant_note_calibration",
        "content": (
            "Instrument calibration note. Field transmitters in the cellhouse are "
            "scaled to a smooth normalized 0-100 percent span before transmission "
            "to the PLC as non-negative analog signals. Describes calibration "
            "procedure only; does not state which quantity a given signal is."
        ),
    },
    # ==== generalization chains: same pointer-follow structure, DIFFERENT code
    #      formats, to show the regex extractor keys on the abstract "reference
    #      code" pattern rather than the literal string "ER-CH-07" ============
    # ---- chain B : Modbus "see Table X-Y" register-map cross-reference --------
    {
        "id": "bridge_modbus_register",
        "source": "plant_note_modbus_map",
        "content": (
            "Modbus register-map note — thickener underflow skid, Modbus unit-id 3 "
            "(mapped to PLC data block DB3). Holding "
            "registers in the 40010-40040 block carry raw normalized 0-100 "
            "percent process values; on the wire they are indistinguishable "
            "integers with no engineering unit attached. This map does NOT list "
            "which quantity each register encodes. For per-register scaling and "
            "the physical role of each normalized value, refer to Table 7-3 (the "
            "register scaling catalogue). Do not infer units from the raw span."
        ),
    },
    {
        "id": "role_reference_modbus",
        "source": "modbus_register_scaling_table",
        # deliberately free of "register / 0-100 / percent / Modbus" wording
        "content": (
            "Table 7-3 — scaling catalogue, decision rule entry. Classification "
            "by temporal signature: "
            "(1) an item with lag-1 autocorrelation above 0.8 and slow integrating "
            "ramps (coefficient of variation typically 0.10-0.20) is the settled-"
            "solids DENSITY, family = density. "
            "(2) an item with lag-1 autocorrelation below 0.5 that mean-reverts "
            "quickly around a midpoint (low coefficient of variation near 0.05) is "
            "the pump FLOW rate, family = flow. "
            "Both entries occupy the identical engineering span, therefore only the "
            "lag-1 autocorrelation distinguishes density from flow in Table 7-3."
        ),
    },
    # ---- chain C : ISA-5.1 P&ID instrument tag -> datasheet hop ---------------
    {
        "id": "bridge_pid_tag",
        "source": "plant_note_pid_loop",
        "content": (
            "P&ID loop note — leaching train, PLC data block DB4. Several "
            "controlled analog loops "
            "report to the PLC as smooth normalized 0-100 percent spans and are "
            "statistically alike on the wire. The P&ID identifies the loop tag but "
            "not the engineering range; to resolve the physical quantity and span "
            "of tag FT-2201 you must consult its instrument datasheet, which holds "
            "the range and role. The 0-100 span alone does not disambiguate it."
        ),
    },
    {
        "id": "role_reference_pid",
        "source": "instrument_datasheet_ft2201",
        # deliberately free of "0-100 / percent / normalized / P&ID / loop" wording
        "content": (
            "FT-2201 instrument datasheet, decision rule entry. Classification by "
            "temporal signature: "
            "(1) an item pinned by a cascade controller with coefficient of "
            "variation below 0.02 and lag-1 autocorrelation above 0.75 is a "
            "TEMPERATURE control point, family = temperature. "
            "(2) an item that wanders with larger coefficient of variation "
            "(typically above 0.05) and slow drift is a tank LEVEL, family = "
            "level. Both entries occupy the identical engineering span, therefore "
            "only the coefficient of variation distinguishes temperature from "
            "level in datasheet FT-2201."
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

    # Verify the two-hop structure: hop-2 must NOT be top-1 on the hop-1 query,
    # but MUST surface once the query includes the bridge pointer phrase.
    from weak_supervision import build_stat_query, compute_stats
    hop1_q = ("zinc hydrometallurgy purification process regulated process "
              "measurement, hundreds-range value, 0-100 percent smooth signal")
    print(f"\n[hop-1 query] {hop1_q[:70]}...")
    for r in col.query(query_texts=[hop1_q], n_results=3)["metadatas"][0]:
        print(f"   -> {r['source']}")

    hop2_q = f"{BRIDGE_POINTER} temporal signature rule potential vs level"
    print(f"\n[hop-2 query] {hop2_q}")
    for r in col.query(query_texts=[hop2_q], n_results=3)["metadatas"][0]:
        print(f"   -> {r['source']}")


if __name__ == "__main__":
    main()
