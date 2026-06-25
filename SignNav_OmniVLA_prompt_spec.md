# SignNav → OmniVLA Prompt Generation — Spec v1

## 0. Purpose
Spec for the module that turns a SignNav reasoning **Decision** into the **language instruction** OmniVLA acts on — the `move toward {target}` prompt, kept inside OmniVLA's training distribution so the VLA executes it reliably. **No code here — this is the spec Claude Code implements from.**

**Scope (today):** the **prompt / language channel only.** Other ways to feed an action — a relative goal pose / waypoint, trajectory tokens, or a classical local planner driving to a point — are **parked** pending alignment with Ajay and Vivek (see §7). This spec is deliberately just the prompt.

---

## 1. OmniVLA's in-distribution prompt format — VERIFIED in the repo (`NHirose/OmniVLA`)
- Every language sample, in **training and inference**, is wrapped in a fixed template, and the instruction is **lowercased**:
  `"What action should the robot take to {lang}?"`
  - refs: `prismatic/models/vlas/openvla.py:53`, `prismatic/vla/datasets/datasets.py:60,254`, `lelan_dataset.py:439`, `cast_dataset.py:115`
- **LeLaN is the only source of language prompts for OmniVLA**, and its `lang` field is built as:
  `lang = "move toward " + {object phrase}`   — `lelan_dataset.py:433`
- OmniVLA's own inference example: `lan_inst_prompt = "move toward blue trash bin"`, i.e. the model is queried with
  `"What action should the robot take to move toward blue trash bin?"`   — `inference/run_omnivla.py:568,356`
- **No-language sentinel** (when relying on pose/image goal only): the exact string `"No language instruction"`.
- **Hard length cap**: if the tokenized prompt exceeds **60 tokens**, the pipeline throws the instruction away and substitutes `"move toward XXXXX"` — `lelan_dataset.py:459`, `cast_dataset.py:127`. ⇒ instructions must be **short**.

**Net:** the safe, in-distribution instruction is
> `move toward {short, visually-grounded object/region phrase}`

Object-centric, optional single appearance/position descriptor, no long relational chains. Real LeLaN examples: *"move toward blue trash bin"*, *"go to the white and grey chair"*. Casing is irrelevant (the interface lowercases).

---

## 2. The mismatch we are bridging
OmniVLA wants *"move toward {something visible right now}"*. SignNav's reasoning produces a decision **off a directory sign** — usually a **direction** toward a goal that may be **out of view** (e.g. the sign says the elevator is left, around the corner). So we usually cannot say "move toward the elevator" — it isn't visible yet.

**Bridge:** emit a short **immediate visual subgoal** — a thing/region visible (or directly ahead) in the *current* frame that, if OmniVLA drives toward it, carries out the decision. (Drive toward "the left hallway" to execute "turn left toward the elevators".)

---

## 3. Design — two pieces
**(a) Reasoner emits the target.** The Reasoner (Qwen) already sees the scene + the sign, so it is the right place to name the visible anchor. Extend its structured output with a **`nav_target`** — a short noun phrase for the immediate thing to head toward, chosen to realize the decision. (This is the `nav_target`/`nav_prompt` field already anticipated on `Decision`.)

**(b) Prompt formatter.** A small **deterministic, model-free** function that maps `Decision → OmniVLA lang field`, enforcing §1's format and constraints. Pure and unit-testable.

(The reasoner naming a target is also reusable if other action inputs are added later — see §7.)

---

## 4. Prompt formatter — spec
**Input** (`Decision`, already produced by the reasoner):
- `action`: enum — `go_straight | turn_left | turn_right | reroute | approach_to_read | continue | stop`
- `nav_target`: short str (may be empty)
- `goal`: str (the journey goal, e.g. "Elevator")

**Output**:
- `omni_lang`: str — the **inner** instruction for OmniVLA's `lang` slot, **always of the form `"move toward {target}"`**, OR a sentinel (below).
- The OmniVLA call site wraps it: `f"What action should the robot take to {omni_lang.lower()}?"`. The formatter returns only the inner phrase; the wrapper is applied at the interface so it matches training exactly.

**Rules**
1. **Shape** — always `"move toward " + target`. Never emit a bare direction ("turn left"); phrase direction *through* a visible region ("the left hallway").
2. **Length** — `target` ≤ ~6 words; whole `omni_lang` ≤ ~8 words (keeps the wrapped prompt far under the 60-token cap). If longer, reduce to head-noun + one descriptor.
3. **Object-grounded** — `target` names a thing/region (doors, hallway, corridor, elevators, stairs, exit), optionally **one** descriptor (a color, or `left/right/ahead`). No multi-clause spatial relations.
4. **Action → target**
   - `go_straight` → the anchor straight ahead (`nav_target`, else `"the hallway ahead"`).
   - `turn_left` / `turn_right` → the visible region that way (`nav_target`, else `"the hallway on the left/right"`).
   - `reroute` → the alternative route's anchor (`nav_target`, e.g. `"the stairs on the right"`).
   - `approach_to_read` / `continue` → `"the directory sign ahead"` (else `"the hallway ahead"`). *(Or handle slow re-read creep as a non-OmniVLA forward primitive — see §8.)*
   - `stop` → **return the `STOP` sentinel / None; do NOT drive OmniVLA.** Halting is the orchestration layer's job, never a prompt.
5. **Goal visibility** — if the goal object *is* the visible anchor, use it (`"the elevators"`); else use the structural affordance toward it (`"the left hallway"`).
6. **Fallbacks** — empty/unusable `nav_target` ⇒ the deterministic per-action default in rule 4. Never emit an empty or over-long prompt.
7. **No-language mode** — when deliberately relying on pose/image-goal modalities, set the language field to the exact sentinel `"No language instruction"` (not a `move toward` phrase).

---

## 5. Reasoner output extension — spec
- Add **`nav_target: str`** to the reasoner's structured output (alongside `action` and the rationale). This is the only new field the prompt channel needs.
- Constraints the reasoner must honor for `nav_target`: a short noun phrase (≤6 words); names something **visible in the current frame or directly ahead**; consistent with `action`; ≤1 descriptor; **no** bare directions, **no** full sentences.
- **Conservative fallback:** if the reasoner can't confidently name an anchor, it leaves `nav_target` empty and the formatter falls back to the per-action default (§4 rule 4). Never block on this field.

So for this channel `Decision` carries: `action`, `nav_target`, `goal`.

Guidance to add to the reasoner's own system prompt (NOT the OmniVLA prompt):
> "After choosing the action, name in ≤6 words the single visible thing or corridor to head toward to carry it out (e.g. 'the left hallway', 'the glass doors ahead', 'the elevators'). If nothing salient is visible, answer 'the hallway ahead'."

---

## 6. Test table (`Decision → expected omni_lang`) — use directly as unit tests
| action | goal | nav_target | → omni_lang |
|---|---|---|---|
| go_straight | Elevator | the elevators ahead | `move toward the elevators ahead` |
| turn_left | Elevator | the left hallway | `move toward the left hallway` |
| turn_right | Restroom | the doors on the right | `move toward the doors on the right` |
| reroute | Elevator | the stairs on the right | `move toward the stairs on the right` |
| go_straight | Exit | *(empty)* | `move toward the hallway ahead` |
| approach_to_read | Elevator | *(empty)* | `move toward the directory sign ahead` |
| stop | — | — | **STOP sentinel — no OmniVLA call** |
| go_straight | Cafe | the brightly lit corridor on the left past the double doors | `move toward the corridor on the left` *(simplified to ≤6 words)* |

Wrapped-form check (one example): `omni_lang = "move toward the left hallway"` → OmniVLA input
`"what action should the robot take to move toward the left hallway?"`

---

## 7. Parked — other action inputs (out of scope today)
Pending alignment with **Ajay and Vivek**, these are **not** being built yet; recorded only so the prompt work stays compatible with them:
- a **relative goal pose / waypoint** (bearing + range) for a classical local planner or OmniVLA's pose-goal mode;
- **trajectory / action tokens** (Vivek's idea — encode a short path as action tokens);
- the **classical + VLA fusion** direction (Ajay: enable the VLA only for complex decisions; basic motion from a planner), for which **ReasonNav** (arXiv 2509.21189 — landmark → coordinate → classical primitives, with SLAM + RGB-D) is the closest related work.

All of these would consume the **same reasoner output** as the prompt channel (the `nav_target` the reasoner already names), so today's prompt work is not wasted if they're added. Revisit once the team has agreed on action inputs.

---

## 8. Open questions — validate against the model / dataset
1. Does OmniVLA ground **region** phrases ("the left hallway") as well as concrete **objects** ("the blue door")? LeLaN is object-centric, so regions may be mildly OOD — though OmniVLA's larger backbone is more OOD-robust. **A/B object vs. region targets** once the challenge bags land.
2. For pure corridor-following with no salient object, is `"move toward the hallway ahead"` reliable, or does that case want a different modality (parked, §7)?
3. Confirm the wrapper + `"move toward"` prefix against the **exact checkpoint** we deploy (verified on repo `main`).