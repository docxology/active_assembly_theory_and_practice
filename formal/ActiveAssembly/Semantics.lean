import ActiveAssembly.NormalizedMachine
import ActiveAssembly.IR

/-!
# A kernel-checked operational core for the assembly demonstrator

This file deliberately formalizes a small, closed-world fragment rather than
pretending to be a complete x86 semantics.  The fragment has a 64-bit
accumulator, modular addition/subtraction, immediate assignment, increment,
and a no-op.  The Python/Capstone/Unicorn pipeline supplies the machine-level
trace; this file supplies independently checked algebra for the quotient
witnesses used by the local P3/P4 protocol.

The trusted base is the Lean kernel plus the imported core/standard library.
No external solver is used by these proofs.  The companion bounded exhaustive
check may use Z3, but a Z3 result is never treated as a proof certificate.
-/

namespace ActiveAssembly

def MOD64 : Nat := 2 ^ 64

theorem mod64_pos : 0 < MOD64 := by
  simp [MOD64]

def Word64 := Fin MOD64

def word64 (value : Nat) : Word64 :=
  ⟨value % MOD64, Nat.mod_lt _ mod64_pos⟩

def add64 (left right : Word64) : Word64 :=
  word64 (left.val + right.val)

def sub64 (left right : Word64) : Word64 :=
  word64 (left.val + MOD64 - right.val)

def inc64 (value : Word64) : Word64 :=
  add64 value (word64 1)

inductive Op where
  | mov (value : Word64)
  | add (value : Word64)
  | sub (value : Word64)
  | inc
  | nop
  | cmp (value : Word64)
  | and (value : Word64)
  | or (value : Word64)
  | xor (value : Word64)
  | movImm (width : Nat) (value : Nat)
  | push (index : Fin 16)
  | pop (index : Fin 16)
  | jcc (condition : JccConditionN) (taken : Nat) (fallthrough : Nat)

/-- Accumulator-level step.  The original five families keep their algebraic
interpretation; the extended families are interpreted as follows: ``cmp`` is
exactly ``sub`` without writeback (the accumulator is unchanged), the logic
operations apply bitwise, and width-aware moves apply the Python width rules.
``push``/``pop``/``jcc`` have no accumulator effect: their register/memory/
control effects are defined by the machine-level ``stepN`` below. -/
def step (op : Op) (state : Word64) : Word64 :=
  match op with
  | .mov value => value
  | .add value => add64 state value
  | .sub value => sub64 state value
  | .inc => inc64 state
  | .nop => state
  | .cmp _ => state
  | .and value => word64 (Nat.land state.val value.val)
  | .or value => word64 (Nat.lor state.val value.val)
  | .xor value => word64 (Nat.xor state.val value.val)
  | .movImm width value =>
      match width with
      | 8 => mov8N state value
      | 16 => mov16N state value
      | 32 => write32N state value
      | _ => word64 value
  | .push _ => state
  | .pop _ => state
  | .jcc _ _ _ => state

/-- Machine-level step over the explicit bounded machine state: the new
instruction families execute with the exact flag definedness, stack bounds,
and fail-closed branch semantics of the Python reference. -/
def stepN (op : Op) (state : MachineStateN) : MachineStateN :=
  match op with
  | .mov value => withResultN state raxIndexN value state.flags
  | .add value => add64N state raxIndexN value
  | .sub value => sub64N state raxIndexN value
  | .inc => incStateN state
  | .nop => state
  | .cmp value => cmp64N state raxIndexN value
  | .and value => and64N state raxIndexN value
  | .or value => or64N state raxIndexN value
  | .xor value => xor64N state raxIndexN value
  | .movImm width value =>
      match width with
      | 8 => withResultN state raxIndexN (mov8N (state.gprs raxIndexN) value) state.flags
      | 16 =>
          withResultN state raxIndexN (mov16N (state.gprs raxIndexN) value) state.flags
      | 32 =>
          withResultN state raxIndexN (write32N (state.gprs raxIndexN) value)
            state.flags
      | _ => withResultN state raxIndexN (word64n value) state.flags
  | .push index => push64N state index
  | .pop index => pop64N state index
  | .jcc condition taken fallthrough =>
      jccStateN state condition taken fallthrough

def runN (program : List Op) (initial : MachineStateN) : MachineStateN :=
  program.foldl (fun state op => stepN op state) initial

def run (program : List Op) (initial : Word64) : Word64 :=
  program.foldl (fun state op => step op state) initial

def ObservationallyEqual (left right : List Op) : Prop :=
  ∀ initial : Word64, run left initial = run right initial

theorem inc_inc_eq_add_two (state : Word64) :
    inc64 (inc64 state) = add64 state (word64 2) := by
  apply Fin.ext
  change (((((state.val + 1) % MOD64) + 1) % MOD64) =
    (state.val + 2) % MOD64)
  rw [Nat.mod_add_mod]

theorem nop_is_identity (state : Word64) : step .nop state = state := by
  rfl

theorem constant_add_equivalence :
    ObservationallyEqual
      [.mov (word64 1), .add (word64 2)]
      [.mov (word64 1), .inc, .inc] := by
  intro initial
  rfl

theorem neutral_nop_equivalence :
    ObservationallyEqual
      [.mov (word64 7), .add (word64 5)]
      [.mov (word64 7), .nop, .add (word64 5)] := by
  intro initial
  rfl

theorem changed_subtraction_is_not_equivalent :
    ¬ ObservationallyEqual
      [.mov (word64 5), .add (word64 3)]
      [.mov (word64 5), .sub (word64 3)] := by
  intro hypothesis
  have witness := hypothesis (word64 0)
  have values := congrArg (fun value : Word64 => value.val) witness
  change (8 : Nat) = 2 at values
  omega

theorem run_nop_append (program : List Op) (initial : Word64) :
    run (program ++ [.nop]) initial = run program initial := by
  unfold run
  rw [List.foldl_append]
  rfl

/-- ``cmp`` is exactly ``sub`` without writeback: identical flags, identical
registers, memory, and program counter. -/
theorem cmp_matches_sub_flags (state : MachineStateN) (index : Fin 16)
    (value : Word64N) :
    (cmp64N state index value).flags = (sub64N state index value).flags ∧
    (cmp64N state index value).gprs = state.gprs ∧
    (cmp64N state index value).memory = state.memory ∧
    (cmp64N state index value).rip = state.rip :=
  ⟨rfl, rfl, rfl, rfl⟩

/-- Adding a value and then subtracting it restores the register file, memory,
and program counter exactly. -/
theorem add_sub_inverse (state : MachineStateN) (index : Fin 16) (value : Word64N) :
    let roundtrip := sub64N (add64N state index value) index value
    roundtrip.gprs = state.gprs ∧ roundtrip.memory = state.memory ∧
      roundtrip.rip = state.rip :=
  ⟨add_sub_inverse_gprs state index value, rfl, rfl⟩

/-- 32-bit zero-extension into the 64-bit word: the upper 32 bits are zero. -/
theorem movzx32_to_64_zero_extension (source : Word64N) :
    (movzxN source 32).val = source.val % 4294967296 ∧
    (movzxN source 32).val < 4294967296 :=
  movzx32N_zero_extension source

/-- Push into the bounded stack window followed by pop restores the pushed
register and the stack pointer. -/
theorem push_pop_roundtrip (state : MachineStateN) (index : Fin 16)
    (hsp : 8 ≤ (state.gprs rspIndexN).val)
    (hstack : state.stackLower + 8 ≤ (state.gprs rspIndexN).val ∧
      (state.gprs rspIndexN).val + 8 ≤ state.stackUpper)
    (hne : index ≠ rspIndexN) :
    (runN [.push index, .pop index] state).gprs index = state.gprs index ∧
    (runN [.push index, .pop index] state).gprs rspIndexN = state.gprs rspIndexN := by
  have hunfold : runN [.push index, .pop index] state
      = pop64N (push64N state index) index := rfl
  rw [hunfold]
  have hroundtrip := push64N_pop64N_roundtrip state index hsp hstack hne
  exact ⟨hroundtrip.1, hroundtrip.2.1⟩

/-- With a fully-defined zero flag, the taken and not-taken branch outcomes
are distinct machine states (distinct rip). -/
theorem jcc_taken_vs_not_taken_distinct (state : MachineStateN)
    (taken fallthrough : Nat) (hne : word64n taken ≠ word64n fallthrough) :
    runN [.jcc JccConditionN.e taken fallthrough] (stateWithZfN state true) ≠
      runN [.jcc JccConditionN.e taken fallthrough] (stateWithZfN state false) := by
  have hunfold : ∀ s : MachineStateN,
      runN [.jcc JccConditionN.e taken fallthrough] s
        = jccStateN s JccConditionN.e taken fallthrough := fun _ => rfl
  rw [hunfold, hunfold]
  exact jcc_zero_flag_branch_distinct state taken fallthrough hne

/-- Logic operations mark AF undefined while CF is defined and cleared,
exactly mirroring ``_set_logic_flags``. -/
theorem flag_definedness_logic_ops (state : MachineStateN) (value : Word64N) :
    (and64N state raxIndexN value).flags.defined afBitN = false ∧
    (or64N state raxIndexN value).flags.defined afBitN = false ∧
    (xor64N state raxIndexN value).flags.defined afBitN = false ∧
    (and64N state raxIndexN value).flags.cf = false ∧
    (and64N state raxIndexN value).flags.defined cfBitN = true := by
  refine ⟨logic_flags_af_undefined _, logic_flags_af_undefined _,
    logic_flags_af_undefined _, (logic_flags_cf_defined_and_clear _).2,
    (logic_flags_cf_defined_and_clear _).1⟩

/-- Arithmetic flag sets after add/sub are fully defined for every bit. -/
theorem arithmetic_flags_defined_after_add_sub (state : MachineStateN)
    (index : Fin 16) (value : Word64N) (bit : Nat) :
    (add64N state index value).flags.defined bit = true ∧
    (sub64N state index value).flags.defined bit = true := by
  have h1 : (add64N state index value).flags = addFlagsN (state.gprs index) value :=
    add64N_flags state index value
  have h2 : (sub64N state index value).flags = subFlagsN (state.gprs index) value :=
    sub64N_flags state index value
  rw [h1, h2]
  exact arithmetic_flags_fully_defined _ _ bit

end ActiveAssembly
