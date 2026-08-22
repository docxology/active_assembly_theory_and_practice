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
def step (op : Op) (state : Word64) : Word64 :=
  match op with
  | .mov value => value
  | .add value => add64 state value
  | .sub value => sub64 state value
  | .inc => inc64 state
  | .nop => state

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
  simp [run, step, inc64, add64, word64, MOD64, Nat.add_mod]

theorem neutral_nop_equivalence :
    ObservationallyEqual
      [.mov (word64 7), .add (word64 5)]
      [.mov (word64 7), .nop, .add (word64 5)] := by
  intro initial
  simp [run, step, add64, word64, MOD64, Nat.add_mod]

theorem changed_subtraction_is_not_equivalent :
    ¬ ObservationallyEqual
      [.mov (word64 5), .add (word64 3)]
      [.mov (word64 5), .sub (word64 3)] := by
  intro hypothesis
  have witness := hypothesis (word64 0)
  have values := congrArg Fin.val witness
  change (8 : Nat) = 2 at values
  omega

theorem run_nop_append (program : List Op) (initial : Word64) :
    run (program ++ [.nop]) initial = run program initial := by
  induction program generalizing initial with
  | nil => simp [run, step]
  | cons head tail inductionHypothesis =>
      simp [run, step, List.foldl_append]

end ActiveAssembly
