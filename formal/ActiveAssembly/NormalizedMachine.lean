/-!
# Normalized bounded machine state

This file is the Lean-side schema for the promotion tranche's explicit
machine state.  It intentionally models a bounded register/memory interface;
it is not a claim that the whole x86-64 ISA has been formalized.  The Python
reference interpreter and the Unicorn receipt use the corresponding fields
and the same width/flag boundary.
-/

namespace ActiveAssembly

def MOD64N : Nat := 2 ^ 64

theorem mod64n_pos : 0 < MOD64N := by
  simp [MOD64N]

def Word64N := Fin MOD64N

def word64n (value : Nat) : Word64N :=
  ⟨value % MOD64N, Nat.mod_lt _ mod64n_pos⟩

inductive MachineStatusN where
  | running
  | halted
  | unsupported
  | boundExceeded
  | exceptional
deriving Repr, DecidableEq

structure FlagsN where
  cf : Bool
  pf : Bool
  af : Bool
  zf : Bool
  sf : Bool
  of : Bool
  defined : Bool → Bool


structure MachineStateN where
  gprs : Fin 16 → Word64N
  rip : Word64N
  flags : FlagsN
  memory : Nat → UInt8
  stackLower : Nat
  stackUpper : Nat
  memoryLower : Nat
  memoryUpper : Nat
  stepIndex : Nat
  status : MachineStatusN


def write32N (_old : Word64N) (value : Nat) : Word64N :=
  word64n (value % (2 ^ 32))

theorem write32_zero_extends (old : Word64N) (value : Nat) :
    (write32N old value).val = value % (2 ^ 32) := by
  change (value % (2 ^ 32)) % MOD64N = value % (2 ^ 32)
  apply Nat.mod_eq_of_lt
  exact Nat.lt_of_lt_of_le (Nat.mod_lt _ (by omega)) (by simp [MOD64N])

def inc64N (value : Word64N) : Word64N :=
  word64n (value.val + 1)

def incStateN (state : MachineStateN) : MachineStateN :=
  { state with
    gprs := fun index => if index = 0 then inc64N (state.gprs index) else state.gprs index
    stepIndex := state.stepIndex + 1 }

theorem inc_preserves_cf (state : MachineStateN) :
    (incStateN state).flags.cf = state.flags.cf := by
  rfl

def adc64N (left right : Word64N) (carry : Bool) : Word64N :=
  word64n (left.val + right.val + if carry then 1 else 0)

theorem adc_zero_carry (left right : Word64N) :
    adc64N left right false = word64n (left.val + right.val) := by
  simp [adc64N]

def memoryInBoundsN (state : MachineStateN) (address width : Nat) : Prop :=
  state.memoryLower ≤ address ∧ address + width ≤ state.memoryUpper

theorem unsupported_is_explicit :
    MachineStatusN.unsupported ≠ MachineStatusN.running := by
  decide

end ActiveAssembly
